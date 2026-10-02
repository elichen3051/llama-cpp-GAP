"""Hermetic inputs and native processes for the article contract tests."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from lib.reference_dataset import canonical_json


GAP = Path(__file__).resolve().parents[1]
REPO = GAP.parents[1]
SCHEMA = "company-article-corpus-v1"
VOCABULARY = {
    "scheme": "llama-vocabulary-sha256-v1", "size": 512, "type": 2,
    "mapping": "a" * 64, "attributes": "b" * 64,
}
RUNTIME = {
    "n_ctx": 32768, "n_batch": 2048, "n_ubatch": 512, "tf_chunk": 2048,
    "n_threads": 8, "metric_threads": 16, "n_gpu_layers": -2,
}
CHECKS = ("manifest", "metrics_files", "lengths", "finite", "targets", "runtime", "reference_parity", "zero_kld")
MODELS = (
    "gemma-4-31b-it", "gemma-4-e4b-it", "glm-4.6v-flash", "internvl3.5-30b-a3b",
    "kimi-vl-a3b-instruct", "kimi-vl-a3b-thinking-2506", "muse-glimmer-30b", "qwen3.5-4b", "qwen3.6-35b-a3b",
)
ARTICLES = (b"First \\n <special>\r\n\n", "Second caf\u00e9\n".encode())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest_json(value):
    return sha256(canonical_json(value).encode("utf-8"))


def digest_tokens(tokens):
    return sha256(struct.pack(f"<{len(tokens)}i", *tokens))


def runtime_flags(runtime=None):
    return [part for key, value in (runtime or RUNTIME).items() for part in ("--" + key.replace("_", "-"), str(value))]


def offline_env(tmp_path):
    return {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "hf-cache"), "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    }


def write_executable(path, body):
    # Resolve the interpreter from PATH; fixtures never embed a machine path.
    path.write_text(f"#!{shutil.which('env')} python3\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def require_cli(name):
    path = GAP / "cli" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing implementation: cli/{name}.py")
    return path


def run_cli(name, args, tmp_path, env=None):
    return subprocess.run(
        [sys.executable, str(require_cli(name)), *map(str, args)], cwd=GAP,
        env=env or offline_env(tmp_path), capture_output=True, text=True, timeout=30,
    )


def make_prepare_inputs(tmp_path, mode="bos", articles=ARTICLES):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = b"".join(articles)
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(raw)
    entries = []
    cursor = 0
    for i, article in enumerate(articles):
        entries.append({"id": f"source-{i}", "title": f"Title {i}", "byte_start": cursor, "byte_end_exclusive": cursor + len(article)})
        cursor += len(article)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"corpus_sha256": sha256(raw), "articles": entries}, indent=2) + "\n")
    ref = tmp_path / "ref.gguf"
    ref.write_bytes(b"fake model")
    tokenizer = write_executable(tmp_path / "llama-tokenize", f"MODE = {mode!r}\n" + r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
raw = sys.stdin.buffer.read()
base = Path(__file__).parent
with (base / "native-calls.jsonl").open("a") as log:
    log.write(json.dumps({"tool": "tokenize", "args": args, "hex": raw.hex(), "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
print("tokenizer diagnostic", file=sys.stderr)
if MODE == "fail-tokenize":
    sys.exit(7)
tokens = [byte + 16 for byte in raw]
if "--no-bos" not in args:
    if MODE == "bos" or MODE == "eos" or MODE == "mixed" and raw.startswith(b"First"):
        tokens.insert(0, 2)
    if MODE == "eos":
        tokens.append(3)
print(json.dumps(tokens))
''')
    scorer = write_executable(tmp_path / "llama-llm-kld", f"MODE = {mode!r}\nVOCABULARY = {VOCABULARY!r}\n" + r'''
import json, os, sys
from pathlib import Path
with (Path(__file__).parent / "native-calls.jsonl").open("a") as log:
    log.write(json.dumps({"tool": "vocab", "args": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
print("vocabulary diagnostic", file=sys.stderr)
assert len(sys.argv) == 3 and sys.argv[1] == "--vocab-identity"
if MODE == "fail-vocab":
    sys.exit(9)
print(json.dumps(VOCABULARY))
''')
    return {"corpus": corpus, "article_index": index, "ref_model": ref, "llama_tokenize": tokenizer, "llama_llm_kld": scorer}


def prepare_args(inputs, out, name="wikitext-2-test-article"):
    flags = [part for key, value in inputs.items() for part in ("--" + key.replace("_", "-"), str(value))]
    return [*flags, "--corpus-name", name, "--out", str(out)]


def prepared_articles(tmp_path, mode="bos", articles=ARTICLES):
    require_cli("prepare_article_corpus")
    inputs = make_prepare_inputs(tmp_path / "inputs", mode, articles)
    out = tmp_path / "prepared"
    env = {**offline_env(tmp_path), "CUDA_VISIBLE_DEVICES": "fixture-visible-device"}
    result = run_cli("prepare_article_corpus", prepare_args(inputs, out), tmp_path, env)
    assert result.returncode == 0, result.stderr
    from datasets import load_from_disk
    rows = list(load_from_disk(str(out / "dataset")))
    return out, json.loads((out / "corpus_articles.json").read_text()), rows, inputs


def article_module():
    return importlib.import_module("lib.article_corpus")
