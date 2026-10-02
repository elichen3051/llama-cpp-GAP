"""Article datasets use the collector's existing generic scoring path."""

import csv
import json
import sys

import numpy as np
import pytest

from article_fakes import RUNTIME, VOCABULARY, offline_env, prepared_articles, runtime_flags, write_executable
from fakes import EXECUTION_IDENTITY, make_records, write_vlmk
from lib.kld_metrics_io import VLMK_VERSION, load_kld_metrics


def setup_collector(tmp_path, monkeypatch, rows, prepared, ubatch=512, allow=False):
    import cli.collect_llm_kld as collector

    for key, value in offline_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(collector, "build_collect_provenance", lambda *args: {"execution_identity": EXECUTION_IDENTITY})
    ref, cand = tmp_path / "ref.gguf", tmp_path / "cand.gguf"
    ref.write_bytes(b"reference fixture")
    cand.write_bytes(b"candidate fixture")
    fixtures = tmp_path / "scorer-fixtures"
    fixtures.mkdir()
    for i, row in enumerate(rows):
        records = make_records(npos=len(row["input_ids"]) - 1, vocab=VOCABULARY["size"], seed=i)
        records["target"] = row["input_ids"][1:]
        write_vlmk(fixtures / f"{i}.bin", records, vocab=VOCABULARY["size"], n_prefill=1, n_past_actual=1)
    (fixtures / "tokens.json").write_text(json.dumps([row["input_ids"] for row in rows]))
    scorer = write_executable(tmp_path / "fake-scorer", f"VERSION = {VLMK_VERSION}\n" + r'''
import json, struct, sys
from pathlib import Path
base = Path(__file__).parent
if sys.argv[1:] == ["--vlmk-version"]:
    print(VERSION)
    sys.exit(0)
(base / "scorer-argv.json").write_text(json.dumps(sys.argv[1:]))
expected = json.loads((base / "scorer-fixtures/tokens.json").read_text())
manifest = Path(sys.argv[sys.argv.index("--manifest") + 1])
entries = [json.loads(line) for line in manifest.read_text().splitlines()]
assert len(entries) == len(expected)
for i, entry in enumerate(entries):
    assert entry["n_prefill"] == 1
    data = Path(entry["tokens_in"]).read_bytes()
    assert list(struct.unpack(f"<{len(data) // 4}i", data)) == expected[i]
    output = Path(entry["output_metrics"])
    output.write_bytes((base / "scorer-fixtures" / f"{i}.bin").read_bytes())
    print("DONE output_metrics=" + str(output) + " wall_s=0.5", flush=True)
''')
    out = tmp_path / "collection"
    argv = [
        "collect_llm_kld.py", "--ref-model", str(ref), "--cand-model", str(cand),
        "--dataset", str(prepared / "dataset"), "--subset", "", "--out", str(out),
        "--llama-llm-kld", str(scorer), *runtime_flags({**RUNTIME, "n_ubatch": ubatch}),
        "--flash-attn", "--start", "0", "--end", "-1", "--num-eval-tokens", "-1",
    ]
    if allow:
        argv.append("--allow-vocab-attr-mismatch")
    monkeypatch.setattr(sys, "argv", argv)
    return collector, out


@pytest.mark.parametrize("mode,ubatch,allow", [("bos", 2048, True), ("no-bos", 512, False)])
def test_article_collection_generic_manifest_metrics_and_runtime(tmp_path, monkeypatch, mode, ubatch, allow):
    prepared, _, rows, _ = prepared_articles(tmp_path, mode)
    collector, out = setup_collector(tmp_path, monkeypatch, rows, prepared, ubatch, allow)
    collector.main()
    with (out / "manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    assert len(manifest) == len(rows)
    expected_files = set()
    for i, (row, record) in enumerate(zip(rows, manifest)):
        assert record["row_idx"] == str(i)
        assert record["item_id"] == row["id"]
        assert record["status"] == "OK"
        assert int(record["n_prefill"]) == 1
        assert int(record["n_answer"]) == int(record["n_eval"]) == len(row["input_ids"]) - 1
        name = f"{i:03d}_{row['id']}.npz"
        expected_files.add(name)
        metrics, header = load_kld_metrics(out / "metrics" / name)
        assert header["n_prefill"] == 1
        assert header["npos"] == len(row["input_ids"]) - 1
        np.testing.assert_array_equal(metrics["target"], row["input_ids"][1:])
    assert {path.name for path in (out / "metrics").iterdir()} == expected_files
    meta = json.loads((out / "collect_meta.json").read_text())
    expected = {
        **RUNTIME, "n_ubatch": ubatch, "kind": "llm_kld_metrics", "perplexity_window": False,
        "num_eval_tokens": -1, "flash_attn": "enabled", "swa_full": False, "allow_vocab_attr_mismatch": allow,
    }
    assert {key: meta[key] for key in expected} == expected
    native = json.loads((tmp_path / "scorer-argv.json").read_text())
    flags = {"-c": 32768, "-b": 2048, "-ub": ubatch, "--tf-chunk": 2048, "-t": 8, "--metric-threads": 12, "-ngl": -2, "--num-eval-tokens": -1}
    for flag, value in flags.items():
        assert native.count(flag) == 1
        assert native[native.index(flag) + 1] == str(value)
    assert "--flash-attn" in native
    assert "--swa-full" not in native and "--perplexity-window" not in native
    assert ("--allow-vocab-attr-mismatch" in native) is allow


def test_corpus_protocol_still_requires_perplexity_window(tmp_path, monkeypatch):
    prepared, _, rows, _ = prepared_articles(tmp_path)
    from datasets import Dataset
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    altered = [{**row, "corpus_protocol": row["article_protocol"]} for row in rows]
    Dataset.from_list(altered).save_to_disk(str(guarded / "dataset"))
    collector, _ = setup_collector(tmp_path, monkeypatch, altered, guarded)
    with pytest.raises(ValueError, match="--perplexity-window"):
        collector.main()
    assert not (tmp_path / "scorer-argv.json").exists()
