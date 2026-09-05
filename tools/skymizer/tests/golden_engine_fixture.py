# Shared fixture + serialization for the ENGINE golden
# (tests/data/paired_compare_engine_golden.json).
#
# Contract: build_golden_payload() must be a pure function of this file and
# paired_compare's engine — no host data, no time, no filesystem — so the
# stored payload pins the ENTIRE compare_items result (every float at
# float.hex() precision, all four --ci-method paths, pooled ladders,
# position strata, multiplicity, TOST) plus the rendered markdown, across
# any refactor. Regenerate ONLY for a deliberate, reviewed semantic change:
#     python3 tests/data/gen_paired_compare_golden.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare.contracts import DEFAULT_CI_METHOD, DEFAULT_METRICS
from compare.engine import compare_items
from compare.render import format_comparison_table

CI_METHODS = ("t", "studentized", "bca", "percentile")
# The metric set the stored golden was generated with (before the EAR_K
# family joined DEFAULT_METRICS). The golden pins the ENGINE's math, not the
# default metric list, so it stays on this fixed tuple.
GOLDEN_METRICS = ("nll", "kld", "reversed_kld", "js_kld", "ear", "same_top_rate", "mse_dp")
assert set(GOLDEN_METRICS) <= set(DEFAULT_METRICS)
BOOTSTRAP_ITERS = 1000
SEED = 20260824
N_ITEMS = 16


def build_fixture():
    """Deterministic per-item scores + per-token columns at production-like
    magnitudes (kld ~1e-3 nats, ear ~0.98, nll ~2), with answer lengths
    spanning all three default position buckets (0-32 / 32-256 / 256+)."""
    rng = np.random.default_rng(SEED)
    keys, scores_a, scores_b, weights = [], [], [], []
    tok_a = {"kld": [], "ear": [], "dp": []}
    tok_b = {"kld": [], "ear": [], "dp": []}
    lengths = [10, 18, 25, 33, 40, 64, 90, 120, 150, 200, 240, 260, 280,
               300, 45, 75]
    for i, T in enumerate(lengths[:N_ITEMS]):
        keys.append(f"{i:03d}_item{i}")
        latent = float(rng.normal(0.0, 0.3))
        ka = np.abs(rng.normal(1.2e-3, 4e-4, T) * (1.0 + 0.2 * latent))
        ka = ka.astype(np.float32)
        kb = (ka * np.float32(1.25) + rng.normal(0, 1e-4, T).astype(np.float32))
        kb = np.abs(kb).astype(np.float32)
        ea = (1.0 - np.abs(rng.normal(0.012, 0.004, T))).astype(np.float32)
        eb = (ea - np.abs(rng.normal(0.008, 0.003, T))).astype(np.float32)
        da = rng.normal(-0.05, 0.4, T).astype(np.float32)
        db = (da + rng.normal(-0.08, 0.1, T).astype(np.float32)).astype(np.float32)
        tok_a["kld"].append(ka); tok_b["kld"].append(kb)
        tok_a["ear"].append(ea); tok_b["ear"].append(eb)
        tok_a["dp"].append(da);  tok_b["dp"].append(db)

        nll_a = 2.0 + latent + float(rng.normal(0, 0.05))
        nll_b = nll_a + 0.01 + float(rng.normal(0, 0.02))
        base = {
            "nll": nll_a,
            "kld": float(ka.mean(dtype=np.float64)),
            "reversed_kld": float(ka.mean(dtype=np.float64)) * 1.1,
            "js_kld": float(ka.mean(dtype=np.float64)) * 0.5,
            "ear": float(ea.mean(dtype=np.float64)),
            "same_top_rate": 0.995 - abs(latent) * 0.01,
            "mse_dp": float((da.astype(np.float64) ** 2).mean()),
            "mean_dp": float(da.mean(dtype=np.float64)),
            "nll_ref": nll_a - 0.02,
            "entropy": 1.8 + latent * 0.1,
        }
        cand = dict(base)
        cand.update({
            "nll": nll_b,
            "kld": float(kb.mean(dtype=np.float64)),
            "reversed_kld": float(kb.mean(dtype=np.float64)) * 1.1,
            "js_kld": float(kb.mean(dtype=np.float64)) * 0.5,
            "ear": float(eb.mean(dtype=np.float64)),
            "same_top_rate": base["same_top_rate"] - 0.002,
            "mse_dp": float((db.astype(np.float64) ** 2).mean()),
            "mean_dp": float(db.mean(dtype=np.float64)),
            "entropy": base["entropy"] + 0.01,
        })
        scores_a.append(base)
        scores_b.append(cand)
        weights.append(T)
    return keys, scores_a, scores_b, weights, tok_a, tok_b


def hexify(obj):
    """Floats -> float.hex() strings, recursively: exact bit-level pinning
    that survives JSON round-trips. Bools stay bools (checked before int)."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return float(obj).hex()
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: hexify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [hexify(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.floating):
        return float(obj).hex()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"hexify: unsupported type {type(obj)!r}")


def build_golden_payload():
    keys, sa, sb, w, ta, tb = build_fixture()
    payload = {"fixture": {"n_items": N_ITEMS, "seed": SEED,
                           "bootstrap_iters": BOOTSTRAP_ITERS,
                           "weights": list(w)},
               "results": {}, "markdown": {}}
    for method in CI_METHODS:
        res = compare_items(
            sa, sb, w, metrics=list(GOLDEN_METRICS),
            confidence_level=0.95, bootstrap_iters=BOOTSTRAP_ITERS,
            seed=SEED, model_a_label="A", model_b_label="B",
            ci_method=method,
            equivalence_margin=2e-4,
            token_metrics_a=ta, token_metrics_b=tb, item_keys=keys)
        payload["results"][method] = hexify(res)
        if method == DEFAULT_CI_METHOD:
            payload["markdown"]["default"] = format_comparison_table(
                res, reference_label="F16")
    return payload
