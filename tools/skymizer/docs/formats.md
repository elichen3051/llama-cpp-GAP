# On-disk formats

The VLMK (per-token metrics) layout and its lossless `.npz` form. Reader:
`kld_metrics_io.py`; writer: the C++ scorers (`skymizer-vlmk-kernel.h`).

### Metric dump format (VLMK)

Little-endian, magic `"VLMK"`. 24-byte header:

```text
offset  size  type    field        notes
─────────────────────────────────────────────────────────────
0       4     uint32  magic        0x564C4D4B ("VLMK")
4       4     uint32  version      2 (v1 = legacy 40-byte records without ear)
8       4     uint32  vocab_size   == llama_vocab_n_tokens (both models)
12      4     uint32  n_positions  number of scored answer positions
16      4     uint32  n_prefill    HF ground-truth sequential prefill length,
                                    echoed from the manifest
20      4     uint32  n_past_actual llama.cpp's OWN position count after
                                    prefill; 0 = not recorded (legacy dump)
24+     ...           n_positions × 44-byte packed records
```

Each record (v2, field order = the packed `kld_record` struct in
`skymizer-vlmk-kernel.h`):

```text
float32  kld, reversed_kld, js_kld, nll_ref, nll_cand,
         entropy_ref, entropy_cand, ear
int32    target, argmax_ref, argmax_cand
```

Version history: **v1** = 40-byte records without the `ear` field; **v2**
(current) inserts `float32 ear` between `entropy_cand` and `target`
(44 bytes). Readers (`kld_metrics_io`, the comparator) accept both
versions; a v1 dir simply has no `ear` metric (`saved_metrics_paired_compare`
drops `ear` from the default report with a persisted warning, or hard-fails if
`--metrics ear` was explicit). The collectors only WRITE v2: they preflight
the scorer binary's `--vlmk-version` before any work (a stale build is
refused with a rebuild hint). An existing v1 row is simply prior output: a
window that includes it is refused as a collision (nothing is deleted), so a
dir never mixes record versions unless an operator clears the old rows by
hand.

> [!IMPORTANT]
> `n_prefill` is the saved sequential reference-prefix length. `n_past_actual` is the decoder position after prefill. For aligned Gemma/Kimi inputs they match. For Qwen/GLM M-RoPE, each image advances positions by the longest merged-grid side, so `n_past_actual` is smaller and cannot alone detect a change to the shorter grid side. The aligned scorer checks each image grid and all reference-prefix IDs before evaluation; the collector also checks `n_past_expected`. A zero `n_past_actual` means an old dump did not record it.

Aligned collections additionally retain `metrics/NNN_ITEM.preprocess.json`, copied from the prepared row's metadata. It records the generation config, interpolation backend/version, original and final image sizes, merged grids, and source/output RGB hashes. `collect_meta.json.image_preprocessing` prevents mixing legacy preprocessing or different backends/package versions in a comparison. The VLMK binary format remains version 2.

Collections made with `--no-image-preprocessing` instead record `image_preprocessing.resize_backend="native"`. Their sidecars retain the original RGB sizes and hashes, without an HF grid or `n_past_expected`. Native and aligned identities cannot be mixed in one collection or paired comparison, and neither is silently equated to historical metadata with no identity.

### `.npz` form

`kld_metrics_io.load_kld_metrics(path)` reads `.bin` or `.npz`;
`collect_kld.py` always converts to `.npz` (tmp + fsync + atomic rename)
and deletes the `.bin`. The mapping is lossless and 1:1: the header fields
are stored as 0-d uint32 arrays (`vocab`, `npos`, `n_prefill`, `version`,
plus `n_past_actual` when recorded) and each record column as its own 1-d
array with its on-disk dtype, so ad-hoc analysis needs nothing but numpy:

```python
import numpy as np
with np.load("tmp/kld-q4km/metrics/000_test_X.npz") as z:
    kld = z["kld"]                                # (npos,) float32
    same_top = z["argmax_ref"] == z["argmax_cand"]
```
