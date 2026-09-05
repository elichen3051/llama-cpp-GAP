# On-disk formats

The VLMK (per-token metrics) layout and its lossless `.npz` form. Reader:
`kld_metrics_io.py`; writer: the C++ scorers (`skymizer-vlmk-kernel.h`).

### Metric dump format (VLMK)

Little-endian, magic `"VLMK"`. 24-byte header:

```text
offset  size  type    field        notes
─────────────────────────────────────────────────────────────
0       4     uint32  magic        0x564C4D4B ("VLMK")
4       4     uint32  version      3 (v2 = 44-byte records without ear_20/10/5;
                                    v1 = legacy 40-byte records without ear)
8       4     uint32  vocab_size   == llama_vocab_n_tokens (both models)
12      4     uint32  n_positions  number of scored answer positions
16      4     uint32  n_prefill    HF ground-truth sequential prefill length,
                                    echoed from the manifest
20      4     uint32  n_past_actual llama.cpp's OWN position count after
                                    prefill; 0 = not recorded (legacy dump)
24+     ...           n_positions × 56-byte packed records
```

Each record (v3, field order = the packed `kld_record` struct in
`skymizer-vlmk-kernel.h`):

```text
float32  kld, reversed_kld, js_kld, nll_ref, nll_cand,
         entropy_ref, entropy_cand, ear, ear_20, ear_10, ear_5
int32    target, argmax_ref, argmax_cand
```

Version history: **v1** = 40-byte records without the `ear` field; **v2**
inserts `float32 ear` between `entropy_cand` and `target` (44 bytes); **v3**
(current) inserts `float32 ear_20, ear_10, ear_5` after `ear` (56 bytes) —
EAR restricted to the reference's top-K slots, both rows renormalized on
exactly those slots (see `docs/compare.md`). Readers (`kld_metrics_io`, the
comparator) accept every version; an older dir simply lacks the newer
metrics (`saved_metrics_paired_compare` drops them from the default report
with a persisted warning, or hard-fails if `--metrics ear_20` etc. was
explicit). The collectors only WRITE v3: they preflight
the scorer binary's `--vlmk-version` before any work (a stale build is
refused with a rebuild hint). An existing v1 row is simply prior output: a
window that includes it is refused as a collision (nothing is deleted), so a
dir never mixes record versions unless an operator clears the old rows by
hand.

> [!IMPORTANT]
> `n_prefill` and `n_past_actual` are **not the same number** and are not
> expected to match. `n_prefill` is the HF ground-truth *sequential* prefill
> length (one entry per image-pad token) echoed from the manifest;
> `n_past_actual` is llama.cpp's own position count after prefill, which under
> M-RoPE counts one position per merged patch group and is much smaller.
>
> The distinction matters: **only `n_past_actual` moves when the vision-token
> budget changes.** Collect one dir at `--image-max-tokens 1024` and another
> at the `-1` default and `n_prefill` is identical on both sides while the
> images really did become different numbers of embeddings. The comparator
> hard-fails on an `n_past_actual` mismatch, which is the content-level
> backstop behind the `collect_meta.json` image-bounds guard. `0` means
> *not recorded* — every dump written before the field existed — and is
> skipped rather than compared against a real count.

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
