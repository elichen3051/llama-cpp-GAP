# On-disk formats

The VLMK (per-token metrics) layout and its lossless `.npz` form. Reader:
`lib/kld_metrics_io.py`; writer: the C++ scorers (`core/company-vlmk-kernel.h`).

### Metric dump format (VLMK)

Little-endian with numeric magic `0x564C4D4B` (the stored bytes are `4b 4d 4c 56`). The format name is VLMK. 24-byte header:

```text
offset  size  type    field        notes
─────────────────────────────────────────────────────────────
0       4     uint32  magic        0x564C4D4B
4       4     uint32  version      5 (v4 = 68-byte; v3 = 56-byte interim; v2 = 44-byte records
                                    without the EAR_K family; v1 = legacy 40-byte
                                    records without ear)
8       4     uint32  vocab_size   == llama_vocab_n_tokens (both models)
12      4     uint32  n_positions  number of scored answer positions
16      4     uint32  n_prefill    Frozen sequential first-target index,
                                    echoed from the manifest
20      4     uint32  n_past_actual Native position count after prefill, or full
                                    window length in perplexity-window mode;
                                    0 = not recorded (legacy dump)
24+     ...           n_positions x 76-byte packed records
```

Each record (v5, field order = the packed `kld_record` struct in
`core/company-vlmk-kernel.h`):

```text
float32  kld, reversed_kld, js_kld, nll_ref, nll_cand,
         entropy_ref, entropy_cand, ear,
         ear_20, ear_10, ear_5,
         ear_20_normalized, ear_10_normalized, ear_5_normalized
int32    target, argmax_ref, argmax_cand
float32  ear_64, ear_64_normalized
```

Version history: **v1** = 40-byte records without the `ear` field; **v2**
inserts `float32 ear` between `entropy_cand` and `target` (44 bytes); **v3**
(a one-build interim) inserts the three RENORMALIZED top-K EARs after `ear`
(56 bytes; read back as `ear_K_normalized`); **v4** inserts
`ear_20, ear_10, ear_5` (sum of `min(p_ref, p_cand)` over the reference's
top-K slots with full-vocab probabilities — the top-K share of `ear`) followed
by `ear_20_normalized, ear_10_normalized, ear_5_normalized` (both rows
renormalized on those K slots first) after `ear` (68 bytes). **v5** (current) appends `ear_64` and
`ear_64_normalized` after the integer IDs, preserving the complete 68-byte
v4 prefix; each record is now 76 bytes. The definitions match the earlier
EAR_K family with K=64; see [metric definitions](compare.md). Readers (`kld_metrics_io`, the comparator) accept every
version; an older dir simply lacks the newer metrics
(`saved_metrics_paired_compare` drops them from the default report with a
persisted warning, or hard-fails if `--metrics ear_64` etc. was explicit). The
collectors only WRITE v5: they preflight
the scorer binary's `--vlmk-version` before any work (a stale build is
refused with a rebuild hint). An existing v1 row is simply prior output: a
window that includes it is refused as a collision (nothing is deleted), so a
dir never mixes record versions unless an operator clears the old rows by
hand.

`n_prefill` is the first scored target's index in the frozen sequential token stream. Native rows use the saved GGUF layout, while legacy HF rows use their prepared compatibility layout. `n_past_actual` is the native position count after multimodal prefill; it can differ under multimodal RoPE. In classic PPL window mode, the complete 512-token window has already been decoded, so `n_prefill=257` and `n_past_actual=512`.

Neither field alone proves identical conditioning. Collection identity also guards image bounds and runtime, and native rows validate the saved image/prompt layout. The comparator checks recorded native positions; a legacy zero means the count was not recorded. Do not change a native dataset's image budget to hide a layout mismatch.

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
