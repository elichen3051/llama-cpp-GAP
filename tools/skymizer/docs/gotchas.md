# Common gotchas


> [!IMPORTANT]
> **Whichever term you are NOT measuring must be held constant.** The
> mmproj is a variable like any other, not something to freeze
> unconditionally:
>
> - **Measuring LLM quantization?** Point the reference and both candidates
>   at the *same* mmproj file. A different mmproj precision (Q8_0 vs F16),
>   or even a different F16 conversion of the same projector, causes
>   vision-token drift that lands in the KLD-family metrics (`kld` /
>   `reversed_kld` / `js_kld` / `same_top_rate` / `mse_dp`) and confounds
>   the LLM-quant signal. Only `nll` (scored at the dataset's target
>   tokens) is largely mmproj-insensitive. If two *builds* ship different
>   mmprojs, the verdict is "package vs package", not "LLM quantization vs
>   LLM quantization".
> - **Measuring the projector?** Then varying the mmproj between A and B is
>   exactly the point — give both arms the *same* LLM GGUF, differ only in
>   the mmproj, and the paired delta is the projector's contribution.
>   `vlm-eval-scripts/00_env.sh` ships that configuration.
>
> Either way the reference and both candidates must share every field the
> collectors treat as identity (`mmproj` included), so a shard that changes
> one is refused — collect into a fresh `--out` dir.

- **Marker count vs `--image` count** — every workflow funnels through
  `prep_vlm_score_from_hf.py` getting `<__media__>` count right. If a
  row fails with `marker count != num_images`, the dataset row itself
  is suspect; rerun the prep script standalone with `--row N` to debug.

- **Vision token drift** — a Q8_0 mmproj can produce slightly different
  vision-token counts than an F16 one for the same image. The drift is
  reflected in KLD; it is **not** a bug (a same-model F16 double-run is
  bit-exact — see [Verifying outputs](troubleshooting.md#verifying-outputs)). What *is*
  refused is drift **within one comparison**: `llama-vlm-kld` hard-fails
  when its two sides end prefill at different positions
  (`--allow-n-past-drift` overrides), and the comparator hard-fails on an
  `n_past_actual` mismatch across dirs — every scored position would
  otherwise be conditioned on a different prefix.

- **Dataset positional indices** — `--row N` (and `--start` / `--end`)
  is the index **after** `--sort-by`, not the original dataset order.
  Use `--item-id` for stable cross-version references.

- **Decode batching moves logits** — `--n-seq-max > 1` (multi-row waves)
  runs GEMM instead of GEMV decode kernels and perturbs logits by FP
  non-associativity (measured ~1e-5/1e-6 per-token); different
  `--tf-chunk` values do the same, and so does the sliding-window KV cache
  mode (`--swa-full`; measured 2026-09-02 on gemma-4-E4B: the same spread
  as changing `--n-ubatch`, up to ~0.15 nll at isolated positions, means
  unchanged). Neither is a bug, but every dir of one comparison must share
  these knobs — the collect-meta identity guard and the comparator both
  enforce it. History: until 2026-09-02 both scorers used llama.cpp's
  low-level default (full SWA cache); since then the default is the common
  CLI's window-sized cache and `--swa-full` restores the old behaviour.
  `collect_meta.json` records `swa_full` from that date on.

- **Paired test needs ≥ 2 items** — a single item gives a zero-width
  bootstrap CI that reports spurious "significant" verdicts;
  `saved_metrics_paired_compare.py` fails closed on `n_items < 2`.
