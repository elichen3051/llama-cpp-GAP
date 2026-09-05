"""The one VLM-only piece of collector state: the vision-budget reporter
that compares mtmd's per-image token bounds against the dataset's own
(used by collect_logits.py and collect_kld.py). Moved verbatim out of the
former collect_llm_common.py, where a vision-only class had no business.
"""

import sys


class VisionBudgetReporter:
    """Compare the vision-token budget mtmd will actually use against the one
    the ground-truth answers were generated under, and say so ONCE per run.

    prep_vlm_score_from_hf.derive_image_token_limits inverts the HF image
    processor config the dataset's answers came from into mtmd-equivalent
    per-image bounds and writes them into every row's meta.json -- and
    nothing read them. Meanwhile the collectors forward the global
    --image-min-tokens/--image-max-tokens flags, whose -1 default lets mtmd
    take its bounds from the mmproj GGUF's own metadata. So llama.cpp's
    vision budget came from the GGUF, the ground truth came from the HF
    config, and no check compared the two.

    A warning, not a refusal: -1 is the shipped default, every collection so
    far used it, and the paired comparison still cancels the preprocessing as
    long as BOTH sides use the same budget (which the compare-time
    image_min_tokens/image_max_tokens guard enforces). What it costs is the
    "conditioned exactly like the ground truth" claim, and that has to be
    visible rather than assumed."""

    def __init__(self, cli_min: int, cli_max: int):
        self.cli = (int(cli_min), int(cli_max))
        self._seen: set[tuple[int, int]] = set()
        self.mismatches: list[str] = []

    def check(self, meta: dict) -> None:
        limits = (meta or {}).get("image_token_limits")
        if not isinstance(limits, dict):
            return
        derived = (int(limits.get("image_min_tokens", -1)),
                   int(limits.get("image_max_tokens", -1)))
        if derived in self._seen:
            return
        self._seen.add(derived)
        if derived == self.cli:
            return
        cli_note = ("(-1 = mtmd's own default for this projector)"
                    if -1 in self.cli else "")
        msg = (f"vision-token budget differs from the dataset's own: this run "
               f"uses --image-min-tokens {self.cli[0]} --image-max-tokens "
               f"{self.cli[1]} {cli_note}, while the ground-truth answers were "
               f"generated with min={derived[0]} max={derived[1]} (derived "
               "from the row's HF image_processor_config). Pass those values "
               "explicitly to use the dataset's own budget; that reproduces "
               "the ground-truth conditioning only for families whose HF "
               "resize mtmd matches (see the family notes in prep "
               "derive_image_token_limits).")
        self.mismatches.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
