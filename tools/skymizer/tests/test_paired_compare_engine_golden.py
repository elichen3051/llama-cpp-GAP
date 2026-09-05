# The whole-engine golden: compare_items' ENTIRE result dict — every float
# at float.hex() precision — for all four --ci-method constructions, plus
# the rendered markdown, against tests/data/paired_compare_engine_golden.json.
#
# This is the refactor oracle the old golden could not be: that one pins
# 2 fields (estimate, ci) of the percentile path only, at 1e-12 tolerance,
# while production defaults to studentized. Here a single reordered
# reduction, RNG draw, Holm tie-break or renderer cell moves SOME hex digit
# and fails exactly.
#
# Regenerate only for a deliberate, reviewed semantic change:
#     python3 tests/data/gen_paired_compare_golden.py
import json
from pathlib import Path

from golden_engine_fixture import CI_METHODS, build_golden_payload

GOLDEN = Path(__file__).parent / "data" / "paired_compare_engine_golden.json"


def test_engine_golden_matches_bit_for_bit():
    stored = json.loads(GOLDEN.read_text(encoding="utf-8"))
    live = build_golden_payload()
    assert set(live["results"]) == set(CI_METHODS)
    for method in CI_METHODS:
        assert live["results"][method] == stored["results"][method], (
            f"ci_method={method}: engine output diverged from the golden "
            "(bit-level). If this is a DELIBERATE semantic change, "
            "regenerate via tests/data/gen_paired_compare_golden.py and "
            "review the diff; a refactor must never trip this.")
    assert live["markdown"] == stored["markdown"]
    assert live["fixture"] == stored["fixture"]
