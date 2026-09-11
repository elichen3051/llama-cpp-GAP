#!/usr/bin/env python3
# Regenerate tests/data/paired_compare_engine_golden.json — the whole-engine
# golden pinning compare_items across all four --ci-method paths at
# float.hex() precision, plus the rendered markdown.
#
# Run ONLY for a deliberate, reviewed semantic change to the engine; a
# refactor must never need this (the test compares exactly).
#
#     python3 tests/data/gen_paired_compare_golden.py
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))          # tests/ (fixture module)
sys.path.insert(0, str(HERE.parents[2]))          # tools/gap (engine)

from golden_engine_fixture import build_golden_payload  # noqa: E402

OUT = HERE.parent / "paired_compare_engine_golden.json"


def main() -> int:
    payload = build_golden_payload()
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    n_floats = json.dumps(payload).count("0x")
    print(f"wrote {OUT} ({n_floats} hex-pinned floats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
