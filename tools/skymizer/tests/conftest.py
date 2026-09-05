# Make the skymizer scripts importable from this tests/ subdirectory.
# `import paired_compare` etc. would otherwise fail because pytest only
# adds the test file's own directory to sys.path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest


@pytest.fixture(autouse=True)
def _isolate_reproducible_report_env(monkeypatch):
    """SKYMIZER_REPRODUCIBLE_REPORT flips --omit-host-metadata's default; a
    value inherited from the invoking shell must not leak into tests that
    assert the unset-default behaviour."""
    monkeypatch.delenv("SKYMIZER_REPRODUCIBLE_REPORT", raising=False)
