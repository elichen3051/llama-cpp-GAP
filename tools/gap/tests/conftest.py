# Make the GAP scripts importable from this tests/ subdirectory.
# `import paired_compare` etc. would otherwise fail because pytest only
# adds the test file's own directory to sys.path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest


@pytest.fixture(autouse=True)
def _isolate_reproducible_report_env(monkeypatch):
    """COMPANY_REPRODUCIBLE_REPORT flips --omit-host-metadata's default; a
    value inherited from the invoking shell must not leak into tests that
    assert the unset-default behaviour."""
    monkeypatch.delenv("COMPANY_REPRODUCIBLE_REPORT", raising=False)


def pytest_addoption(parser):
    parser.addoption("--checks-csv", metavar="PATH", help="Write one check,passed,detail CSV row per executed test")


def pytest_configure(config):
    path = config.getoption("--checks-csv")
    if path:
        config.pluginmanager.register(_ChecksCSV(Path(path)), "gap-checks-csv")


class _ChecksCSV:
    def __init__(self, path):
        self.path = path
        self.reports = {}

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, []).append(report)

    def pytest_sessionfinish(self, session, exitstatus):
        import csv

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["check", "passed", "detail"])
            for nodeid, reports in self.reports.items():
                problem = next((r for r in reports if r.failed), None)
                if problem is None:
                    problem = next((r for r in reports if r.skipped), None)
                passed = problem is None and any(r.when == "call" and r.passed for r in reports)
                detail = "passed" if passed else "test did not complete"
                if problem is not None:
                    crash = getattr(problem.longrepr, "reprcrash", None)
                    reason = crash.message if crash else str(problem.longrepr)
                    detail = f"{problem.when} {problem.outcome}: {' '.join(reason.split())}"[:300]
                writer.writerow([nodeid, str(passed).lower(), detail])
