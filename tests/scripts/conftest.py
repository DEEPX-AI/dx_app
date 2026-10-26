"""tests/scripts is hermetic (U-74): no test configures, builds or deletes
anything in the real build/, build_*/, bin/, lib/ or include/, and none
rewrites a tracked file. build.sh runs only in build_sh_sandbox.sh's
throwaway tree. See hermetic_guard.py for what is compared."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import hermetic_guard
# TARGET's known failures as strict xfails, when --known-failures names the
# list (scripts/ci_checks.sh does); see known_failures.py.
from known_failures import (  # noqa: F401 - pytest hooks
    pytest_addoption, pytest_collection_modifyitems, pytest_configure)

ROOT = Path(__file__).resolve().parents[2]

# Scripts these tests import or run must not drop __pycache__ into the tree.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

_TRACKED = hermetic_guard.tracked_files(ROOT)


def pytest_sessionstart(session):
    problem = hermetic_guard.tracked_list_problem(ROOT, _TRACKED)
    if problem:
        pytest.exit(problem, returncode=4)


@pytest.fixture(autouse=True)
def _real_tree_untouched(request):
    before = hermetic_guard.snapshot(ROOT, _TRACKED)
    yield
    changes = hermetic_guard.diff(before, hermetic_guard.snapshot(ROOT, _TRACKED))
    if changes:
        shown = "\n  ".join(changes[:20])
        more = "" if len(changes) <= 20 else "\n  ... and {} more".format(len(changes) - 20)
        pytest.fail("{} changed the real repository (tests/scripts must be hermetic, "
                    "U-74):\n  {}{}".format(request.node.nodeid, shown, more), pytrace=False)
