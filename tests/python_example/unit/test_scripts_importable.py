"""Every ``scripts/*.py`` must import cleanly.

``scripts/verify_examples.py`` shipped importing ``scripts.audit_models`` and
``scripts.model_audit.*`` -- six modules that never existed in this tree. It was
the tail end of a plan whose first chunk was never landed, and because nothing
invokes it (no workflow, Makefile, run_tc.sh path, req_test, or doc), the broken
import sat there unnoticed until someone ran the file by hand.

A missing top-level import is a pure source fact: it needs no NPU, no models,
and no build, so it can be caught the moment the file lands.

Each script is imported in its own subprocess. Importing executes module-level
code, so an in-process import would leak ``sys.modules`` / ``sys.path`` state
between scripts and let one script's imports satisfy the next one's -- exactly
the failure mode this test exists to catch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Executed in the child: import the file by path with scripts/ on sys.path, so a
# sibling import (``from model_audit import ...``) resolves the same way it does
# when the script is run directly.
_CHILD = """
import importlib.util, sys
sys.path.insert(0, {scripts_dir!r})
spec = importlib.util.spec_from_file_location("_under_test", {script!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""


def _scripts() -> list[Path]:
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if not p.name.startswith("_"))


def test_scripts_dir_is_not_empty():
    """Guard against the parametrised test silently passing on an empty list."""
    assert _scripts(), f"no scripts/*.py found under {SCRIPTS_DIR}"


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_script_imports_cleanly(script: Path):
    result = subprocess.run(
        [sys.executable, "-c", _CHILD.format(scripts_dir=str(SCRIPTS_DIR), script=str(script))],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    # A module that calls sys.exit() at import time (argparse at module scope)
    # still proves its imports resolved, so only a non-SystemExit traceback fails.
    if result.returncode != 0 and "SystemExit" not in result.stderr:
        pytest.fail(
            f"{script.relative_to(PROJECT_ROOT)} failed to import:\n{result.stderr.strip()}"
        )
