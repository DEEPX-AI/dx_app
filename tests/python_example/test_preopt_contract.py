# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The pre-optimized output contract, checked against the NPU.

Run as a subprocess, not in-process: ``conftest.py`` installs a Mock for ``dx_engine``
during ``pytest_configure``, which is what the unit tests want and the opposite of
what a hardware test needs. ``scripts/verify_preopt_contract.py`` holds the actual
check; this file is the pytest entry point for it, and it skips rather than fails when
the pre-optimized models are not on this machine.

Why it matters: 63 of the 147 DX Model Zoo 2_5_0 additions are pre-optimized YOLO
models whose .dxnn files are not published yet, so they cannot be run individually.
They share exactly three output contracts, and dx_yolo26 ships one real model for each
-- so this is the only hardware evidence that group can have, and it covers all of it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = PROJECT_ROOT / "scripts" / "verify_preopt_contract.py"

# The models are looked up where every other model lives, PROJECT_ROOT/assets/models;
# override with DXAPP_PREOPT_MODELS_DIR when they live elsewhere.
DEFAULT_MODELS_DIR = PROJECT_ROOT / "assets" / "models"
MODELS_DIR = Path(os.environ.get("DXAPP_PREOPT_MODELS_DIR", str(DEFAULT_MODELS_DIR)))

REQUIRED = (
    "pre_optimized_yolo26-n-od.dxnn",
    "pre_optimized_yolo26n-pose.dxnn",
    "pre_optimized_yolo26n-seg.dxnn",
)


@pytest.mark.e2e
@pytest.mark.e2e_image
def test_preopt_contract_holds_on_real_hardware():
    where = ("assets/models" if MODELS_DIR == DEFAULT_MODELS_DIR else str(MODELS_DIR))
    for name in REQUIRED:
        if not (MODELS_DIR / name).is_file():
            pytest.skip(f"pre-optimized model {name} not in {where} "
                        "(or set DXAPP_PREOPT_MODELS_DIR to a directory holding it)")

    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--models-dir", str(MODELS_DIR)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "RESULT: PASS" in combined, combined
    # The three contracts the 63 unpublished variants depend on.
    for shape in ("[1, K, 6]", "[1, K, 57]", "[1, K, 38]"):
        assert shape in combined, f"{shape} was not verified:\n{combined}"
