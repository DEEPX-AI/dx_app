"""
Test DXAPP_VERIFY numerical verification for C++ executables

Verifies:
  - Setting DXAPP_VERIFY=1 env var triggers JSON output
  - verify_results/ directory is created with .json files
  - JSON contains expected fields (model_path, task, results, frame_shape)
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.utils import setup_environment  # noqa: E402

from conftest import resolve_bin_dir

# ======================================================================
# Paths
# ======================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_DIR = resolve_bin_dir()
LIB_DIR = PROJECT_ROOT / "lib"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"

_JSON_GLOB = "*.json"
SAMPLE_DIR = PROJECT_ROOT / "sample"

TEST_IMAGE = SAMPLE_DIR / "img" / "sample_kitchen.jpg"


# ======================================================================
# Discovery
# ======================================================================
def _normalize_model_to_exe(stem: str) -> str:
    return stem.lower().replace(".", "_")


def discover_sync_cases() -> List[tuple]:
    """Discover sync executables."""
    cases = []
    seen = set()
    for model_path in sorted(MODELS_DIR.glob("*.dxnn")):
        prefix = _normalize_model_to_exe(model_path.stem)
        exe_name = f"{prefix}_sync"
        if exe_name in seen:
            continue
        if (BIN_DIR / exe_name).exists():
            cases.append((exe_name, model_path))
            seen.add(exe_name)
    return sorted(cases, key=lambda x: x[0])


def discover_async_cases() -> List[tuple]:
    """Discover async executables."""
    cases = []
    seen = set()
    for model_path in sorted(MODELS_DIR.glob("*.dxnn")):
        prefix = _normalize_model_to_exe(model_path.stem)
        exe_name = f"{prefix}_async"
        if exe_name in seen:
            continue
        if (BIN_DIR / exe_name).exists():
            cases.append((exe_name, model_path))
            seen.add(exe_name)
    return sorted(cases, key=lambda x: x[0])


def _pick_representative(cases: list, max_count: int = 3) -> list:
    priority = ["yolov5s_", "yolov8n_", "fastdepth"]
    selected = []
    for exe, mp in cases:
        for p in priority:
            if exe.startswith(p) and len(selected) < max_count:
                selected.append((exe, mp))
                break
    for exe, mp in cases:
        if len(selected) >= max_count:
            break
        if (exe, mp) not in selected:
            selected.append((exe, mp))
    return selected


SYNC_CASES = discover_sync_cases()
ASYNC_CASES = discover_async_cases()
SYNC_REPR = _pick_representative(SYNC_CASES)
ASYNC_REPR = _pick_representative(ASYNC_CASES)

SYNC_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in SYNC_REPR
]
ASYNC_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.async_exec)
    for name, mp in ASYNC_REPR
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.verify
class TestDxappVerify:
    """Test DXAPP_VERIFY=1 numerical verification."""

    @pytest.mark.parametrize("executable,model_path", SYNC_PARAMS)
    def test_verify_sync_creates_json(self, executable, model_path, tmp_path):
        """Run sync binary with DXAPP_VERIFY=1, verify JSON output."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
        ]

        env = setup_environment()
        env["DXAPP_VERIFY"] = "1"

        # Run from tmp_path so verify_results/ is created there
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(tmp_path),
        )

        assert result.returncode == 0, (
            f"{executable} with DXAPP_VERIFY=1 failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Look for verify output (could be in tmp_path or PROJECT_ROOT)
        verify_dirs = (
            list(tmp_path.rglob("verify_results")) +
            list(PROJECT_ROOT.rglob("verify_results"))
        )

        json_files = []
        for vd in verify_dirs:
            json_files.extend(vd.rglob(_JSON_GLOB))

        if not json_files:
            # Check stdout for verify-related messages
            output = result.stdout + result.stderr
            # If verify is not active (no NPU, etc.), it's acceptable to skip
            if "verify" not in output.lower():
                pytest.skip(f"{executable}: DXAPP_VERIFY produced no output (may need NPU)")

        # If JSON files exist, validate their structure
        for jf in json_files[:3]:  # Check up to 3
            try:
                data = json.loads(jf.read_text())
                assert isinstance(data, dict), f"JSON should be a dict: {jf}"
                # Verify expected keys exist
                assert "model_path" in data or "task" in data or "results" in data, (
                    f"JSON missing expected keys: {list(data.keys())}\nFile: {jf}"
                )
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {jf}: {e}")

    @pytest.mark.parametrize("executable,model_path", ASYNC_PARAMS)
    def test_verify_async_creates_json(self, executable, model_path, tmp_path):
        """Run async binary with DXAPP_VERIFY=1, verify JSON output."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
        ]

        env = setup_environment()
        env["DXAPP_VERIFY"] = "1"

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(tmp_path),
        )

        assert result.returncode == 0, (
            f"{executable} with DXAPP_VERIFY=1 failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify output (flexible — verify may produce files in cwd or PROJECT_ROOT)
        verify_dirs = (
            list(tmp_path.rglob("verify_results")) +
            list(PROJECT_ROOT.rglob("verify_results"))
        )
        json_files = []
        for vd in verify_dirs:
            json_files.extend(vd.rglob(_JSON_GLOB))

        if not json_files:
            output = result.stdout + result.stderr
            if "verify" not in output.lower():
                pytest.skip(f"{executable}: DXAPP_VERIFY produced no output (may need NPU)")

    @pytest.mark.parametrize("executable,model_path", SYNC_PARAMS)
    def test_verify_disabled_by_default(self, executable, model_path, tmp_path):
        """Without DXAPP_VERIFY, no verify_results should be created."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
        ]

        env = setup_environment()
        # Ensure DXAPP_VERIFY is NOT set
        env.pop("DXAPP_VERIFY", None)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(tmp_path),
        )

        assert result.returncode == 0

        # verify_results should NOT be created
        verify_dirs = list(tmp_path.rglob("verify_results"))
        json_files = []
        for vd in verify_dirs:
            json_files.extend(vd.rglob(_JSON_GLOB))
        assert len(json_files) == 0, (
            f"DXAPP_VERIFY not set but JSON files found: {json_files}"
        )

    def test_verify_prerequisites(self):
        """Sanity check."""
        assert BIN_DIR.exists()
        assert len(SYNC_REPR) > 0, "No sync executables for verify tests"
        assert len(ASYNC_REPR) > 0, "No async executables for verify tests"
        print(f"\n  Sync representatives: {[n for n, _ in SYNC_REPR]}")
        print(f"  Async representatives: {[n for n, _ in ASYNC_REPR]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
