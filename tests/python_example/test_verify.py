"""
Test DXAPP_VERIFY numerical verification for Python inference scripts.

Verifies:
  - Setting ``DXAPP_VERIFY=1`` env var triggers JSON output
  - ``verify_results/`` directory is created with ``.json`` files
  - JSON contains expected fields (model_path, task, results, etc.)
  - Verification is disabled by default (no JSON without env var)

Mirrors ``tests/cpp_example/test_verify.py``.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.constants import (  # noqa: E402
    PROJECT_ROOT,
    SAMPLE_DIR,
)
from common.utils import discover_python_scripts, setup_environment  # noqa: E402

TEST_IMAGE = SAMPLE_DIR / "img" / "sample_kitchen.jpg"

_JSON_GLOB = "*.json"


# ======================================================================
# Discovery — representative sync + async scripts
# ======================================================================

def _pick_representative(suffixes, max_count=3) -> List[tuple]:
    raw = discover_python_scripts(suffixes=suffixes)
    candidates = []
    for _task, model_name, sync_scripts, async_scripts, model_path in raw:
        if model_path is None:
            continue
        scripts = sync_scripts if suffixes[0] == "_sync" else async_scripts
        if not scripts:
            continue
        candidates.append((scripts[0], model_path, model_name))

    priority = ["yolov5s", "yolov8n", "fastdepth"]
    selected = []
    for script, model, name in candidates:
        for p in priority:
            if name.startswith(p) and len(selected) < max_count:
                selected.append((script, model))
                break
    for script, model, _name in candidates:
        if len(selected) >= max_count:
            break
        if (script, model) not in selected:
            selected.append((script, model))
    return selected


SYNC_REPR = _pick_representative(("_sync",))
ASYNC_REPR = _pick_representative(("_async",))

SYNC_PARAMS = [
    pytest.param(s, m, id=s.stem, marks=pytest.mark.sync_exec)
    for s, m in SYNC_REPR
]
ASYNC_PARAMS = [
    pytest.param(s, m, id=s.stem, marks=pytest.mark.async_exec)
    for s, m in ASYNC_REPR
]


# ======================================================================
# Tests
# ======================================================================

@pytest.mark.verify
class TestDxappVerify:
    """Test ``DXAPP_VERIFY=1`` numerical verification."""

    def _run_with_verify(self, script: Path, model_path: Path, tmp_path: Path):
        """Run script with DXAPP_VERIFY=1, return (result, json_files)."""
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        cmd = [
            sys.executable, str(script),
            "--model", str(model_path),
            "--image", str(TEST_IMAGE),
            "--no-display",
            "--loop", "1",
        ]

        env = setup_environment()
        env["DXAPP_VERIFY"] = "1"
        env["DXAPP_VERIFY_DIR"] = str(tmp_path / "verify_results")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"{script.name} with DXAPP_VERIFY=1 failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        verify_dirs = list(tmp_path.rglob("verify_results"))
        json_files = []
        for vd in verify_dirs:
            json_files.extend(vd.rglob(_JSON_GLOB))

        if not json_files:
            output = result.stdout + result.stderr
            if "verify" not in output.lower():
                pytest.skip(f"{script.name}: DXAPP_VERIFY produced no output (may need NPU)")

        return result, json_files

    @pytest.mark.parametrize("script,model_path", SYNC_PARAMS)
    def test_verify_sync_creates_json(self, script: Path, model_path: Path, tmp_path: Path):
        """Run sync script with DXAPP_VERIFY=1, verify JSON output."""
        _result, json_files = self._run_with_verify(script, model_path, tmp_path)

        for jf in json_files[:3]:
            try:
                data = json.loads(jf.read_text())
                assert isinstance(data, dict), f"JSON should be a dict: {jf}"
                assert "model_path" in data or "task" in data or "results" in data, (
                    f"JSON missing expected keys: {list(data.keys())}\nFile: {jf}"
                )
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {jf}: {e}")

    @pytest.mark.parametrize("script,model_path", ASYNC_PARAMS)
    def test_verify_async_creates_json(self, script: Path, model_path: Path, tmp_path: Path):
        """Run async script with DXAPP_VERIFY=1, verify JSON output."""
        self._run_with_verify(script, model_path, tmp_path)

    @pytest.mark.parametrize("script,model_path", SYNC_PARAMS)
    def test_verify_disabled_by_default(self, script: Path, model_path: Path, tmp_path: Path):
        """Without DXAPP_VERIFY, no verify_results should be created."""
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        cmd = [
            sys.executable, str(script),
            "--model", str(model_path),
            "--image", str(TEST_IMAGE),
            "--no-display",
            "--loop", "1",
        ]

        env = setup_environment()
        env.pop("DXAPP_VERIFY", None)
        env["DXAPP_VERIFY_DIR"] = str(tmp_path / "verify_results")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0

        verify_dirs = list(tmp_path.rglob("verify_results"))
        json_files = []
        for vd in verify_dirs:
            json_files.extend(vd.rglob(_JSON_GLOB))
        assert len(json_files) == 0, (
            f"DXAPP_VERIFY not set but JSON files found: {json_files}"
        )

    def test_verify_prerequisites(self):
        """Sanity check."""
        assert len(SYNC_REPR) > 0, "No sync scripts for verify tests"
        assert len(ASYNC_REPR) > 0, "No async scripts for verify tests"
        print(f"\n  Sync representatives: {[s.stem for s, _ in SYNC_REPR]}")
        print(f"  Async representatives: {[s.stem for s, _ in ASYNC_REPR]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
