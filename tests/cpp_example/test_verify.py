"""DXAPP_VERIFY: every downloaded C++ example, sync and async, dumps its real results (U-30, U-31).

Each run uses its task's sample input and a fresh temp directory as both cwd
and DXAPP_VERIFY_DIR, so a JSON left by another run can never satisfy it. It
must leave exactly one <stem>.json whose payload is non-empty (every sample
yields results: checked for all 37 downloaded models on 2026-09-29) and a
one-record <stem>.frames.jsonl equal to it. Expected time: about 2 minutes
(3 x 37 runs of under 2 s each).
"""
import sys
from pathlib import Path

import pytest

# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.proc import run_bounded  # noqa: E402
from test_helpers.utils import (  # noqa: E402
    discover_cpp_model_cases, resolve_cpp_exe_input, setup_environment)
from test_helpers.verify import payload_nonempty, read_verify_frames, read_verify_json  # noqa: E402

from conftest import PROJECT_ROOT, resolve_bin_dir

BIN_DIR = resolve_bin_dir()
DEFAULT_INPUT = PROJECT_ROOT / "sample" / "img" / "sample_kitchen.jpg"
SYNC_CASES = discover_cpp_model_cases("_sync", BIN_DIR)
ASYNC_CASES = discover_cpp_model_cases("_async", BIN_DIR)


def _run(executable, model_path, workdir, verify):
    test_input = resolve_cpp_exe_input(executable, default=DEFAULT_INPUT)
    if not test_input.exists():
        pytest.skip("sample input missing: {}".format(test_input))
    env = setup_environment()
    env.pop("DXAPP_VERIFY", None)
    if verify:
        env["DXAPP_VERIFY"] = "1"
    env["DXAPP_VERIFY_DIR"] = str(workdir / "verify")
    result = run_bounded(
        [str(BIN_DIR / executable), "-m", str(model_path), "-i", str(test_input), "--no-display", "-l", "1"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(workdir))
    assert result.returncode == 0, "{} rc={}\n{}".format(
        executable, result.returncode, (result.stdout + result.stderr)[-1500:])


def _check_dump(executable, workdir):
    verify_dir = workdir / "verify"
    data = read_verify_json(verify_dir)
    for key in ("task", "model", "model_path", "image_height", "image_width"):
        assert key in data, "{}: no '{}' in {}".format(executable, key, sorted(data))
    assert data["image_height"] > 0 and data["image_width"] > 0, (executable, data["image_height"], data["image_width"])
    assert payload_nonempty(data), "{} dumped no results: {}".format(executable, sorted(data))
    frames = read_verify_frames(verify_dir)
    assert len(frames) == 1 and frames[0]["frame"] == 0
    record = dict(frames[0])
    del record["frame"]
    assert record == data


@pytest.mark.verify
@pytest.mark.sync_exec
@pytest.mark.parametrize("executable,model_path", SYNC_CASES, ids=[e for e, _ in SYNC_CASES])
def test_verify_sync_dumps_real_results(executable, model_path, tmp_path):
    _run(executable, model_path, tmp_path, verify=True)
    _check_dump(executable, tmp_path)


@pytest.mark.verify
@pytest.mark.async_exec
@pytest.mark.parametrize("executable,model_path", ASYNC_CASES, ids=[e for e, _ in ASYNC_CASES])
def test_verify_async_dumps_real_results(executable, model_path, tmp_path):
    _run(executable, model_path, tmp_path, verify=True)
    _check_dump(executable, tmp_path)


@pytest.mark.verify
@pytest.mark.parametrize("executable,model_path", SYNC_CASES, ids=[e for e, _ in SYNC_CASES])
def test_verify_disabled_by_default(executable, model_path, tmp_path):
    _run(executable, model_path, tmp_path, verify=False)
    verify_dir = tmp_path / "verify"
    assert not verify_dir.exists() or not any(verify_dir.iterdir())


@pytest.mark.verify
def test_verify_prerequisites():
    assert BIN_DIR.exists()
    assert SYNC_CASES and ASYNC_CASES, "no downloaded model has a built example"
