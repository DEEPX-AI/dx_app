"""DXAPP_VERIFY: every Python example script of a downloaded model dumps real results (U-30, U-31).

Every ``*_sync*`` / ``*_async*`` script is run - ``*_cpp_postprocess`` and
``*_ort_off`` variants included - on its task's sample input with a fresh
DXAPP_VERIFY_DIR. The record must be non-empty and never the ``repr``
fallback. Expected time: about 3 minutes (~230 runs of under 1 s each).
"""
import sys
from pathlib import Path

import pytest

# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.constants import PROJECT_ROOT, SAMPLE_DIR  # noqa: E402
from test_helpers.proc import example_python, run_bounded  # noqa: E402
from test_helpers.utils import (  # noqa: E402
    discover_python_scripts, resolve_image_for_model, setup_environment)
from test_helpers.verify import payload_nonempty, read_verify_frames, read_verify_json  # noqa: E402

DEFAULT_INPUT = SAMPLE_DIR / "img" / "sample_kitchen.jpg"


def _cases(mode):
    cases = []
    for task, name, sync_scripts, async_scripts, model in discover_python_scripts():
        if model is None:
            continue
        rel = resolve_image_for_model(name, task)
        test_input = PROJECT_ROOT / rel if rel else DEFAULT_INPUT
        for script in (sync_scripts if mode == "sync" else async_scripts):
            cases.append(pytest.param(script, model, test_input, id=script.stem))
    return cases


SYNC_CASES = _cases("sync")
ASYNC_CASES = _cases("async")


def _run(script, model, test_input, workdir, verify):
    env = setup_environment()
    env.pop("DXAPP_VERIFY", None)
    if verify:
        env["DXAPP_VERIFY"] = "1"
    env["DXAPP_VERIFY_DIR"] = str(workdir / "verify")
    result = run_bounded(
        [example_python(), str(script), "--model", str(model), "--image", str(test_input),
         "--no-display", "--loop", "1"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, "{} rc={}\n{}".format(
        script.name, result.returncode, (result.stdout + result.stderr)[-1500:])


def _check_dump(script, workdir):
    verify_dir = workdir / "verify"
    data = read_verify_json(verify_dir)
    assert "repr" not in data and "result_type" not in data, "{} has no serializer: {}".format(
        script.name, data.get("result_type"))
    for key in ("task", "model", "model_path", "image_height", "image_width"):
        assert key in data, "{}: no '{}'".format(script.name, key)
    assert payload_nonempty(data), "{} dumped no results: {}".format(script.name, sorted(data))
    frames = read_verify_frames(verify_dir)
    assert len(frames) == 1 and frames[0]["frame"] == 0
    record = dict(frames[0])
    del record["frame"]
    assert record == data, "{}: the frames.jsonl record differs from the JSON".format(script.name)


@pytest.mark.verify
@pytest.mark.sync_exec
@pytest.mark.parametrize("script,model,test_input", SYNC_CASES)
def test_verify_sync_dumps_real_results(script, model, test_input, tmp_path):
    _run(script, model, test_input, tmp_path, verify=True)
    _check_dump(script, tmp_path)


@pytest.mark.verify
@pytest.mark.async_exec
@pytest.mark.parametrize("script,model,test_input", ASYNC_CASES)
def test_verify_async_dumps_real_results(script, model, test_input, tmp_path):
    _run(script, model, test_input, tmp_path, verify=True)
    _check_dump(script, tmp_path)


@pytest.mark.verify
@pytest.mark.parametrize("script,model,test_input", SYNC_CASES)
def test_verify_disabled_by_default(script, model, test_input, tmp_path):
    _run(script, model, test_input, tmp_path, verify=False)
    verify_dir = tmp_path / "verify"
    assert not verify_dir.exists() or not any(verify_dir.iterdir())


@pytest.mark.verify
def test_verify_prerequisites():
    assert SYNC_CASES and ASYNC_CASES, "no downloaded model has a Python example"
