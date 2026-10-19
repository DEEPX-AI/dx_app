"""Smoke tests for the multi-model graph CLI on Windows (SP6 U-37).

NOT BUILT OR RUN ON WINDOWS HERE: the development host has no MSVC. Off
Windows every test skips (tests/scripts/test_platform_paths.py checks that).
On Windows they need build.bat's output in bin\\Release or bin\\ and nothing
else - no NPU: --help, --list-models and --check never open the device.
The expectations are what the Linux CLI does.
"""
import sys

import pytest

from conftest import BIN_DIR, PROJECT_ROOT, run_command

pytestmark = [
    pytest.mark.graph,
    pytest.mark.skipif(sys.platform != "win32", reason="Windows only (graph CLI smoke test)"),
]

GRAPHS = PROJECT_ROOT / "src" / "cpp_example" / "multi_model_graph"
EXECUTABLES = ("multi_model_graph_sync.exe", "multi_model_graph_async.exe")


@pytest.fixture(params=EXECUTABLES)
def graph_cli(request):
    exe = BIN_DIR / request.param
    if not exe.exists():
        pytest.skip(f"{exe} not built (run build.bat)")
    return exe


def test_help_exits_zero(graph_cli):
    result = run_command([str(graph_cli), "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Usage:" in result.stdout


def test_list_models_names_a_detector(graph_cli):
    result = run_command([str(graph_cli), "--list-models"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "yolov8-n_640x640" in result.stdout


def test_check_validates_a_sample_without_the_npu(graph_cli):
    result = run_command([str(graph_cli), "--check", str(GRAPHS / "cascade_od_attr.json")],
                         timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_option_is_a_usage_error(graph_cli):
    result = run_command([str(graph_cli), "--no-such-option"])
    assert result.returncode == 2, result.stdout + result.stderr
