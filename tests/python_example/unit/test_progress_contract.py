"""The C++ front-end timing contracts (tests/progress/*.cpp) build and pass.

The runners include src/cpp_example/common/utility/progress.hpp and run_clock.hpp; this compiles
the contract tests against them with the host compiler, so a change to a header is checked without
a full dx_app build.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TEST_CPP = ROOT / "tests" / "progress" / "test_progress.cpp"
UTILITY = ROOT / "src" / "cpp_example" / "common" / "utility"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
@pytest.mark.parametrize("name, passed", [
    ("test_progress.cpp", "All Progress contract tests PASSED"),
    ("test_run_clock.cpp", "All run clock contract tests PASSED"),
])
def test_progress_contract_builds_and_passes(tmp_path, name, passed):
    exe = tmp_path / "contract"
    build = subprocess.run(["g++", "-std=c++17", "-O1", "-I", str(UTILITY), str(TEST_CPP.with_name(name)),
                            "-o", str(exe), "-lpthread"], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert passed in run.stdout
