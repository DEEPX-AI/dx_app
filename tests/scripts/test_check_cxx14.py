"""Regression test for the C++14 conformance guard (spec B1/B2).

`scripts/check_cxx14.sh` compiles `common/utility/interrupt_flag.hpp`,
`common_util.hpp`, `run_dir.hpp` and `ordered_queue.hpp` standalone under
`-std=gnu++14 -Werror`. This catches two classes of regression:

  - B1: a C++17 extension (e.g. a structured binding) creeping back into any
    of the four headers (`-Werror=c++17-extensions` in the header check, and
    `-Werror=c++17-extensions` from cmake/dxapp_cxx14.cmake for real
    builds).
  - B2: `dxapp::g_interrupted()` losing its single shared definition again
    (declared-but-never-defined under a TU that only includes
    `common_util.hpp` without `interrupt_flag.hpp`/`run_dir.hpp`).

Requires g++ and the opencv4 pkg headers (same prerequisites as
`scripts/check_header_odr.sh`); skipped if unavailable so this doesn't fail
on a machine without a C++ toolchain. Skipped without g++, opencv4 or the
dxrt headers.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_cxx14.sh"


def _dxrt_header() -> Path:
    return Path(os.environ.get("DXRT_INCLUDE_DIR", "/usr/local/include")) / "dxrt" / "dxrt_api.h"


def _toolchain_available() -> bool:
    # common_util.hpp and run_dir.hpp (two of the four checked headers)
    # include <dxrt/dxrt_api.h>: no dxrt headers, no check (the GitHub
    # runner has none; scripts/ci_checks.sh reports SKIP there).
    return (shutil.which("g++") is not None and Path("/usr/include/opencv4").is_dir()
            and _dxrt_header().is_file())


needs_toolchain = pytest.mark.skipif(
    not _toolchain_available(), reason="g++, opencv4 or the dxrt headers not available")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                          cwd=str(ROOT), timeout=120)


def _copy_utility(tmp_path: Path) -> Path:
    """A copy of common/utility/ under tmp_path/src/cpp_example/: the tree
    --root checks. Every checked header lives there (ordered_queue.hpp,
    added in SP3, includes only std headers). The real headers are never
    written."""
    utility = tmp_path / "src" / "cpp_example" / "common" / "utility"
    shutil.copytree(ROOT / "src" / "cpp_example" / "common" / "utility", utility)
    return utility


@needs_toolchain
def test_check_cxx14_passes_on_the_real_tree():
    completed = _run()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "check_cxx14 OK" in completed.stdout


@needs_toolchain
def test_check_cxx14_passes_on_an_unbroken_copy(tmp_path):
    _copy_utility(tmp_path)
    completed = _run("--root", str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "check_cxx14 OK" in completed.stdout


@needs_toolchain
def test_check_cxx14_fails_if_a_structured_binding_is_reintroduced(tmp_path):
    """Break a COPY of common_util.hpp's C++14 rewrite; the guard must fail."""
    common_util = _copy_utility(tmp_path) / "common_util.hpp"
    original = common_util.read_text()
    broken = original.replace(
        "const std::pair<int, int> screen_res = getScreenResolution();\n"
        "        int target_w = screen_res.first / 2;\n"
        "        int target_h = screen_res.second / 2;",
        "auto [screen_w, screen_h] = getScreenResolution();\n"
        "        int target_w = screen_w / 2;\n"
        "        int target_h = screen_h / 2;",
    )
    assert broken != original, "fixture did not find the C++14 rewrite to break"
    common_util.write_text(broken)
    completed = _run("--root", str(tmp_path))
    assert completed.returncode != 0
    assert "FAILED: common/utility/common_util.hpp" in completed.stdout + completed.stderr


def test_check_cxx14_rejects_an_unknown_argument():
    completed = _run("--no-such-option")
    assert completed.returncode == 2
    assert "unknown argument" in completed.stderr


def test_check_cxx14_rejects_a_stray_argument_after_root(tmp_path):
    (tmp_path / "src" / "cpp_example" / "common" / "utility").mkdir(parents=True)
    completed = _run("--root", str(tmp_path), "--extra")
    assert completed.returncode == 2
    assert "unknown argument: --extra" in completed.stderr
