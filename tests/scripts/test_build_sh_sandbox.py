"""build.sh --clean (E4) and quoted user paths (E5), run in a sandbox.

Every case runs a copy of build.sh in a temporary tree (see
build_sh_sandbox.sh) with cmake, python3 and sudo stubbed. Nothing here
touches the repository's own build/, bin/, lib/ or include/, and sudo is
never the real one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from build_sh_sandbox import CROSS, NATIVE, Sandbox, needs_toolchain_file

pytestmark = needs_toolchain_file


@pytest.fixture()
def sandbox(tmp_path):
    return Sandbox(tmp_path / "sb")


def _stub_python(path: Path, log: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/bash\necho "$*" >> "{log}"\n'
                    '[ "$1" = "--version" ] && { echo "Python 3.12.0"; exit 0; }\nexit 1\n')
    path.chmod(0o755)
    return path


# ── E4: --clean ───────────────────────────────────────────────────────────────

def test_clean_native_removes_build_dir_and_bin_lib_include(sandbox):
    sandbox.run("--clean", "--target", "foo_sync")
    assert "CMake configuration failed" in sandbox.output  # stopped at the stub
    assert not sandbox.marker(f"build_{NATIVE}")   # emptied (re-created by mkdir -p)
    for d in ("bin", "lib", "include"):
        assert not sandbox.exists(d), d
    assert sandbox.marker(f"build_{CROSS}")         # another arch's build dir stays
    assert sandbox.lines("sudo") == []
    assert "try to clean again with 'sudo'" not in sandbox.output


def test_clean_cross_removes_only_its_build_dir(sandbox):
    sandbox.run("--clean", "--arch", CROSS, "--target", "foo_sync")
    assert "CMake configuration failed" in sandbox.output
    assert not sandbox.marker(f"build_{CROSS}")
    for d in ("bin", "lib", "include"):
        assert sandbox.marker(d), f"{d}/ must survive a cross build's --clean"
    assert sandbox.marker(f"build_{NATIVE}")
    assert sandbox.lines("sudo") == []


def test_no_clean_deletes_nothing(sandbox):
    sandbox.run("--target", "foo_sync")
    for d in (f"build_{NATIVE}", f"build_{CROSS}", "bin", "lib", "include"):
        assert sandbox.marker(d), d
    assert sandbox.lines("sudo") == []


def test_clean_retries_with_sudo_as_separate_commands(sandbox):
    sandbox.run("--clean", "--target", "foo_sync", SANDBOX_RM_FAILS="1")
    assert "try to clean again with 'sudo'" in sandbox.output
    assert sandbox.lines("rm_denied"), "the unprivileged rm must have been tried first"
    sudo = sandbox.lines("sudo")
    assert sudo and all(line.startswith("rm -rf ") for line in sudo), sudo
    assert not any("&&" in line or "[" in line for line in sudo), sudo
    assert not sandbox.marker(f"build_{NATIVE}")
    for d in ("bin", "lib", "include"):
        assert not sandbox.exists(d), d
    assert "Failed to clean build directory\n" not in sandbox.output


def test_clean_cross_sudo_retry_keeps_bin_lib_include(sandbox):
    sandbox.run("--clean", "--arch", CROSS, "--target", "foo_sync", SANDBOX_RM_FAILS="1")
    assert "try to clean again with 'sudo'" in sandbox.output
    assert not sandbox.marker(f"build_{CROSS}")
    for d in ("bin", "lib", "include"):
        assert sandbox.marker(d), d
    assert all(" bin" not in line for line in sandbox.lines("sudo"))


# ── E5: quoted user paths ─────────────────────────────────────────────────────

def test_configure_argv_for_space_free_paths_is_unchanged(sandbox, tmp_path):
    # Recorded from the pre-fix build.sh (task report, E5): the same argv.
    py = _stub_python(tmp_path / "py" / "python3", tmp_path / "py.log")
    sandbox.run("--python_exec", str(py), "--target", "foo_sync")
    assert sandbox.cmake_argv == [
        "..",
        f"-DDXAPP_PYTHON={py}",
        f"-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.{NATIVE}.cmake",
        "-DCMAKE_VERBOSE_MAKEFILE=false",
        "-DCMAKE_BUILD_TYPE=release",
        "-DCMAKE_GENERATOR=Ninja",
    ]


def test_venv_path_with_a_space_reaches_cmake_as_one_argument(sandbox, tmp_path):
    venv = tmp_path / "my venv"
    _stub_python(venv / "bin" / "python", tmp_path / "venv_py.log")
    (venv / "bin" / "activate").write_text(f'VIRTUAL_ENV="{venv}"\nexport VIRTUAL_ENV\n')
    sandbox.run("--venv_path", str(venv), "--target", "foo_sync")
    assert "CMake configuration failed" in sandbox.output, sandbox.output
    assert f"-DDXAPP_PYTHON={venv}/bin/python" in sandbox.cmake_argv
    assert "No such file" not in sandbox.output, sandbox.output


def test_python_exec_with_a_space_is_one_word_everywhere(sandbox, tmp_path):
    log = tmp_path / "spaced_py.log"
    py = _stub_python(tmp_path / "py dir" / "python3", log)
    sandbox.run("--clean", "--python_exec", str(py), "--target", "foo_sync")
    assert f"-DDXAPP_PYTHON={py}" in sandbox.cmake_argv
    calls = log.read_text().splitlines()
    assert "--version" in calls                      # the version probe
    assert "-m pip show dx_postprocess" in calls     # --clean's uninstall probe
    assert "command not found" not in sandbox.output, sandbox.output
    assert "No such file" not in sandbox.output, sandbox.output


def test_repo_path_with_a_space_and_v3codec_dir(tmp_path):
    sb = Sandbox(tmp_path / "dir with space" / "sb")
    codec = sb.repo / "third_party" / "v3_codec"
    codec.mkdir(parents=True)
    sb.run("--v3codec", "--target", "foo_sync")
    assert "CMake configuration failed" in sb.output, sb.output
    assert "-DUSE_V3_CODEC=True" in sb.cmake_argv
    assert f"-DV3_CODEC_DIR={codec}" in sb.cmake_argv


# ── U-38: a cross build does not pip-install for the host ─────────────────

def test_cross_build_skips_the_host_pip_install_of_dx_postprocess(sandbox):
    sandbox.run("--all", "--arch", CROSS, SANDBOX_CMAKE_RC="0", SANDBOX_INSTALL_TREE="1")
    assert "dx_postprocess skipped" in sandbox.output, sandbox.output
    assert not any("pip" in line for line in sandbox.lines("python3")), sandbox.lines("python3")


def test_native_build_still_installs_dx_postprocess(sandbox):
    sandbox.run("--all", SANDBOX_CMAKE_RC="0", SANDBOX_INSTALL_TREE="1")
    assert any(line.startswith("-m pip install") for line in sandbox.lines("python3")), sandbox.output
    assert "dx_postprocess skipped" not in sandbox.output
