from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from build_sh_sandbox import CROSS, NATIVE, Sandbox, needs_toolchain_file


ROOT = Path(__file__).resolve().parents[2]
RESOLVER = ROOT / "scripts" / "build_target_resolver.sh"


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_minimal_targets_include_sync_and_async_for_run_demo_entries():
    result = run_bash(f"source {RESOLVER}; dxapp_resolve_minimal_targets")

    assert result.returncode == 0, result.stderr
    targets = result.stdout.split()
    assert "yolov7_640x640_sync" in targets
    assert "yolov7_640x640_async" in targets
    assert "resnet50_224x224_sync" in targets
    assert "resnet50_224x224_async" in targets
    assert len(targets) == len(set(targets))


def test_category_targets_are_derived_from_cpp_sources():
    result = run_bash(
        f"source {RESOLVER}; dxapp_resolve_category_targets image_classification"
    )

    assert result.returncode == 0, result.stderr
    targets = result.stdout.split()
    assert "alexnet_224x224_sync" in targets
    assert "alexnet_224x224_async" in targets
    assert "resnet50_224x224_sync" in targets
    assert "resnet50_224x224_async" in targets
    assert all(target.endswith(("_sync", "_async")) for target in targets)
    assert "yolov7_640x640_sync" not in targets


def test_category_list_excludes_support_directories():
    result = run_bash(f"source {RESOLVER}; dxapp_list_categories")

    assert result.returncode == 0, result.stderr
    categories = result.stdout.split()
    assert "image_classification" in categories
    assert "object_detection" in categories
    assert "common" not in categories
    assert "__pycache__" not in categories


def test_unknown_category_returns_dxapp_error():
    result = run_bash(
        f"source {RESOLVER}; dxapp_resolve_category_targets not_a_category"
    )

    assert result.returncode != 0
    assert "[DXAPP] [ERROR] Unknown category: not_a_category" in result.stderr


# ── build.sh itself: only ever in the sandbox (U-74) ──────────────────────
# The real build.sh re-configures build_<arch>/ and deletes
# build_<arch>/release, so no test runs it in the repository.


@pytest.fixture()
def sandbox(tmp_path):
    return Sandbox(tmp_path / "sb")


@needs_toolchain_file
def test_build_sh_rejects_conflicting_target_modes(sandbox):
    sandbox.run("--target", "yolov7_sync", "--minimal")
    assert sandbox.rc != 0
    assert "[DXAPP] [ERROR]" in sandbox.output
    assert "Use only one of --all, --minimal, --target, or --category." in sandbox.output
    assert sandbox.cmake_calls == []


@needs_toolchain_file
def test_build_sh_category_requires_value(sandbox):
    sandbox.run("--category")
    assert sandbox.rc != 0
    assert "[DXAPP] [ERROR]" in sandbox.output
    assert "--category requires a category name or 'list'." in sandbox.output
    assert sandbox.cmake_calls == []


@needs_toolchain_file
def test_build_sh_unknown_category_returns_dxapp_error(sandbox):
    sandbox.run("--category", "not_a_category", SANDBOX_SRC_TREE="1")
    assert sandbox.rc != 0
    assert "[DXAPP] [ERROR] Unknown category: not_a_category" in sandbox.output
    assert sandbox.cmake_calls == []


@needs_toolchain_file
def test_build_sh_category_list_prints_without_configuring_cmake(sandbox):
    sandbox.run("--category", "list", SANDBOX_SRC_TREE="1")
    assert sandbox.rc == 0, sandbox.output
    assert "image_classification" in sandbox.output
    assert "object_detection" in sandbox.output
    assert "cmake args" not in sandbox.output
    assert sandbox.cmake_calls == []


@needs_toolchain_file
def test_build_sh_target_list_configures_then_lists_phony_targets(sandbox):
    sandbox.run("--target", "list", SANDBOX_CMAKE_RC="0")
    assert sandbox.rc == 0, sandbox.output
    assert "Available build targets:" in sandbox.output
    listed = sandbox.output.split("Available build targets:", 1)[1].split()
    assert "yolov7_sync" in listed and "yolov7_async" in listed
    for hidden in ("edit_cache", "rebuild_cache", "install", "list_install_components"):
        assert hidden not in listed, hidden
    assert sandbox.cmake_argv[0] == ".."
    assert sandbox.lines("ninja") == ["-t targets"]
    # The step U-74 ran against the real tree: every configure starts by
    # deleting the native build's release/ tree - here, the sandbox's.
    assert not sandbox.exists(f"build_{NATIVE}/release")
    assert sandbox.marker(f"build_{CROSS}/release")


@needs_toolchain_file
def test_build_sh_help_includes_build_selection_examples(sandbox):
    sandbox.run("--help")
    assert sandbox.rc == 0
    assert "./build.sh --minimal --type Release" in sandbox.output
    assert "./build.sh --category object_detection --type Release" in sandbox.output
    assert "./build.sh --category list" in sandbox.output


def test_build_sh_minimal_installs_dx_postprocess_before_target_exit():
    text = (ROOT / "build.sh").read_text(encoding="utf-8")

    assert "install_dx_postprocess_module()" in text

    target_exit_start = text.index(
        "# Install dx_postprocess (full pip build) for minimal builds; "
        "skip for explicit --target / --category builds")
    full_build_start = text.index("if [ -e $build_dir/release/bin ]")
    target_exit_block = text[target_exit_start:full_build_start]

    assert 'if [ "${build_minimal}" = "true" ]; then' in target_exit_block
    assert "install_dx_postprocess_module" in target_exit_block
    assert target_exit_block.index("install_dx_postprocess_module") < target_exit_block.index("exit 0")
    assert "popd" not in target_exit_block


@needs_toolchain_file
def test_build_sh_empty_category_fails_before_cmake_configure(sandbox):
    sandbox.run("--category", "dxapp_empty_test_category", SANDBOX_SRC_TREE="1",
                SANDBOX_EMPTY_CATEGORY="dxapp_empty_test_category")
    assert sandbox.rc != 0, "Expected non-zero exit for empty category"
    assert "[DXAPP] [ERROR]" in sandbox.output
    assert "No build targets resolved." in sandbox.output
    assert "cmake args" not in sandbox.output, "Should fail before CMake configure"
    assert sandbox.cmake_calls == []
    assert not (ROOT / "src" / "cpp_example" / "dxapp_empty_test_category").exists()
