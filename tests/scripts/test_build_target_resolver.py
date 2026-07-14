from __future__ import annotations

import subprocess
from pathlib import Path


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
    assert "yolov7_sync" in targets
    assert "yolov7_async" in targets
    assert "resnet50_sync" in targets
    assert "resnet50_async" in targets
    assert len(targets) == len(set(targets))


def test_category_targets_are_derived_from_cpp_sources():
    result = run_bash(
        f"source {RESOLVER}; dxapp_resolve_category_targets classification"
    )

    assert result.returncode == 0, result.stderr
    targets = result.stdout.split()
    assert "alexnet_sync" in targets
    assert "alexnet_async" in targets
    assert "resnet50_sync" in targets
    assert "resnet50_async" in targets
    assert all(target.endswith(("_sync", "_async")) for target in targets)
    assert "yolov7_sync" not in targets


def test_category_list_excludes_support_directories():
    result = run_bash(f"source {RESOLVER}; dxapp_list_categories")

    assert result.returncode == 0, result.stderr
    categories = result.stdout.split()
    assert "classification" in categories
    assert "object_detection" in categories
    assert "common" not in categories
    assert "__pycache__" not in categories


def test_unknown_category_returns_dxapp_error():
    result = run_bash(
        f"source {RESOLVER}; dxapp_resolve_category_targets not_a_category"
    )

    assert result.returncode != 0
    assert "[DXAPP] [ERROR] Unknown category: not_a_category" in result.stderr


def test_build_sh_rejects_conflicting_target_modes():
    result = subprocess.run(
        ["bash", "build.sh", "--target", "yolov7_sync", "--minimal"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "[DXAPP] [ERROR]" in combined
    assert "Use only one of --target, --minimal, or --category." in combined


def test_build_sh_category_requires_value():
    result = subprocess.run(
        ["bash", "build.sh", "--category"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "[DXAPP] [ERROR]" in combined
    assert "--category requires a category name or 'list'." in combined


def test_build_sh_unknown_category_returns_dxapp_error():
    result = subprocess.run(
        ["bash", "build.sh", "--category", "not_a_category"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "[DXAPP] [ERROR] Unknown category: not_a_category" in combined


def test_build_sh_category_list_prints_without_configuring_cmake():
    result = subprocess.run(
        ["bash", "build.sh", "--category", "list"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "classification" in result.stdout
    assert "object_detection" in result.stdout
    assert "cmake args" not in result.stdout


def test_build_sh_target_list_still_uses_existing_path():
    result = subprocess.run(
        ["bash", "build.sh", "--target", "list"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert "Available build targets:" in combined


def test_build_sh_help_includes_build_selection_examples():
    result = subprocess.run(
        ["./build.sh", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "./build.sh --minimal --type Release" in combined
    assert "./build.sh --category object_detection --type Release" in combined
    assert "./build.sh --category list" in combined


def test_build_sh_minimal_installs_dx_postprocess_before_target_exit():
    text = (ROOT / "build.sh").read_text(encoding="utf-8")

    assert "install_dx_postprocess_module()" in text

    target_exit_start = text.index("# Skip dx_postprocess installation for target builds")
    full_build_start = text.index("if [ -e $build_dir/release/bin ]")
    target_exit_block = text[target_exit_start:full_build_start]

    assert 'if [ "${build_minimal}" = "true" ]; then' in target_exit_block
    assert "install_dx_postprocess_module" in target_exit_block
    assert target_exit_block.index("install_dx_postprocess_module") < target_exit_block.index("exit 0")
    assert "popd" not in target_exit_block


def test_build_sh_empty_category_fails_before_cmake_configure():
    empty_category_dir = ROOT / "src" / "cpp_example" / "dxapp_empty_test_category"
    
    try:
        # Create empty category directory
        empty_category_dir.mkdir(parents=True, exist_ok=True)
        
        # Run build.sh with the empty category
        result = subprocess.run(
            ["bash", "build.sh", "--category", "dxapp_empty_test_category"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        
        combined = result.stdout + result.stderr
        
        # Assert non-zero exit
        assert result.returncode != 0, "Expected non-zero exit for empty category"
        
        # Assert error message appears
        assert "[DXAPP] [ERROR]" in combined, "Expected [DXAPP] [ERROR] tag"
        assert "No build targets resolved." in combined, "Expected 'No build targets resolved.' message"
        
        # Assert failure happens before CMake configure
        assert "cmake args" not in combined, "Should fail before CMake configure"
        
    finally:
        # Cleanup: remove the empty directory
        if empty_category_dir.exists():
            empty_category_dir.rmdir()
