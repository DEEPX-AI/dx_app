from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "extract_sln_package.py"
SCRATCH = ROOT / ".cache" / "dxapp_extract_sln_tests"


def run_extract(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def reset_scratch(name: str) -> Path:
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def teardown_module():
    shutil.rmtree(SCRATCH, ignore_errors=True)


def load_module():
    spec = importlib.util.spec_from_file_location("extract_sln_package", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_category_model_creates_skeleton():
    output_dir = reset_scratch("category_model")
    result = run_extract(
        "classification/resnet50",
        "--output-dir",
        str(output_dir),
        "--no-generate-sln",
    )

    assert result.returncode == 0, result.stderr
    package_dir = output_dir / "sln" / "classification" / "resnet50"
    assert (package_dir / "CMakeLists.txt").is_file()
    assert (package_dir / "build.bat").is_file()
    assert (package_dir / "README.md").is_file()
    assert (package_dir / "cmake" / "dxapp_package_deps.cmake").is_file()
    assert (package_dir / "cmake" / "dxapp_package_deps.bat").is_file()
    assert (package_dir / "src" / "resnet50_sync.cpp").is_file()
    assert (package_dir / "src" / "resnet50_async.cpp").is_file()
    assert (package_dir / "factory" / "resnet50_factory.hpp").is_file()
    assert (package_dir / "common").is_dir()
    assert (package_dir / "utility" / "common_util.cpp").is_file()
    assert (package_dir / "extern" / "cxxopts.hpp").is_file()
    assert "draft Visual Studio/CMake package skeleton" in (
        package_dir / "README.md"
    ).read_text(encoding="utf-8")
    assert "Visual Studio 17 2022" in (package_dir / "build.bat").read_text(
        encoding="utf-8"
    )
    cmake_text = (package_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "find_package(OpenCV REQUIRED" in cmake_text
    assert "find_library(DXRT_LIB dxrt" in cmake_text
    assert "target_link_libraries" in cmake_text
    assert "dxapp_package_deps.cmake" in cmake_text


def test_generated_cmake_copies_windows_dll_dirs_at_build_time():
    output_dir = reset_scratch("dll_copy")
    result = run_extract(
        "classification/resnet50",
        "--output-dir",
        str(output_dir),
        "--no-generate-sln",
    )

    assert result.returncode == 0, result.stderr
    cmake_text = (
        output_dir / "sln" / "classification" / "resnet50" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "file(GLOB DXRT_DLLS" not in cmake_text
    assert "file(GLOB VCPKG_DLLS" not in cmake_text
    assert "-P" in cmake_text
    assert "dxapp_copy_runtime_dir.cmake" in cmake_text
    assert "${CMAKE_COMMAND} -E copy_directory" not in cmake_text
    assert 'if(EXISTS "${DXRT_INSTALLED_DIR}/bin")' not in cmake_text
    assert 'EXISTS "${VCPKG_INSTALLED_DIR}/x64-windows/bin"' not in cmake_text
    assert '"${DXRT_INSTALLED_DIR}/bin"' in cmake_text
    assert '"${VCPKG_INSTALLED_DIR}/x64-windows/bin"' in cmake_text

    copy_script = (
        output_dir
        / "sln"
        / "classification"
        / "resnet50"
        / "cmake"
        / "dxapp_copy_runtime_dir.cmake"
    )
    assert copy_script.is_file()
    copy_script_text = copy_script.read_text(encoding="utf-8")
    assert 'if(EXISTS "${DXAPP_RUNTIME_SRC_DIR}")' in copy_script_text
    assert 'file(GLOB _dxapp_runtime_dlls "${DXAPP_RUNTIME_SRC_DIR}/*.dll")' in copy_script_text
    assert "-E copy_if_different" in copy_script_text
    assert "-E copy_directory" not in copy_script_text
    assert "message(WARNING" in copy_script_text


def test_extract_unique_basename_resolves_model():
    output_dir = reset_scratch("unique_basename")
    result = run_extract(
        "resnet50",
        "--output-dir",
        str(output_dir),
        "--no-generate-sln",
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "sln" / "classification" / "resnet50").is_dir()


def test_extract_unknown_model_returns_dxapp_error():
    output_dir = reset_scratch("unknown_model")
    result = run_extract(
        "missing_model",
        "--output-dir",
        str(output_dir),
        "--no-generate-sln",
    )

    assert result.returncode != 0
    assert "[DXAPP] [ERROR] Model not found: missing_model" in result.stderr


def test_resolve_basename_rejects_ambiguous_model(capsys):
    module = load_module()
    cpp_root = reset_scratch("fake_cpp_example") / "cpp_example"
    (cpp_root / "category_a" / "same_model").mkdir(parents=True)
    (cpp_root / "category_b" / "same_model").mkdir(parents=True)

    with pytest.raises(SystemExit):
        module.resolve_model("same_model", cpp_root)

    captured = capsys.readouterr()
    assert "[DXAPP] [ERROR] Ambiguous model name: same_model" in captured.err
    assert "category_a/same_model" in captured.err
    assert "category_b/same_model" in captured.err


def test_extract_package_requires_sync_source(capsys):
    """Extract package must fail if sync source is missing."""
    module = load_module()
    
    # Create fake model directory without sync source
    model_dir = SCRATCH / "nosync_test" / "category" / "nosync"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ModelRef
    ref = module.ModelRef(category="category", model="nosync", path=model_dir)
    
    # Create output directory
    output_dir = reset_scratch("missing_sync_output")
    
    # Call extract_package - should exit with error
    with pytest.raises(SystemExit):
        module.extract_package(ref, output_dir)
    
    captured = capsys.readouterr()
    assert "[DXAPP] [ERROR] Sync source not found:" in captured.err
    assert "nosync_sync.cpp" in captured.err


def test_resolve_model_rejects_path_traversal(capsys):
    """resolve_model must reject category/model inputs that resolve outside cpp_example_dir."""
    module = load_module()
    
    # Create repo-relative scratch fake tree
    root = reset_scratch("path_traversal")
    cpp_root = root / "cpp_example"
    outside = root / "outside" / "escape"
    
    # Create both directories
    cpp_root.mkdir(parents=True)
    outside.mkdir(parents=True)
    
    # Try to access outside directory using path traversal
    with pytest.raises(SystemExit):
        module.resolve_model("../outside/escape", cpp_root)
    
    captured = capsys.readouterr()
    assert "[DXAPP] [ERROR] Invalid model path: ../outside/escape" in captured.err


def test_generate_solution_uses_cmake_and_returns_sln(monkeypatch):
    module = load_module()
    package_dir = reset_scratch("generate_solution") / "package"
    build_dir = package_dir / "build"
    package_dir.mkdir(parents=True)

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        assert command[:5] == [
            "cmake",
            "-S",
            str(package_dir),
            "-B",
            str(build_dir),
        ]
        build_dir.mkdir(parents=True)
        (build_dir / "dxapp_resnet50_sln_package.sln").write_text("", encoding="utf-8")
        return Completed()

    monkeypatch.setattr(module.shutil, "which", lambda name: "cmake" if name == "cmake" else None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.generate_solution(package_dir) == build_dir / "dxapp_resnet50_sln_package.sln"
