from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "generate_build_bat.py"
SCRATCH = ROOT / ".cache" / "dxapp_build_selection_tests"


def run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def scratch_output(name: str) -> Path:
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH / name


def teardown_module():
    shutil.rmtree(SCRATCH, ignore_errors=True)


def load_module():
    spec = importlib.util.spec_from_file_location("generate_build_bat", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_targets_emit_target_only_build_and_copy():
    output = scratch_output("build_internal.bat")

    result = run_generator(
        "--targets",
        "yolov7_sync",
        "yolov7_async",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "--target yolov7_sync yolov7_async" in text
    assert "cmake --install" not in text
    assert "for /R \"%BUILD_DIR%\" %%F in (yolov7_sync.exe)" in text
    assert "for /R \"%BUILD_DIR%\" %%F in (yolov7_async.exe)" in text


def test_minimal_resolves_run_demo_targets():
    output = scratch_output("build_internal.bat")

    result = run_generator("--minimal", "--output", str(output))

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "--target" in text
    assert "yolov7_sync" in text
    assert "resnet50_async" in text
    assert "cmake --install" not in text


def test_category_requires_value():
    result = run_generator("--category")

    assert result.returncode != 0
    assert "expected one argument" in result.stderr


def test_category_build_resolves_category_targets():
    output = scratch_output("build_internal.bat")

    result = run_generator("--category", "classification", "--output", str(output))

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "--target" in text
    assert "alexnet_sync" in text
    assert "resnet50_sync" in text
    assert "yolov7_sync" not in text


def test_unknown_category_returns_dxapp_error():
    result = run_generator("--category", "not_a_category")

    assert result.returncode != 0
    assert "[DXAPP] [ERROR] Unknown category: not_a_category" in (
        result.stdout + result.stderr
    )


def test_category_list_prints_categories_without_writing_output():
    output = scratch_output("build_internal.bat")

    result = run_generator("--category", "list", "--output", str(output))

    assert result.returncode == 0, result.stderr
    assert "classification" in result.stdout
    assert "object_detection" in result.stdout
    assert not output.exists()


def test_conflicting_generator_modes_are_rejected():
    result = run_generator("--targets", "yolov7_sync", "--minimal")

    assert result.returncode != 0
    assert "[DXAPP] [ERROR] Use only one of --targets, --minimal, or --category." in (
        result.stdout + result.stderr
    )


def test_targets_with_hyphens_sanitize_batch_variable_names():
    """배치 변수명은 sanitize되어야 하지만 실제 타겟명과 exe 파일명은 원본 유지"""
    output = scratch_output("build_internal.bat")

    result = run_generator(
        "--targets",
        "my-target_v2",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    
    # 원본 타겟명이 CMake 빌드 명령에 사용되어야 함
    assert "--target my-target_v2" in text
    
    # 원본 exe 파일명이 검색에 사용되어야 함
    assert "for /R \"%BUILD_DIR%\" %%F in (my-target_v2.exe)" in text
    
    # 배치 변수명은 sanitize되어야 함 (하이픈 -> 언더스코어)
    assert "FOUND_my_target_v2" in text


def test_run_bat_executes_bare_relative_filename(tmp_path, monkeypatch):
    """Regression test for a Windows cmd.exe quoting bug in run_bat().

    subprocess.run(f'"{path}"', shell=True) wraps the string in ANOTHER pair
    of quotes internally (cmd.exe /c "<args>"). If `path` is a bare relative
    filename with no directory component (e.g. "build_internal.bat", which is
    the default --output used by build.bat), the resulting command line has
    four quote characters total, which defeats cmd.exe's special-case rule
    for stripping a single surrounding quote pair around an executable name.
    This previously made every real `build.bat --all/--minimal/--category`
    invocation fail with: '"build_internal.bat"' is not recognized...
    run_bat() must resolve the path to an absolute one before invoking it.
    """
    module = load_module()

    marker = tmp_path / "marker.txt"
    bat_file = tmp_path / "regression_run.bat"
    bat_file.write_text(
        f'@echo off\r\necho done> "{marker}"\r\nexit /b 0\r\n', encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    # Pass a BARE relative filename (no "./" prefix) — this is exactly how
    # build.bat's default `--output build_internal.bat` reaches run_bat().
    rc = module.run_bat(Path(bat_file.name))

    assert rc == 0
    assert marker.exists(), "run_bat() failed to execute a bare relative .bat filename"
    assert marker.read_text(encoding="utf-8").strip() == "done"
