from __future__ import annotations

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
    
    # sanitize 안 된 변수명은 없어야 함
    assert "FOUND_my-target_v2" not in text
