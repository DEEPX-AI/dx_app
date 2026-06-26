"""
Test --save / --save-dir functionality for C++ executables

Verifies:
  - run_dir creation with timestamp-based directory structure
  - run_info.txt metadata file generation
  - VideoWriter output (video save mode)
  - Image save output (image save mode)
  - initVideoWriter XVID→mp4v fallback
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.utils import setup_environment  # noqa: E402

from conftest import resolve_bin_dir

# ======================================================================
# Paths
# ======================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_DIR = resolve_bin_dir()
LIB_DIR = PROJECT_ROOT / "lib"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
SAMPLE_DIR = PROJECT_ROOT / "sample"

TEST_IMAGE = SAMPLE_DIR / "img" / "sample_kitchen.jpg"
TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"


# ======================================================================
# Discovery — reuse same logic as test_e2e.py
# ======================================================================
def _normalize_model_to_exe(stem: str) -> str:
    return stem.lower().replace(".", "_")


def discover_sync_cases() -> List[tuple]:
    """Discover (executable_name, model_path) pairs for sync executables."""
    cases = []
    seen = set()
    for model_path in sorted(MODELS_DIR.glob("*.dxnn")):
        prefix = _normalize_model_to_exe(model_path.stem)
        exe_name = f"{prefix}_sync"
        if exe_name in seen:
            continue
        if (BIN_DIR / exe_name).exists():
            cases.append((exe_name, model_path))
            seen.add(exe_name)
    return sorted(cases, key=lambda x: x[0])


def _pick_representative(cases: list, max_count: int = 3) -> list:
    """Pick a small representative subset to keep tests fast."""
    # Prefer one detection, one classification, one other
    priority_prefixes = ["yolov5s_sync", "yolov8n_sync", "fastdepth"]
    selected = []
    for exe, mp in cases:
        for p in priority_prefixes:
            if exe.startswith(p) and len(selected) < max_count:
                selected.append((exe, mp))
                break
    # Fill remaining
    for exe, mp in cases:
        if len(selected) >= max_count:
            break
        if (exe, mp) not in selected:
            selected.append((exe, mp))
    return selected


# ======================================================================
# Task-specific save verification cases
# ======================================================================
# 각 태스크(depth, face, hand, obb, pose, seg, semantic_seg)별 대표 모델
TASK_SAVE_CASES: List[tuple] = []

_TASK_EXE_IMAGE_MAP = {
    "detection":    ("yolov5s_sync",                "sample/img/sample_dog.jpg"),
    "depth":        ("fastdepth_1_sync",            "sample/img/sample_kitchen.jpg"),
    "face":         ("yolov7s_face_sync",           "sample/img/sample_face.jpg"),
    "hand":         ("handlandmarklite_1_sync",     "sample/img/sample_hand.jpg"),
    "obb":          ("yolo26s_obb_sync",            "sample/dota8_test/P0177.png"),
    "pose":         ("yolov8m_pose_sync",           "sample/img/sample_people.jpg"),
    "seg":          ("yolov5s_seg_sync",            "sample/img/sample_street.jpg"),
    "semantic_seg": ("segformer_b0_512x1024_sync",  "sample/img/sample_street.jpg"),
}

for _task, (_exe_name, _img_rel) in _TASK_EXE_IMAGE_MAP.items():
    _exe_path = BIN_DIR / _exe_name
    _img_path = PROJECT_ROOT / _img_rel
    if _exe_path.exists():
        # model 파일 자동 매칭: exe_name에서 _sync 제거 후 models에서 탐색
        _model_stem = _exe_name.replace("_sync", "")
        _model_candidates = list(MODELS_DIR.glob("*.dxnn"))
        _model_path = None
        for _mp in _model_candidates:
            if _mp.stem.lower().replace(".", "_").replace("-", "_") == _model_stem:
                _model_path = _mp
                break
        if _model_path:
            TASK_SAVE_CASES.append((_task, _exe_name, _model_path, _img_path))

TASK_SAVE_PARAMS = [
    pytest.param(task, exe, mp, img, id=f"{task}_{exe}",
                 marks=pytest.mark.sync_exec)
    for task, exe, mp, img in TASK_SAVE_CASES
]


SYNC_CASES = discover_sync_cases()
REPRESENTATIVE_CASES = _pick_representative(SYNC_CASES)
SAVE_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in REPRESENTATIVE_CASES
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.save_mode
class TestSaveMode:
    """Test --save and --save-dir CLI options."""

    @pytest.mark.parametrize("executable,model_path", SAVE_PARAMS)
    def test_image_save_creates_run_dir(self, executable, model_path, tmp_path):
        """Run with --save --save-dir, verify run_dir structure for image input."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        save_dir = tmp_path / "save_test"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"{executable} failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify run_dir was created under save_dir
        assert save_dir.exists(), f"save_dir not created: {save_dir}"

        # Find the run directory (should contain a timestamp-based subdir)
        run_dirs = list(save_dir.rglob("run_info.txt"))
        assert len(run_dirs) >= 1, (
            f"No run_info.txt found under {save_dir}\n"
            f"Contents: {list(save_dir.rglob('*'))}"
        )

        # Verify run_info.txt content
        run_info = run_dirs[0]
        run_info_text = run_info.read_text()
        assert "model" in run_info_text.lower() or "Model" in run_info_text, (
            f"run_info.txt missing model info:\n{run_info_text[:300]}"
        )

    @pytest.mark.parametrize("executable,model_path", SAVE_PARAMS)
    def test_video_save_creates_output(self, executable, model_path, tmp_path):
        """Run with --save on video input, verify video file is produced."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        # Skip face models (too slow for video)
        if "face" in executable.lower():
            pytest.skip(f"{executable}: face model too slow for video save test")

        save_dir = tmp_path / "video_save"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"{executable} video save failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify video output file was created (.mp4 or .mov)
        video_files = list(save_dir.rglob("*.mp4")) + list(save_dir.rglob("*.mov"))
        assert len(video_files) >= 1, (
            f"No .mp4/.mov output file found under {save_dir}\n"
            f"Contents: {list(save_dir.rglob('*'))}"
        )

        # Verify file is non-empty
        for vf in video_files:
            assert vf.stat().st_size > 0, f"Video file is empty: {vf}"

    @pytest.mark.parametrize("executable,model_path", SAVE_PARAMS)
    def test_run_info_contains_metadata(self, executable, model_path, tmp_path):
        """Verify run_info.txt contains expected metadata fields."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        save_dir = tmp_path / "metadata_test"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0

        run_infos = list(save_dir.rglob("run_info.txt"))
        assert len(run_infos) >= 1

        content = run_infos[0].read_text()

        # Check for expected metadata fields (run_info.txt uses 'script:' not 'executable:')
        expected_fields = ["script", "model", "input"]
        for field in expected_fields:
            assert field.lower() in content.lower(), (
                f"run_info.txt missing '{field}' field:\n{content[:500]}"
            )

    def test_save_mode_prerequisites(self):
        """Sanity: verify test prerequisites."""
        assert BIN_DIR.exists(), f"Bin directory not found: {BIN_DIR}"
        assert MODELS_DIR.exists(), f"Models directory not found: {MODELS_DIR}"
        assert len(REPRESENTATIVE_CASES) > 0, "No executables discovered for save mode tests"
        print(f"\n  Representative cases: {len(REPRESENTATIVE_CASES)}")
        for name, _ in REPRESENTATIVE_CASES:
            print(f"    - {name}")


# ======================================================================
# Task-specific output file verification tests
# ======================================================================
@pytest.mark.save_mode
class TestSaveOutputFiles:
    """Verify that --save actually produces image/video output files (jpg/png/mp4/avi)."""

    @pytest.mark.parametrize("task,executable,model_path,image_path", TASK_SAVE_PARAMS)
    def test_image_save_produces_output_file(
        self, task, executable, model_path, image_path, tmp_path
    ):
        """Run with --save on image input, verify output image file (jpg/png) is produced."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        save_dir = tmp_path / f"save_{task}_img"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(image_path),
            "--no-display",
            "-l", "1",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"[{task}] {executable} failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # save_dir가 생성되었는지
        assert save_dir.exists(), f"[{task}] save_dir not created: {save_dir}"

        # 실제 이미지 출력 파일 (jpg/png) 검증
        image_outputs = (
            list(save_dir.rglob("*.jpg"))
            + list(save_dir.rglob("*.jpeg"))
            + list(save_dir.rglob("*.png"))
        )
        assert len(image_outputs) >= 1, (
            f"[{task}] No output image file (jpg/png) found under {save_dir}\n"
            f"All files: {[str(f.relative_to(save_dir)) for f in save_dir.rglob('*') if f.is_file()]}"
        )

        # 파일 크기 > 0 확인
        for img_file in image_outputs:
            assert img_file.stat().st_size > 0, (
                f"[{task}] Output image is empty (0 bytes): {img_file.name}"
            )
            print(f"  [{task}] saved: {img_file.name} ({img_file.stat().st_size} bytes)")

    @pytest.mark.parametrize("task,executable,model_path,image_path", TASK_SAVE_PARAMS)
    def test_video_save_produces_output_file(
        self, task, executable, model_path, image_path, tmp_path
    ):
        """Run with --save on video input, verify output video file (mp4/avi) is produced."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        save_dir = tmp_path / f"save_{task}_video"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"[{task}] {executable} video save failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # 실제 비디오 출력 파일 (mp4/avi/mov) 검증
        video_outputs = (
            list(save_dir.rglob("*.mp4"))
            + list(save_dir.rglob("*.avi"))
            + list(save_dir.rglob("*.mov"))
        )
        assert len(video_outputs) >= 1, (
            f"[{task}] No output video file (mp4/avi/mov) found under {save_dir}\n"
            f"All files: {[str(f.relative_to(save_dir)) for f in save_dir.rglob('*') if f.is_file()]}"
        )

        # 파일 크기 > 0 확인
        for vid_file in video_outputs:
            assert vid_file.stat().st_size > 0, (
                f"[{task}] Output video is empty (0 bytes): {vid_file.name}"
            )
            print(f"  [{task}] saved: {vid_file.name} ({vid_file.stat().st_size} bytes)")

    def test_task_coverage(self):
        """Sanity: verify which tasks are covered by save output tests."""
        expected_tasks = {"detection", "depth", "face", "hand", "obb", "pose", "seg", "semantic_seg"}
        covered_tasks = {task for task, _, _, _ in TASK_SAVE_CASES}
        missing = expected_tasks - covered_tasks
        if missing:
            print(f"\n  WARNING: Missing task coverage (binary/model not found): {missing}")
        print(f"\n  Covered tasks: {sorted(covered_tasks)}")
        assert len(covered_tasks) >= 1, "No task-specific save tests could be configured"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
