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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.utils import (  # noqa: E402
    setup_environment,
    discover_cpp_executables,
    cpp_exe_task_map,
    resolve_cpp_exe_input,
    stream_rejecting_cpp_cases,
)
from test_helpers.constants import (  # noqa: E402
    IMAGE_ONLY_TASKS,
    STREAM_REJECTING_TASKS_CPP,
)

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


def _test_input_for(executable: str) -> Path:
    """Per-task ``-i`` input: 3D detection needs the KITTI LiDAR ``.bin``, etc.

    ``TEST_IMAGE`` is only the fallback for tasks without a dedicated sample —
    hardcoding it fed ``sfa3d_608x608`` a JPG, which its runner rejects (rc=255).
    """
    return resolve_cpp_exe_input(executable, default=TEST_IMAGE)

# Save outputs under a stable, inspectable dir instead of pytest's throwaway
# /tmp ``tmp_path`` — so the produced images/videos can be opened and reviewed
# after the run. Cleaned per-test on start (see the ``artifacts_dir`` fixture).
ARTIFACTS_ROOT = PROJECT_ROOT / ".test_artifacts" / "cpp_save_mode"

# Tasks whose --save output is a feature vector / embedding rather than a
# rendered image or video: there is no image/video file to verify (the run
# still produces run_info.txt). Keep the run assertion, skip the file check.
NO_VISUAL_OUTPUT_TASKS = {"embedding", "reid"}


@pytest.fixture
def artifacts_dir(request):
    """Per-test output dir under ``.test_artifacts`` (replaces ``tmp_path``).

    Named after the test/param id so outputs are easy to find, and wiped at the
    start of each run so a stale file from a previous run can't cause a false
    pass (or a false failure).
    """
    safe = re.sub(r"[^\w.-]", "_", request.node.name)
    d = ARTIFACTS_ROOT / safe
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


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
# 태스크 카테고리마다 대표 sync 바이너리 1개를, 실제 빌드된 것 중에서 동적으로
# 선택한다. 이전에는 exe/모델 이름을 하드코딩했는데(_TASK_EXE_IMAGE_MAP), 모델이
# 재배포/개명되면 매칭이 전부 깨져 커버리지가 0이 되고 test_task_coverage 가
# 하드 실패했다(run_tc_sonar red). 공용 discover_cpp_executables 를 써서
# 빌드/모델 유무를 자동으로 따라가게 한다.
def _build_task_save_cases() -> List[tuple]:
    """Return one ``(task, exe_name, model_path, image_path)`` per task category.

    Uses :func:`discover_cpp_executables` so the set tracks whatever single-model
    ``*_sync`` binaries + matching ``.dxnn`` files are actually present, instead
    of a hardcoded name map that goes stale when models are renamed/re-published
    (the naive exact-match matcher missed variant suffixes like ``_1``/``-1``,
    so e.g. ``yolov5s`` never matched ``YOLOV5S_1.dxnn``). Multi-model
    executables (no single ``-m`` model) are skipped — the save tests launch
    with ``-m <model>``. Existence is re-checked against the test launch bin dir
    (``resolve_bin_dir()``) so it stays consistent with how the tests run.
    """
    cases: List[tuple] = []
    seen: set = set()
    for task, exe_name, model_args, is_multi, image_rel in discover_cpp_executables(suffixes=("_sync",)):
        if task in seen or is_multi:
            continue
        if len(model_args) != 2 or model_args[0] != "-m":
            continue
        if not (BIN_DIR / exe_name).exists():
            continue
        cases.append((task, exe_name, Path(model_args[1]), PROJECT_ROOT / image_rel))
        seen.add(task)
    return sorted(cases, key=lambda c: c[0])


TASK_SAVE_CASES: List[tuple] = _build_task_save_cases()

TASK_SAVE_PARAMS = [
    pytest.param(task, exe, mp, img, id=f"{task}_{exe}",
                 marks=pytest.mark.sync_exec)
    for task, exe, mp, img in TASK_SAVE_CASES
]


SYNC_CASES = discover_sync_cases()
REPRESENTATIVE_CASES = _pick_representative(SYNC_CASES)

# exe_name → task category, used to partition params by input-source capability.
EXE_TASK_MAP = cpp_exe_task_map(suffixes=("_sync",))


# Image save runs on every representative model. Video save runs only on
# stream-capable ones: image-only tasks (SDKREQ-517) are filtered out up-front
# (proactive) so the video test never even parametrizes a model that can't take
# ``-v`` — no runtime skip, no false failure.
IMAGE_SAVE_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in REPRESENTATIVE_CASES
]
VIDEO_SAVE_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in REPRESENTATIVE_CASES
    if EXE_TASK_MAP.get(name) not in IMAGE_ONLY_TASKS
]
# Negative test: one model per task whose runner HARD-REJECTS stream input, to
# assert the SDKREQ-517 exclusion is actually enforced (not merely skipped).
IMAGE_ONLY_REJECT_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in stream_rejecting_cpp_cases(STREAM_REJECTING_TASKS_CPP, BIN_DIR)
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.save_mode
class TestSaveMode:
    """Test --save and --save-dir CLI options."""

    @pytest.mark.parametrize("executable,model_path", IMAGE_SAVE_PARAMS)
    def test_image_save_creates_run_dir(self, executable, model_path, artifacts_dir):
        """Run with --save --save-dir, verify run_dir structure for image input."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        test_input = _test_input_for(executable)
        if not test_input.exists():
            pytest.skip(f"Test input not found: {test_input}")

        save_dir = artifacts_dir / "save_test"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(test_input),
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

    @pytest.mark.parametrize("executable,model_path", VIDEO_SAVE_PARAMS)
    def test_video_save_creates_output(self, executable, model_path, artifacts_dir):
        """Run with --save on video input, verify video file is produced.

        Image-only tasks (SDKREQ-517) are excluded from ``VIDEO_SAVE_PARAMS``
        up-front, so this only ever runs on stream-capable models. That they
        reject ``-v`` is asserted separately by ``test_image_only_rejects_video``.
        """
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        # Skip face models (too slow for video)
        if "face" in executable.lower():
            pytest.skip(f"{executable}: face model too slow for video save test")

        save_dir = artifacts_dir / "video_save"
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

    @pytest.mark.parametrize("executable,model_path", IMAGE_SAVE_PARAMS)
    def test_run_info_contains_metadata(self, executable, model_path, artifacts_dir):
        """Verify run_info.txt contains expected metadata fields."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        test_input = _test_input_for(executable)
        if not test_input.exists():
            pytest.skip(f"Test input not found: {test_input}")

        save_dir = artifacts_dir / "metadata_test"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(test_input),
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

    @pytest.mark.parametrize("executable,model_path", IMAGE_ONLY_REJECT_PARAMS)
    def test_image_only_rejects_video(self, executable, model_path):
        """SDKREQ-517: image-only single-model examples must REJECT stream input.

        Positive counterpart to ``test_video_save_creates_output``: instead of
        skipping image-only models, verify their runners refuse ``-v`` with a
        non-zero exit and the documented "image input only" message. The guard
        fires before the inference engine is constructed, so this needs no NPU.
        """
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")

        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            env=env, cwd=str(PROJECT_ROOT),
        )

        task = EXE_TASK_MAP.get(executable)
        # Two valid rejection forms, both satisfying SDKREQ-517 (video refused):
        #   (a) runtime guard — the example registers -v then refuses it with
        #       "...supports image input only..." (sfa3d, dope, ...)
        #   (b) the example never registers the -v option, so the CLI parser
        #       fails with "Option 'v' does not exist" (embedding/reid examples).
        # rc != 0 alone is too weak (a crash / missing model also exits non-zero),
        # so require rc != 0 AND a recognised rejection signature.
        stderr_low = result.stderr.lower()
        rejected = ("image input only" in stderr_low) or ("does not exist" in stderr_low)
        assert result.returncode != 0 and rejected, (
            f"[{task}] {executable}: expected -v to be rejected (image-only task, "
            f"SDKREQ-517) but got rc={result.returncode}\n"
            f"STDERR: {result.stderr[-500:]}"
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
        self, task, executable, model_path, image_path, artifacts_dir
    ):
        """Run with --save on image input, verify output image file (jpg/png) is produced."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        save_dir = artifacts_dir / f"save_{task}_img"
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

        # embedding/reid 등은 feature vector 출력이라 렌더된 이미지 파일이 없다.
        # 실행(rc==0)과 run_dir 생성까지만 확인하고 이미지 파일 검증은 건너뛴다.
        if task in NO_VISUAL_OUTPUT_TASKS:
            run_infos = list(save_dir.rglob("run_info.txt"))
            assert len(run_infos) >= 1, f"[{task}] no run_info.txt produced under {save_dir}"
            pytest.skip(f"[{task}] produces a feature vector, no output image to verify")

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
        self, task, executable, model_path, image_path, artifacts_dir
    ):
        """Run with --save on video input, verify output video file (mp4/avi) is produced."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")
        # Image-only task categories have no video/stream save path; face models
        # are too slow for a video save run (mirrors TestSaveMode.test_video_save).
        if task in IMAGE_ONLY_TASKS:
            pytest.skip(f"[{task}] image-only task; no video save path")
        if "face" in task.lower() or "face" in executable.lower():
            pytest.skip(f"[{task}] {executable}: face model too slow for video save test")

        save_dir = artifacts_dir / f"save_{task}_video"
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
        """Report which task categories are covered by save-output tests.

        Cases are discovered dynamically (see ``_build_task_save_cases``), so
        this reflects whatever ``*_sync`` binaries + matching ``.dxnn`` files the
        current build actually provides. When none are discoverable (e.g. a
        coverage build that produced only a subset, or a machine without matching
        models) there is nothing to verify — skip, matching the per-test
        ``pytest.skip`` behaviour throughout this module, instead of hard-failing
        the whole suite (which turned an environment gap into a red
        ``run_tc_sonar`` pipeline).
        """
        covered_tasks = {task for task, _, _, _ in TASK_SAVE_CASES}
        print(f"\n  Covered tasks ({len(covered_tasks)}): {sorted(covered_tasks)}")
        if not covered_tasks:
            pytest.skip(
                "No single-model *_sync binaries + matching .dxnn discovered "
                "in this build; nothing to verify."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
