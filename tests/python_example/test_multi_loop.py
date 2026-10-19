"""
Multi-loop tests for Python inference scripts.

Verifies that ``--loop 2`` on a video exits 0 and processes the clip twice:
the performance summary's ``Total Frames`` is ~2x the frames one pass reads
(the C++ test's 1.5-2.5 window). Tests both sync and async scripts.

Mirrors ``tests/cpp_example/test_multi_loop.py``.
"""

import functools
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.proc import example_python, run_bounded  # noqa: E402
from test_helpers.constants import IMAGE_ONLY_TASKS, PROJECT_ROOT  # noqa: E402
from test_helpers.utils import (  # noqa: E402
    discover_python_scripts, py_script_task, py_variant_image_only, setup_environment,
)


# Two passes exercise the reopen between loops (as the C++ multi-loop test,
# -l 2). A short clip keeps every model inside the budget. The slowest measured
# Python throughput is ppmatting-hrnet-w48 from Model Zoo 2_5_0 (~0.65 FPS):
# 2 x 192 frames here took 551-601 s. superpoint_sync (~1.1 FPS) took ~350 s.
_SHORT_VIDEO = PROJECT_ROOT / "assets" / "videos" / "person-pair-hallway.mp4"  # 192 frames
_TEST_VIDEO = (_SHORT_VIDEO if _SHORT_VIDEO.exists()
               else PROJECT_ROOT / "assets" / "videos" / "dance-group.mov")
_LOOP_COUNT = 2
# 1200 s holds the slowest script on the 192-frame clip with a 2x margin; the
# budget scales with the clip actually used (dance-group.mov, 478 frames,
# gets ~3000 s).
_BASE_TIMEOUT_S = 1200
_BASE_FRAMES = 192

_DECODE_FRAMES = (
    "import sys, cv2\n"
    "cap = cv2.VideoCapture(sys.argv[1])\n"
    "n = 0\n"
    "while cap.read()[0]:\n"
    "    n += 1\n"
    "print(n)\n"
)


@functools.lru_cache(maxsize=None)
def _clip_frames(video: Path) -> Optional[int]:
    """Frames one pass over *video* reads, decoded by the examples' own
    interpreter (the pytest process may have a mocked cv2)."""
    try:
        result = run_bounded([example_python(), "-c", _DECODE_FRAMES, str(video)],
                             capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _timeout_for(frames: int) -> int:
    return max(_BASE_TIMEOUT_S, int(math.ceil(_BASE_TIMEOUT_S * frames / _BASE_FRAMES)))


def _parse_total_frames(output: str) -> Optional[int]:
    """'Total Frames : N' from the performance summary (the last one: the
    summary comes last)."""
    matches = re.findall(r"Total\s+Frames\s*:?\s*(\d+)", output, re.IGNORECASE)
    return int(matches[-1]) if matches else None


# ---------------------------------------------------------------------------
# Discovery — sync + async scripts with a model
# ---------------------------------------------------------------------------

def _discover_scripts() -> List[pytest.param]:
    raw = discover_python_scripts(suffixes=("_sync", "_async"))
    params = []
    for _task, _model_name, sync_scripts, async_scripts, model_path in raw:
        if model_path is None:
            continue
        for script, mode in ([(s, "sync") for s in sync_scripts] + [(s, "async") for s in async_scripts]):
            marker = pytest.mark.sync_exec if mode == "sync" else pytest.mark.async_exec
            params.append(pytest.param(
                script, model_path,
                id=script.stem,
                marks=marker,
            ))
    return params


SCRIPT_PARAMS = _discover_scripts()

if not SCRIPT_PARAMS:
    pytest.skip(
        "No models found in assets/models/ — run setup_sample_models.sh first",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers pinned without hardware
# ---------------------------------------------------------------------------

def test_timeout_scales_with_the_clip():
    assert _timeout_for(_BASE_FRAMES) == _BASE_TIMEOUT_S
    assert _timeout_for(100) == _BASE_TIMEOUT_S           # never below the base
    assert _timeout_for(478) >= 1.5 * 870                 # dance-group.mov, slowest ~870 s
    # ppmatting-hrnet-w48 (Model Zoo 2_5_0) took 551-601 s for 2 x 192 frames.
    assert _timeout_for(_BASE_FRAMES) >= 1.5 * 601


def test_total_frames_is_the_summary_s_last_count():
    output = "Total frames: 192\n...\n Total Frames    :    384\n"
    assert _parse_total_frames(output) == 384
    assert _parse_total_frames("no summary") is None


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.multi_loop
@pytest.mark.parametrize("script,model", SCRIPT_PARAMS)
def test_multi_loop_video(script: Path, model: Path):
    """Run script with --loop 2 on video: exit 0, and the clip processed twice."""
    if not _TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {_TEST_VIDEO}")

    # Image-only tasks (3d_object_detection/sfa3d, embedding, reid, …) reject
    # --video: their runners sys.exit(1) (SDKREQ-517 excludes them from stream
    # inference). Scripts live at src/python_example/<task>/<family>/<variant>/,
    # so the task is the first directory under src/python_example/.
    task = py_script_task(script)
    if task in IMAGE_ONLY_TASKS:
        pytest.skip(f"[{task}] {script.name}: image-only task; no video/stream path")
    # A variant of a video-capable task can still be image-only (casvit under
    # image_classification): the runner reads the variant's config.json.
    if py_variant_image_only(script):
        pytest.skip(f"[{task}] {script.name}: image-only variant (config.json); "
                    "no video/stream path")

    clip_frames = _clip_frames(_TEST_VIDEO)
    assert clip_frames, f"could not count the frames of {_TEST_VIDEO}"
    timeout_s = _timeout_for(clip_frames)

    cmd = [
        example_python(), str(script),
        "--model", str(model),
        "--video", str(_TEST_VIDEO),
        "--loop", str(_LOOP_COUNT),
        "--no-display",
    ]

    try:
        result = run_bounded(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=setup_environment(),
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} --loop {_LOOP_COUNT} timed out after {timeout_s} s "
                    f"({clip_frames}-frame clip)")

    assert result.returncode == 0, (
        f"{script.name}: multi-loop FAILED (exit {result.returncode})\n"
        f"CMD    : {' '.join(cmd)}\n"
        f"stdout : {result.stdout[-2000:]}\nstderr : {result.stderr[-2000:]}"
    )

    output = result.stdout + result.stderr
    frames = _parse_total_frames(output)
    assert frames is not None, (
        f"{script.name}: no 'Total Frames' in the summary\nstdout : {result.stdout[-2000:]}")
    ratio = frames / clip_frames
    assert 1.5 <= ratio <= 2.5, (
        f"{script.name}: --loop {_LOOP_COUNT} processed {frames} frames; one pass over "
        f"{_TEST_VIDEO.name} is {clip_frames} (ratio {ratio:.2f}, expected ~2)")
