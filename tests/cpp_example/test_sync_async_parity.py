"""Async runners deliver the same per-frame results as sync, in input order (U-03, U-29).

DXAPP_VERIFY writes one record per delivered frame to <stem>.frames.jsonl.
DXRT_TASK_MAX_LOAD=40 raises dxrt's I/O buffer count (default: about 7 jobs in
flight) to the runners' ASYNC_BUFFER_SIZE, so "max" depth really has up to 40
frames in flight (YOLOPv2 runs at 20: see MAX_LOAD). No runner option is
involved: dxrt reads the variable. Every downloaded video-capable async runner
is compared at that depth (SWEEP), and the image-only super-resolution, 3D and
embedding runners in image mode. Expected time: about 5 minutes.
"""
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import IMAGE_ONLY_TASKS  # noqa: E402
from test_helpers.proc import run_bounded  # noqa: E402
from test_helpers.utils import (  # noqa: E402
    cpp_exe_task_map, discover_cpp_model_cases, load_registry, resolve_cpp_exe_input,
    setup_environment)
from test_helpers.verify import (  # noqa: E402
    first_frame_difference, inflight_max, make_video_clip, read_verify_frames)

from conftest import PROJECT_ROOT, resolve_bin_dir

BIN_DIR = resolve_bin_dir()
SOURCE_VIDEO = PROJECT_ROOT / "assets" / "videos" / "blackbox-city-road.mp4"
CLIP_FRAMES = 120
FULL_DEPTH_ENV = {"DXRT_TASK_MAX_LOAD": "40"}
MODELS = {exe[: -len("_sync")]: model
          for exe, model in discover_cpp_model_cases("_sync", BIN_DIR)}

# A plain detector, per-frame masks (YOLOPv2) and SuperPoint (keypoints and
# descriptors per frame; the port has no SuperPoint tracker).
NAMED = ["yolov8-n_640x640", "yolopv2_384x640", "superpoint_480x640"]
# DXRT_TASK_MAX_LOAD for the "max" depth, where it is not 40 (FULL_DEPTH_ENV).
# YOLOPv2: dxrt cannot allocate 30-40 I/O buffers for it on this device. At 40 it logs
# "Buffer count reduced from 40 to 31 due to NPU memory limit (4211081216 bytes)"
# and then fails with "Dynamic IPC task init failed" (SERVICE_IO, rc 255) before
# the first frame; 30 and 31 fail the same way (probed 2026-09-30). 20 works,
# and YOLOPv2's own decode limits it to about 13 in flight anyway, so 20 still
# runs it at its full depth. Other models that cannot get 40 buffers run at 40:
# dxrt logs the same reduction warning and uses the count it could allocate.
MAX_LOAD = {"yolopv2_384x640": "20"}
# In-flight depth each reaches at its MAX_LOAD on the clip (measured: yolov8-n 40
# at 40; yolopv2 13-14 at 40 on 2026-09-29, 13 at 20 on 2026-09-30; superpoint
# 14-16 at 40 on the port, 2026-10-01, where 8d0b748's runner is bounded by the
# reader, not the NPU). YOLOPv2 and SuperPoint only have to exceed dxrt's
# default of 7.
MIN_DEPTH_AT_MAX = {"yolov8-n_640x640": 40, "yolopv2_384x640": 8, "superpoint_480x640": 8}


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not SOURCE_VIDEO.exists():
        pytest.skip("source video missing: {}".format(SOURCE_VIDEO))
    path = tmp_path_factory.mktemp("clip") / "clip120.avi"
    assert make_video_clip(SOURCE_VIDEO, path, CLIP_FRAMES) == CLIP_FRAMES
    return path


def run_frames(exe, model, input_args, workdir, extra_env=None, timeout=600):
    """Run *exe* with DXAPP_VERIFY; return (per-frame records, stdout + stderr)."""
    exe_path = BIN_DIR / exe
    if not exe_path.exists():
        pytest.skip("binary not built: {}".format(exe))
    verify_dir = Path(workdir) / ("verify_" + exe)
    env = setup_environment()
    env["DXAPP_VERIFY"] = "1"
    env["DXAPP_VERIFY_DIR"] = str(verify_dir)
    env.update(extra_env or {})
    result = run_bounded([str(exe_path), "-m", str(model)] + list(input_args) + ["--no-display"],
                         capture_output=True, text=True, timeout=timeout, env=env, cwd=str(workdir))
    output = result.stdout + result.stderr
    assert result.returncode == 0, "{} rc={}\n{}".format(exe, result.returncode, output[-2000:])
    return read_verify_frames(verify_dir), output


def max_depth_env(name):
    """The child environment that runs *name* at its full dxrt depth."""
    return {"DXRT_TASK_MAX_LOAD": MAX_LOAD.get(name, FULL_DEPTH_ENV["DXRT_TASK_MAX_LOAD"])}


def _model(name):
    if name not in MODELS:
        pytest.skip("model not downloaded for {}".format(name))
    return MODELS[name]


@pytest.mark.verify
@pytest.mark.parametrize("depth", ["default", "max"])
@pytest.mark.parametrize("name", NAMED)
def test_async_frames_equal_sync_frames(name, depth, clip, tmp_path):
    model = _model(name)
    video = ["-v", str(clip), "-l", "1"]
    expected, _ = run_frames(name + "_sync", model, video, tmp_path)
    got, output = run_frames(name + "_async", model, video, tmp_path,
                             max_depth_env(name) if depth == "max" else None)
    assert len(expected) == CLIP_FRAMES
    assert [f["frame"] for f in got] == list(range(CLIP_FRAMES))
    difference = first_frame_difference(expected, got)
    assert difference is None, "{} async at {} depth: {}".format(name, depth, difference)
    if depth == "max":
        seen = inflight_max(output)
        assert seen is not None and seen >= MIN_DEPTH_AT_MAX[name], (
            "{} reached only {} in flight".format(name, seen))


# Review Focus 4: tickets and frame numbers continue across --loop passes.
@pytest.mark.verify
def test_async_loops_continue_the_frame_order(clip, tmp_path):
    model = _model("yolov8-n_640x640")
    video = ["-v", str(clip), "-l", "2"]
    expected, _ = run_frames("yolov8-n_640x640_sync", model, video, tmp_path)
    got, _ = run_frames("yolov8-n_640x640_async", model, video, tmp_path, FULL_DEPTH_ENV)
    assert len(expected) == 2 * CLIP_FRAMES
    difference = first_frame_difference(expected, got)
    assert difference is None, difference


# Review Focus 1: Ctrl-C while frames are buffered out of order.
@pytest.mark.verify
def test_async_interrupt_mid_video_keeps_order_and_finalizes_the_video(clip, tmp_path):
    import cv2
    model = _model("yolov8-n_640x640")
    expected, _ = run_frames("yolov8-n_640x640_sync", model, ["-v", str(clip), "-l", "1"], tmp_path)
    exe = BIN_DIR / "yolov8-n_640x640_async"
    verify_dir = tmp_path / "verify_interrupted"
    out_dir = tmp_path / "out"
    env = setup_environment()
    env.update({"DXAPP_VERIFY": "1", "DXAPP_VERIFY_DIR": str(verify_dir)})
    env.update(FULL_DEPTH_ENV)
    proc = subprocess.Popen(
        [str(exe), "-m", str(model), "-v", str(clip), "-l", "20", "--no-display",
         "-s", "--save-dir", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env, cwd=str(tmp_path), start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        ready = False
        while time.monotonic() < deadline and proc.poll() is None:
            found = list(verify_dir.glob("*.frames.jsonl"))
            if found and found[0].read_text().count("\n") >= 60:
                ready = True
                break
            time.sleep(0.1)
        assert ready, "fewer than 60 frames delivered within 60 s"
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, output[-2000:]
    got = read_verify_frames(verify_dir)
    assert [f["frame"] for f in got] == list(range(len(got)))
    for record in got:  # loop k replays the clip, so frame i is clip frame i % CLIP_FRAMES
        want = dict(expected[record["frame"] % CLIP_FRAMES])
        want["frame"] = record["frame"]
        assert record == want, "delivered frame {} is not clip frame {}".format(
            record["frame"], record["frame"] % CLIP_FRAMES)
    videos = list(out_dir.rglob("output.*"))
    assert len(videos) == 1, videos
    cap = cv2.VideoCapture(str(videos[0]))
    written = 0
    while cap.read()[0]:
        written += 1
    cap.release()
    assert written == len(got), "saved {} frames, delivered {}".format(written, len(got))


_TASKS = cpp_exe_task_map()
# Registry rows that take no video although their task does (casvit-t_224x224:
# an embedding model filed under image_classification).
_IMAGE_ONLY_VARIANTS = {e.get("variant") for e in load_registry() if e.get("image_only")}
# Every downloaded async runner that takes video. Image-only tasks (IMAGE_ONLY_TASKS,
# super_resolution included: ESPCN is tiled, thousands of 17x17 tiles per video
# frame) and image-only rows are left out; the SR, 3D and embedding runners are
# compared in image mode below.
SWEEP = sorted(name for name in MODELS
               if _TASKS.get(name + "_async") not in IMAGE_ONLY_TASKS
               and name not in _IMAGE_ONLY_VARIANTS and not name.startswith("espcn"))
SWEEP_FRAMES = 60


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory):
    if not SOURCE_VIDEO.exists():
        pytest.skip("source video missing: {}".format(SOURCE_VIDEO))
    path = tmp_path_factory.mktemp("short_clip") / "clip60.avi"
    assert make_video_clip(SOURCE_VIDEO, path, SWEEP_FRAMES) == SWEEP_FRAMES
    return path


@pytest.mark.verify
@pytest.mark.parametrize("name", SWEEP)
def test_every_async_runner_equals_sync_at_full_depth(name, short_clip, tmp_path):
    model = _model(name)
    video = ["-v", str(short_clip), "-l", "1"]
    expected, _ = run_frames(name + "_sync", model, video, tmp_path)
    got, output = run_frames(name + "_async", model, video, tmp_path, max_depth_env(name))
    print("\n  {}: {} frames, in flight max {}".format(name, len(got), inflight_max(output)))
    difference = first_frame_difference(expected, got)
    assert difference is None, "{}: {}".format(name, difference)


IMAGE_MODE = [
    # (name, input): the SR runners on their low-res sample (ESPCN is tiled:
    # 3 frames); SFA3D and the stateful embedding reference over a directory
    # of inputs.
    ("espcn-x4_17x17", None, ["-l", "3"]),
    ("realesrgan-x2_192x192", None, []),
    ("sfa3d_608x608", PROJECT_ROOT / "sample" / "kitti" / "velodyne", []),
    ("arcface_mobilefacenet_112x112", PROJECT_ROOT / "sample" / "img" / "face_pair", []),
]


@pytest.mark.verify
@pytest.mark.parametrize("name,path,extra", IMAGE_MODE, ids=[c[0] for c in IMAGE_MODE])
def test_image_mode_async_equals_sync(name, path, extra, tmp_path):
    model = _model(name)
    source = path if path is not None else resolve_cpp_exe_input(name + "_sync")
    if source is None or not Path(source).exists():
        pytest.skip("input missing for {}".format(name))
    args = ["-i", str(source)] + extra
    expected, _ = run_frames(name + "_sync", model, args, tmp_path)
    got, _ = run_frames(name + "_async", model, args, tmp_path, max_depth_env(name))
    assert expected, "{} dumped nothing".format(name)
    difference = first_frame_difference(expected, got)
    assert difference is None, "{}: {}".format(name, difference)
