"""Python async examples deliver the same per-frame results as sync, in input order (U-03).

The AsyncRunner waits for results in submit order (one wait worker, FIFO
queues), so this holds by construction; this test is the hardware proof.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import ASSETS_DIR, PROJECT_ROOT  # noqa: E402
from test_helpers.proc import example_python, run_bounded  # noqa: E402
from test_helpers.utils import discover_python_scripts, setup_environment  # noqa: E402
from test_helpers.verify import first_frame_difference, make_video_clip, read_verify_frames  # noqa: E402

SOURCE_VIDEO = ASSETS_DIR / "videos" / "blackbox-city-road.mp4"
CLIP_FRAMES = 120
# yolov5-s, yolopv2, superpoint, then one model per postprocessor family (U-26).
NAMED = ["yolov5-s_640x640", "yolopv2_384x640", "superpoint_480x640",
         "retinaface_mobilenet-0.25_640x640", "ulfgfd-slim_240x320", "yolov5-n-face_640x640",
         "yolov7-s-face_640x640", "yolact_regnet-x800mf_512x512", "yolo26-n-seg_640x640",
         "yolov5-n-seg_640x640", "damoyolo-t_640x640", "efficientdet-d1_640x640",
         "nanodet-plus_416x416", "ssd-mobilenetv2-lite_300x300", "yolov10-n_640x640",
         "yolov7_640x640_nodecode", "yolov9-m_640x640", "yolox-t_416x416",
         "centerpose_repvgg-a0_416x416", "yolo26-n-pose_640x640", "yolov5-s6-pose_640x640",
         "yolov5-s_640x640_ppu", "yolov8-n_640x640_ppu",
         "bisenetv1_1024x2048", "segformer_mit-b0_512x1024"]
SCRIPTS = {name: (sync_scripts, async_scripts, model)
           for _task, name, sync_scripts, async_scripts, model in discover_python_scripts()}


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not SOURCE_VIDEO.exists():
        pytest.skip("source video missing: {}".format(SOURCE_VIDEO))
    path = tmp_path_factory.mktemp("clip") / "clip120.avi"
    assert make_video_clip(SOURCE_VIDEO, path, CLIP_FRAMES) == CLIP_FRAMES
    return path


def _script(name, mode):
    sync_scripts, async_scripts, model = SCRIPTS.get(name, ([], [], None))
    wanted = "{}_{}.py".format(name, mode)
    found = [s for s in (sync_scripts + async_scripts) if s.name == wanted]
    if model is None or not found:
        pytest.skip("{} or its model is missing".format(wanted))
    return found[0], model


def _frames(script, model, clip, workdir):
    verify_dir = Path(workdir) / ("verify_" + script.stem)
    env = setup_environment()
    env["DXAPP_VERIFY"] = "1"
    env["DXAPP_VERIFY_DIR"] = str(verify_dir)
    result = run_bounded([example_python(), str(script), "--model", str(model), "--video", str(clip),
                          "--no-display", "--loop", "1"],
                         capture_output=True, text=True, timeout=900, env=env, cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    return read_verify_frames(verify_dir)


@pytest.mark.verify
@pytest.mark.parametrize("name", NAMED)
def test_python_async_frames_equal_sync_frames(name, clip, tmp_path):
    sync_script, model = _script(name, "sync")
    async_script, _ = _script(name, "async")
    expected = _frames(sync_script, model, clip, tmp_path)
    got = _frames(async_script, model, clip, tmp_path)
    assert len(expected) == CLIP_FRAMES
    difference = first_frame_difference(expected, got)
    assert difference is None, "{}: {}".format(name, difference)
