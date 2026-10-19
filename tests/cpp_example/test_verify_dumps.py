"""DXAPP_VERIFY regressions, checked at run time (U-30, U-31).

* async classification/detection once dumped a moved-from, empty vector;
* sync restoration once dumped an empty 0x0 record; the tiled-SR path did
  until SP3, and async tiled ESPCN and both SFA3D runners wrote nothing;
* YOLOPv2 dumped boxes only.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.proc import run_bounded  # noqa: E402
from test_helpers.utils import dxnn_for_exe, resolve_cpp_exe_input, setup_environment, strip_variant_suffix  # noqa: E402
from test_helpers.verify import read_verify_json  # noqa: E402

from conftest import PROJECT_ROOT, resolve_bin_dir

BIN_DIR = resolve_bin_dir()


def _model_for(executable):
    """The variant's .dxnn when both it and the binary exist, else None."""
    if not (BIN_DIR / executable).is_file():
        return None
    return dxnn_for_exe(strip_variant_suffix(executable))


def _dump(executable, tmp_path):
    model = _model_for(executable)
    if model is None:
        pytest.skip("{} or its model is missing".format(executable))
    test_input = resolve_cpp_exe_input(executable, default=PROJECT_ROOT / "sample" / "img" / "sample_kitchen.jpg")
    env = setup_environment()
    env["DXAPP_VERIFY"] = "1"
    env["DXAPP_VERIFY_DIR"] = str(tmp_path / "verify")
    result = run_bounded([str(BIN_DIR / executable), "-m", str(model), "-i", str(test_input),
                          "--no-display", "-l", "1"],
                         capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    return read_verify_json(tmp_path / "verify")


@pytest.mark.verify
@pytest.mark.parametrize("executable", ["resnet50_224x224_async", "mobilenetv2_224x224_async"])
def test_async_classification_dumps_its_classes(executable, tmp_path):
    data = _dump(executable, tmp_path)
    assert data["classifications"] and data["top_k_confs"]


@pytest.mark.verify
@pytest.mark.parametrize("executable", ["yolov8-n_640x640_async", "yolov5-s_640x640_async"])
def test_async_detection_dumps_its_boxes(executable, tmp_path):
    assert _dump(executable, tmp_path)["detections"]


@pytest.mark.verify
@pytest.mark.parametrize("executable", ["dncnn-50_512x512_sync", "espcn-x4_17x17_sync", "espcn-x4_17x17_async"])
def test_restoration_dumps_output_stats_at_the_input_size(executable, tmp_path):
    data = _dump(executable, tmp_path)
    assert data["image_height"] > 0 and data["image_width"] > 0
    assert data["output_stats"]["shape"] == data["output_shape"]


@pytest.mark.verify
@pytest.mark.parametrize("executable", ["sfa3d_608x608_sync", "sfa3d_608x608_async"])
def test_3d_detection_dumps_its_boxes(executable, tmp_path):
    data = _dump(executable, tmp_path)
    assert data["task"] == "3d_object_detection"
    assert data["detections"] and {"bev", "center", "dims", "yaw"} <= set(data["detections"][0])


@pytest.mark.verify
@pytest.mark.parametrize("executable", ["yolopv2_384x640_sync", "yolopv2_384x640_async"])
def test_yolopv2_dumps_its_masks(executable, tmp_path):
    data = _dump(executable, tmp_path)
    for key in ("drivable_stats", "lane_stats"):
        shape = data[key]["shape"]
        assert len(shape) == 2 and min(shape) > 0, (key, shape)
    assert "detections" in data
