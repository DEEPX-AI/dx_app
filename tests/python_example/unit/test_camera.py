"""Raspberry Pi camera support: common/utility/camera.py and its call sites.

Mirrors the camera cases in src/cpp_example/common/test/common_unit_test.cpp.
"""

import logging
import subprocess

import cv2
import pytest

from common.utility import camera

LIST_TWO_CAMERAS = (
    "Available cameras\n"
    "-----------------\n"
    "0 : imx708_wide [4608x2592 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a)\n"
    "    Modes: 'SRGGB10_CSI2P' : 1536x864 [120.13 fps - (768, 432)/3072x1728 crop]\n"
    "                             2304x1296 [56.03 fps - (0, 0)/4608x2592 crop]\n"
    "\n"
    "1 : imx219 [3280x2464 8-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@80000/imx219@10)\n"
    "    Modes: 'SRGGB8' : 640x480 [103.33 fps - (1000, 752)/1280x960 crop]\n"
)
CAM0_ID = "/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a"
CAM1_ID = "/base/axi/pcie@120000/rp1/i2c@80000/imx219@10"


# --- pure helpers ----------------------------------------------------------

def test_raspberry_pi_model_string():
    assert camera.is_raspberry_pi_model("Raspberry Pi 5 Model B Rev 1.0\0")
    assert camera.is_raspberry_pi_model("raspberry pi 4 model b")
    assert not camera.is_raspberry_pi_model("Radxa ROCK 5B")
    assert not camera.is_raspberry_pi_model("")


def test_is_raspberry_pi_reads_the_device_tree(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.write_bytes(b"Raspberry Pi 5 Model B Rev 1.0\0")
    monkeypatch.setattr(camera, "_DEVICE_TREE_MODEL_PATHS",
                        (str(tmp_path / "missing"), str(model)))
    assert camera.is_raspberry_pi()
    model.write_bytes(b"Generic x86 board\0")
    assert not camera.is_raspberry_pi()


@pytest.mark.parametrize("name, csi", [
    ("rp1-cfe", True), ("unicam", True), ("pispbe", True), ("bcm2835-isp", True),
    # The legacy V4L2 capture driver opens through OpenCV's V4L2 backend.
    ("bcm2835-v4l2", False),
    ("UVC Camera (046d:0825)", False), ("", False),
])
def test_csi_node_names(name, csi):
    assert camera.is_rpi_csi_device_name(name) is csi


def test_csi_video_node_reads_sysfs(tmp_path, monkeypatch):
    (tmp_path / "video0").mkdir()
    (tmp_path / "video0" / "name").write_text("rp1-cfe\n")
    (tmp_path / "video8").mkdir()
    (tmp_path / "video8" / "name").write_text("UVC Camera (046d:0825)\n")
    monkeypatch.setattr(camera, "_V4L2_SYSFS_DIR", str(tmp_path))
    assert camera.v4l2_device_name(0) == "rp1-cfe"
    assert camera.is_rpi_csi_video_node(0)
    assert not camera.is_rpi_csi_video_node(8)
    assert camera.v4l2_device_name(3) == ""
    assert not camera.is_rpi_csi_video_node(3)


def test_libcamera_id_comes_from_the_camera_list():
    assert camera.parse_libcamera_camera_id(LIST_TWO_CAMERAS, 0) == CAM0_ID
    assert camera.parse_libcamera_camera_id(LIST_TWO_CAMERAS, 1) == CAM1_ID
    assert camera.parse_libcamera_camera_id(LIST_TWO_CAMERAS, 2) == ""
    assert camera.parse_libcamera_camera_id("No cameras available!\n", 0) == ""
    assert camera.parse_libcamera_camera_id("", 0) == ""


def test_libcamera_pipeline_string():
    assert camera.build_libcamera_pipeline("", 1280, 720) == (
        "libcamerasrc ! video/x-raw,width=1280,height=720,format=NV12"
        " ! videoconvert ! video/x-raw,format=BGR"
        " ! appsink drop=true max-buffers=2")
    assert camera.build_libcamera_pipeline(
        "/base/soc/i2c0mux/i2c@1/imx477@1a", 640, 480) == (
        'libcamerasrc camera-name="/base/soc/i2c0mux/i2c@1/imx477@1a"'
        " ! video/x-raw,width=640,height=480,format=NV12"
        " ! videoconvert ! video/x-raw,format=BGR"
        " ! appsink drop=true max-buffers=2")


def test_gstreamer_from_build_information():
    assert camera.build_info_has_gstreamer(
        "  Video I/O:\n    GStreamer:                   YES (1.26.0)\n")
    assert not camera.build_info_has_gstreamer(
        "  Video I/O:\n    GStreamer:                   NO\n")
    assert not camera.build_info_has_gstreamer(
        "  Video I/O:\n    FFMPEG:                      YES\n")


# --- backend selection -----------------------------------------------------

class _Captures:
    """Stand-in for cv2.VideoCapture that records how it was opened."""

    def __init__(self, opens=lambda args: True):
        self.calls = []
        self._opens = opens

    def factory(self):
        record = self

        class FakeCapture:
            def __init__(self, *args):
                record.calls.append(args)
                self._opened = bool(args) and record._opens(args)

            def isOpened(self):
                return self._opened

            def release(self):
                self._opened = False

        return FakeCapture


@pytest.fixture
def captures(monkeypatch):
    rec = _Captures()
    monkeypatch.setattr(camera.cv2, "VideoCapture", rec.factory())
    return rec


@pytest.fixture
def on_pi(monkeypatch, tmp_path):
    """A Raspberry Pi whose /dev/video0 is a CSI node and /dev/video8 a USB webcam."""
    (tmp_path / "video0").write_text("")
    (tmp_path / "video8").write_text("")
    monkeypatch.setattr(camera, "is_raspberry_pi", lambda: True)
    monkeypatch.setattr(camera, "is_rpi_csi_video_node", lambda index: index == 0)
    monkeypatch.setattr(camera, "_DEV_VIDEO_FMT", str(tmp_path / "video{}"))
    monkeypatch.setattr(camera, "opencv_has_gstreamer", lambda: True)
    monkeypatch.setattr(camera, "has_libcamera_gst_element", lambda: True)
    monkeypatch.setattr(camera, "list_libcamera_cameras", lambda: LIST_TWO_CAMERAS)


def test_off_a_pi_the_index_opens_unchanged(captures, monkeypatch):
    monkeypatch.setattr(camera, "is_raspberry_pi", lambda: False)
    cap = camera.open_camera_capture(0)
    assert cap.isOpened()
    assert captures.calls == [(0,)]


def test_a_file_or_url_source_is_never_routed(captures, on_pi):
    camera.open_camera_capture("clip.mp4")
    camera.open_camera_capture("rtsp://host/stream")
    assert captures.calls == [("clip.mp4",), ("rtsp://host/stream",)]


def test_pi_csi_node_opens_libcamera(captures, on_pi):
    cap = camera.open_camera_capture(0)
    assert cap.isOpened()
    assert captures.calls == [(camera.build_libcamera_pipeline(CAM0_ID, 1280, 720),
                               cv2.CAP_GSTREAMER)]


def test_pi_camera_source_size_reaches_the_pipeline(captures, on_pi):
    camera.open_camera_capture(0, 640, 480)
    assert captures.calls == [(camera.build_libcamera_pipeline(CAM0_ID, 640, 480),
                               cv2.CAP_GSTREAMER)]


def test_pi_index_without_a_node_is_libcamera_camera_n(captures, on_pi):
    camera.open_camera_capture(1)  # no /dev/video1 in the fixture
    assert captures.calls == [(camera.build_libcamera_pipeline(CAM1_ID, 1280, 720),
                               cv2.CAP_GSTREAMER)]


def test_pi_usb_webcam_keeps_v4l2(captures, on_pi):
    cap = camera.open_camera_capture(8)
    assert cap.isOpened()
    assert captures.calls == [(8,)]


def test_pi_usb_webcam_that_fails_falls_back_to_libcamera(monkeypatch, on_pi):
    rec = _Captures(opens=lambda args: args != (8,))
    monkeypatch.setattr(camera.cv2, "VideoCapture", rec.factory())
    monkeypatch.setattr(camera, "list_libcamera_cameras",
                        lambda: LIST_TWO_CAMERAS.replace("1 : imx219", "8 : imx219"))
    assert camera.open_camera_capture(8).isOpened()
    assert rec.calls == [(8,), (camera.build_libcamera_pipeline(CAM1_ID, 1280, 720),
                                cv2.CAP_GSTREAMER)]


def test_pi_without_gstreamer_in_opencv_says_so(captures, on_pi, monkeypatch, caplog):
    monkeypatch.setattr(camera, "opencv_has_gstreamer", lambda: False)
    with caplog.at_level(logging.ERROR, logger=camera.__name__):
        assert not camera.open_camera_capture(0).isOpened()
    assert "GStreamer: NO" in caplog.text
    assert all(len(args) < 2 for args in captures.calls)  # no pipeline tried


def test_pi_without_the_libcamera_plugin_says_how_to_install(captures, on_pi,
                                                             monkeypatch, caplog):
    monkeypatch.setattr(camera, "has_libcamera_gst_element", lambda: False)
    with caplog.at_level(logging.ERROR, logger=camera.__name__):
        assert not camera.open_camera_capture(0).isOpened()
    assert "gstreamer1.0-libcamera" in caplog.text
    assert all(len(args) < 2 for args in captures.calls)


def test_pi_camera_not_in_the_list_is_refused(captures, on_pi, caplog):
    with caplog.at_level(logging.ERROR, logger=camera.__name__):
        assert not camera.open_camera_capture(5).isOpened()
    assert "no camera 5" in caplog.text
    assert all(len(args) < 2 for args in captures.calls)


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("rpicam-hello", 30),
                                   OSError("exec format error")])
def test_a_list_tool_that_cannot_answer_gives_no_listing(monkeypatch, error):
    """A hung or unrunnable rpicam-hello is "no listing" (None), not an empty
    one, so camera 0 still opens libcamerasrc's first camera."""
    monkeypatch.setattr(camera.shutil, "which", lambda tool: "/usr/bin/" + tool)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(camera.subprocess, "run", fail)
    assert camera.list_libcamera_cameras() is None


def test_pi_without_a_list_tool(captures, on_pi, monkeypatch, caplog):
    monkeypatch.setattr(camera, "list_libcamera_cameras", lambda: None)
    assert camera.open_camera_capture(0).isOpened()
    assert captures.calls[-1] == (camera.build_libcamera_pipeline("", 1280, 720),
                                  cv2.CAP_GSTREAMER)
    with caplog.at_level(logging.ERROR, logger=camera.__name__):
        assert not camera.open_camera_capture(1).isOpened()
    assert "rpicam-hello" in caplog.text


# --- call sites ------------------------------------------------------------

class _Unopened:
    def isOpened(self):
        return False

    def release(self):
        pass


def _record_opens(monkeypatch, module):
    opened = []

    def fake_open(source, *args):
        opened.append((source,) + args)
        return _Unopened()

    monkeypatch.setattr(module, "open_camera_capture", fake_open)
    return opened


def test_sync_stream_opens_the_camera_through_the_helper(monkeypatch):
    from common.runner import sync_runner
    opened = _record_opens(monkeypatch, sync_runner)
    runner = sync_runner.SyncRunner.__new__(sync_runner.SyncRunner)
    with pytest.raises(RuntimeError, match="Failed to open source: 3"):
        runner._stream_inference_once(3, display=False, save_enabled=False, run_dir=None)
    assert opened == [(3,)]


def test_async_sr_stream_opens_the_camera_through_the_helper(monkeypatch):
    from common.runner import async_runner
    opened = _record_opens(monkeypatch, async_runner)
    runner = async_runner.AsyncRunner.__new__(async_runner.AsyncRunner)
    assert runner._run_stream_sr_once(2, "camera:2", False, False) is None
    assert opened == [(2,)]


def test_async_verbose_probe_opens_the_camera_through_the_helper(monkeypatch):
    from common.runner import async_runner
    opened = _record_opens(monkeypatch, async_runner)

    class _Stop(Exception):
        pass

    def stop(_uri):
        raise _Stop()

    monkeypatch.setattr(async_runner.InputFactory, "create", staticmethod(stop))
    runner = async_runner.AsyncRunner.__new__(async_runner.AsyncRunner)
    runner._verbose = True
    runner._save = False
    runner._dump_tensors = False
    runner._loop = 1
    with pytest.raises(_Stop):
        runner._run_stream(4, display=False)
    assert opened == [(4,)]


def test_camera_source_opens_through_the_helper(monkeypatch):
    from common.inputs import camera_source

    class _Opened:
        def isOpened(self):
            return True

        def set(self, prop, value):
            return False

        def get(self, prop):
            return 0.0

        def release(self):
            pass

    seen = []

    def fake_open(device_id, width=None, height=None):
        seen.append((device_id, width, height))
        return _Opened()

    monkeypatch.setattr(camera_source, "open_camera_capture", fake_open)
    camera_source.CameraSource(1, 640, 480)
    assert seen == [(1, 640, 480)]


def test_the_helper_is_exported_from_common_utility():
    from common import utility
    assert utility.open_camera_capture is camera.open_camera_capture
    assert "open_camera_capture" in utility.__all__
