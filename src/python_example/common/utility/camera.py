"""
Open a camera by index: USB webcams and the Raspberry Pi CSI camera.

On current Raspberry Pi OS the CSI camera is managed by libcamera, not the
legacy V4L2 capture interface, so OpenCV has to read it through the
libcamerasrc GStreamer element. open_camera_capture() takes that path on a
Raspberry Pi and is plain cv2.VideoCapture(index) everywhere else.

Mirrors src/cpp_example/common/utility/camera.hpp - keep the two in sync.
"""

import logging
import os
import re
import shutil
import subprocess
from typing import Optional, Union

import cv2

logger = logging.getLogger(__name__)

# libcamera capture size when the caller does not ask for one.
LIBCAMERA_DEFAULT_WIDTH = 1280
LIBCAMERA_DEFAULT_HEIGHT = 720

_DEVICE_TREE_MODEL_PATHS = ("/proc/device-tree/model",
                            "/sys/firmware/devicetree/base/model")
_V4L2_SYSFS_DIR = "/sys/class/video4linux"
_DEV_VIDEO_FMT = "/dev/video{}"
_RPI_CSI_NODE_TAGS = ("rp1-cfe", "unicam", "pispbe", "bcm2835")
_CAMERA_LIST_TOOLS = ("rpicam-hello", "libcamera-hello")
# "0 : imx708 [4608x2592 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a)"
_CAMERA_LINE = re.compile(r"^\s*(\d{1,4})\s*:.*\(([^()]*)\)\s*$")


def is_raspberry_pi_model(model: str) -> bool:
    """True when a device-tree model string names a Raspberry Pi."""
    # device-tree strings are NUL-terminated; drop any embedded/trailing NULs.
    return "raspberry pi" in model.replace("\0", "").lower()


def is_raspberry_pi() -> bool:
    """True when the host is a Raspberry Pi (device-tree model). Safe elsewhere."""
    for path in _DEVICE_TREE_MODEL_PATHS:
        try:
            with open(path, "rb") as f:
                model = f.read().decode("utf-8", errors="ignore")
        except OSError:
            continue
        if is_raspberry_pi_model(model):
            return True
    return False


def v4l2_device_name(index: int) -> str:
    """V4L2 driver/card name of /dev/video<index>, lower-cased ("" if unknown)."""
    try:
        with open(os.path.join(_V4L2_SYSFS_DIR, f"video{index}", "name")) as f:
            return f.readline().strip().lower()
    except OSError:
        return ""


def is_rpi_csi_device_name(name: str) -> bool:
    """True for the V4L2 name of a Raspberry Pi CSI/ISP node.

    rp1-cfe, unicam, pispbe and bcm2835-* nodes emit raw/processing streams
    that OpenCV's V4L2 backend cannot consume; they must go through libcamera.
    USB webcams (uvcvideo, etc.) report other names.
    """
    lowered = name.lower()
    # The legacy bcm2835-v4l2 capture driver works through OpenCV's V4L2
    # backend, so it is NOT a libcamera-only node despite the bcm2835 prefix.
    if "bcm2835-v4l2" in lowered:
        return False
    return any(tag in lowered for tag in _RPI_CSI_NODE_TAGS)


def is_rpi_csi_video_node(index: int) -> bool:
    """True when /dev/video<index> is a Raspberry Pi CSI/ISP node."""
    return is_rpi_csi_device_name(v4l2_device_name(index))


def parse_libcamera_camera_id(listing: str, index: int) -> str:
    """libcamera ID of camera ``index`` in ``rpicam-hello --list-cameras`` output.

    The ID is the last parenthesised field of the camera's line. Mode lines
    ("Modes: ...", "2304x1296 [...]") never match. "" when ``index`` is not listed.
    """
    for line in listing.splitlines():
        match = _CAMERA_LINE.match(line)
        if match and int(match.group(1)) == index:
            return match.group(2)
    return ""


def build_info_has_gstreamer(build_info: str) -> bool:
    """True when OpenCV build information reports "GStreamer: YES"."""
    for line in build_info.splitlines():
        _, key, value = line.partition("GStreamer:")
        if key:
            return "YES" in value
    return False


def opencv_has_gstreamer() -> bool:
    """True when this cv2 can open GStreamer pipelines (pip wheels cannot)."""
    return build_info_has_gstreamer(cv2.getBuildInformation())


def build_libcamera_pipeline(camera_id: str = "",
                             width: int = LIBCAMERA_DEFAULT_WIDTH,
                             height: int = LIBCAMERA_DEFAULT_HEIGHT) -> str:
    """libcamerasrc pipeline string for OpenCV (BGR appsink).

    An empty ``camera_id`` lets libcamerasrc take its first camera. The caps
    ask for a processed (ISP) NV12 stream: without an explicit format
    libcamerasrc negotiates a raw Bayer stream that videoconvert cannot consume
    ("Internal data stream error", seen on a Raspberry Pi). videoconvert then
    produces the BGR frames OpenCV's appsink expects.
    """
    pipeline = "libcamerasrc"
    if camera_id:
        # camera-name takes the libcamera ID, not an index.
        pipeline += f' camera-name="{camera_id}"'
    return (f"{pipeline} ! video/x-raw,width={width},height={height},format=NV12"
            " ! videoconvert ! video/x-raw,format=BGR"
            " ! appsink drop=true max-buffers=2")


def list_libcamera_cameras() -> Optional[str]:
    """Output of ``rpicam-hello --list-cameras`` (``libcamera-hello`` on older
    Raspberry Pi OS). None when the listing is unavailable (no tool installed,
    or the tool could not run / timed out) so the caller falls back to
    libcamerasrc's first camera for index 0 instead of failing hard."""
    for tool in _CAMERA_LIST_TOOLS:
        if shutil.which(tool) is None:
            continue
        try:
            proc = subprocess.run([tool, "--list-cameras"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout
    return None


def has_libcamera_gst_element() -> bool:
    """True when the libcamerasrc GStreamer element is installed."""
    if shutil.which("gst-inspect-1.0") is None:
        return False
    try:
        return subprocess.run(["gst-inspect-1.0", "libcamerasrc"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _open_libcamera_capture(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera ``index`` through libcamerasrc; logs why on failure."""
    if not opencv_has_gstreamer():
        logger.error(
            f"Raspberry Pi camera (index {index}) is read through GStreamer, but "
            "this OpenCV was built without it\n"
            "    (cv2.getBuildInformation() shows \"GStreamer: NO\"; the pip "
            "opencv-python wheel is built that way).\n"
            "    Use an OpenCV built with GStreamer.")
        return cv2.VideoCapture()
    # Fail fast if the libcamera GStreamer element is missing: opening a pipeline
    # that references an unknown element otherwise blocks for tens of seconds.
    if not has_libcamera_gst_element():
        logger.error(
            f"Raspberry Pi camera (index {index}) needs the libcamera GStreamer "
            "plugin, which is not installed.\n"
            "    Install it with: sudo apt-get install -y gstreamer1.0-libcamera\n"
            "    (or re-run dx_app/install.sh), then verify: rpicam-hello --list-cameras")
        return cv2.VideoCapture()

    camera_id = ""
    listing = list_libcamera_cameras()
    if listing is not None:
        camera_id = parse_libcamera_camera_id(listing, index)
        if not camera_id:
            logger.error(f"libcamera has no camera {index}. "
                         f"rpicam-hello --list-cameras reports:\n{listing}")
            return cv2.VideoCapture()
    elif index > 0:
        logger.error(
            f"Selecting Raspberry Pi camera {index} needs rpicam-hello (or "
            "libcamera-hello) to look up its libcamera ID.\n"
            "    Install it with: sudo apt-get install -y rpicam-apps-lite")
        return cv2.VideoCapture()

    pipeline = build_libcamera_pipeline(camera_id, width, height)
    logger.info(f"Using Raspberry Pi libcamera pipeline for camera {index}:\n"
                f"    {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        logger.info("Raspberry Pi camera opened via libcamera.")
        return cap
    cap.release()
    logger.error(f"Failed to open Raspberry Pi camera (index {index}) through "
                 "libcamera.\n"
                 "    Check that it is detected: rpicam-hello --list-cameras")
    return cv2.VideoCapture()


def open_camera_capture(source: Union[int, str],
                        width: Optional[int] = None,
                        height: Optional[int] = None) -> cv2.VideoCapture:
    """Open ``source`` like cv2.VideoCapture, supporting the Raspberry Pi camera.

    A file path, URL, or any index off a Raspberry Pi opens exactly as
    cv2.VideoCapture(source). On a Raspberry Pi a camera index follows its
    V4L2 node, so a CSI raw node is never probed with the wrong backend:
      - CSI/ISP node, or no /dev/video<index> -> libcamerasrc GStreamer pipeline
      - any other node (USB webcam)           -> V4L2, then libcamera if that fails
    ``width`` / ``height`` size the libcamera stream (default 1280x720); on the
    V4L2 path callers set those properties after opening, as before.
    Check ``isOpened()`` on the result, as with cv2.VideoCapture.
    """
    # bool is an int subclass; only a real camera index takes the Pi path.
    if not isinstance(source, int) or isinstance(source, bool) or not is_raspberry_pi():
        return cv2.VideoCapture(source)
    use_libcamera = (is_rpi_csi_video_node(source)
                     or not os.path.exists(_DEV_VIDEO_FMT.format(source)))
    if not use_libcamera:
        cap = cv2.VideoCapture(source)  # USB webcam / capture card on the Pi.
        if cap.isOpened():
            return cap
        cap.release()
    return _open_libcamera_capture(source, width or LIBCAMERA_DEFAULT_WIDTH,
                                   height or LIBCAMERA_DEFAULT_HEIGHT)
