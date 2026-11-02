"""
OpenCV video I/O defaults that must be in the environment before cv2 is imported.

opencv-python reads OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS once, when cv2 is
imported; setting it later has no effect. common/__init__.py therefore calls
default_msmf_hw_transforms_off() before any of its modules imports cv2, so this
module must not import cv2 itself.

Mirrors defaultMsmfHwTransformsOff() in
src/cpp_example/common/utility/camera.hpp (the C++ OpenCV build reads the
variable each time a capture opens, so C++ sets it right before a camera opens).
"""

import os
from typing import MutableMapping, Optional

MSMF_HW_TRANSFORMS_ENV = "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"


def default_msmf_hw_transforms_off(environ: Optional[MutableMapping[str, str]] = None,
                                   on_windows: Optional[bool] = None) -> None:
    """Windows: turn OpenCV's MSMF hardware transforms off unless the user chose.

    With them on, the MSMF backend takes about 7 s to open a USB webcam; off,
    about 0.15 s at the same frame rate. A value the user set (e.g. 1 to keep
    them on) is left alone.
    """
    if environ is None:
        environ = os.environ
    if on_windows is None:
        on_windows = os.name == "nt"
    if on_windows:
        environ.setdefault(MSMF_HW_TRANSFORMS_ENV, "0")
