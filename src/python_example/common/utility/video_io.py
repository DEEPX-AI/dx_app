"""Video output helpers.

cv2.VideoWriter accepts only frames whose size matches the size it was opened
with — any other frame is discarded *silently*, with no exception and no return
value to check. `writer.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)` cannot be relied
on to detect this: several OpenCV builds (notably the GStreamer backend) return
0. Callers therefore pass the size they opened the writer with.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def writer_frame_size(writer, size_hint: Optional[Tuple[int, int]] = None):
    """(width, height) of `writer`'s frames, or None when undeterminable."""
    if writer is None:
        return None
    w = int(writer.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w > 0 and h > 0:
        return w, h
    if size_hint and size_hint[0] > 0 and size_hint[1] > 0:
        return int(size_hint[0]), int(size_hint[1])
    return None


def write_video_frame(writer, frame: Optional[np.ndarray],
                      size_hint: Optional[Tuple[int, int]] = None) -> None:
    """Write `frame` to `writer`, resizing it to the writer's frame size first.

    `size_hint` is the (width, height) the writer was opened with; it is used
    whenever the writer itself cannot report its size.
    """
    if writer is None or frame is None or getattr(frame, "size", 0) == 0:
        return
    size = writer_frame_size(writer, size_hint)
    if size is not None and (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size)
    writer.write(frame)
