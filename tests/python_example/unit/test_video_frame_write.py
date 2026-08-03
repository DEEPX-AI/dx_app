"""Unit tests for write_video_frame().

cv2.VideoWriter silently discards frames whose size differs from the size the
writer was opened with, and ``writer.get(CAP_PROP_FRAME_WIDTH/HEIGHT)`` returns
0 on several OpenCV builds (e.g. the GStreamer backend). The runners relied on
that getter to decide whether to resize, so a render whose canvas differs from
the input size (super-resolution side-by-side panel) produced an empty video
with no error at all.
"""

import cv2
import numpy as np

from common.utility.video_io import write_video_frame


def _decoded_frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _open_writer(path, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    return writer


def test_oversized_frame_is_resized_and_written(tmp_path):
    out = tmp_path / "out.mp4"
    size = (320, 180)
    writer = _open_writer(out, size)
    canvas = np.full((360, 640, 3), 128, np.uint8)  # 2x the writer size
    for _ in range(5):
        write_video_frame(writer, canvas, size)
    writer.release()
    assert _decoded_frames(out) == 5


def test_matching_frame_is_written_unchanged(tmp_path):
    out = tmp_path / "out.mp4"
    size = (320, 180)
    writer = _open_writer(out, size)
    frame = np.zeros((180, 320, 3), np.uint8)
    for _ in range(3):
        write_video_frame(writer, frame, size)
    writer.release()
    assert _decoded_frames(out) == 3


def test_no_size_hint_still_writes_when_writer_reports_size(tmp_path):
    """Without a hint we fall back to the writer's own (possibly 0) report."""
    out = tmp_path / "out.mp4"
    size = (320, 180)
    writer = _open_writer(out, size)
    frame = np.zeros((180, 320, 3), np.uint8)
    write_video_frame(writer, frame, None)
    writer.release()
    assert _decoded_frames(out) == 1


def test_none_frame_and_none_writer_are_ignored(tmp_path):
    out = tmp_path / "out.mp4"
    writer = _open_writer(out, (320, 180))
    write_video_frame(writer, None, (320, 180))
    write_video_frame(None, np.zeros((180, 320, 3), np.uint8), (320, 180))
    writer.release()
    assert _decoded_frames(out) == 0
