"""Helpers for DXAPP_VERIFY tests: a short clip, the per-frame records, the async summary.

Every DXAPP_VERIFY run writes ``<stem>.json`` (last frame) and
``<stem>.frames.jsonl`` (one record per delivered frame, with ``"frame": n``).
Comparing a sync and an async ``frames.jsonl`` record by record catches a frame
delivered out of order and a result drawn from another frame.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple


def make_video_clip(src: Path, dst: Path, frames: int,
                    size: Tuple[int, int] = (960, 540), skip: int = 30) -> int:
    """Write *frames* frames of *src* (after the first *skip*) to *dst* as MJPG AVI at *size*.

    Returns the number of frames written. MJPG keeps every frame an I-frame, so
    the sync and async runners decode exactly the same pixels.
    """
    import cv2
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return 0
    for _ in range(skip):
        cap.read()
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    written = 0
    while written < frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, size))
        written += 1
    writer.release()
    cap.release()
    return written


def read_verify_json(verify_dir: Path) -> dict:
    """The one ``<stem>.json`` in *verify_dir* (AssertionError otherwise)."""
    files = sorted(p for p in Path(verify_dir).glob("*.json"))
    assert len(files) == 1, "expected one verify JSON in {}, found {}".format(
        verify_dir, [p.name for p in files])
    return json.loads(files[0].read_text())


def read_verify_frames(verify_dir: Path) -> List[dict]:
    """The records of the one ``<stem>.frames.jsonl`` in *verify_dir*."""
    files = sorted(Path(verify_dir).glob("*.frames.jsonl"))
    assert len(files) == 1, "expected one frames.jsonl in {}, found {}".format(
        verify_dir, [p.name for p in files])
    return [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]


_INFLIGHT_MAX = re.compile(r"Infer Inflight Max\s*:\s*(\d+)")


def inflight_max(output: str) -> Optional[int]:
    """``Infer Inflight Max`` from an async runner's performance summary."""
    found = _INFLIGHT_MAX.findall(output)
    return int(found[-1]) if found else None


def first_frame_difference(expected: List[dict], actual: List[dict]) -> Optional[str]:
    """None when the two record lists are equal, else where they first differ."""
    if len(expected) != len(actual):
        return "frame count differs: {} expected, {} got".format(len(expected), len(actual))
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            keys = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
            return "frame {} differs in {}".format(index, keys)
    return None


def payload_nonempty(data: dict) -> bool:
    """True when a verify record carries results, not just the image size."""
    for key in ("detections", "classifications", "class_ids", "output_stats_list"):
        if data.get(key):
            return True
    embedding = data.get("embedding")
    if isinstance(embedding, dict) and embedding.get("dim", 0) > 0:
        return True
    # output_stats: raw arrays; drivable_stats / lane_stats: YOLOPv2 masks, which a
    # frame without vehicles still carries.
    return any(isinstance(data.get(key), dict) and bool(data[key].get("shape"))
               for key in ("output_stats", "drivable_stats", "lane_stats"))
