"""
IoU Tracker (SORT-lite)

A lightweight, dependency-free multi-object tracker for assigning stable
``track_id`` values to per-frame bounding boxes. Used to keep instance colors
consistent across frames (same object → same color) in instance segmentation
visualizers.

Association is greedy IoU matching between the previous frame's tracks and the
current frame's boxes. There is no Kalman filter — for the color-stability use
case, frame-to-frame IoU is sufficient and avoids extra dependencies (numpy
only). A track that is not matched survives for ``max_age`` frames before being
dropped, so a briefly-occluded object keeps its id/color when it reappears.
"""

import numpy as np
from typing import List, Sequence


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = inter_x2 - inter_x1
    ih = inter_y2 - inter_y1
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    """Internal state for a single tracked object."""

    __slots__ = ("id", "box", "time_since_update", "hits")

    def __init__(self, track_id: int, box: List[float]):
        self.id = track_id
        self.box = list(box)
        self.time_since_update = 0
        self.hits = 1


class IoUTracker:
    """
    Greedy IoU multi-object tracker (SORT without Kalman).

    Args:
        iou_threshold: minimum IoU for a current box to match an existing track.
        max_age: number of consecutive frames a track may go unmatched before
            it is removed (its id/color is reserved during this window).

    Usage:
        tracker = IoUTracker()
        track_ids = tracker.update([[x1, y1, x2, y2], ...])
        # track_ids[i] is the stable id assigned to boxes[i]
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self._tracks: List[_Track] = []
        self._next_id = 0

    def reset(self) -> None:
        """Clear all track state (e.g. between independent image sequences)."""
        self._tracks = []
        self._next_id = 0

    def update(self, boxes: Sequence[Sequence[float]]) -> List[int]:
        """Associate current-frame boxes with existing tracks.

        Returns a list of ``track_id`` values aligned to ``boxes`` (order
        preserved), assigning a fresh id to any box that matches no track.
        """
        n = len(boxes)
        track_ids: List[int] = [-1] * n
        if n == 0:
            self._age_and_prune(matched_tracks=set())
            return track_ids

        # Build candidate (iou, box_idx, track_idx) triples above threshold.
        candidates = []
        for ti, track in enumerate(self._tracks):
            for bi in range(n):
                iou = _iou(track.box, boxes[bi])
                if iou >= self.iou_threshold:
                    candidates.append((iou, bi, ti))

        # Greedy match: highest IoU first, one-to-one.
        candidates.sort(key=lambda c: c[0], reverse=True)
        matched_boxes = set()
        matched_tracks = set()
        for _iou_val, bi, ti in candidates:
            if bi in matched_boxes or ti in matched_tracks:
                continue
            matched_boxes.add(bi)
            matched_tracks.add(ti)
            track = self._tracks[ti]
            track.box = list(boxes[bi])
            track.time_since_update = 0
            track.hits += 1
            track_ids[bi] = track.id

        # Unmatched boxes → new tracks.
        for bi in range(n):
            if bi in matched_boxes:
                continue
            track = _Track(self._next_id, list(boxes[bi]))
            self._next_id += 1
            self._tracks.append(track)
            # Newly created track is matched this frame.
            matched_tracks.add(len(self._tracks) - 1)
            track_ids[bi] = track.id

        self._age_and_prune(matched_tracks)
        return track_ids

    def _age_and_prune(self, matched_tracks: set) -> None:
        """Increment age of unmatched tracks and drop expired ones."""
        surviving: List[_Track] = []
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.time_since_update += 1
            if track.time_since_update <= self.max_age:
                surviving.append(track)
        self._tracks = surviving
