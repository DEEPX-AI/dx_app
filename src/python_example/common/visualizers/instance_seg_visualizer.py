"""
Instance Segmentation Visualizer

Draws bounding boxes with semi-transparent masks overlaid on the image.
Each instance gets a color that is stable across frames: a lightweight IoU
tracker assigns a persistent ``track_id`` to every box, and the color is chosen
by ``track_id`` (not by detection order). This keeps the same object the same
color for both its box and mask across frames. Used by YOLOv8-seg, YOLOv5-seg,
YOLO26-seg, YOLACT, FastSAM.
"""

import numpy as np
import cv2
from typing import List

from ..base import IVisualizer
from ..trackers import IoUTracker


# 20 distinct colors for instance segmentation
INSTANCE_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


class InstanceSegVisualizer(IVisualizer):
    """Visualizer for instance segmentation with mask overlay.

    Args:
        label_set: named label set (e.g. 'coco80') when ``labels`` is not given.
        labels: explicit class-name list; overrides ``label_set``.
        show_boxes: draw bounding boxes + labels (False → mask-only, e.g. FastSAM).
        enable_tracking: when True (default) color is bound to a per-object
            ``track_id`` from an IoU tracker so colors stay stable across
            frames; when False, color falls back to detection index.
        iou_threshold / max_age: IoU tracker tuning (see IoUTracker).
    """

    def __init__(self, label_set: str = 'coco80', labels: List[str] = None,
                 show_boxes: bool = True, enable_tracking: bool = True,
                 iou_threshold: float = 0.3, max_age: int = 30):
        self.label_set = label_set
        self.labels = labels if labels is not None else self._load_labels(label_set)
        self.color_palette = INSTANCE_COLORS
        self.show_boxes = show_boxes
        self.enable_tracking = enable_tracking
        self.tracker = IoUTracker(iou_threshold=iou_threshold,
                                  max_age=max_age) if enable_tracking else None

    def _load_labels(self, label_set: str) -> List[str]:
        if label_set == 'coco80':
            return [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
        return []

    def _assign_track_ids(self, results: List) -> None:
        """Run the tracker over the current boxes and stamp results in place.

        Color is later chosen from ``track_id`` so the same object keeps its
        color across frames regardless of detection order.
        """
        if self.tracker is None:
            return
        boxes = [r.box for r in results]
        track_ids = self.tracker.update(boxes)
        for r, tid in zip(results, track_ids):
            r.track_id = tid

    def _color_for(self, result, index: int):
        """Pick a palette color: by track_id when tracking, else by index."""
        key = result.track_id if (self.enable_tracking
                                   and getattr(result, 'track_id', -1) >= 0) else index
        return INSTANCE_COLORS[key % len(INSTANCE_COLORS)]

    def visualize(self, image: np.ndarray, results: List) -> np.ndarray:
        output = image.copy()
        overlay = image.copy()

        # Assign stable track ids for the current frame before drawing so that
        # both the mask and the box of one object use the same color.
        self._assign_track_ids(results)

        # For class-agnostic models (1 class), sort by area descending so large
        # segments are drawn first and smaller ones overlay on top — cleaner look.
        draw_order = list(range(len(results)))
        if len(self.labels) == 1 and len(results) > 1:
            draw_order.sort(key=lambda i: -((
                results[i].box[2] - results[i].box[0]) * (
                results[i].box[3] - results[i].box[1])))

        img_h, img_w = image.shape[:2]
        for i in draw_order:
            r = results[i]
            color = self._color_for(r, i)
            x1, y1, x2, y2 = [int(v) for v in r.box]

            # Draw mask overlay. The postprocessor zeroes each instance mask
            # outside its bbox, so restrict the (per-instance, full-frame)
            # boolean index + write to the bbox ROI — this is output-identical
            # but avoids allocating/scanning a full H*W array per instance.
            # A small margin absorbs sub-pixel spread from mask resizing.
            if hasattr(r, 'mask') and r.mask is not None and r.mask.size > 0:
                mask = r.mask
                if mask.shape[:2] == image.shape[:2]:
                    m = 2
                    rx1, ry1 = max(0, x1 - m), max(0, y1 - m)
                    rx2, ry2 = min(img_w, x2 + m), min(img_h, y2 + m)
                    if rx2 > rx1 and ry2 > ry1:
                        roi = mask[ry1:ry2, rx1:rx2]
                        overlay[ry1:ry2, rx1:rx2][roi > 0] = color

            # Draw bounding box and label (skip for mask-only mode)
            if self.show_boxes:
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{r.class_id}"
                if self.labels and r.class_id < len(self.labels):
                    label = self.labels[r.class_id]
                # Prefix a stable track id (#N) when tracking is active.
                id_prefix = ""
                if self.enable_tracking and getattr(r, 'track_id', -1) >= 0:
                    id_prefix = f"#{r.track_id} "
                # For class-agnostic models (1 class), show instance index instead
                if len(self.labels) == 1:
                    text = f"{id_prefix}{label} {i+1} {r.confidence:.2f}"
                else:
                    text = f"{id_prefix}{label} {r.confidence:.2f}"

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(output, text, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend mask overlay
        output = cv2.addWeighted(overlay, 0.4, output, 0.6, 0)

        return output
