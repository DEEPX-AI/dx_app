"""
YOLOPv2 Panoptic Driving Perception Visualizer

Blends drivable area and lane masks, then draws detection bounding boxes.
"""

import numpy as np
import cv2
from typing import List

from ..base import IVisualizer, DetectionResult
from ..utility import get_labels


class YOLOPv2Visualizer(IVisualizer):
    """
    Visualizer for YOLOPv2 panoptic driving perception results.

    Overlays:
      - Drivable area mask (green tint)
      - Lane line mask (red tint)
      - Object detection bounding boxes (COCO labels)
    """

    def __init__(
        self,
        drivable_alpha: float = 0.4,
        lane_alpha: float = 0.5,
        box_score_threshold: float = 0.25,
        label_set: str = "coco80",
    ):
        self.drivable_alpha = drivable_alpha
        self.lane_alpha = lane_alpha
        self.box_score_threshold = box_score_threshold
        self._labels = get_labels(label_set)

    def _blend_mask(
        self,
        output: np.ndarray,
        mask: np.ndarray,
        color: tuple,
        alpha: float,
    ) -> np.ndarray:
        """Blend a binary mask into the image with a given color."""
        if mask is None or mask.size == 0:
            return output
        h, w = output.shape[:2]
        # resize mask to image size if needed
        if mask.shape[:2] != (h, w):
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask.astype(np.uint8)

        overlay = output.copy()
        overlay[mask_resized > 0] = color
        return cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)

    def visualize(self, image: np.ndarray, results: list) -> np.ndarray:
        output = image.copy()

        for result in results:
            # Overlay drivable area mask (green)
            if result.drivable_mask is not None:
                output = self._blend_mask(output, result.drivable_mask, (0, 180, 0), self.drivable_alpha)

            # Overlay lane mask (red)
            if result.lane_mask is not None:
                output = self._blend_mask(output, result.lane_mask, (0, 0, 200), self.lane_alpha)

            # Draw detection boxes
            for det in result.detections:
                if det.confidence < self.box_score_threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in det.box]
                color = (0, 255, 255)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                # Use class_name from postprocessor ("vehicle"), not COCO80 label
                label_name = det.class_name if det.class_name else str(det.class_id)
                label = f"{label_name}: {det.confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ly = y1 - 5 if y1 - 5 > lh else y1 + lh + 5
                cv2.rectangle(output, (x1, ly - lh - 2), (x1 + lw, ly + 2), color, -1)
                cv2.putText(output, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return output
