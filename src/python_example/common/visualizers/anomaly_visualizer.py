# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Visualizer for anomaly-detection feature responses."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..base import AnomalyResult, IVisualizer


class AnomalyVisualizer(IVisualizer):
    """Overlay the anomaly response on the frame, with its relative severity.

    The severity is labelled "relative" on the frame itself, and that wording is
    deliberate in both cases it serves. EfficientAD now runs as the ensemble it is --
    teacher, student and autoencoder together -- but its published score divides by
    q_st/q_ae quantiles fitted on the training set, which no .dxnn carries. PatchCore
    is still a single network here, because its metric needs a memory bank of training
    features. So the number ranks frames against each other for one model set, and
    printing a bare "score" would invite exactly the comparison it cannot support.
    """

    def __init__(self, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET):
        self.alpha = float(alpha)
        self.colormap = colormap

    def visualize(self, image: np.ndarray,
                  results: List[AnomalyResult]) -> np.ndarray:
        output = image.copy()
        if not results:
            return output

        result = results[0]
        heatmap = result.heatmap
        if heatmap is None or heatmap.size == 0:
            return output

        height, width = output.shape[:2]
        if heatmap.shape[:2] != (height, width):
            heatmap = cv2.resize(heatmap, (width, height),
                                 interpolation=cv2.INTER_LINEAR)
        coloured = cv2.applyColorMap(
            np.clip(heatmap * 255.0, 0, 255).astype(np.uint8), self.colormap)
        output = cv2.addWeighted(output, 1.0 - self.alpha, coloured, self.alpha, 0.0)

        label = f"relative severity {result.score:.4f}"
        if result.channels:
            label += f"  ({result.channels} ch)"

        # A dark plate behind one stroke, not an outline pass plus a fill pass:
        # cv2.getTextSize's width depends on THICKNESS (312 px at 1, 332 px at 3 for
        # this label), so drawing the same string twice at different thicknesses
        # desynchronises the glyph advance and the two copies drift apart across the
        # line. The plate is also what every other visualizer in this tree uses.
        scale, thickness = 0.7, 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        origin = (12, 12 + text_h)
        plate = output.copy()
        cv2.rectangle(plate, (origin[0] - 6, origin[1] - text_h - 6),
                      (origin[0] + text_w + 6, origin[1] + baseline + 4),
                      (0, 0, 0), -1)
        cv2.addWeighted(plate, 0.5, output, 0.5, 0.0, output)
        cv2.putText(output, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        return output
