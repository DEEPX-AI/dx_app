# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Alpha matte visualisation: original, matte, and the composite it enables.

Why this exists rather than reusing ``SemanticSegmentationVisualizer``: that class
reads ``SegmentationResult.mask`` as CLASS IDS and paints it with a 19-colour
Cityscapes palette. Handing it a matting result renders the thresholded 0/1 map in two
arbitrary palette colours and throws the matte away -- the picture looks like a
segmentation and hides the only thing matting produces, a *continuous* opacity.

So this reads ``alpha`` and shows three panels:

``original``   what went in.
``alpha``      the matte itself, 0-255 greyscale. Soft edges (hair, motion blur) are
               the whole point of matting and are only visible here.
``composite``  the foreground over a checkerboard, which is what the matte is FOR.
               A checkerboard rather than a flat colour because it makes partial
               opacity legible instead of guessable.
"""
from __future__ import annotations

from typing import Any, List

import cv2
import numpy as np

from ..base import IVisualizer

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FG = (245, 245, 245)
_SHADOW = (20, 20, 20)


def checkerboard(height: int, width: int, square: int = 16,
                 light: int = 160, dark: int = 110) -> np.ndarray:
    """The transparency backdrop, built without allocating per-pixel Python."""
    ys = (np.arange(height) // square)[:, None]
    xs = (np.arange(width) // square)[None, :]
    board = np.where((ys + xs) % 2 == 0, light, dark).astype(np.uint8)
    return np.repeat(board[:, :, None], 3, axis=2)


class MattingVisualizer(IVisualizer):
    """``SegmentationResult`` carrying ``alpha`` -> original | matte | composite."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.square = int(cfg.get("checker_size", 16))
        self.show_panels = bool(cfg.get("show_panels", True))

    @staticmethod
    def _caption(img: np.ndarray, text: str) -> None:
        cv2.putText(img, text, (10, 26), _FONT, 0.66, _SHADOW, 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, 26), _FONT, 0.66, _FG, 1, cv2.LINE_AA)

    def _alpha_of(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """The matte at frame resolution, in [0,1] float32.

        Falls back to the thresholded ``mask`` only when ``alpha`` is absent, so a
        postprocessor that does not produce a matte still renders something honest
        (a hard cut) rather than crashing.
        """
        alpha = getattr(result, "alpha", None)
        if alpha is None:
            mask = np.asarray(getattr(result, "mask", np.array([])))
            if mask.size == 0:
                return np.ones(frame.shape[:2], np.float32)
            alpha = mask.astype(np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        h, w = frame.shape[:2]
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0)

    def composite(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Foreground over the checkerboard: ``a*F + (1-a)*B``, the matting equation."""
        back = checkerboard(frame.shape[0], frame.shape[1], self.square)
        a = alpha[:, :, None]
        return (frame.astype(np.float32) * a
                + back.astype(np.float32) * (1.0 - a)).astype(np.uint8)

    def visualize(self, frame: np.ndarray, results: List[Any]) -> np.ndarray:
        if not results:
            return frame.copy()
        alpha = self._alpha_of(frame, results[0])
        comp = self.composite(frame, alpha)
        if not self.show_panels:
            return comp

        matte = cv2.cvtColor((alpha * 255.0).round().astype(np.uint8),
                             cv2.COLOR_GRAY2BGR)
        panels = [frame.copy(), matte, comp]
        for img, text in zip(panels, ("original", "alpha matte", "composite")):
            self._caption(img, text)
        # A 1px divider keeps the three panels readable when the images are pale.
        div = np.full((frame.shape[0], 2, 3), 60, np.uint8)
        return np.hstack([panels[0], div, panels[1], div, panels[2]])
