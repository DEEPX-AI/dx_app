# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""PP-PicoDet postprocessor: a GFL head delivered one tensor per pyramid level.

PicoDet and NanoDet-Plus share the same head -- anchor-free, class scores plus a
Distribution Focal Loss box regression -- so :class:`NanoDetPostprocessor` looks like
the obvious reuse, and that is what this family was wired to first. It does not fit,
for a packaging reason rather than an algorithmic one: NanoDet emits ONE concatenated
tensor ``(1, N, num_classes + 4*(reg_max+1))``, and PicoDet emits a separate score and
box tensor per level.

MEASURED on DX-RT 3.5.0, pp-shituv1-mainbody-detection_640x640 at 640x640::

    (1, 6400, 1)  (1, 1600, 1)  (1, 400, 1)  (1, 100, 1)     class scores, 1 class
    (1, 6400, 32) (1, 1600, 32) (1, 400, 32) (1, 100, 32)    box distributions

Four levels, anchor counts 6400/1600/400/100 -- so strides 8/16/32/64 at 640 -- and
32 = 4 sides x (reg_max + 1) with reg_max = 7.

Two properties of this export, both measured rather than assumed:

* the scores are **already sigmoid** (range 0.0005..0.2615 on a real image), so
  applying one again would shrink every score toward 0.5;
* the distributions are **not** softmaxed (per-side sums near 0, not 1), so the DFL
  integration has to softmax them itself.

Levels are matched to strides by anchor count, never by tensor order: nothing
guarantees the runtime hands them over largest-grid-first.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..base import DetectionResult, IPostprocessor, PreprocessContext


def _shapes(outputs: Sequence[np.ndarray]) -> str:
    return ", ".join("(" + ", ".join(str(d) for d in o.shape) + ")" for o in outputs)


class PicoDetPostprocessor(IPostprocessor):
    """Per-level GFL decoding for PP-PicoDet (PP-ShiTu's mainbody detector)."""

    def __init__(self, input_width: int = 640, input_height: int = 640,
                 config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.conf_threshold = float(
            self.config.get("conf_threshold",
                            self.config.get("score_threshold", 0.3)))
        self.nms_threshold = float(self.config.get("nms_threshold", 0.5))
        self.num_classes = int(self.config.get("num_classes", 1))

    # ------------------------------------------------------------------ pairing
    def _levels(self, outputs: Sequence[np.ndarray]
                ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """``[(scores, distributions, stride)]``, paired by anchor count.

        Paired by count and not by position: the two families of tensors may arrive
        interleaved or in either order, and a positional assumption would silently
        pair a level's scores with another level's boxes.
        """
        scores: dict = {}
        boxes: dict = {}
        for out in outputs:
            if out.ndim != 3 or out.shape[0] != 1:
                continue
            anchors, channels = int(out.shape[1]), int(out.shape[2])
            if channels % 4 == 0 and channels >= 8:
                boxes[anchors] = out[0]
            else:
                scores[anchors] = out[0]

        levels = []
        for anchors in sorted(set(scores) & set(boxes), reverse=True):
            grid = int(round(anchors ** 0.5))
            if grid * grid != anchors:
                continue                    # not a square grid: not a pyramid level
            levels.append((scores[anchors], boxes[anchors],
                           self.input_width // grid))
        if not levels:
            raise ValueError(
                "[DXAPP] [ERROR] PicoDetPostprocessor - no matching per-level score "
                "and box tensors found.\n"
                f"  Got: {_shapes(outputs)}\n"
                "  Expected one (1, N, num_classes) and one (1, N, 4*(reg_max+1)) per "
                "pyramid level, with N a square anchor count. A SINGLE concatenated "
                "(1, N, num_classes + 4*(reg_max+1)) tensor is NanoDet's packaging -- "
                "use NanoDetPostprocessor for that."
            )
        return levels

    # ------------------------------------------------------------------ decoding
    @staticmethod
    def _integrate(distribution: np.ndarray, reg_max: int) -> np.ndarray:
        """DFL: softmax over the bins, then the expected bin index, per side."""
        bins = distribution.reshape(-1, 4, reg_max + 1).astype(np.float32)
        bins -= bins.max(axis=2, keepdims=True)          # stable softmax
        np.exp(bins, out=bins)
        bins /= np.maximum(bins.sum(axis=2, keepdims=True), 1e-12)
        weights = np.arange(reg_max + 1, dtype=np.float32)
        return bins @ weights                            # (n, 4) in grid units

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[DetectionResult]:
        boxes_xyxy: List[np.ndarray] = []
        confidences: List[np.ndarray] = []
        class_ids: List[np.ndarray] = []

        for scores, distribution, stride in self._levels(outputs):
            reg_max = distribution.shape[1] // 4 - 1
            best = scores.max(axis=1)
            keep = np.flatnonzero(best >= self.conf_threshold)
            if keep.size == 0:
                continue
            distances = self._integrate(distribution[keep], reg_max) * stride

            grid = self.input_width // stride
            cx = (keep % grid + 0.5) * stride
            cy = (keep // grid + 0.5) * stride
            boxes_xyxy.append(np.stack([cx - distances[:, 0], cy - distances[:, 1],
                                        cx + distances[:, 2], cy + distances[:, 3]],
                                       axis=1))
            confidences.append(best[keep])
            class_ids.append(scores[keep].argmax(axis=1))

        if not boxes_xyxy:
            return []
        boxes_all = np.concatenate(boxes_xyxy)
        conf_all = np.concatenate(confidences)
        class_all = np.concatenate(class_ids)

        # Undo the letterbox, clipping as every other postprocessor here does.
        gain = max(ctx.scale, 1e-6)
        boxes_all[:, 0::2] = np.clip((boxes_all[:, 0::2] - ctx.pad_x) / gain,
                                     0, ctx.original_width - 1)
        boxes_all[:, 1::2] = np.clip((boxes_all[:, 1::2] - ctx.pad_y) / gain,
                                     0, ctx.original_height - 1)

        order = np.argsort(-conf_all, kind="stable")
        areas = (np.clip(boxes_all[:, 2] - boxes_all[:, 0], 0, None)
                 * np.clip(boxes_all[:, 3] - boxes_all[:, 1], 0, None))
        kept: List[int] = []
        remaining = list(order)
        while remaining:
            best_idx = remaining.pop(0)
            kept.append(best_idx)
            if not remaining or self.nms_threshold >= 1.0:
                if self.nms_threshold >= 1.0:
                    kept.extend(remaining)
                break
            rest = np.asarray(remaining)
            x1 = np.maximum(boxes_all[best_idx, 0], boxes_all[rest, 0])
            y1 = np.maximum(boxes_all[best_idx, 1], boxes_all[rest, 1])
            x2 = np.minimum(boxes_all[best_idx, 2], boxes_all[rest, 2])
            y2 = np.minimum(boxes_all[best_idx, 3], boxes_all[rest, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            union = areas[best_idx] + areas[rest] - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
            remaining = [r for r, survives in zip(remaining, iou < self.nms_threshold)
                         if survives]

        return [
            DetectionResult(box=[float(v) for v in boxes_all[i]],
                            confidence=float(conf_all[i]),
                            class_id=int(class_all[i]))
            for i in kept
        ]

    def get_model_name(self) -> str:
        return "picodet"
