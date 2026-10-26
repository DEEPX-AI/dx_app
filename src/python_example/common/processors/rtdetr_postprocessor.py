# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Postprocessors for the RT-DETR family (RT-DETR, RT-DETRv2, RT-DETRv3, mask-RT-DETR).

RT-DETR is NMS-free: the decoder emits a fixed number of queries (300 by default),
each with one box and per-class logits, and the standard postprocess sigmoids the
logits and takes the top-k over the flattened (query x class) scores. Its
preprocessing is a plain resize with no letterbox, so ``ctx`` carries
``scale_x``/``scale_y`` and zero padding.

MEASURED on DX-RT 3.5.0, which is the first runtime able to load these v9 models:

    rtdetr-r18vd-6x_640x640            (300, 6)
    mask-rtdetr-hgnetv2-s-6x_640x640   (100, 6) + (100, 640, 640) int32 0/1

-- so the ``paddle_nms`` layout below is what these models actually emit, in model-input
pixels, with NO batch axis. The ``split`` layout is kept for an export without the
postprocess op. Both are accepted, and anything else is refused:

``split`` -- the raw decoder head, from an export WITHOUT the postprocess op::

    boxes  (1, N, 4)   cxcywh, normalised to [0, 1]
    logits (1, N, C)   per-class, before sigmoid

``paddle_nms`` -- PaddleDetection's own ``DETRPostProcess`` folded into the export::

    bbox   (1, N, 6)   [class_id, score, x1, y1, x2, y2] in model-input pixels

The layout is inferred from the shapes, and that inference is the risky part: a
``(1, 300, 6)`` tensor is indistinguishable by shape from a pre-optimized row table,
which orders the same six columns differently (``[box, score, class]`` rather than
``[class, score, box]``). Reading one as the other yields plausible-looking nonsense
rather than an error, so ``config["layout"]`` forces the choice and every failure
message names that key alongside the shapes it actually saw.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..base import (DetectionResult, InstanceSegResult, IPostprocessor,
                    PreprocessContext)

LAYOUT_SPLIT = "split"
LAYOUT_PADDLE = "paddle_nms"
LAYOUT_AUTO = "auto"

# PaddleDetection's DETRPostProcess column order.
P_CLASS, P_SCORE, P_X1, P_Y1, P_X2, P_Y2 = range(6)


def _shapes(outputs: Sequence[np.ndarray]) -> str:
    return ", ".join("(" + ", ".join(str(d) for d in o.shape) + ")" for o in outputs)


def _needs_sigmoid(scores: np.ndarray) -> bool:
    """True when *scores* are logits rather than probabilities.

    RT-DETR emits logits, but a model compiled with the sigmoid folded in emits
    probabilities, and squashing those a second time silently halves every score
    (sigmoid(0.93) = 0.717) without ever raising.
    """
    return bool(scores.min() < 0.0 or scores.max() > 1.0)


class _RTDETRBase(IPostprocessor):
    """Shared query decoding for the RT-DETR family."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.score_threshold = float(
            self.config.get("score_threshold", self.config.get("conf_threshold", 0.4)))
        # 1.0 == off, which is the correct default: the model is NMS-free.
        self.nms_threshold = float(self.config.get("nms_threshold", 1.0))
        self.top_k = int(self.config.get("top_k", 300))
        self.layout = str(self.config.get("layout", LAYOUT_AUTO))
        if self.layout not in (LAYOUT_AUTO, LAYOUT_SPLIT, LAYOUT_PADDLE):
            raise ValueError(
                f"[DXAPP] [ERROR] {type(self).__name__} - unknown layout "
                f"{self.layout!r}; expected one of {LAYOUT_AUTO!r}, "
                f"{LAYOUT_SPLIT!r}, {LAYOUT_PADDLE!r}."
            )

    # ------------------------------------------------------------ tensor finding
    @staticmethod
    def _find_split(outputs: Sequence[np.ndarray]
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(boxes, logits)`` by shape: (1, N, 4) plus (1, N, C>4) with the same N."""
        boxes = next((o for o in outputs
                      if o.ndim == 3 and o.shape[0] == 1 and o.shape[2] == 4), None)
        if boxes is None:
            return None
        queries = boxes.shape[1]
        logits = next((o for o in outputs
                       if o.ndim == 3 and o.shape[0] == 1
                       and o.shape[1] == queries and o.shape[2] > 4), None)
        if logits is None:
            return None
        return boxes[0], logits[0]

    @staticmethod
    def _find_paddle(outputs: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        """The ``[N, 6]`` table PaddleDetection's own postprocess emits.

        MEASURED on DX-RT 3.5.0: rtdetr-r18vd-6x emits ``(300, 6)`` and mask-rtdetr
        ``(100, 6)`` -- TWO dimensions, no batch axis. The first version of this
        required ``ndim == 3`` and rejected every real model. Both forms are accepted
        now, because nothing guarantees the next compile drops the axis too.
        """
        for out in outputs:
            if out.ndim == 2 and out.shape[1] == 6:
                return out
            if out.ndim == 3 and out.shape[0] == 1 and out.shape[2] == 6:
                return out[0]
        return None

    def _fail(self, outputs: Sequence[np.ndarray], reason: str) -> "NoReturn":
        raise ValueError(
            f"[DXAPP] [ERROR] {type(self).__name__} - {reason}\n"
            f"  Got: {_shapes(outputs)}\n"
            f"  Expected either the raw decoder head -- boxes (1, N, 4) cxcywh in "
            f"[0,1] plus logits (1, N, num_classes) -- or PaddleDetection's own "
            f"postprocess output, bbox (1, N, 6) as [class, score, x1, y1, x2, y2].\n"
            f"  Set config[\"layout\"] to \"{LAYOUT_SPLIT}\" or \"{LAYOUT_PADDLE}\" to "
            f"force one; a (1, N, 6) tensor cannot be told apart from a pre-optimized "
            f"row table by shape alone, and the two order their columns differently."
        )

    # ------------------------------------------------------------ decoding
    def _rows(self, outputs: Sequence[np.ndarray], ctx: PreprocessContext
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(boxes_xyxy_original, scores, class_ids, query_index)``, score-descending."""
        if self.layout in (LAYOUT_AUTO, LAYOUT_SPLIT):
            split = self._find_split(outputs)
            if split is not None:
                return self._decode_split(*split, ctx)
            if self.layout == LAYOUT_SPLIT:
                self._fail(outputs, f'layout "{LAYOUT_SPLIT}" was requested but no '
                                    "(1, N, 4) + (1, N, C>4) pair is present.")

        if self.layout in (LAYOUT_AUTO, LAYOUT_PADDLE):
            table = self._find_paddle(outputs)
            if table is not None:
                return self._decode_paddle(table, ctx)
            if self.layout == LAYOUT_PADDLE:
                self._fail(outputs, f'layout "{LAYOUT_PADDLE}" was requested but no '
                                    "(1, N, 6) table is present.")

        self._fail(outputs, "no recognised RT-DETR output layout.")

    def _decode_split(self, boxes: np.ndarray, logits: np.ndarray,
                      ctx: PreprocessContext):
        scores_all = logits.astype(np.float32)
        if _needs_sigmoid(scores_all):
            scores_all = 1.0 / (1.0 + np.exp(-scores_all))

        # Top-k over the flattened (query x class) scores, as the reference
        # postprocess does, so one query may survive under two classes.
        flat = scores_all.reshape(-1)
        keep = np.flatnonzero(flat >= self.score_threshold)
        if keep.size == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return empty, np.empty(0, np.float32), np.empty(0, np.int32), \
                np.empty(0, np.int32)
        if keep.size > self.top_k:
            keep = keep[np.argsort(-flat[keep], kind="stable")[:self.top_k]]
        keep = keep[np.argsort(-flat[keep], kind="stable")]

        num_classes = scores_all.shape[1]
        queries = keep // num_classes
        class_ids = (keep % num_classes).astype(np.int32)
        scores = flat[keep].astype(np.float32)

        # cxcywh normalised to [0,1] -> xyxy in ORIGINAL pixels. RT-DETR normalises
        # against the input, and the input is a plain stretch of the whole frame, so
        # the same ratios apply directly to the original size.
        cx, cy, w, h = (boxes[queries, i].astype(np.float32) for i in range(4))
        ow, oh = float(ctx.original_width), float(ctx.original_height)
        xyxy = np.stack([(cx - w / 2.0) * ow, (cy - h / 2.0) * oh,
                         (cx + w / 2.0) * ow, (cy + h / 2.0) * oh], axis=1)
        return self._clip(xyxy, ctx), scores, class_ids, queries.astype(np.int32)

    def _decode_paddle(self, table: np.ndarray, ctx: PreprocessContext):
        scores = table[:, P_SCORE].astype(np.float32)
        # The table is fixed-size and pads with zero rows, whose score is 0.0, so the
        # threshold is floored above zero exactly as in the pre-optimized decode.
        threshold = max(self.score_threshold, float(np.finfo(np.float32).tiny))
        keep = np.flatnonzero(scores >= threshold)
        if keep.size == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return empty, np.empty(0, np.float32), np.empty(0, np.int32), \
                np.empty(0, np.int32)
        keep = keep[np.argsort(-scores[keep], kind="stable")][:self.top_k]

        # Corners are in model-input pixels; undo the per-axis stretch.
        sx = ctx.scale_x if ctx.scale_x > 0 else max(ctx.scale, 1e-6)
        sy = ctx.scale_y if ctx.scale_y > 0 else max(ctx.scale, 1e-6)
        xyxy = np.stack([table[keep, P_X1] / sx, table[keep, P_Y1] / sy,
                         table[keep, P_X2] / sx, table[keep, P_Y2] / sy],
                        axis=1).astype(np.float32)
        return (self._clip(xyxy, ctx), scores[keep],
                table[keep, P_CLASS].astype(np.int32), keep.astype(np.int32))

    @staticmethod
    def _clip(xyxy: np.ndarray, ctx: PreprocessContext) -> np.ndarray:
        xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0.0, max(ctx.original_width - 1, 0))
        xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0.0, max(ctx.original_height - 1, 0))
        return xyxy

    def _merge_duplicates(self, boxes: np.ndarray) -> np.ndarray:
        """Class-agnostic greedy NMS. Off by default -- the model is NMS-free."""
        if self.nms_threshold >= 1.0 or boxes.shape[0] == 0:
            return np.arange(boxes.shape[0], dtype=np.int32)
        areas = (np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
                 * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))
        order = list(range(boxes.shape[0]))
        kept: List[int] = []
        while order:
            best = order.pop(0)
            kept.append(best)
            if not order:
                break
            rest = np.asarray(order)
            x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            union = areas[best] + areas[rest] - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
            order = [o for o, survives in zip(order, iou < self.nms_threshold)
                     if survives]
        return np.asarray(kept, dtype=np.int32)


class RTDETRPostprocessor(_RTDETRBase):
    """RT-DETR / RT-DETRv2 / RT-DETRv3 object detection."""

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[DetectionResult]:
        boxes, scores, class_ids, _ = self._rows(outputs, ctx)
        return [
            DetectionResult(box=[float(v) for v in boxes[i]],
                            confidence=float(scores[i]),
                            class_id=int(class_ids[i]))
            for i in self._merge_duplicates(boxes)
        ]

    def get_model_name(self) -> str:
        return "rtdetr"


class MaskRTDETRPostprocessor(_RTDETRBase):
    """mask-RT-DETR instance segmentation: one mask logit map per query."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        super().__init__(input_width, input_height, config)
        self.mask_threshold = float(self.config.get("mask_threshold", 0.5))

    def _find_masks(self, outputs: Sequence[np.ndarray],
                    queries: int) -> np.ndarray:
        """The per-query masks, ``[N, H, W]``.

        MEASURED on DX-RT 3.5.0: mask-rtdetr-hgnetv2-s emits ``(100, 640, 640)`` of
        int32 whose values are 0 and 1 -- already binary, already at the model input
        resolution. Not logits, not prototypes: applying a sigmoid to an integer 1
        would give 0.73 and thresholding that at 0.5 would still work, which is exactly
        the kind of accident that hides a wrong assumption. The 4-D ``(1, N, H, W)``
        form is accepted too.
        """
        for out in outputs:
            if out.ndim == 4 and out.shape[0] == 1 and out.shape[1] >= queries:
                return out[0]
            if out.ndim == 3 and out.shape[0] >= queries:
                return out
        raise ValueError(
            f"[DXAPP] [ERROR] {type(self).__name__} - per-query mask tensor "
            f"(N>={queries}, H, W) not found in the model outputs.\n"
            f"  Got: {_shapes(outputs)}\n"
            "  mask-RT-DETR emits the query table AND one mask per query; only the "
            "table was found."
        )

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[InstanceSegResult]:
        boxes, scores, class_ids, query_index = self._rows(outputs, ctx)
        if boxes.shape[0] == 0:
            # Still demand the mask tensor, so a wired-up-wrong model is reported
            # rather than looking like "nothing detected".
            self._find_masks(outputs, 1)
            return []
        masks = self._find_masks(outputs, int(query_index.max()) + 1)

        results = []
        for i in self._merge_duplicates(boxes):
            raw = masks[query_index[i]].astype(np.float32)
            # An already-binary int32 mask needs no sigmoid; a float logit map does.
            probability = 1.0 / (1.0 + np.exp(-raw)) if _needs_sigmoid(raw) else raw
            # INTER_LINEAR for both forms. numpy's astype CONVERTS an int32 mask
            # rather than reinterpreting its bits (unlike a C++ float* cast over the
            # same buffer), so a 0/1 mask arrives here as 0.0/1.0 and interpolating
            # before the 0.5 threshold simply gives a smoother edge.
            full = cv2.resize(probability, (ctx.original_width, ctx.original_height),
                              interpolation=cv2.INTER_LINEAR)
            box = boxes[i]
            binary = (full > self.mask_threshold).astype(np.uint8)
            # Confine to the box, as every other instance-segmentation postprocessor
            # in this tree does: a query's mask logits are not bounded by its box.
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2 = min(ctx.original_width, int(np.ceil(box[2])))
            y2 = min(ctx.original_height, int(np.ceil(box[3])))
            binary[:y1, :] = 0
            binary[y2:, :] = 0
            binary[:, :x1] = 0
            binary[:, x2:] = 0

            results.append(InstanceSegResult(
                box=[float(v) for v in box],
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
                mask=binary,
            ))
        return results

    def get_model_name(self) -> str:
        return "mask_rtdetr"
