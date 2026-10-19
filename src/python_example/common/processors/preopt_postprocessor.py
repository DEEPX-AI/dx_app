# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Postprocessors for the pre-optimized YOLO models (top-k selected in the model).

A pre-optimized ``.dxnn`` carries the tail of the end-to-end head inside the model:
the DFL integration runs on the NPU and a CPU task -- executed by DXRT through ONNX
Runtime -- applies sigmoid to the class logits, keeps the top-k (300) anchors by their
best class score, keeps the top-k (anchor, class) pairs of those, decodes boxes and
keypoints and gathers the mask coefficients. What the application receives is a
fixed-size table of K rows in model-input (letterboxed) pixel coordinates, sorted by
score descending::

    preopt_output [1, K, C]      C = 6 + extra
      [x1, y1, x2, y2, score, class_id, extra...]
      extra: detection    -> none                        (C = 6)
             pose         -> 17 x (x, y, visibility)     (C = 57)
             segmentation -> 32 mask coefficients        (C = 38)
                             + prototypes [1, 32, 160, 160]

Measured on an M1 with DXRT v3.4.2 against dx_yolo26's three pre-optimized models:
det (1, 300, 6), pose (1, 300, 57), seg (1, 300, 38) + (1, 32, 160, 160).

The second top-k is taken over the flattened (anchor x class) scores, as in
Ultralytics' end-to-end postprocess, so one anchor may appear several times with
different classes -- the same box as "car" and as "truck". The model itself is
NMS-free; the class-agnostic NMS here exists only to merge those duplicates. Set
``nms_threshold >= 1.0`` to keep every row.

Rows of a partly filled table are all zero, and a zero row has score 0.0. The score
test is therefore ``>=`` against a threshold floored above zero, because a literal
0.0 threshold would otherwise admit 300 boxes at the origin.

The C++ peer (``src/cpp_example/common/processors/preopt_topk_postprocessor.hpp``)
also decodes the raw per-stride NPU tensors (``preopt_bbox_tr_i``,
``preopt_cls_tr_i``, ...) for when DXRT runs without ONNX Runtime and the CPU task is
skipped. That path is deliberately NOT implemented here: every model measured produced
the table, and a second decode path that nothing exercises is a liability rather than
a feature. :func:`find_row_table` raises with the actual tensor shapes when the table
is absent, which names that situation exactly instead of guessing at it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..base import (InstanceSegResult, IPostprocessor, DetectionResult, Keypoint,
                    PoseResult, PreprocessContext)

X1, Y1, X2, Y2, SCORE, CLASS, EXTRA = 0, 1, 2, 3, 4, 5, 6


def _shapes(outputs: Sequence[np.ndarray]) -> str:
    return ", ".join("(" + ", ".join(str(d) for d in o.shape) + ")" for o in outputs)


def find_row_table(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """The ``[K, C]`` table produced by the model's CPU task, batch dim removed.

    Identified by shape, never by position: ``(1, K, C)`` with ``K > C >= 6``. The
    prototypes are 4-D, and a non-pre-optimized detection head would be ``(1, 4+nc,
    N)`` with the long axis last, so neither is mistaken for the table.
    """
    for out in outputs:
        if out.ndim == 3 and out.shape[0] == 1 and out.shape[1] > out.shape[2] >= EXTRA:
            return out[0]
    raise ValueError(
        "[DXAPP] [ERROR] pre-optimized row table (1, K, C>=6) not found in the model "
        f"outputs. Got: {_shapes(outputs)}. This postprocessor requires a model "
        "compiled with the pre-optimize (top-k) output configuration, and a DXRT "
        "built with ONNX Runtime so the model's CPU task actually runs."
    )


def find_prototypes(outputs: Sequence[np.ndarray], num_mask_coefs: int) -> np.ndarray:
    """The ``[C, H, W]`` mask prototypes of a pre-optimized segmentation model."""
    for out in outputs:
        if out.ndim == 4 and num_mask_coefs in (out.shape[1], out.shape[-1]):
            proto = np.squeeze(out, axis=0)
            if proto.shape[0] != num_mask_coefs and proto.shape[-1] == num_mask_coefs:
                proto = np.transpose(proto, (2, 0, 1))      # HWC -> CHW
            return proto
    raise ValueError(
        "[DXAPP] [ERROR] mask prototype tensor (1, "
        f"{num_mask_coefs}, H, W) not found in the model outputs. "
        f"Got: {_shapes(outputs)}. A pre-optimized segmentation model emits the row "
        "table AND the prototypes; only the table was found."
    )


def select_rows(table: np.ndarray, score_threshold: float,
                nms_threshold: float) -> np.ndarray:
    """Row indices at or above the threshold, best first, duplicates merged."""
    # A fixed-size table pads with zero rows, so never let a 0.0 threshold admit them.
    threshold = max(float(score_threshold), float(np.finfo(np.float32).tiny))
    keep = np.flatnonzero(table[:, SCORE] >= threshold)
    if keep.size == 0:
        return keep
    keep = keep[np.argsort(-table[keep, SCORE], kind="stable")]
    if nms_threshold >= 1.0:
        return keep

    boxes = table[keep, X1:Y2 + 1].astype(np.float32)
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    areas = widths * heights

    order = list(range(keep.size))
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
        inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)
        order = [o for o, survives in zip(order, iou < nms_threshold) if survives]
    return keep[kept]


def _scale_box(row: np.ndarray, ctx: PreprocessContext) -> List[float]:
    """Undo the letterbox, clipping to the original image as every runner expects."""
    gain = max(ctx.scale, 1e-6)
    return [
        float(np.clip((row[X1] - ctx.pad_x) / gain, 0, ctx.original_width - 1)),
        float(np.clip((row[Y1] - ctx.pad_y) / gain, 0, ctx.original_height - 1)),
        float(np.clip((row[X2] - ctx.pad_x) / gain, 0, ctx.original_width - 1)),
        float(np.clip((row[Y2] - ctx.pad_y) / gain, 0, ctx.original_height - 1)),
    ]


class _PreoptBase(IPostprocessor):
    """Shared configuration for the three pre-optimized postprocessors."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.score_threshold = float(
            self.config.get("score_threshold", self.config.get("conf_threshold", 0.3)))
        self.nms_threshold = float(self.config.get("nms_threshold", 0.45))
        self.top_k = int(self.config.get("top_k", 300))

    def _table(self, outputs: Sequence[np.ndarray], min_cols: int) -> np.ndarray:
        table = find_row_table(outputs)
        if table.shape[1] < min_cols:
            raise ValueError(
                f"[DXAPP] [ERROR] {type(self).__name__} - the row table has "
                f"{table.shape[1]} columns, expected at least {min_cols}. "
                f"Got: {_shapes(outputs)}."
            )
        # top_k bounds how many rows are considered, mirroring the model's own K.
        return table[:self.top_k]


class PreoptDetectionPostprocessor(_PreoptBase):
    """Detection: rows are ``[x1, y1, x2, y2, score, class_id]``."""

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[DetectionResult]:
        table = self._table(outputs, EXTRA)
        results = []
        for i in select_rows(table, self.score_threshold, self.nms_threshold):
            row = table[i]
            results.append(DetectionResult(
                box=_scale_box(row, ctx),
                confidence=float(row[SCORE]),
                class_id=int(row[CLASS]),
            ))
        return results

    def get_model_name(self) -> str:
        return "yolo_preopt"


class PreoptPosePostprocessor(_PreoptBase):
    """Pose: rows carry ``num_keypoints x (x, y, visibility)`` after the six columns."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        super().__init__(input_width, input_height, config)
        self.num_keypoints = int(self.config.get("num_keypoints", 17))

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[PoseResult]:
        table = self._table(outputs, EXTRA + 3 * self.num_keypoints)
        gain = max(ctx.scale, 1e-6)
        results = []
        for i in select_rows(table, self.score_threshold, self.nms_threshold):
            row = table[i]
            triplets = row[EXTRA:EXTRA + 3 * self.num_keypoints].reshape(-1, 3)
            keypoints = [
                Keypoint(
                    x=float(np.clip((kp[0] - ctx.pad_x) / gain,
                                    0, ctx.original_width - 1)),
                    y=float(np.clip((kp[1] - ctx.pad_y) / gain,
                                    0, ctx.original_height - 1)),
                    confidence=float(kp[2]),
                )
                for kp in triplets
            ]
            results.append(PoseResult(
                box=_scale_box(row, ctx),
                confidence=float(row[SCORE]),
                class_id=int(row[CLASS]),
                keypoints=keypoints,
            ))
        return results

    def get_model_name(self) -> str:
        return "yolo_preopt_pose"


class PreoptSegPostprocessor(_PreoptBase):
    """Instance segmentation: rows carry mask coefficients; prototypes come separately."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        super().__init__(input_width, input_height, config)
        self.num_mask_coefs = int(self.config.get("num_mask_coefs", 32))
        self.mask_threshold = float(self.config.get("mask_threshold", 0.5))

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[InstanceSegResult]:
        # Resolve the prototypes FIRST: a caller who forgot them should be told that,
        # not handed an empty result list that looks like "nothing detected".
        proto = find_prototypes(outputs, self.num_mask_coefs)
        table = self._table(outputs, EXTRA + self.num_mask_coefs)
        keep = select_rows(table, self.score_threshold, self.nms_threshold)
        if keep.size == 0:
            return []

        channels, proto_h, proto_w = proto.shape
        coefs = table[keep, EXTRA:EXTRA + self.num_mask_coefs].astype(np.float32)
        # Same math as YOLOv8InstanceSegPostprocessor: sigmoid(coefs @ prototypes),
        # upsampled to the model input, cropped to the box in INPUT coordinates, then
        # unpadded and resized to the original frame.
        masks = 1.0 / (1.0 + np.exp(-(coefs @ proto.reshape(channels, -1))))
        masks = masks.reshape(-1, proto_h, proto_w)

        gain = max(ctx.scale, 1e-6)
        unpad_w = int(round(ctx.original_width * gain))
        unpad_h = int(round(ctx.original_height * gain))
        left, top = int(ctx.pad_x), int(ctx.pad_y)

        results = []
        for slot, i in enumerate(keep):
            row = table[i]
            scaled = cv2.resize(masks[slot], (self.input_width, self.input_height),
                                interpolation=cv2.INTER_LINEAR)
            bx1 = max(0, int(row[X1]))
            by1 = max(0, int(row[Y1]))
            bx2 = min(self.input_width, int(row[X2]))
            by2 = min(self.input_height, int(row[Y2]))
            scaled[:by1, :] = 0.0
            scaled[by2:, :] = 0.0
            scaled[:, :bx1] = 0.0
            scaled[:, bx2:] = 0.0

            cropped = scaled[top:top + unpad_h, left:left + unpad_w]
            if cropped.size:
                mask = cv2.resize(cropped, (ctx.original_width, ctx.original_height),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                mask = np.zeros((ctx.original_height, ctx.original_width),
                                dtype=np.float32)

            results.append(InstanceSegResult(
                box=_scale_box(row, ctx),
                confidence=float(row[SCORE]),
                class_id=int(row[CLASS]),
                mask=(mask > self.mask_threshold).astype(np.uint8),
            ))
        return results

    def get_model_name(self) -> str:
        return "yolo_preopt_seg"
