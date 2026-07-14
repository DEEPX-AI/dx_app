"""
MediaPipe Hand (Palm) Detector Postprocessor

Decodes MediaPipe palm-detection SSD outputs (NHWC feature-map format) into
hand bounding boxes with confidence scores.

Model outputs (NHWC):
  - regressor_palm_8:    [1, 24, 24, 36]   stride-8  (2 anchors × 18 values)
  - classifier_palm_8:   [1, 24, 24, 2]    stride-8  (2 anchors × 1 score)
  - regressor_palm_16:   [1, 12, 12, 108]  stride-16 (6 anchors × 18 values)
  - classifier_palm_16:  [1, 12, 12, 6]    stride-16 (6 anchors × 1 score)

Total anchors = 24×24×2 + 12×12×6 = 1152 + 864 = 2016

Anchor generation follows MediaPipe SsdAnchorsCalculator:
  strides = [8, 16, 16, 16], 2 anchors per cell per layer.
  stride-16 has 3 layers → 6 anchors per cell.

Decoding follows MediaPipe TensorsToDetectionsCalculator.
Box expansion (box_scale, box_shift) follows DetectionsToRects.
"""

import math
import numpy as np
import cv2
from typing import List, Optional

from ..base import IPostprocessor, PreprocessContext
from .face_postprocessor import FaceResult


# ── Anchor generation (identical to dx-modelzoo custom_ops.py) ──────────────

_STRIDES = (8, 16, 16, 16)
_NUM_LAYERS = 4
_ANCHOR_OFFSET = 0.5
_ANCHORS_PER_CELL = 2
_KP_WRIST = 0
_KP_MIDDLE = 2
_ANCHOR_CACHE: dict = {}


def _generate_palm_anchors(input_size: int = 192) -> np.ndarray:
    if input_size in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[input_size]
    anchors = []
    layer_id = 0
    while layer_id < _NUM_LAYERS:
        last_same = layer_id
        repeats = 0
        while last_same < _NUM_LAYERS and _STRIDES[last_same] == _STRIDES[layer_id]:
            repeats += _ANCHORS_PER_CELL
            last_same += 1
        stride = _STRIDES[layer_id]
        feature_map = math.ceil(input_size / stride)
        for y in range(feature_map):
            for x in range(feature_map):
                cx = (x + _ANCHOR_OFFSET) / feature_map
                cy = (y + _ANCHOR_OFFSET) / feature_map
                for _ in range(repeats):
                    anchors.append((cx, cy))
        layer_id = last_same
    arr = np.asarray(anchors, dtype=np.float32)
    _ANCHOR_CACHE[input_size] = arr
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100.0, 100.0)))


class MediaPipeHandPostprocessor(IPostprocessor):
    """
    Postprocessor for MediaPipe Palm/Hand Detector.

    Reshapes NHWC feature-map outputs, generates SSD anchors, decodes
    box/keypoint offsets, applies sigmoid + NMS, and expands palm box to
    full-hand bounding box.
    """

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_size = input_width  # model is square 192×192
        self.config = config or {}
        self.conf_thres = float(self.config.get('conf_threshold', self.config.get('score_threshold', 0.5)))
        self.iou_thres = float(self.config.get('nms_threshold', 0.3))
        self.box_scale = float(self.config.get('box_scale', 2.0))
        self.box_shift = float(self.config.get('box_shift', 0.3))

    def _parse_outputs(self, outputs):
        """
        Parse outputs into (reg [N,18], scores [N]).
        Handles both NHWC feature-map format and pre-flattened format.
        """
        tensors = [np.squeeze(o) for o in outputs]

        reg, score = None, None

        # First try flat format (already reshaped by dxrt): last_dim==18 or last_dim==1
        for t in tensors:
            if t.ndim == 2:
                if t.shape[-1] == 18 and reg is None:
                    reg = t
                elif t.shape[-1] == 1 and score is None:
                    score = t.reshape(-1)
            elif t.ndim == 1 and score is None:
                score = t

        if reg is not None and score is not None:
            return reg, score

        # Fallback: NHWC feature-map format [H, W, C]
        regs, clss = [], []
        for t in tensors:
            if t.ndim != 3:
                continue
            h, w, c = t.shape
            if c % 18 == 0:
                a = c // 18
                regs.append((h * w * a, t.reshape(h, w, a, 18).reshape(-1, 18)))
            elif c <= 8:
                a = c
                clss.append((h * w * a, t.reshape(-1)))

        if not regs or not clss:
            return None, None

        regs.sort(key=lambda x: -x[0])
        clss.sort(key=lambda x: -x[0])
        reg = np.concatenate([r[1] for r in regs], axis=0)
        score = np.concatenate([c[1] for c in clss], axis=0)
        return reg, score

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[FaceResult]:
        reg, score_logits = self._parse_outputs(outputs)
        if reg is None or score_logits is None:
            return []

        anchors = _generate_palm_anchors(self.input_size)
        n = min(len(anchors), reg.shape[0], score_logits.shape[0])
        anchors = anchors[:n]
        reg = reg[:n]
        scores = _sigmoid(score_logits[:n])

        scale = float(self.input_size)
        cx = reg[:, 0] / scale + anchors[:, 0]
        cy = reg[:, 1] / scale + anchors[:, 1]
        w = np.abs(reg[:, 2]) / scale
        h = np.abs(reg[:, 3]) / scale
        s = np.maximum(w, h)

        # Hand orientation keypoints (wrist=0, middle-finger MCP=2)
        kx0 = reg[:, 4 + 2 * _KP_WRIST]     / scale + anchors[:, 0]
        ky0 = reg[:, 4 + 2 * _KP_WRIST + 1] / scale + anchors[:, 1]
        kx2 = reg[:, 4 + 2 * _KP_MIDDLE]     / scale + anchors[:, 0]
        ky2 = reg[:, 4 + 2 * _KP_MIDDLE + 1] / scale + anchors[:, 1]
        dx = kx2 - kx0
        dy = ky2 - ky0
        norm = np.hypot(dx, dy) + 1e-9
        dx /= norm
        dy /= norm

        # Expand palm → full-hand box
        side = s * self.box_scale
        ccx = cx + self.box_shift * side * dx
        ccy = cy + self.box_shift * side * dy
        x1 = (ccx - side / 2.0)
        y1 = (ccy - side / 2.0)
        x2 = (ccx + side / 2.0)
        y2 = (ccy + side / 2.0)

        mask = scores > self.conf_thres
        x1, y1, x2, y2, scores_f = x1[mask], y1[mask], x2[mask], y2[mask], scores[mask]
        if len(scores_f) == 0:
            return []

        # NMS (normalized coords)
        boxes_xywh = np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_f.tolist(), self.conf_thres, self.iou_thres)
        if len(indices) == 0:
            return []
        keep = np.array(indices).reshape(-1)

        # Back to pixel coords
        ow = ctx.original_width
        oh = ctx.original_height
        if ctx.scale_x > 0 and ctx.scale_y > 0:
            px = ow  # direct: normalized → original pixel (no extra scale needed)
            py = oh
        else:
            px = ow
            py = oh

        results = []
        for i in keep:
            bx1 = float(np.clip(x1[i] * ow, 0, ow - 1))
            by1 = float(np.clip(y1[i] * oh, 0, oh - 1))
            bx2 = float(np.clip(x2[i] * ow, 0, ow - 1))
            by2 = float(np.clip(y2[i] * oh, 0, oh - 1))
            results.append(FaceResult(
                box=[bx1, by1, bx2, by2],
                confidence=float(scores_f[i]),
                class_id=0,
                keypoints=[],
            ))
        return results

    def get_model_name(self) -> str:
        return "mediapipe_hand_detector"
