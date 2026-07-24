"""
SFA3D Postprocessor

Decodes SFA3D model output tensors into 3D detection results.

Supported layouts:
  1) Multi-head NPU output (5 tensors): hm_cen, cen_offset, direction, z_coor, dim
  2) Single fused tensor (C, H, W) with 11 channels
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..base import IPostprocessor, PreprocessContext

logger = logging.getLogger(__name__)

# KITTI object classes — order must match SFA3D training (Pedestrian=0, Car=1, Cyclist=2)
SFA3D_CLASSES = ["Pedestrian", "Car", "Cyclist"]

# BEV grid parameters (must match preprocessor / kitti_config boundary)
_X_MIN, _X_MAX = 0.0, 50.0
_Y_MIN, _Y_MAX = -25.0, 25.0
_VELO_Z_MIN = -2.73  # boundary['minZ'] in official SFA3D kitti_config


@dataclass
class Detection3DResult:
    """Single 3D detection in KITTI-like format."""
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bev_x: float = 0.0
    bev_y: float = 0.0
    bev_w: float = 0.0
    bev_h: float = 0.0
    x3d: float = 0.0
    y3d: float = 0.0
    z3d: float = 0.0
    dim_h: float = 0.0
    dim_w: float = 0.0
    dim_l: float = 0.0
    yaw: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _topk_heatmap(heatmap: np.ndarray, k: int = 50):
    """Extract top-K peaks from a (C, H, W) heatmap."""
    num_classes, h, w = heatmap.shape
    heatmap_flat = heatmap.reshape(num_classes, -1)
    results = []
    for cls in range(num_classes):
        scores = heatmap_flat[cls]
        if k >= len(scores):
            topk_idx = np.argsort(-scores)
        else:
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        for idx in topk_idx:
            row = idx // w
            col = idx % w
            results.append((cls, scores[idx], row, col))
    return results


class SFA3DPostprocessor(IPostprocessor):
    """Decode SFA3D model outputs into 3D detections."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self._input_width = input_width
        self._input_height = input_height
        self.config = config or {}
        self._score_threshold = self.config.get(
            "score_threshold", self.config.get("conf_threshold", 0.3))
        self._max_detections = self.config.get("max_detections", 100)
        self._num_classes = len(SFA3D_CLASSES)

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[Detection3DResult]:
        if not outputs:
            return []
        try:
            return self._decode(outputs, ctx)
        except Exception:
            logger.warning(
                "SFA3D postprocessor: could not decode outputs "
                f"(shapes: {[o.shape for o in outputs]}). Returning empty.",
                exc_info=True,
            )
            return []

    def _decode(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[Detection3DResult]:
        if len(outputs) >= 5:
            heads = self._split_multi_head(outputs)
            if heads is not None:
                return self._decode_heads(*heads, ctx=ctx)

        out = _squeeze_batch(outputs[0])
        if out.ndim == 2:
            return self._decode_flat(out)
        if out.ndim != 3:
            return []

        c, h, w = out.shape
        expected_c = self._num_classes + 8
        if c < expected_c:
            return self._decode_heuristic(out, h, w, ctx)

        heatmap = _sigmoid(out[:self._num_classes])
        offset = out[self._num_classes:self._num_classes + 2]
        z_coord = out[self._num_classes + 2]
        dims = out[self._num_classes + 3:self._num_classes + 6]
        yaw_im = out[self._num_classes + 6]
        yaw_re = out[self._num_classes + 7]
        return self._decode_heads(
            heatmap, offset, z_coord, dims, yaw_im, yaw_re, h, w, ctx)

    def _split_multi_head(
        self, outputs: List[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Map 5-head NPU outputs by channel count."""
        tensors = [_squeeze_batch(o) for o in outputs[:5]]
        by_channels = {}
        for tensor in tensors:
            if tensor.ndim != 3:
                continue
            by_channels.setdefault(tensor.shape[0], []).append(tensor)

        if self._num_classes not in by_channels:
            return None
        hm_candidates = by_channels[self._num_classes]
        if len(hm_candidates) < 2:
            return None

        heatmap = hm_candidates[0]
        dims = hm_candidates[1]
        two_ch = by_channels.get(2, [])
        one_ch = by_channels.get(1, [])
        if len(two_ch) < 2 or len(one_ch) < 1:
            return None

        offset = two_ch[0]
        direction = two_ch[1]
        z_coord = one_ch[0]
        h, w = heatmap.shape[1], heatmap.shape[2]
        return (
            _sigmoid(heatmap),
            offset,
            z_coord,
            dims,
            direction[0],
            direction[1],
            h,
            w,
        )

    def _decode_heads(
        self,
        heatmap: np.ndarray,
        offset: np.ndarray,
        z_coord: np.ndarray,
        dims: np.ndarray,
        yaw_im: np.ndarray,
        yaw_re: np.ndarray,
        grid_h: int,
        grid_w: int,
        ctx: PreprocessContext,
    ) -> List[Detection3DResult]:
        peaks = _topk_heatmap(heatmap, k=self._max_detections)
        x_res = (_X_MAX - _X_MIN) / grid_h
        y_res = (_Y_MAX - _Y_MIN) / grid_w
        scale_x = self._input_width / max(grid_w, 1)
        scale_y = self._input_height / max(grid_h, 1)

        detections: List[Detection3DResult] = []
        for cls_id, score, row, col in peaks:
            if score < self._score_threshold:
                continue

            cx = (col + float(offset[0, row, col])) * scale_x
            cy = (row + float(offset[1, row, col])) * scale_y

            x3d = _X_MAX - (row + float(offset[1, row, col])) * x_res
            y3d = _Y_MIN + (col + float(offset[0, row, col])) * y_res
            z3d = float(z_coord[row, col]) if z_coord.ndim == 2 else float(z_coord[0, row, col])
            z3d += _VELO_Z_MIN  # network regresses (z - minZ); convert to velodyne metres

            # SFA3D regresses (h, w, l) in metres — no exp (see official evaluation_utils.py).
            dim_h = float(dims[0, row, col])
            dim_w = float(dims[1, row, col])
            dim_l = float(dims[2, row, col])
            yaw = float(np.arctan2(yaw_im[row, col], yaw_re[row, col]))

            bev_w_px = (dim_w / y_res) * scale_x
            bev_h_px = (dim_l / x_res) * scale_y

            detections.append(Detection3DResult(
                class_id=cls_id,
                class_name=SFA3D_CLASSES[cls_id] if cls_id < len(SFA3D_CLASSES) else f"cls_{cls_id}",
                confidence=float(score),
                bev_x=cx, bev_y=cy,
                bev_w=bev_w_px, bev_h=bev_h_px,
                x3d=x3d, y3d=y3d, z3d=z3d,
                dim_h=dim_h, dim_w=dim_w, dim_l=dim_l,
                yaw=yaw,
            ))

        detections.sort(key=lambda d: -d.confidence)
        return detections[:self._max_detections]

    def _decode_heuristic(self, out: np.ndarray, h: int, w: int,
                          ctx: PreprocessContext) -> List[Detection3DResult]:
        heatmap = _sigmoid(out[0:1])
        peaks = _topk_heatmap(heatmap, k=self._max_detections)
        scale_x = self._input_width / max(w, 1)
        scale_y = self._input_height / max(h, 1)
        x_res = (_X_MAX - _X_MIN) / h
        y_res = (_Y_MAX - _Y_MIN) / w

        detections = []
        for cls_id, score, row, col in peaks:
            if score < self._score_threshold:
                continue
            detections.append(Detection3DResult(
                class_id=0,
                class_name="Object",
                confidence=float(score),
                bev_x=float(col) * scale_x,
                bev_y=float(row) * scale_y,
                bev_w=10.0, bev_h=10.0,
                x3d=_X_MAX - row * x_res,
                y3d=_Y_MIN + col * y_res,
                z3d=0.0,
            ))
        detections.sort(key=lambda d: -d.confidence)
        return detections[:self._max_detections]

    def _decode_flat(self, out: np.ndarray) -> List[Detection3DResult]:
        detections = []
        for row in out:
            if len(row) < 7:
                continue
            score = float(row[1])
            if score < self._score_threshold:
                continue
            cls_id = int(row[0])
            detections.append(Detection3DResult(
                class_id=cls_id,
                class_name=SFA3D_CLASSES[cls_id] if cls_id < len(SFA3D_CLASSES) else f"cls_{cls_id}",
                confidence=score,
                x3d=float(row[2]), y3d=float(row[3]), z3d=float(row[4]),
                dim_h=float(row[5]), dim_w=float(row[6]),
                dim_l=float(row[7]) if len(row) > 7 else 0.0,
                yaw=float(row[8]) if len(row) > 8 else 0.0,
            ))
        detections.sort(key=lambda d: -d.confidence)
        return detections[:self._max_detections]

    def get_model_name(self) -> str:
        return "sfa3d"
