"""
RetinaFace Postprocessor

Anchor-based face detection with 5-point landmarks.

Supports two output formats automatically:
  1. NHWC feature-map format (this model):
       9 tensors: [1, H, W, A*4], [1, H, W, A*2], [1, H, W, A*10]  × 3 strides
  2. Flattened format (some compiled models):
       3 tensors: [1, N, 4], [1, N, 2], [1, N, 10]

Anchors: 2 per location, strides [8, 16, 32], min_sizes [[16,32],[64,128],[256,512]].
Box decoding uses variance = [0.1, 0.2].
"""

import numpy as np
import cv2
from itertools import product
from typing import List, Optional, Tuple

from ..base import IPostprocessor, PreprocessContext, Keypoint
from .face_postprocessor import FaceResult


class RetinaFacePostprocessor(IPostprocessor):
    """
    Postprocessor for RetinaFace anchor-based face detection.
    Handles both NHWC feature-map and pre-flattened output formats.
    """

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}

        self.score_threshold = self.config.get('score_threshold', 0.5)
        self.nms_threshold = self.config.get('nms_threshold', 0.4)
        self.variance = self.config.get('variance', [0.1, 0.2])
        self.strides = self.config.get('strides', [8, 16, 32])
        self.min_sizes = self.config.get('min_sizes', [[16, 32], [64, 128], [256, 512]])
        self.num_anchors = self.config.get('num_anchors', 2)
        self.top_k = self.config.get('top_k', 750)

        self._priors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Prior generation
    # ------------------------------------------------------------------

    def _generate_priors(self) -> np.ndarray:
        anchors = []
        for k, stride in enumerate(self.strides):
            feat_h = (self.input_height + stride - 1) // stride
            feat_w = (self.input_width + stride - 1) // stride
            min_s = self.min_sizes[k] if k < len(self.min_sizes) else self.min_sizes[-1]
            for i, j in product(range(feat_h), range(feat_w)):
                for min_size in min_s:
                    cx = (j + 0.5) * stride / self.input_width
                    cy = (i + 0.5) * stride / self.input_height
                    s_kx = min_size / self.input_width
                    s_ky = min_size / self.input_height
                    anchors.append([cx, cy, s_kx, s_ky])
        return np.array(anchors, dtype=np.float32)

    @property
    def priors(self) -> np.ndarray:
        if self._priors is None:
            self._priors = self._generate_priors()
        return self._priors

    # ------------------------------------------------------------------
    # Format detection & tensor parsing
    # ------------------------------------------------------------------

    def _is_nhwc_format(self, outputs: List[np.ndarray]) -> bool:
        """Return True if outputs are NHWC feature-map format (ndim==4, last_dim != 4/2/10 singletons)."""
        for o in outputs:
            arr = np.squeeze(o)
            if arr.ndim == 3:  # H x W x C after squeeze
                return True
        return False

    def _parse_nhwc_outputs(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Parse NHWC feature-map tensors into flattened [N, k] arrays.
        Groups by feature-map spatial size, then identifies bbox/class/lmk by last-dim ratio.
        """
        # Squeeze batch dim; expect shape [H, W, C]
        tensors = [np.squeeze(o) for o in outputs]
        tensors = [t for t in tensors if t.ndim == 3]

        # Group tensors by (H, W)
        groups: dict = {}
        for t in tensors:
            key = (t.shape[0], t.shape[1])
            groups.setdefault(key, []).append(t)

        all_bbox, all_cls, all_lmk = [], [], []

        for (fh, fw), group in sorted(groups.items(), key=lambda x: -x[0][0]):
            bbox_t = cls_t = lmk_t = None
            for t in group:
                c = t.shape[2]
                per_anchor = c // self.num_anchors
                if per_anchor == 4:
                    bbox_t = t
                elif per_anchor == 2:
                    cls_t = t
                elif per_anchor == 10:
                    lmk_t = t

            if bbox_t is None or cls_t is None:
                continue

            n = fh * fw * self.num_anchors

            # Reshape [H, W, A*k] → [H*W*A, k]
            # Interleave anchors: for each cell, anchors come in order
            def _reshape(t, k):
                # t: [H, W, A*k] → [H, W, A, k] → [H*W*A, k]
                return t.reshape(fh, fw, self.num_anchors, k).reshape(-1, k)

            all_bbox.append(_reshape(bbox_t, 4))
            all_cls.append(_reshape(cls_t, 2))
            if lmk_t is not None:
                all_lmk.append(_reshape(lmk_t, 10))

        if not all_bbox:
            return None, None, None

        bbox = np.concatenate(all_bbox, axis=0)
        cls  = np.concatenate(all_cls,  axis=0)
        lmk  = np.concatenate(all_lmk,  axis=0) if all_lmk else None
        return bbox, cls, lmk

    def _parse_flat_outputs(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Parse pre-flattened outputs (last dim: 4=bbox, 2=cls, 10=lmk)."""
        bbox_t = cls_t = lmk_t = None
        for t in [np.squeeze(o) for o in outputs]:
            if t.ndim < 2:
                continue
            d = t.shape[-1]
            if d == 4 and bbox_t is None:
                bbox_t = t
            elif d == 2 and cls_t is None:
                cls_t = t
            elif d == 10 and lmk_t is None:
                lmk_t = t
        return bbox_t, cls_t, lmk_t

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _filter_topk(self, bbox_t, score_t, lmk_t, priors):
        if score_t.shape[-1] == 2:
            scores = self._softmax(score_t)[:, 1]
        else:
            scores = score_t.flatten()

        n = min(bbox_t.shape[0], scores.shape[0], priors.shape[0])
        bbox_t, scores, priors = bbox_t[:n], scores[:n], priors[:n]
        if lmk_t is not None:
            lmk_t = lmk_t[:n]

        order = scores.argsort()[::-1][:self.top_k]
        scores, bbox_t, priors = scores[order], bbox_t[order], priors[order]
        if lmk_t is not None:
            lmk_t = lmk_t[order]

        mask = scores >= self.score_threshold
        return bbox_t[mask], scores[mask], lmk_t[mask] if lmk_t is not None else None, priors[mask]

    def _decode_boxes(self, bbox_t, priors):
        var0, var1 = self.variance
        cx = priors[:, 0] + bbox_t[:, 0] * var0 * priors[:, 2]
        cy = priors[:, 1] + bbox_t[:, 1] * var0 * priors[:, 3]
        w  = priors[:, 2] * np.exp(bbox_t[:, 2] * var1)
        h  = priors[:, 3] * np.exp(bbox_t[:, 3] * var1)
        x1 = (cx - w / 2) * self.input_width
        y1 = (cy - h / 2) * self.input_height
        x2 = (cx + w / 2) * self.input_width
        y2 = (cy + h / 2) * self.input_height
        return x1, y1, x2, y2

    def _decode_landmarks(self, lmk_t, priors):
        if lmk_t is None:
            return None
        var0 = self.variance[0]
        decoded = np.zeros_like(lmk_t)
        for k in range(5):
            decoded[:, k*2]   = (priors[:,0] + lmk_t[:,k*2]   * var0 * priors[:,2]) * self.input_width
            decoded[:, k*2+1] = (priors[:,1] + lmk_t[:,k*2+1] * var0 * priors[:,3]) * self.input_height
        return decoded

    def _build_results(self, keep, x1, y1, x2, y2, face_scores, lmk_decoded, ctx):
        pad_x, pad_y = ctx.pad_x, ctx.pad_y
        ow, oh = ctx.original_width - 1, ctx.original_height - 1
        # Use independent scale_x/scale_y for stretch-resize; fall back to uniform scale
        if ctx.scale_x > 0 and ctx.scale_y > 0:
            gain_x, gain_y = ctx.scale_x, ctx.scale_y
        else:
            gain_x = gain_y = max(ctx.scale, 1e-6)

        results = []
        for idx in keep:
            bx1 = float(np.clip((x1[idx] - pad_x) / gain_x, 0, ow))
            by1 = float(np.clip((y1[idx] - pad_y) / gain_y, 0, oh))
            bx2 = float(np.clip((x2[idx] - pad_x) / gain_x, 0, ow))
            by2 = float(np.clip((y2[idx] - pad_y) / gain_y, 0, oh))

            keypoints = []
            if lmk_decoded is not None:
                for kp in lmk_decoded[idx].reshape(5, 2):
                    kx = float(np.clip((kp[0] - pad_x) / gain_x, 0, ow))
                    ky = float(np.clip((kp[1] - pad_y) / gain_y, 0, oh))
                    keypoints.append(Keypoint(x=kx, y=ky, confidence=1.0))

            results.append(FaceResult(
                box=[bx1, by1, bx2, by2],
                confidence=float(face_scores[idx]),
                class_id=0,
                keypoints=keypoints,
            ))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[FaceResult]:
        if self._is_nhwc_format(outputs):
            bbox_t, score_t, lmk_t = self._parse_nhwc_outputs(outputs)
        else:
            bbox_t, score_t, lmk_t = self._parse_flat_outputs(outputs)

        if bbox_t is None or score_t is None:
            return []

        bbox_t, face_scores, lmk_t, priors = self._filter_topk(
            bbox_t, score_t, lmk_t, self.priors)
        if face_scores.size == 0:
            return []

        x1, y1, x2, y2 = self._decode_boxes(bbox_t, priors)

        boxes_xywh = np.column_stack([x1, y1, x2 - x1, y2 - y1])
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), face_scores.tolist(),
            self.score_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []

        keep = np.array(indices).reshape(-1)
        lmk_decoded = self._decode_landmarks(lmk_t, priors)
        return self._build_results(keep, x1, y1, x2, y2, face_scores, lmk_decoded, ctx)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def get_model_name(self) -> str:
        return "retinaface"
