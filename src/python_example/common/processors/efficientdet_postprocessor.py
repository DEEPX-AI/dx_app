"""
EfficientDet Postprocessor

Handles EfficientDet-D0~D6 detection model outputs.

EfficientDet (TF-style post-NMS) outputs 2 tensors:
  - output[0]: [1, N, 4]   bounding boxes (normalized [ymin, xmin, ymax, xmax])
  - output[1]: [1, N]      class scores or [1, N, C+1] softmax

Auto-detects output format:
  A) TF post-processed: 2 tensors with boxes and class scores
  B) SSD-like: 2 tensors with softmax scores and box coordinates
  C) Multi-output (BiFPN features): attempts best-effort parsing

Coordinates are in [ymin, xmin, ymax, xmax] normalized (0-1) format,
converted to [x1, y1, x2, y2] pixel coordinates internally.
"""

import numpy as np
import cv2
from typing import List

from ..base import IPostprocessor, DetectionResult, PreprocessContext


class EfficientDetPostprocessor(IPostprocessor):
    """
    Postprocessor for EfficientDet detection models.

    Handles multiple output formats from different EfficientDet variants.
    """

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}

        self.score_threshold = self.config.get('score_threshold', 0.3)
        self.nms_threshold = self.config.get('nms_threshold', 0.45)
        self.num_classes = self.config.get('num_classes', 90)
        self.has_background = self.config.get('has_background', True)

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[DetectionResult]:
        """
        Process EfficientDet outputs.

        Supports:
          - 2-tensor format: [boxes, scores] or [scores, boxes]
          - 4-tensor TF format: [boxes, classes, scores, num_detections]

        Args:
            outputs: Model output tensors
            ctx: Preprocessing context

        Returns:
            List of DetectionResult
        """
        if len(outputs) == 4:
            return self._process_tf_format(outputs, ctx)
        elif len(outputs) == 2:
            return self._process_two_tensor(outputs, ctx)
        else:
            # Multi-output (BiFPN features) — attempt raw feature decode
            return self._process_multi_output(outputs, ctx)

    def _process_tf_format(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[DetectionResult]:
        """
        TF SavedModel format: boxes, classes, scores, num_detections
        """
        squeezed = [np.squeeze(o) for o in outputs]
        boxes, scores, classes, num_det = self._parse_tf_tensors(squeezed)

        if boxes is None or scores is None:
            return []

        if num_det is not None:
            boxes = boxes[:num_det]
            scores = scores[:num_det]
            if classes is not None:
                classes = classes[:num_det]

        return self._decode_results(boxes, scores, classes, ctx)

    def _classify_tf_tensor(self, t, boxes):
        """Return (role, value) for a single tensor: 'num_det', 'boxes', 'score_class', or (None, None)."""
        if t.ndim == 0 or (t.ndim == 1 and t.size == 1):
            return 'num_det', int(t.item()) if t.size == 1 else int(t)
        if t.ndim == 2 and t.shape[-1] == 4:
            return 'boxes', t
        if t.ndim == 1 and boxes is not None and t.size == boxes.shape[0]:
            return 'score_class', t
        return None, None

    def _parse_tf_tensors(self, squeezed):
        """Identify boxes, scores, classes, and num_detections from TF tensor list."""
        boxes = scores = classes = None
        num_det = None
        for t in squeezed:
            role, val = self._classify_tf_tensor(t, boxes)
            if role == 'num_det':
                num_det = val
            elif role == 'boxes':
                boxes = val
            elif role == 'score_class':
                if scores is None:
                    scores = val
                else:
                    classes = val
        return boxes, scores, classes, num_det

    def _process_two_tensor(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[DetectionResult]:
        """
        2-tensor format.
        Identify boxes (last dim=4) and scores tensor.
        """
        t0 = np.squeeze(outputs[0])
        t1 = np.squeeze(outputs[1])
        boxes_raw, scores_raw = self._determine_boxes_and_scores(t0, t1)

        if boxes_raw.ndim == 1:
            boxes_raw = boxes_raw.reshape(boxes_raw.size // 4, 4)

        scores, class_ids = self._parse_score_tensor(scores_raw)
        return self._decode_results(boxes_raw, scores, class_ids, ctx)

    def _determine_boxes_and_scores(self, t0: np.ndarray, t1: np.ndarray):
        """Identify which tensor is boxes (last dim=4) and which is scores."""
        if t0.ndim >= 2 and t0.shape[-1] == 4:
            return t0, t1
        if t1.ndim >= 2 and t1.shape[-1] == 4:
            return t1, t0
        # Fallback: larger tensor is scores
        return (t1, t0) if t0.size > t1.size else (t0, t1)

    def _parse_score_tensor(self, scores_raw: np.ndarray):
        """Parse score tensor into (scores, class_ids) arrays."""
        if scores_raw.ndim == 1:
            return scores_raw, np.zeros(len(scores_raw), dtype=int)
        if scores_raw.ndim == 2 and scores_raw.shape[-1] > 1:
            fg = scores_raw[:, 1:] if self.has_background else scores_raw
            return np.max(fg, axis=1), np.argmax(fg, axis=1)
        scores = scores_raw.flatten()
        return scores, np.zeros(len(scores), dtype=int)

    def _classify_tensor(self, s: np.ndarray, boxes_cand, scores_cand):
        """Classify a squeezed tensor as boxes, scores, or neither. Returns (boxes_cand, scores_cand)."""
        if s.ndim == 2 and s.shape[-1] == 4 and boxes_cand is None:
            return s, scores_cand
        if s.ndim == 1 and scores_cand is None:
            return boxes_cand, s
        if s.ndim == 2 and s.shape[-1] > 4 and scores_cand is None:
            fg = s[:, 1:] if self.has_background else s
            return boxes_cand, np.max(fg, axis=1)
        return boxes_cand, scores_cand

    def _process_multi_output(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[DetectionResult]:
        """
        Multi-output (BiFPN feature outputs). Best-effort: look for box-like and score-like tensors.
        """
        boxes_cand = None
        scores_cand = None

        for t in outputs:
            s = np.squeeze(t)
            boxes_cand, scores_cand = self._classify_tensor(s, boxes_cand, scores_cand)

        if boxes_cand is not None and scores_cand is not None:
            n = min(len(boxes_cand), len(scores_cand))
            return self._decode_results(
                boxes_cand[:n], scores_cand[:n], None, ctx)

        return []

    def _decode_results(self, boxes: np.ndarray, scores: np.ndarray,
                         class_ids, ctx: PreprocessContext) -> List[DetectionResult]:
        """
        Convert boxes/scores/classes to DetectionResult list.

        boxes: [N, 4] in [ymin, xmin, ymax, xmax] normalized or pixel format
        scores: [N] confidence scores
        class_ids: [N] or None
        """
        if class_ids is None:
            class_ids = np.zeros(len(scores), dtype=int)

        # Filter by threshold
        mask = scores >= self.score_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Determine if normalized (values mostly in [0, 2])
        is_normalized = np.percentile(np.abs(boxes), 95) < 2.0

        if is_normalized:
            # [ymin, xmin, ymax, xmax] normalized → [x1, y1, x2, y2] pixel
            x1 = boxes[:, 1] * self.input_width
            y1 = boxes[:, 0] * self.input_height
            x2 = boxes[:, 3] * self.input_width
            y2 = boxes[:, 2] * self.input_height
        else:
            # Already pixel coordinates; check if [ymin, xmin, ymax, xmax] or [x1,y1,x2,y2]
            # Heuristic: if col1 > col0 for most rows, it's xmin>ymin which is unlikely
            # Just assume [x1, y1, x2, y2]
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        boxes_pixel = np.column_stack([x1, y1, x2, y2])

        # NMS
        boxes_xywh = np.column_stack([
            boxes_pixel[:, 0], boxes_pixel[:, 1],
            boxes_pixel[:, 2] - boxes_pixel[:, 0],
            boxes_pixel[:, 3] - boxes_pixel[:, 1],
        ])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(),
            self.score_threshold, self.nms_threshold)

        if len(indices) == 0:
            return []

        keep = np.array(indices).reshape(-1)

        # Scale back to original coordinates
        # EfficientDet typically uses simple_resize preprocessing
        if ctx.pad_x == 0 and ctx.pad_y == 0:
            sx = ctx.original_width / self.input_width
            sy = ctx.original_height / self.input_height
        else:
            sx = sy = None

        results = []
        for idx in keep:
            box = self._scale_box(boxes_pixel[idx].copy(), ctx, sx, sy)
            results.append(DetectionResult(
                box=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                confidence=float(scores[idx]),
                class_id=int(class_ids[idx]),
            ))

        return results

    def _scale_box(self, box: np.ndarray, ctx, sx, sy) -> np.ndarray:
        """Scale a box from input space to original image coordinates."""
        if sx is not None:
            box[0] = np.clip(box[0] * sx, 0, ctx.original_width - 1)
            box[1] = np.clip(box[1] * sy, 0, ctx.original_height - 1)
            box[2] = np.clip(box[2] * sx, 0, ctx.original_width - 1)
            box[3] = np.clip(box[3] * sy, 0, ctx.original_height - 1)
        else:
            gain = max(ctx.scale, 1e-6)
            box[0] = np.clip((box[0] - ctx.pad_x) / gain, 0, ctx.original_width - 1)
            box[1] = np.clip((box[1] - ctx.pad_y) / gain, 0, ctx.original_height - 1)
            box[2] = np.clip((box[2] - ctx.pad_x) / gain, 0, ctx.original_width - 1)
            box[3] = np.clip((box[3] - ctx.pad_y) / gain, 0, ctx.original_height - 1)
        return box

    def get_model_name(self) -> str:
        return "efficientdet"
