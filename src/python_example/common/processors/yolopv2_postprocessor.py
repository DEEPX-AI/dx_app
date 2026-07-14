"""
YOLOPv2 Panoptic Driving Perception Postprocessor

Input:  UINT8 [1, 384, 640, 3]
Outputs:
  det0:     FLOAT [1, 255, 48, 80]
  det1:     FLOAT [1, 255, 24, 40]
  det2:     FLOAT [1, 255, 12, 20]
  output_4: FLOAT [1, 2, 384, 640] — drivable area
  output_5: FLOAT [1, 1, 384, 640] — lane line

Reference: dx-modelzoo/src/dx_modelzoo/models/cv/panoptic_driving_perception/yolopv2/custom_ops.py
"""

import numpy as np
import cv2
from typing import List
from dataclasses import dataclass

from ..base import IPostprocessor, PreprocessContext, DetectionResult


@dataclass
class YOLOPv2Result:
    """Result from YOLOPv2 panoptic driving perception."""
    detections: List[DetectionResult]
    drivable_mask: np.ndarray
    lane_mask: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _nms_agnostic(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Class-agnostic greedy NMS. boxes in xyxy."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


class YOLOPv2Postprocessor(IPostprocessor):
    """Postprocessor for YOLOPv2 panoptic driving perception.

    Key design decisions (aligned with custom_ops.py):
    - Only detects vehicles (class index 3), not all 80 COCO classes.
    - Uses class-agnostic NMS.
    - Restores box coordinates using letterbox scale + pad from PreprocessContext.
    - Returns masks resized to original image resolution for display.
    """

    ANCHORS = [
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]],
    ]
    STRIDES = [8, 16, 32]
    NUM_CLASSES = 80
    VEHICLE_CLASS_INDEX = 3
    LANE_THRESHOLD = 0.5

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.conf_threshold = float(self.config.get("conf_threshold", 0.25))
        self.nms_threshold = float(self.config.get("nms_threshold", 0.45))
        self.max_det = int(self.config.get("max_det", 300))

    def _decode_head(self, pred: np.ndarray, anchors, stride: int) -> tuple:
        """Decode one detection head. Returns (boxes_xyxy, scores) in model-input coords."""
        pred = np.squeeze(pred, axis=0)
        num_anchors = len(anchors)
        height, width = pred.shape[1], pred.shape[2]
        # [na, no, H, W] → [na, H, W, no]
        pred = pred.reshape(num_anchors, 5 + self.NUM_CLASSES, height, width)
        pred = pred.transpose(0, 2, 3, 1)
        pred = _sigmoid(pred)

        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

        boxes_list, scores_list = [], []
        for a_idx, (aw, ah) in enumerate(anchors):
            head = pred[a_idx]
            cx = (head[..., 0] * 2.0 - 0.5 + grid[..., 0]) * stride
            cy = (head[..., 1] * 2.0 - 0.5 + grid[..., 1]) * stride
            bw = (head[..., 2] * 2.0) ** 2 * aw
            bh = (head[..., 3] * 2.0) ** 2 * ah
            obj = head[..., 4]

            # Vehicle class only (index 3), matching custom_ops.py vehicle_class_index
            cls_conf = head[..., 5 + self.VEHICLE_CLASS_INDEX]
            score = obj * cls_conf

            mask = score >= self.conf_threshold
            if not np.any(mask):
                continue
            cx_m, cy_m = cx[mask], cy[mask]
            bw_m, bh_m = bw[mask], bh[mask]
            x1 = cx_m - bw_m / 2
            y1 = cy_m - bh_m / 2
            x2 = cx_m + bw_m / 2
            y2 = cy_m + bh_m / 2
            boxes_list.append(np.stack([x1, y1, x2, y2], axis=1))
            scores_list.append(score[mask])

        if not boxes_list:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return (
            np.concatenate(boxes_list, axis=0).astype(np.float32),
            np.concatenate(scores_list, axis=0).astype(np.float32),
        )

    def _decode_seg(self, output: np.ndarray, ctx: PreprocessContext, is_lane: bool) -> np.ndarray:
        """Decode a segmentation head and return mask resized to original image size."""
        arr = np.squeeze(output, axis=0)
        if is_lane:
            seg = (arr[0] > self.LANE_THRESHOLD).astype(np.uint8)
        else:
            seg = np.argmax(arr, axis=0).astype(np.uint8)

        # Compute content region (undo letterbox padding)
        content_h = int(round(ctx.original_height * ctx.scale))
        content_w = int(round(ctx.original_width * ctx.scale))
        top = ctx.pad_y
        left = ctx.pad_x
        top = max(0, min(top, seg.shape[0] - content_h))
        left = max(0, min(left, seg.shape[1] - content_w))
        seg_content = seg[top: top + content_h, left: left + content_w]

        return cv2.resize(
            seg_content,
            (ctx.original_width, ctx.original_height),
            interpolation=cv2.INTER_NEAREST,
        )

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext):
        det_outputs, seg_outputs = [], []
        for output in outputs:
            if output.shape[1] == 255:
                det_outputs.append(output)
            else:
                seg_outputs.append(output)

        # Sort detection heads: largest grid (smallest stride) first
        det_outputs.sort(key=lambda x: -(x.shape[2] * x.shape[3]))

        all_boxes, all_scores = [], []
        for pred, anchors, stride in zip(det_outputs, self.ANCHORS, self.STRIDES):
            boxes, scores = self._decode_head(pred, anchors, stride)
            if boxes.shape[0] > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)

        detections = []
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)

            # Undo letterbox: (model_coord - pad) / scale → original coords
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - ctx.pad_x) / ctx.scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - ctx.pad_y) / ctx.scale
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ctx.original_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ctx.original_height)

            keep = _nms_agnostic(boxes, scores, self.nms_threshold)
            keep = keep[: self.max_det]
            for idx in keep:
                detections.append(DetectionResult(
                    box=[
                        float(boxes[idx, 0]),
                        float(boxes[idx, 1]),
                        float(boxes[idx, 2]),
                        float(boxes[idx, 3]),
                    ],
                    class_id=self.VEHICLE_CLASS_INDEX,
                    class_name="vehicle",
                    confidence=float(scores[idx]),
                ))

        drivable_mask = np.zeros((ctx.original_height, ctx.original_width), dtype=np.uint8)
        lane_mask = np.zeros((ctx.original_height, ctx.original_width), dtype=np.uint8)
        for output in seg_outputs:
            ch = output.shape[1]
            if ch == 2:
                drivable_mask = self._decode_seg(output, ctx, is_lane=False)
            elif ch == 1:
                lane_mask = self._decode_seg(output, ctx, is_lane=True)

        return [YOLOPv2Result(detections=detections, drivable_mask=drivable_mask, lane_mask=lane_mask)]

    def get_model_name(self) -> str:
        return "yolopv2"
