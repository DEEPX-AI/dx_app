"""
Generic fast semantic segmentation postprocessor.

This postprocessor is optimized for reduced-resolution class logits. It computes
argmax at the model output resolution first, then resizes the single-channel
class map to the original image size. This avoids upsampling every class logit
channel to full resolution.
"""

from typing import List, Optional

import cv2
import numpy as np

from ..base import IPostprocessor, PreprocessContext, SegmentationResult


class FastSegmentationPostprocessor(IPostprocessor):
    """Fast semantic segmentation postprocessor for low-resolution logits."""

    def __init__(
        self,
        input_width: int = 2048,
        input_height: int = 1024,
        config: Optional[dict] = None,
        *,
        num_classes: Optional[int] = None,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.num_classes = num_classes or self.config.get("num_classes", 19)

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[SegmentationResult]:
        if not outputs:
            return []

        class_map = self._to_class_map(outputs[0])
        if class_map is None:
            return []

        class_map = self._resize_to_original(class_map, ctx)
        return [self._make_result(class_map)]

    def _to_class_map(self, output: np.ndarray) -> Optional[np.ndarray]:
        """Convert NCHW/NHWC logits or pre-argmaxed outputs to a class map."""
        if output.ndim == 4:
            logits = output[0]
            if logits.shape[0] == 1 or np.issubdtype(logits.dtype, np.integer):
                return logits.squeeze().astype(np.int32)
            if (logits.ndim == 3 and logits.shape[2] < logits.shape[0]
                    and logits.shape[2] < logits.shape[1]):
                logits = np.transpose(logits, (2, 0, 1))
            return np.argmax(logits, axis=0).astype(np.int32)

        if output.ndim == 3:
            if output.shape[0] == 1 or np.issubdtype(output.dtype, np.integer):
                return output.squeeze().astype(np.int32)
            if output.shape[2] < output.shape[0] and output.shape[2] < output.shape[1]:
                output = np.transpose(output, (2, 0, 1))
            return np.argmax(output, axis=0).astype(np.int32)

        if output.ndim == 2:
            return output.astype(np.int32)

        return None

    def _resize_to_original(self, class_map: np.ndarray, ctx: PreprocessContext) -> np.ndarray:
        """Resize the single-channel class map to original image size."""
        if ctx is None or ctx.original_width <= 0 or ctx.original_height <= 0:
            return class_map

        if ctx.pad_x == 0 and ctx.pad_y == 0:
            if (class_map.shape[1] == ctx.original_width
                    and class_map.shape[0] == ctx.original_height):
                return class_map
            return cv2.resize(
                class_map.astype(np.float32),
                (ctx.original_width, ctx.original_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        gain = max(ctx.scale, 1e-6)
        unpad_h = int(round(ctx.original_height * gain))
        unpad_w = int(round(ctx.original_width * gain))
        ratio_h = class_map.shape[0] / float(max(ctx.input_height or self.input_height, 1))
        ratio_w = class_map.shape[1] / float(max(ctx.input_width or self.input_width, 1))
        top = int(round(int(ctx.pad_y) * ratio_h))
        left = int(round(int(ctx.pad_x) * ratio_w))
        crop_h = int(round(unpad_h * ratio_h))
        crop_w = int(round(unpad_w * ratio_w))
        bottom = min(top + crop_h, class_map.shape[0])
        right = min(left + crop_w, class_map.shape[1])
        cropped = class_map[max(top, 0):bottom, max(left, 0):right]
        if cropped.size == 0:
            return class_map
        return cv2.resize(
            cropped.astype(np.float32),
            (ctx.original_width, ctx.original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

    @staticmethod
    def _make_result(class_map: np.ndarray) -> SegmentationResult:
        height, width = class_map.shape
        unique_classes = np.unique(class_map).tolist()
        return SegmentationResult(
            mask=class_map,
            width=width,
            height=height,
            class_ids=unique_classes,
            class_names=[],
        )

    def get_model_name(self) -> str:
        return "fast_segmentation"
