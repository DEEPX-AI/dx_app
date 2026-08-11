"""
Simple Resize Preprocessor

Direct resize without aspect ratio preservation.
Used by EfficientNet, DeepLabV3, etc.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional

from ..base import IPreprocessor, PreprocessContext


class SimpleResizePreprocessor(IPreprocessor):
    """
    Simple resize preprocessor.
    
    Directly resizes image to target size without padding.
    Suitable for classification and semantic segmentation models.
    
    Args:
        input_width: Model input width
        input_height: Model input height
        normalize_float: If True, return float32 NCHW [0,1] tensor
                         (for models that require normalized float input)
        nhwc: If True with normalize_float, return float32 HWC [0,1]
              tensor for NHWC models
        mean: Optional per-channel mean to subtract (after resize, before output).
              If provided, output is float32 regardless of normalize_float.
              Applied in the channel order of the output (see bgr parameter).
        std:  Optional per-channel std to divide by (after mean subtraction).
              Defaults to [1.0, 1.0, 1.0] (no division).
        bgr:  If True (default False), keep BGR channel order (no RGB conversion).
              Use for models trained on BGR images (e.g. RetinaFace, many OpenCV models).
        store_original: If True, keep a copy of the input BGR frame in
              ``ctx.original_image`` (needed only for color restoration, e.g. ESPCN).
              Off by default: the copy costs ~0.5-1.1 ms/frame at 1080p and no
              model that uses this preprocessor reads it.
        store_normalized: If True, also populate ``ctx.normalized_input``
              (float32 CHW [0,1]) on the uint8/mean-std paths. Off by default:
              building it costs ~1.7-3.1 ms/frame at 768x768 and only the
              Zero-DCE enhancement family reads it — that family sets
              ``normalize_float=True``, which populates it regardless.
    """

    def __init__(self, input_width: int, input_height: int,
                 normalize_float: bool = False, nhwc: bool = False,
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 bgr: bool = False,
                 store_original: bool = False,
                 store_normalized: bool = False):
        self._input_width = input_width
        self._input_height = input_height
        self._normalize_float = normalize_float
        self._nhwc = nhwc
        self._mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self._std = np.array(std, dtype=np.float32) if std is not None else None
        self._bgr = bgr
        self._store_original = store_original
        self._store_normalized = store_normalized
    
    def process(self, input_image: np.ndarray) -> Tuple[np.ndarray, PreprocessContext]:
        """
        Preprocess image with simple resize.
        
        Args:
            input_image: Input image (BGR, HWC format)
            
        Returns:
            Tuple of (preprocessed_image, context)
        """
        ctx = PreprocessContext()
        ctx.original_height = input_image.shape[0]
        ctx.original_width = input_image.shape[1]
        ctx.input_width = self._input_width
        ctx.input_height = self._input_height
        # For simple direct resize we have independent scale factors per axis
        # because the aspect ratio may change (stretch). Use scale_x/scale_y
        # for inverse mapping back to original image coordinates.
        ctx.scale_x = float(self._input_width) / float(input_image.shape[1])
        ctx.scale_y = float(self._input_height) / float(input_image.shape[0])
        # Keep `scale` for backwards compatibility (set to geometric mean).
        ctx.scale = min(ctx.scale_x, ctx.scale_y)
        ctx.pad_x = 0
        ctx.pad_y = 0
        if self._store_original:
            ctx.original_image = input_image.copy()  # BGR original for color restoration

        # Color conversion
        if self._bgr:
            img = input_image  # Keep BGR as-is
        else:
            img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Direct resize
        resized = cv2.resize(img, (self._input_width, self._input_height), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Store normalized input (RGB float32 [0,1] CHW) only for models that
        # actually read it — building it is ~1.7-3.1 ms/frame at 768x768 and
        # the input stage is what caps async throughput.
        resized_float = None
        if self._normalize_float or self._store_normalized:
            resized_float = resized.astype(np.float32) / 255.0
            ctx.normalized_input = np.transpose(resized_float, (2, 0, 1))  # HWC → CHW

        # Mean/std normalization path (overrides normalize_float)
        if self._mean is not None:
            out = resized.astype(np.float32) - self._mean
            if self._std is not None:
                out = out / self._std
            if not self._nhwc:
                out = np.transpose(out, (2, 0, 1))  # HWC → CHW
            return out, ctx

        if self._normalize_float:
            if self._nhwc:
                return resized_float, ctx
            return ctx.normalized_input, ctx
        return resized, ctx
    
    def get_input_width(self) -> int:
        return self._input_width
    
    def get_input_height(self) -> int:
        return self._input_height
