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
        mean_target: If set, apply a per-channel GAIN that moves the resized image's
              own channel means to this value, before any mean/std step. Off by
              default (None) and it must stay that way: it is an illumination
              correction for inputs far outside a model's expected brightness, and
              applying it unconditionally changes every in-distribution result.

              MEASURED on both ppmatting variants over six subjects (foreground
              fraction at alpha>0.5, target 110): it rescues the one pathological
              input and degrades several that already work.

                subject (frame mean)     base    target=110
                person_a1 (209, white
                  shirt on white studio) 10.57%  13.97%   <- +3.4pp, and inside the
                                                             detector's person box
                                                             41.6% -> 54.2% with the
                                                             head's exact-zero share
                                                             92.4% -> 67.1%
                person_b  (182)          19.61%  18.92%   <- worse
                dog       (107)          27.18%  27.23%   <- unchanged
                face_a1   (130)          83.17%  80.45%   <- worse
                distinctions/dog         25.30%  20.10%   <- clearly worse
                distinctions/face         3.11%   1.01%   <- clearly worse

              So it is gated rather than unconditional: pair it with
              `mean_target_above` so only an over-bright input is corrected. A contrast
              stretch was tried first and is strictly worse -- at x1.5 the head band
              goes to 100% exact zeros -- so it is the channel MEAN that matters here,
              not the spread.
        mean_target_above: Only apply `mean_target` when the resized image's overall
              mean EXCEEDS this value; otherwise the frame passes through untouched.
              This is what makes the correction safe to enable by default for a family:
              of the six measured subjects only person_a1 sits above 195 (frame mean
              209), and the five that the ungated gain degraded (84-182) are left
              exactly as they were.
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
                 store_normalized: bool = False,
                 mean_target: Optional[float] = None,
                 mean_target_above: Optional[float] = None):
        self._input_width = input_width
        self._input_height = input_height
        self._normalize_float = normalize_float
        self._nhwc = nhwc
        self._mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self._std = np.array(std, dtype=np.float32) if std is not None else None
        self._bgr = bgr
        self._store_original = store_original
        self._store_normalized = store_normalized
        self._mean_target = float(mean_target) if mean_target is not None else None
        self._mean_target_above = (float(mean_target_above)
                                   if mean_target_above is not None else None)
    
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

        # Optional illumination gain (see `mean_target` above). Computed on the RESIZED
        # image, which is what the model actually sees, and applied per channel so a
        # colour cast is corrected too rather than only overall brightness.
        if self._mean_target is not None:
            flat = resized.reshape(-1, resized.shape[2])
            channel_mean = flat.mean(axis=0)
            if (self._mean_target_above is None
                    or float(channel_mean.mean()) > self._mean_target_above):
                gain = self._mean_target / np.maximum(channel_mean, 1e-6)
                # np.rint, not a bare astype: astype TRUNCATES toward zero while
                # OpenCV's saturate_cast (which the C++ preprocessor's convertTo uses)
                # ROUNDS. Measured, the two disagree by 1 on half of all input values,
                # which flips matte decisions along every edge and showed up as a
                # cross-tree foreground difference of 14.26% vs 14.34%.
                resized = np.clip(np.rint(resized.astype(np.float32) * gain),
                                  0.0, 255.0).astype(np.uint8)
        
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
