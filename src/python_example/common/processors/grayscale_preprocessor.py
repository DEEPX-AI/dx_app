"""
Grayscale Preprocessor

Converts BGR input to single-channel grayscale and resizes.
Used for models like DnCNN that expect [1, H, W, 1] uint8 input.
Also stores the normalized (float [0,1]) version in ctx for denoising.
"""

import cv2
import numpy as np
from typing import Tuple

from ..base import IPreprocessor, PreprocessContext
from ..utility.colorspace import bgr_to_y_limited

Y_MODE_GRAY = "gray"
Y_MODE_BT601_LIMITED = "bt601_limited"


class GrayscaleResizePreprocessor(IPreprocessor):
    """
    Preprocessor for grayscale models:
      1. Convert BGR → single channel
      2. Resize to target size
      3. Add channel dim → [H, W, 1]
      4. Store normalized version in ctx for post-processing usage

    Args:
        store_original: If True, saves the original BGR image in ctx.original_image.
                        Needed for ESPCN-style color restoration (YCrCb merge).
        y_mode: Which single-channel definition to produce.

            ``"gray"`` (default) — OpenCV ``COLOR_BGR2GRAY``, full range
            [0, 255]. This is the historical behaviour and stays the default so
            that every existing caller (DnCNN, SuperPoint, …) is unaffected.

            ``"bt601_limited"`` — MATLAB ``rgb2ycbcr`` Y, limited range
            [16, 235]. Required by models trained through the MATLAB SR
            pipeline: ESPCN's training set is built with
            ``bgr_to_ycbcr(..., only_use_y_channel=True)``, so a full-range Y
            lands outside the model's training domain. Opt in per model —
            never change the default.
    """

    def __init__(self, input_width: int, input_height: int, store_original: bool = False,
                 y_mode: str = Y_MODE_GRAY):
        if y_mode not in (Y_MODE_GRAY, Y_MODE_BT601_LIMITED):
            raise ValueError(
                f"y_mode must be '{Y_MODE_GRAY}' or '{Y_MODE_BT601_LIMITED}', "
                f"got {y_mode!r}")
        self.input_width = input_width
        self.input_height = input_height
        self.store_original = store_original
        self.y_mode = y_mode

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, PreprocessContext]:
        h, w = image.shape[:2]

        # Convert to the single channel this model was trained on
        if len(image.shape) == 3 and image.shape[2] == 3:
            if self.y_mode == Y_MODE_BT601_LIMITED:
                gray = bgr_to_y_limited(image)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            gray = image[:, :, 0]

        # Resize
        resized = cv2.resize(gray, (self.input_width, self.input_height))

        # Store normalized version in context (for DnCNN: denoised = input - residual)
        # The resize above is a *stretch* (no letterbox), so the context must
        # carry per-axis scale factors — consumers that map model-space
        # coordinates back to the source frame (SuperPoint keypoints via
        # scale_to_original) are a no-op without them.
        scale_x = float(self.input_width) / float(w) if w > 0 else 0.0
        scale_y = float(self.input_height) / float(h) if h > 0 else 0.0
        ctx = PreprocessContext(
            original_width=w,
            original_height=h,
            input_width=self.input_width,
            input_height=self.input_height,
            scale_x=scale_x,
            scale_y=scale_y,
            # Keep `scale` for backwards compatibility (uniform-scale readers).
            scale=min(scale_x, scale_y) if (scale_x > 0 and scale_y > 0) else 1.0,
            pad_x=0,
            pad_y=0,
        )
        ctx.normalized_input = resized.astype(np.float32) / 255.0
        if self.store_original:
            ctx.original_image = image  # BGR, for YCrCb color restoration

        # Model input: [H, W, 1] uint8
        input_tensor = resized[:, :, np.newaxis]

        return input_tensor, ctx

    def get_input_width(self) -> int:
        return self.input_width

    def get_input_height(self) -> int:
        return self.input_height
