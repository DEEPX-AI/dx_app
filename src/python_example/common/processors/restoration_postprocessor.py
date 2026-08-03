"""
Image Restoration Postprocessor

Handles image restoration models (DnCNN, etc.) that output a single-channel
or multi-channel image in NCHW format:
  - output[0]: [1, C, H, W]  restored/denoised image

DnCNN outputs the denoised image directly (values in [0, 1] range).
"""

import cv2
import numpy as np
from typing import List
from dataclasses import dataclass

from ..base import IPostprocessor, PreprocessContext


@dataclass
class RestorationResult:
    """Result from image restoration model."""
    output_image: np.ndarray  # restored image in HWC uint8 format


def restore_source_geometry(image: np.ndarray, ctx, model_in_w: int,
                            model_in_h: int) -> np.ndarray:
    """Map a model-space restoration output back onto the source geometry.

    Upscaling models (RealESRGAN) have a fixed square input, and the
    preprocessor stretches the frame into it, so the raw output carries the
    *model input* aspect ratio (192x192 -> 768x768) instead of the source one.
    Undoing that per-axis stretch gives ``original_size * upscale_factor``,
    which is what the tiled ESPCN path produces for the same task.

    Same-size restoration models (DnCNN denoising, upscale factor 1) are left
    untouched — there is no upscale to express and their callers expect the
    model-space image.
    """
    if image is None or ctx is None or image.size == 0:
        return image
    orig_w = int(getattr(ctx, "original_width", 0) or 0)
    orig_h = int(getattr(ctx, "original_height", 0) or 0)
    if orig_w <= 0 or orig_h <= 0 or model_in_w <= 0 or model_in_h <= 0:
        return image

    out_h, out_w = image.shape[:2]
    scale_x = out_w / float(model_in_w)
    scale_y = out_h / float(model_in_h)
    if scale_x <= 1.0 and scale_y <= 1.0:
        return image

    target_w = max(1, int(round(orig_w * scale_x)))
    target_h = max(1, int(round(orig_h * scale_y)))
    if (target_w, target_h) == (out_w, out_h):
        return image
    interp = cv2.INTER_AREA if (target_w < out_w and target_h < out_h) \
        else cv2.INTER_CUBIC
    return cv2.resize(image, (target_w, target_h), interpolation=interp)


class DnCNNPostprocessor(IPostprocessor):
    """
    Postprocessor for DnCNN denoising model.

    DnCNN outputs the denoised image directly in [1, 1, H, W] format.
    Values are clamped to [0, 1] and scaled to uint8.
    """

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext):
        """
        Process DnCNN output.

        Args:
            outputs: Model output shape [1, 1, H, W] — denoised image in [0, 1]
            ctx: PreprocessContext (contains preprocessed input info)

        Returns:
            RestorationResult with denoised image
        """
        raw = np.squeeze(outputs[0])  # [H, W] or [C, H, W] or [H, W, C] for color

        # CHW → HWC for color models, but skip if already HWC (NHWC models like UNet)
        if raw.ndim == 3:
            if raw.shape[2] <= 4 and raw.shape[0] > 4:
                pass  # Already HWC (e.g. unet output [256, 256, 3])
            else:
                raw = np.transpose(raw, (1, 2, 0))  # CHW → HWC

        # Model outputs denoised image directly — use as-is
        denoised = raw

        # Normalize to uint8: handle different output value ranges
        dmin, dmax = float(denoised.min()), float(denoised.max())
        if dmax - dmin < 1e-6:
            denoised_uint8 = np.zeros_like(denoised, dtype=np.uint8)
        elif dmin >= -0.1 and dmax <= 1.1:
            # [0, 1] range — standard DnCNN output
            denoised_uint8 = (np.clip(denoised, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif dmin >= -1.0 and dmax <= 256.0:
            # Roughly [0, 255] range
            denoised_uint8 = np.clip(denoised, 0.0, 255.0).astype(np.uint8)
        else:
            # Arbitrary range — min-max normalize to [0, 255]
            denoised_uint8 = ((denoised - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

        return [RestorationResult(output_image=denoised_uint8)]

    def get_model_name(self) -> str:
        return "dncnn"


class RealESRGANPostprocessor(IPostprocessor):
    """
    Postprocessor for RealESRGAN color super-resolution model.

    Output: [1, 3, H*scale, W*scale] NCHW float in [0, 1] range.
    Converts to HWC uint8 for display.
    """

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext):
        raw = outputs[0]
        raw = np.squeeze(raw)
        if raw.ndim == 3 and raw.shape[0] == 3:
            raw = np.transpose(raw, (1, 2, 0))  # CHW → HWC (RGB order)
        # Match C++ DnCNNPostprocessor: round (not floor) and emit BGR so that
        # downstream cv2.imwrite writes correct colors. The model output is RGB
        # (preprocessor converts BGR→RGB), so swap R/B back to BGR here.
        out_uint8 = np.clip(raw, 0.0, 1.0) * 255.0
        out_uint8 = np.round(out_uint8).astype(np.uint8)
        if out_uint8.ndim == 3 and out_uint8.shape[2] == 3:
            out_uint8 = out_uint8[:, :, ::-1].copy()  # RGB → BGR
        # The preprocessor stretched the frame into the square model input, so
        # undo that here — otherwise the result carries the model's 1:1 ratio.
        out_uint8 = restore_source_geometry(out_uint8, ctx, self.input_width,
                                            self.input_height)
        return [RestorationResult(output_image=out_uint8)]

    def get_model_name(self) -> str:
        return "realesrgan"
