"""
Image Restoration Postprocessor - DX-APP v3.0.0

Handles image restoration models (DnCNN, etc.) that output a single-channel
or multi-channel image in NCHW format:
  - output[0]: [1, C, H, W]  restored/denoised image

DnCNN outputs the denoised image directly (values in [0, 1] range).
"""

import numpy as np
from typing import List
from dataclasses import dataclass

from ..base import IPostprocessor, PreprocessContext


@dataclass
class RestorationResult:
    """Result from image restoration model."""
    output_image: np.ndarray  # restored image in HWC uint8 format


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
        raw = np.squeeze(outputs[0])  # [H, W] or [C, H, W] for color

        # CHW → HWC for color models (e.g. dncnn_color_blind outputs [3, H, W])
        if raw.ndim == 3:
            raw = np.transpose(raw, (1, 2, 0))

        # Model outputs denoised image directly — use as-is
        denoised = raw

        # Clip and scale to uint8
        denoised = np.clip(denoised, 0.0, 1.0)
        denoised_uint8 = (denoised * 255.0).astype(np.uint8)

        return [RestorationResult(output_image=denoised_uint8)]

    def get_model_name(self) -> str:
        return "dncnn"
