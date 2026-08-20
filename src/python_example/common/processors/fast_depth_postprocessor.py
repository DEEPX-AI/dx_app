"""
Generic fast depth estimation postprocessor.

Two differences from :class:`DepthEstimationPostprocessor`:

* min/max + rescale + uint8 cast run as **one** SIMD pass
  (``cv2.minMaxLoc`` + ``cv2.convertScaleAbs``) instead of three temporary
  float arrays over the full depth map.
* the colormap is produced **at model resolution** and NOT resized. The depth
  visualizer resizes to the frame size unconditionally, so the standard path
  resizes the 3-channel colormap twice; the fast path resizes once.

Measured 2.2-3.0x on the postprocess step and 1.2-2.6x end-to-end
(postprocess + visualize) for a 768x768 model.

Like every fast variant this is a path-B approximation: dropping one bilinear
resize shifts pixels by up to ~3/255 versus the standard path.
"""

from typing import List, Optional

import cv2
import numpy as np

from ..base import IPostprocessor, PreprocessContext
from .depth_postprocessor import DepthResult


class FastDepthEstimationPostprocessor(IPostprocessor):
    """Fast monocular-depth postprocessor (single-pass normalize, no resize)."""

    def __init__(self, input_width: int, input_height: int,
                 config: Optional[dict] = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.colormap = self.config.get('colormap', cv2.COLORMAP_MAGMA)

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[DepthResult]:
        """
        Process depth estimation output.

        Args:
            outputs: [depth_tensor]  shape [1, 1, H, W]
            ctx: PreprocessContext (unused — the visualizer owns the resize)

        Returns:
            DepthResult with the raw depth map and a model-resolution colormap
        """
        if not outputs:
            return []

        depth = np.squeeze(outputs[0])
        # cv2 needs a contiguous single-channel float32 buffer.
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        if not depth.flags['C_CONTIGUOUS']:
            depth = np.ascontiguousarray(depth)

        d_min, d_max, _, _ = cv2.minMaxLoc(depth)
        depth_range = d_max - d_min
        if depth_range > 1e-6:
            # (d - d_min) / range * 255 in a single pass. Values are already
            # >= 0 after the shift, so convertScaleAbs' abs() is a no-op.
            depth_norm = cv2.convertScaleAbs(
                depth, alpha=255.0 / depth_range,
                beta=-255.0 * d_min / depth_range)
        else:
            depth_norm = np.zeros(depth.shape, dtype=np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, self.colormap)

        return [DepthResult(
            depth_map=depth,
            depth_colormap=depth_color,
        )]

    def get_model_name(self) -> str:
        return "fast_depth"
