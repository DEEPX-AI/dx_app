# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""PP-Matting postprocessor: a continuous alpha matte.

PaddleSeg's matting models emit one channel -- the foreground opacity of every pixel
-- rather than a class map, so this is written to that output shape and refuses
anything else. MEASURED on both published .dxnn (DX Model Zoo 2_5_0): input uint8
[1,512,512,3], output float32 [1,1,512,512] already in [0,1].

Matting is trimap-free here, so it needs a subject separable from its background.
Measured foreground coverage over six sample subjects: the beach portrait 19.6% and
the dog 27.2% matte cleanly, while a white shirt and skin against a white studio
backdrop collapses to 10.6% (composition) / 1.6% (distinctions) with the face and
hands punched out. That is the model's limit, not a preprocessing bug -- the examples
default to subjects each variant resolves.

The result carries both forms on purpose:

``mask``  the alpha thresholded to a 0/1 class map. ``SemanticSegmentationVisualizer``
          reads ``mask`` as class IDs, so handing it a continuous alpha would render
          the matte through a 19-colour class palette.
``alpha`` the matte itself, in [0,1] at the original resolution, for anything that
          wants to composite with it rather than look at it.

A multi-class segmentation head is refused rather than sliced: taking channel 0 of a
19-class Cityscapes output would produce a believable image and be wrong.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..base import IPostprocessor, PreprocessContext, SegmentationResult


def _shapes(outputs: Sequence[np.ndarray]) -> str:
    return ", ".join("(" + ", ".join(str(d) for d in o.shape) + ")" for o in outputs)


class PPMattingPostprocessor(IPostprocessor):
    """Single-channel alpha matte -> (class map, alpha) at the original resolution."""

    def __init__(self, input_width: int = 512, input_height: int = 512,
                 config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.alpha_threshold = float(self.config.get("alpha_threshold", 0.5))

    def _find_alpha(self, outputs: Sequence[np.ndarray]) -> np.ndarray:
        """The HxW alpha map, whatever single-channel layout it arrives in."""
        for out in outputs:
            if out.ndim == 4 and out.shape[0] == 1:
                if out.shape[1] == 1:
                    return out[0, 0]                     # NCHW
                if out.shape[3] == 1:
                    return out[0, :, :, 0]               # NHWC
            elif out.ndim == 3 and out.shape[0] == 1:
                return out[0]                            # N H W
            elif out.ndim == 2:
                return out
        raise ValueError(
            "[DXAPP] [ERROR] PPMattingPostprocessor - no single-channel alpha map "
            f"found in the model outputs.\n  Got: {_shapes(outputs)}\n"
            "  A matting model emits one channel: (1, 1, H, W), (1, H, W, 1) or "
            "(1, H, W). A multi-class segmentation head is refused here on purpose -- "
            "silently taking its channel 0 would yield a believable but wrong matte. "
            "Use SemanticSegmentationPostprocessor for a class map."
        )

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[SegmentationResult]:
        alpha = self._find_alpha(outputs).astype(np.float32)
        # A model compiled without the final sigmoid emits logits; squash those, and
        # leave a map that is already a probability alone.
        if alpha.min() < 0.0 or alpha.max() > 1.0:
            alpha = 1.0 / (1.0 + np.exp(-alpha))

        if (alpha.shape[1], alpha.shape[0]) != (ctx.original_width,
                                                ctx.original_height):
            alpha = cv2.resize(alpha, (ctx.original_width, ctx.original_height),
                               interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0)

        return [SegmentationResult(
            mask=(alpha > self.alpha_threshold).astype(np.uint8),
            alpha=alpha,
            width=ctx.original_width,
            height=ctx.original_height,
            class_ids=[0, 1],
            class_names=["background", "foreground"],
        )]

    def get_model_name(self) -> str:
        return "ppmatting"
