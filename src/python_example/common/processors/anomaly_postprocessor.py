# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Anomaly-detection feature response -- deliberately not an anomaly score.

The four anomaly models in the DX Model Zoo 2_5_0 additions are single networks out of
multi-network methods:

* **EfficientAD-M** ships as three separate ``.dxnn`` files. Its anomaly map is
  ``0.5*norm(mean((teacher - student[:384])**2)) + 0.5*norm(mean((ae - student[384:])**2))``
  -- it needs all three at once.
* **PatchCore** needs a memory bank of training-set features to turn its embedding
  into a distance.

dx_app has no in-tree multi-model application pattern (the multi-model cascade work is
staged under ``dx-agent-dev/``, not ``src/``), so each variant gets a single-model
feature-extraction example: the per-pixel magnitude of the feature response, rendered
as a heatmap. That is a real, working application -- it just is not the published
EfficientAD or PatchCore metric, and ``AnomalyResult.score`` documents that it is a
relative severity only.

None of the four ``.dxnn`` files is published (403), so the decode is written to the
shape these architectures emit -- a ``(1, C, H, W)`` feature map, PDN features for
EfficientAD and WideResNet layer features for PatchCore -- and raises with the actual
shapes for anything else.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..base import AnomalyResult, IPostprocessor, PreprocessContext


def _shapes(outputs: Sequence[np.ndarray]) -> str:
    return ", ".join("(" + ", ".join(str(d) for d in o.shape) + ")" for o in outputs)


class AnomalyFeaturePostprocessor(IPostprocessor):
    """Feature map -> normalised per-pixel response at the original resolution."""

    def __init__(self, input_width: int = 256, input_height: int = 256,
                 config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        # The percentile reported as `score`. 99 rather than max, because a single hot
        # pixel of quantisation noise would otherwise dominate every frame.
        self.score_percentile = float(self.config.get("score_percentile", 99.0))

    def _find_features(self, outputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, int]:
        """``(features_CHW, channels)``, accepting either channel order.

        The spatial grid of these models is square -- 64x64 at a 256x256 input, 28x28
        at 224x224 -- which is what tells the two layouts apart::

            (1, C, H, W)  the last two axes are the equal pair
            (1, H, W, C)  the FIRST two are

        Channels-first wins when all three are equal (a 64-channel 64x64 map), because
        that is the layout DXRT emits for these networks; the alternative would be to
        guess, and a wrong guess here transposes the heatmap without erroring.
        """
        for out in outputs:
            if out.ndim != 4 or out.shape[0] != 1:
                continue
            _, a, b, c = out.shape
            if a == b and b != c:
                return np.transpose(out[0], (2, 0, 1)), int(c)   # NHWC
            return out[0], int(a)                                # NCHW
        raise ValueError(
            "[DXAPP] [ERROR] AnomalyFeaturePostprocessor - no (1, C, H, W) feature "
            f"map found in the model outputs.\n  Got: {_shapes(outputs)}\n"
            "  EfficientAD's teacher/student/autoencoder and PatchCore's backbone all "
            "emit a spatial feature map; a flat embedding cannot be turned into a "
            "per-pixel response."
        )

    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[AnomalyResult]:
        features, channels = self._find_features(outputs)
        features = features.astype(np.float32)

        # Per-pixel L2 norm across channels: the magnitude of the feature response,
        # which is what every distance-based anomaly method reduces to per location.
        response = np.sqrt(np.sum(features * features, axis=0))

        low, high = float(response.min()), float(response.max())
        span = high - low
        # A featureless map (all channels constant) has no contrast to normalise.
        # Reporting it as uniformly hot would be the opposite of the truth.
        normalised = ((response - low) / span) if span > 1e-12 \
            else np.zeros_like(response)

        heatmap = cv2.resize(normalised, (ctx.original_width, ctx.original_height),
                             interpolation=cv2.INTER_LINEAR)
        heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

        return [AnomalyResult(
            heatmap=heatmap,
            score=float(np.percentile(heatmap, self.score_percentile)),
            channels=channels,
        )]

    def get_model_name(self) -> str:
        return "anomaly_feature"
