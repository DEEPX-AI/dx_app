"""
Yolo26DepthN Factory — YOLO26-Depth-N monocular depth estimation (DX-M1 NPU).

Source model: `yolo26-depth-n_768x768.dxnn` — a single dense depth
output `depth` [1,1,768,768] float32.

Preprocessing: the compiled model consumes a **uint8 NHWC** tensor
(`images` [1,768,768,3]) — the `pixel/255` normalization is folded into the
compiled model, so the app must NOT normalize. `SimpleResizePreprocessor`
without mean/std returns the resized RGB uint8 image, which is exactly the
buffer the runner hands to the engine.
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IDepthEstimationFactory
from common.processors import DepthEstimationPostprocessor, SimpleResizePreprocessor
from common.visualizers import DepthVisualizer


class Yolo26DepthNFactory(IDepthEstimationFactory):
    """Factory for YOLO26-Depth-N components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # uint8 RGB NHWC — no mean/std: normalization is baked into the model.
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return DepthEstimationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DepthVisualizer()

    def get_model_name(self) -> str:
        return "yolo26_depth_n"

    def get_task_type(self) -> str:
        return "depth_estimation"
