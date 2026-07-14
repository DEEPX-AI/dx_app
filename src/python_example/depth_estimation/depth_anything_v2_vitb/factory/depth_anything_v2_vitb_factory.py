"""
Depth_anything_v2_vitb Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IDepthEstimationFactory
from common.processors import DepthEstimationPostprocessor, SimpleResizePreprocessor
from common.visualizers import DepthVisualizer


class Depth_anything_v2_vitbFactory(IDepthEstimationFactory):
    """Factory for creating Depth_anything_v2_vitb components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # Depth Anything V2 expects ImageNet-normalized float input:
        #   (pixel/255 - mean) / std, mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] (RGB)
        # SimpleResizePreprocessor applies mean/std on the [0,255] range, so scale by 255.
        imagenet_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        imagenet_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        return SimpleResizePreprocessor(
            input_width, input_height, mean=imagenet_mean, std=imagenet_std)

    def create_postprocessor(self, input_width: int, input_height: int):
        return DepthEstimationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DepthVisualizer()

    def get_model_name(self) -> str:
        return "depth_anything_v2_vitb"

    def get_task_type(self) -> str:
        return "depth_estimation"
