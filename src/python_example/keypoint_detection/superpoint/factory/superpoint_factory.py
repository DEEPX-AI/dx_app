"""
SuperPoint Keypoint Detection Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base.i_factory import _FactoryConfigMixin
from common.processors import GrayscaleResizePreprocessor, SuperPointPostprocessor
from common.visualizers import SuperPointVisualizer


class SuperpointFactory(_FactoryConfigMixin):
    """Factory for creating SuperPoint keypoint detection components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return GrayscaleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return SuperPointPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return SuperPointVisualizer()

    def get_model_name(self) -> str:
        return "superpoint"

    def get_task_type(self) -> str:
        return "keypoint_detection"
