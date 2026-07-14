"""
YOLOPv2 Panoptic Driving Perception Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base.i_factory import _FactoryConfigMixin
from common.processors import LetterboxPreprocessor, YOLOPv2Postprocessor
from common.visualizers import YOLOPv2Visualizer


class Yolopv2Factory(_FactoryConfigMixin):
    """Factory for creating YOLOPv2 panoptic driving perception components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOPv2Postprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return YOLOPv2Visualizer()

    def get_model_name(self) -> str:
        return "yolopv2"

    def get_task_type(self) -> str:
        return "panoptic_driving_perception"
