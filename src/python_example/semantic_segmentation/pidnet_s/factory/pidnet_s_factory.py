"""
Pidnet_s Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import ISegmentationFactory
from common.processors import SemanticSegmentationPostprocessor, SimpleResizePreprocessor
from common.visualizers import SemanticSegmentationVisualizer


class Pidnet_sFactory(ISegmentationFactory):
    """Factory for creating Pidnet_s components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return SemanticSegmentationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return SemanticSegmentationVisualizer()

    def get_model_name(self) -> str:
        return "pidnet_s"

    def get_task_type(self) -> str:
        return "semantic_segmentation"
