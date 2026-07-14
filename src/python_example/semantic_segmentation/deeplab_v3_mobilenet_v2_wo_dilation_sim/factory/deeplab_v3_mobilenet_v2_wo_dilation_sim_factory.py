"""
Deeplab_v3_mobilenet_v2_wo_dilation_sim Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import ISegmentationFactory
from common.processors import SemanticSegmentationPostprocessor, SimpleResizePreprocessor
from common.visualizers import SemanticSegmentationVisualizer


class Deeplab_v3_mobilenet_v2_wo_dilation_simFactory(ISegmentationFactory):
    """Factory for creating Deeplab_v3_mobilenet_v2_wo_dilation_sim components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return SemanticSegmentationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return SemanticSegmentationVisualizer()

    def get_model_name(self) -> str:
        return "deeplab_v3_mobilenet_v2_wo_dilation_sim"

    def get_task_type(self) -> str:
        return "semantic_segmentation"