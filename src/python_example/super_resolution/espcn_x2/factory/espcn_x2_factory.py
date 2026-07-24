"""
Espcn_x2 Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IRestorationFactory
from common.processors import ESPCNPostprocessor, GrayscaleResizePreprocessor
from common.visualizers import SuperResolutionVisualizer


class Espcn_x2Factory(IRestorationFactory):
    """Factory for creating Espcn_x2 components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return GrayscaleResizePreprocessor(input_width, input_height, store_original=True)

    def create_postprocessor(self, input_width: int, input_height: int):
        return ESPCNPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return SuperResolutionVisualizer()

    def get_model_name(self) -> str:
        return "espcn_x2"

    def get_task_type(self) -> str:
        return "super_resolution"
