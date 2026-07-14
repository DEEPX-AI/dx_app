"""
Fastvit_sa24 Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IClassificationFactory
from common.processors import ClassificationPostprocessor, SimpleResizePreprocessor
from common.visualizers import ClassificationVisualizer


class Fastvit_sa24Factory(IClassificationFactory):
    """Factory for creating Fastvit_sa24 components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return ClassificationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return ClassificationVisualizer()

    def get_model_name(self) -> str:
        return "fastvit_sa24"

    def get_task_type(self) -> str:
        return "classification"
