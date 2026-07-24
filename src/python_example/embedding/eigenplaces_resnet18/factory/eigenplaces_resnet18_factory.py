"""
Eigenplaces_resnet18 Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IEmbeddingFactory
from common.processors import ArcFacePostprocessor, SimpleResizePreprocessor
from common.visualizers import EmbeddingVisualizer


class Eigenplaces_resnet18Factory(IEmbeddingFactory):
    """Factory for creating Eigenplaces_resnet18 components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return ArcFacePostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return EmbeddingVisualizer()

    def get_model_name(self) -> str:
        return "eigenplaces_resnet18"

    def get_task_type(self) -> str:
        return "embedding"
