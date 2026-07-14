"""
Rn50x16_openai Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IEmbeddingFactory
from common.processors import CLIPImagePostprocessor, SimpleResizePreprocessor
from common.visualizers import EmbeddingVisualizer


class Rn50x16_openaiFactory(IEmbeddingFactory):
    """Factory for creating Rn50x16_openai components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height, normalize_float=True)

    def create_postprocessor(self, input_width: int, input_height: int):
        return CLIPImagePostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return EmbeddingVisualizer()

    def get_model_name(self) -> str:
        return "rn50x16_openai"

    def get_task_type(self) -> str:
        return "embedding"