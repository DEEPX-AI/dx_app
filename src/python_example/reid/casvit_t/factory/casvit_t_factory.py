"""
Person Re-ID Factory
"""

from common.base import IEmbeddingFactory
from common.processors import SimpleResizePreprocessor, ArcFacePostprocessor
from common.visualizers import EmbeddingVisualizer


class Casvit_tFactory(IEmbeddingFactory):
    """Factory for creating CasViT Person Re-ID components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return ArcFacePostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return EmbeddingVisualizer()
    
    def get_model_name(self) -> str:
        return "casvit_t"
    
    def get_task_type(self) -> str:
        return "reid"

