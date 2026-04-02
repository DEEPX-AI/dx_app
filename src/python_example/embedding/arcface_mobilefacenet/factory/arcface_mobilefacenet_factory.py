"""
ArcFace Face Embedding Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import IEmbeddingFactory
from common.processors import SimpleResizePreprocessor, ArcFacePostprocessor
from common.visualizers import EmbeddingVisualizer


class Arcface_mobilefacenetFactory(IEmbeddingFactory):
    """Factory for creating ArcFace-MobileFaceNet components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return ArcFacePostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return EmbeddingVisualizer()
    
    def get_model_name(self) -> str:
        return "arcface_mobilefacenet"
    
    def get_task_type(self) -> str:
        return "embedding"

