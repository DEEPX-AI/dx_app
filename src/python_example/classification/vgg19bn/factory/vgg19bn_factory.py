"""
Efficientnet Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import IClassificationFactory
from common.processors import SimpleResizePreprocessor, ClassificationPostprocessor
from common.visualizers import ClassificationVisualizer


class Vgg19bnFactory(IClassificationFactory):
    """Factory for creating Vgg19Bn components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return ClassificationPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return ClassificationVisualizer()
    
    def get_model_name(self) -> str:
        return "vgg19bn"
    
    def get_task_type(self) -> str:
        return "classification"
