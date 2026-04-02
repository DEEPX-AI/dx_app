"""
Deeplabv3 Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import ISegmentationFactory
from common.processors import SimpleResizePreprocessor, SemanticSegmentationPostprocessor
from common.visualizers import SemanticSegmentationVisualizer


class Deeplabv3Factory(ISegmentationFactory):
    """Factory for creating DeepLabV3Plus-MobileNet components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return SemanticSegmentationPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return SemanticSegmentationVisualizer()
    
    def get_model_name(self) -> str:
        return "deeplabv3plusmobilenet"
    
    def get_task_type(self) -> str:
        return "semantic_segmentation"
