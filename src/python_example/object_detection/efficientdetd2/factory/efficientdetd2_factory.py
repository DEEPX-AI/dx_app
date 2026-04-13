"""
EfficientDet Factory
"""

from common.base import IDetectionFactory
from common.processors import SimpleResizePreprocessor, EfficientDetPostprocessor
from common.visualizers import DetectionVisualizer


class Efficientdetd2Factory(IDetectionFactory):
    """Factory for creating EfficientDet-D2 components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return EfficientDetPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "efficientdetd2"
    
    def get_task_type(self) -> str:
        return "object_detection"
