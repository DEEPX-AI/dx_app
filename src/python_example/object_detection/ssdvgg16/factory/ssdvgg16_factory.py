"""
SSDMV1 Factory
"""

from common.base import IDetectionFactory
from common.processors import SimpleResizePreprocessor, SSDPostprocessor
from common.visualizers import DetectionVisualizer


class Ssdvgg16Factory(IDetectionFactory):
    """Factory for creating SSD MobileNet V1 components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.config.setdefault('num_classes', 20)
        self.config.setdefault('has_background', True)
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return SSDPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer(label_set='voc')
    
    def get_model_name(self) -> str:
        return "ssdvgg16"
    
    def get_task_type(self) -> str:
        return "object_detection"
