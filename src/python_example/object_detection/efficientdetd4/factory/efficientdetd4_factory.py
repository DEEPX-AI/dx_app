"""
Yolov8 Factory
"""

from common.base import IDetectionFactory
from common.processors import SimpleResizePreprocessor, EfficientDetPostprocessor
from common.visualizers import DetectionVisualizer


class Efficientdetd4Factory(IDetectionFactory):
    """Factory for creating YOLOv8n components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return EfficientDetPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "efficientdetd4"
    
    def get_task_type(self) -> str:
        return "object_detection"
