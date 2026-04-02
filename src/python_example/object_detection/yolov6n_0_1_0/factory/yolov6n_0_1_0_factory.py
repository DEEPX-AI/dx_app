"""
Yolov5 Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOv5Postprocessor
from common.visualizers import DetectionVisualizer


class Yolov6n_0_1_0Factory(IDetectionFactory):
    """Factory for creating YOLOv6n_0_1_0 components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv5Postprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yolov6n_0_1_0"
    
    def get_task_type(self) -> str:
        return "object_detection"
