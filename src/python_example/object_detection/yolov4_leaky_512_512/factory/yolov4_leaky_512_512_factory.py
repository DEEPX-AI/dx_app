"""
YOLOv4 Leaky Factory
"""

from common.base import IDetectionFactory
from common.processors import SimpleResizePreprocessor, YOLOv5Postprocessor
from common.visualizers import DetectionVisualizer


class Yolov4_leaky_512_512Factory(IDetectionFactory):
    """Factory for creating YOLOv4-Leaky components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv5Postprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yolov4_leaky_512_512"
    
    def get_task_type(self) -> str:
        return "object_detection"
