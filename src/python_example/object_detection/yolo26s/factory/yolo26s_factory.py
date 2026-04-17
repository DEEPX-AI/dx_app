"""
Yolo26s Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOv8Postprocessor
from common.visualizers import DetectionVisualizer


class Yolo26sFactory(IDetectionFactory):
    """Factory for creating YOLOv26s components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv8Postprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yolo26s"
    
    def get_task_type(self) -> str:
        return "object_detection"
