"""
Yolov5 Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOv5Postprocessor
from common.visualizers import DetectionVisualizer


class Yolov5s_c3tr_640Factory(IDetectionFactory):
    """Factory for creating YOLOv5n components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv5Postprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yolov5s_c3tr_640"
    
    def get_task_type(self) -> str:
        return "object_detection"
