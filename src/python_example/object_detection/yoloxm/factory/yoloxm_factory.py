"""
Yolox Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOXPostprocessor
from common.visualizers import DetectionVisualizer


class YoloxmFactory(IDetectionFactory):
    """Factory for creating YOLOX-S components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOXPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yoloxm"
    
    def get_task_type(self) -> str:
        return "object_detection"
