"""
Yolov26OBB Factory
"""

from common.base import IOBBFactory
from common.processors import LetterboxPreprocessor, OBBPostprocessor
from common.visualizers import OBBVisualizer


class Yolo26l_obbFactory(IOBBFactory):
    """Factory for creating YOLOv26l-OBB components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return OBBPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return OBBVisualizer()
    
    def get_model_name(self) -> str:
        return "yolo26l_obb"
    
    def get_task_type(self) -> str:
        return "obb_detection"
