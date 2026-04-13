"""
Yolov8seg Factory
"""

from common.base import IInstanceSegFactory
from common.processors import LetterboxPreprocessor, YOLOv8InstanceSegPostprocessor
from common.visualizers import InstanceSegVisualizer


class Yolo26l_segFactory(IInstanceSegFactory):
    """Factory for creating YOLOv8n-SEG components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv8InstanceSegPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return InstanceSegVisualizer()
    
    def get_model_name(self) -> str:
        return "yolo26l_seg"
    
    def get_task_type(self) -> str:
        return "instance_segmentation"
