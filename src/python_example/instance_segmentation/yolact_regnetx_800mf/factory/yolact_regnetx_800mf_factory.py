"""
yolact_regnetx_800mf Factory
"""

from common.base import IInstanceSegFactory
from common.processors import SimpleResizePreprocessor, YOLACTPostprocessor
from common.visualizers import InstanceSegVisualizer


class Yolact_regnetx_800mfFactory(IInstanceSegFactory):
    """Factory for creating YOLACT-RegNetX components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLACTPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return InstanceSegVisualizer()
    
    def get_model_name(self) -> str:
        return "yolact_regnetx_800mf"
    
    def get_task_type(self) -> str:
        return "instance_segmentation"

