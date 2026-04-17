"""
FastDepth Factory
"""

from common.base import IDepthEstimationFactory
from common.processors import SimpleResizePreprocessor, DepthEstimationPostprocessor
from common.visualizers import DepthVisualizer


class Scdepthv3Factory(IDepthEstimationFactory):
    """Factory for creating FastDepth depth estimation components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return DepthEstimationPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DepthVisualizer()
    
    def get_model_name(self) -> str:
        return "scdepthv3"
    
    def get_task_type(self) -> str:
        return "depth_estimation"
