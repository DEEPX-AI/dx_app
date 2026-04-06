"""
DnCNN Factory
"""

from common.base import IRestorationFactory
from common.processors import GrayscaleResizePreprocessor, DnCNNPostprocessor
from common.visualizers import RestorationVisualizer


class Dncnn_50Factory(IRestorationFactory):
    """Factory for creating DnCNN denoising components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return GrayscaleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return DnCNNPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return RestorationVisualizer()
    
    def get_model_name(self) -> str:
        return "dncnn_50"
    
    def get_task_type(self) -> str:
        return "image_denoising"
