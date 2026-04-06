"""
Zero-DCE Low-Light Enhancement Factory
"""

from common.base import IRestorationFactory
from common.processors import SimpleResizePreprocessor, ZeroDCEPostprocessor
from common.visualizers import EnhancementVisualizer


class Zero_dceFactory(IRestorationFactory):
    """Factory for creating Zero-DCE components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height, normalize_float=True)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return ZeroDCEPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return EnhancementVisualizer()
    
    def get_model_name(self) -> str:
        return "zero_dce"
    
    def get_task_type(self) -> str:
        return "image_enhancement"

