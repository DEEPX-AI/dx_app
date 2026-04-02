"""
ESPCN x2 Super-Resolution Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import IRestorationFactory
from common.processors import GrayscaleResizePreprocessor, ESPCNPostprocessor
from common.visualizers import SuperResolutionVisualizer


class Espcn_x4Factory(IRestorationFactory):
    """Factory for creating ESPCN-x4 components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return GrayscaleResizePreprocessor(input_width, input_height, store_original=True)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return ESPCNPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return SuperResolutionVisualizer()
    
    def get_model_name(self) -> str:
        return "espcn_x4"
    
    def get_task_type(self) -> str:
        return "super_resolution"

