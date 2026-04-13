"""
NanoDet Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, NanoDetPostprocessor
from common.visualizers import DetectionVisualizer


class NanodetplusmFactory(IDetectionFactory):
    """Factory for creating NanoDet-RepVGG components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        config = {**self.config, 'reg_max': 7}
        return NanoDetPostprocessor(input_width, input_height, config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "nanodetplusm"
    
    def get_task_type(self) -> str:
        return "object_detection"
