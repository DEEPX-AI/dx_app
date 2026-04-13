"""
YOLOX-S PPU Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor
from common.processors.ppu_postprocessor import YOLOXPPUPostprocessor
from common.visualizers import DetectionVisualizer


class Yolox_s_2_ppuFactory(IDetectionFactory):
    """Factory for creating YOLOX-S-PPU components (anchor-free)."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOXPPUPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yolox_s_2_ppu"
    
    def get_task_type(self) -> str:
        return "object_detection"
