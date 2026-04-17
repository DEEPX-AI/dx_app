"""
3DDFA v2 Face Alignment Factory
"""

from common.base import IFaceAlignmentFactory
from common.processors import SimpleResizePreprocessor, TDDFAPostprocessor
from common.visualizers import FaceAlignmentVisualizer


class N3ddfa_v2_mobilnet0_5_120x120Factory(IFaceAlignmentFactory):
    """Factory for creating 3DDFA v2 MobileNet 0.5 face alignment components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return TDDFAPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return FaceAlignmentVisualizer()
    
    def get_model_name(self) -> str:
        return "3ddfa_v2_mobilnet0_5_120x120"
    
    def get_task_type(self) -> str:
        return "face_alignment"

