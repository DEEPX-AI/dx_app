"""
Scrfd Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import IFaceFactory
from common.processors import LetterboxPreprocessor, SCRFDPostprocessor
from common.visualizers import FaceVisualizer


class Scrfd2_5gFactory(IFaceFactory):
    """Factory for creating SCRFD-2_5G components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return SCRFDPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return FaceVisualizer()
    
    def get_model_name(self) -> str:
        return "scrfd2_5g"
    
    def get_task_type(self) -> str:
        return "face_detection"
    
    def get_num_keypoints(self) -> int:
        """SCRFD uses 5-point facial landmarks."""
        return 5
