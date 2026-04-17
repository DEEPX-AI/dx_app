"""
ULFG Face Detector Factory
"""

from common.base import IFaceFactory
from common.processors import SimpleResizePreprocessor, ULFGPostprocessor
from common.visualizers import FaceVisualizer


class Ulfgfd_slim_320Factory(IFaceFactory):
    """Factory for creating ULFG-Slim face detection components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return ULFGPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return FaceVisualizer()
    
    def get_model_name(self) -> str:
        return "ulfgfd_slim_320"
    
    def get_task_type(self) -> str:
        return "face_detection"
    
    def get_num_keypoints(self) -> int:
        """ULFG does not output keypoints."""
        return 0
