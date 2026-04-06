"""
retinaface_mobilenet0_25_640 Factory
"""

from common.base import IFaceFactory
from common.processors import LetterboxPreprocessor, RetinaFacePostprocessor
from common.visualizers import FaceVisualizer


class Retinaface_mobilenet0_25_640Factory(IFaceFactory):
    """Factory for creating RetinaFace-MobileNet components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return RetinaFacePostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return FaceVisualizer()
    
    def get_model_name(self) -> str:
        return "retinaface_mobilenet0_25_640"
    
    def get_task_type(self) -> str:
        return "face_detection"

    def get_num_keypoints(self) -> int:
        """RetinaFace uses 5-point facial landmarks."""
        return 5

