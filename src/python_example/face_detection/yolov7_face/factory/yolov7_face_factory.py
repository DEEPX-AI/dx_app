"""
Yolov5face Factory
"""

from common.base import IFaceFactory
from common.processors import LetterboxPreprocessor, YOLOv5FacePostprocessor
from common.visualizers import FaceVisualizer


class Yolov7_faceFactory(IFaceFactory):
    """Factory for creating YOLOv7-Face components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv5FacePostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return FaceVisualizer()
    
    def get_model_name(self) -> str:
        return "yolov7_face"
    
    def get_task_type(self) -> str:
        return "face_detection"
    
    def get_num_keypoints(self) -> int:
        """YOLOv5Face uses 5-point facial landmarks."""
        return 5
