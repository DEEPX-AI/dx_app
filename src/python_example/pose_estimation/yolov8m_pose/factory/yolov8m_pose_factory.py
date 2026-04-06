"""YOLOv8m-Pose Factory"""

from common.base import IPoseFactory
from common.processors import LetterboxPreprocessor, YOLOv8PosePostprocessor
from common.visualizers import PoseVisualizer


class Yolov8m_poseFactory(IPoseFactory):
    """Factory for creating YOLOv8m-Pose components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv8PosePostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return PoseVisualizer()
    
    def get_model_name(self) -> str:
        return "yolov8m_pose"
    
    def get_task_type(self) -> str:
        return "pose_estimation"
    
    def get_num_keypoints(self) -> int:
        """YOLOv8Pose uses COCO 17-point body keypoints."""
        return 17
