"""YOLOv8s-Pose Factory"""

from common.base import IPoseFactory
from common.processors import LetterboxPreprocessor, CenterPosePostprocessor
from common.visualizers import PoseVisualizer


class Centerpose_regnetx_800mfFactory(IPoseFactory):
    """Factory for creating YOLOv8s-Pose components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return CenterPosePostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return PoseVisualizer()
    
    def get_model_name(self) -> str:
        return "centerpose_regnetx_800mf"
    
    def get_task_type(self) -> str:
        return "pose_estimation"
    
    def get_num_keypoints(self) -> int:
        """CenterPose uses 8 3D bounding box keypoints."""
        return 8
