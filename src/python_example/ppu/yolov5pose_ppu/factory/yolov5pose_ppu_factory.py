"""
Yolov5posePpu Factory
"""

from common.base import IPoseFactory
from common.processors import LetterboxPreprocessor, YOLOv5PosePPUPostprocessor
from common.visualizers import PoseVisualizer


class Yolov5posePpuFactory(IPoseFactory):
    """Factory for creating YOLOv5Pose-PPU components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv5PosePPUPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return PoseVisualizer()

    def get_model_name(self) -> str:
        return "yolov5pose_ppu"

    def get_task_type(self) -> str:
        return "pose_estimation"

    def get_num_keypoints(self) -> int:
        """YOLOv5Pose uses 17 COCO keypoints."""
        return 17
