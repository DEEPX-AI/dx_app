"""
Yolo11x_pose Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IPoseFactory
from common.processors import LetterboxPreprocessor, YOLOv8PosePostprocessor
from common.visualizers import PoseVisualizer


class Yolo11x_poseFactory(IPoseFactory):
    """Factory for creating Yolo11x_pose components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv8PosePostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return PoseVisualizer()

    def get_model_name(self) -> str:
        return "yolo11x_pose"

    def get_task_type(self) -> str:
        return "pose_estimation"

    def get_num_keypoints(self) -> int:
        return 17
