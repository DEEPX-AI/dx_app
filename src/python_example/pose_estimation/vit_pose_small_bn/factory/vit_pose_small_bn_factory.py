"""
VitPose Small BN Pose Estimation Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IPoseFactory
from common.processors import SimpleResizePreprocessor, VitPosePostprocessor
from common.visualizers import PoseVisualizer


class Vit_pose_small_bnFactory(IPoseFactory):
    """Factory for creating VitPose-Small-BN components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height, normalize_float=True)

    def create_postprocessor(self, input_width: int, input_height: int):
        return VitPosePostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return PoseVisualizer()

    def get_model_name(self) -> str:
        return "vit_pose_small_bn"

    def get_task_type(self) -> str:
        return "pose_estimation"

    def get_num_keypoints(self) -> int:
        return 17
