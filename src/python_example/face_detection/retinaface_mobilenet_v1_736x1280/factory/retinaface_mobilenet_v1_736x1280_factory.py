"""
Retinaface_mobilenet_v1_736x1280 Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IFaceFactory
from common.processors import RetinaFacePostprocessor, SimpleResizePreprocessor
from common.visualizers import FaceVisualizer


class Retinaface_mobilenet_v1_736x1280Factory(IFaceFactory):
    """Factory for creating Retinaface_mobilenet_v1_736x1280 components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(
            input_width,
            input_height,
            nhwc=True,
            mean=[104.0, 117.0, 123.0],
            bgr=True,
        )

    def create_postprocessor(self, input_width: int, input_height: int):
        return RetinaFacePostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return FaceVisualizer()

    def get_model_name(self) -> str:
        return "retinaface_mobilenet_v1_736x1280"

    def get_task_type(self) -> str:
        return "face_detection"

    def get_num_keypoints(self) -> int:
        return 5
