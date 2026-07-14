"""
MediaPipe Hand (Palm) Detector Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IFaceFactory
from common.processors import MediaPipeHandPostprocessor, SimpleResizePreprocessor
from common.visualizers import FaceVisualizer


class Mediapipe_hand_detectorFactory(IFaceFactory):
    """Factory for MediaPipe palm/hand detection model (192x192 UINT8 NHWC)."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # Model takes UINT8 NHWC RGB - SimpleResizePreprocessor returns uint8 RGB by default
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return MediaPipeHandPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return FaceVisualizer(label="Hand")

    def get_model_name(self) -> str:
        return "mediapipe_hand_detector"

    def get_task_type(self) -> str:
        return "hand_detection"

    def get_num_keypoints(self) -> int:
        return 7
