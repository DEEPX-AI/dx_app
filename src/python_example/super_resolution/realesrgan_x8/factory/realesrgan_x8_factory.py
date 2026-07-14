"""RealESRGAN x8 Super-Resolution Factory"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IRestorationFactory
from common.processors import SimpleResizePreprocessor, RealESRGANPostprocessor
from common.visualizers import SuperResolutionVisualizer


class Realesrgan_x8Factory(IRestorationFactory):
    """Factory for creating RealESRGAN-x8 components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return RealESRGANPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return SuperResolutionVisualizer()

    def get_model_name(self) -> str:
        return "realesrgan_x8"

    def get_task_type(self) -> str:
        return "super_resolution"
