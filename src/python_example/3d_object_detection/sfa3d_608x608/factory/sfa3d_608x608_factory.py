"""
SFA3D 608x608 Quantized Lite Factory

Creates matching preprocessor, postprocessor, and visualizer for SFA3D
3D object detection from LiDAR point clouds.
"""

from common.base import IDetectionFactory
from common.processors import SFA3DBEVPreprocessor, SFA3DPostprocessor
from common.visualizers import SFA3DVisualizer


class Sfa3d608x608Factory(IDetectionFactory):
    """Factory for SFA3D 608x608 quantized lite model."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SFA3DBEVPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return SFA3DPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return SFA3DVisualizer()

    def get_model_name(self) -> str:
        return "sfa3d_608x608"

    def get_task_type(self) -> str:
        return "3d_detection"

