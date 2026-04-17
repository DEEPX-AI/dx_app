"""
Scrfd500mPpu Factory
"""

from common.base import IFaceFactory
from common.processors import LetterboxPreprocessor, SCRFDPPUPostprocessor
from common.visualizers import FaceVisualizer


class Scrfd500mPpuFactory(IFaceFactory):
    """Factory for creating SCRFD500M-PPU components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return SCRFDPPUPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return FaceVisualizer()

    def get_model_name(self) -> str:
        return "scrfd500m_ppu"

    def get_task_type(self) -> str:
        return "face_detection"

    def get_num_keypoints(self) -> int:
        """SCRFD uses 5-point facial landmarks."""
        return 5
