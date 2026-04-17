"""
Yolov10nPpu Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOv10PPUPostprocessor
from common.visualizers import DetectionVisualizer


class Yolov10nPpuFactory(IDetectionFactory):
    """Factory for creating YOLOv10N-PPU components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv10PPUPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DetectionVisualizer()

    def get_model_name(self) -> str:
        return "yolov10n_ppu"

    def get_task_type(self) -> str:
        return "object_detection"
