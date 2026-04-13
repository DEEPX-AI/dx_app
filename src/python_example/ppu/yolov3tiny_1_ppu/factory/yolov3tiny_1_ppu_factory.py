"""
YOLOv3Tiny PPU Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOv5PPUPostprocessor
from common.visualizers import DetectionVisualizer


# YOLOv3Tiny anchors (different from YOLOv5 defaults)
YOLOV3TINY_ANCHORS = {
    16: [[10, 14], [23, 27], [37, 58]],
    32: [[81, 82], [135, 169], [344, 319]],
}


class Yolov3tiny_1_ppuFactory(IDetectionFactory):
    """Factory for creating YOLOv3Tiny-PPU components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        # YOLOv3Tiny has 2 heads: layer_idx=0 → stride 16, layer_idx=1 → stride 32
        cfg = {**self.config, 'anchors': YOLOV3TINY_ANCHORS, 'strides': [16, 32]}
        return YOLOv5PPUPostprocessor(input_width, input_height, cfg)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yolov3tiny_1_ppu"
    
    def get_task_type(self) -> str:
        return "object_detection"
