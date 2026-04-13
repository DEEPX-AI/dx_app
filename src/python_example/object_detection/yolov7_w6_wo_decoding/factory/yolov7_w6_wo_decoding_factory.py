"""
Yolov7_w6_wo_decoding Factory
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOv5Postprocessor
from common.visualizers import DetectionVisualizer


# YOLOv7-W6 anchors — 4 detection heads (P3/P4/P5/P6)
YOLOV7_W6_ANCHORS = {
    8:  [[19, 27],  [44, 40],   [38, 94]],
    16: [[96, 68],  [86, 152],  [180, 137]],
    32: [[140, 301],[303, 264], [238, 542]],
    64: [[436, 615],[739, 380], [925, 792]],
}


class Yolov7_w6_wo_decodingFactory(IDetectionFactory):
    """Factory for creating YOLOv7-W6 (without decoding) components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        cfg = {**self.config, 'anchors': YOLOV7_W6_ANCHORS}
        return YOLOv5Postprocessor(input_width, input_height, cfg)
    
    def create_visualizer(self):
        return DetectionVisualizer()
    
    def get_model_name(self) -> str:
        return "yolov7_w6_wo_decoding"
    
    def get_task_type(self) -> str:
        return "object_detection"
