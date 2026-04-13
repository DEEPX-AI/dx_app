"""
Yolov8seg Factory
"""

from common.base import IInstanceSegFactory
from common.processors import LetterboxPreprocessor, YOLOv8InstanceSegPostprocessor
from common.visualizers import InstanceSegVisualizer


class Fastsam_sFactory(IInstanceSegFactory):
    """Factory for creating FastSAM-S components (class-agnostic segmentation)."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        # FastSAM (Segment Anything): large objects score ~0.5-0.7 so we need
        # a lower score threshold. Higher NMS threshold preserves overlapping
        # segments which is essential for class-agnostic panoptic output.
        config = {**self.config,
                  'num_classes': 1,
                  'score_threshold': 0.5,
                  'nms_threshold': 0.65}
        return YOLOv8InstanceSegPostprocessor(input_width, input_height, config)
    
    def create_visualizer(self):
        return InstanceSegVisualizer(labels=["segment"], show_boxes=False)
    
    def get_model_name(self) -> str:
        return "fastsam_s"
    
    def get_task_type(self) -> str:
        return "instance_segmentation"
