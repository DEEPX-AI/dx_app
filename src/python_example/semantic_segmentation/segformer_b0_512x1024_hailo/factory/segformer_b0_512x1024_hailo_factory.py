"""
SegFormer Semantic Segmentation Factory
"""

from common.base import ISegmentationFactory
from common.processors import SimpleResizePreprocessor, SegFormerPostprocessor
from common.visualizers import SemanticSegmentationVisualizer


class Segformer_b0_512x1024_hailoFactory(ISegmentationFactory):
    """Factory for creating SegFormer-B0 components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return SegFormerPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return SemanticSegmentationVisualizer()
    
    def get_model_name(self) -> str:
        return "segformer_b0_512x1024_hailo"
    
    def get_task_type(self) -> str:
        return "semantic_segmentation"

