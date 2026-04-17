"""
Face Attribute ResNet18 Factory
"""

from common.base import IClassificationFactory
from common.processors import SimpleResizePreprocessor, AttributePostprocessor
from common.processors.attribute_postprocessor import CELEBA_40_LABELS
from common.visualizers import AttributeVisualizer


class Face_attr_resnet_v1_18Factory(IClassificationFactory):
    """Factory for creating face attribute recognition components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return AttributePostprocessor(input_width, input_height, self.config,
                                      labels=CELEBA_40_LABELS)
    
    def create_visualizer(self):
        return AttributeVisualizer()
    
    def get_model_name(self) -> str:
        return "face_attr_resnet_v1_18"
    
    def get_task_type(self) -> str:
        return "attribute_recognition"
