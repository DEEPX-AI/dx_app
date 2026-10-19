"""Family-specific processor construction that a variant config cannot express.

These variants pass COMPUTED arguments -- an ``imagenet_mean`` constant, a
``[m * 255.0 for m in mean]`` comprehension, a PPU anchor table -- so they cannot be
reduced to literal JSON. Their original factory modules are carried over verbatim
below and the family factory delegates to them, which preserves behaviour by
construction rather than by a re-derivation that could drift.
"""

# ---- carried over verbatim from src/python_example/attribute_recognition/face_attr_resnet_v1_18/factory/face_attr_resnet_v1_18_factory.py ----
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


# variant -> the factory class that builds it
FACTORIES = {
    "faceattr_resnetv1-18_218x178": Face_attr_resnet_v1_18Factory,
}
