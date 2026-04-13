"""
UNet-MobileNetV2 Factory

UNet-MobileNetV2 semantic segmentation model.
Output: [1, 256, 256, 3] float32 logits (3 classes, NHWC layout).
"""

import numpy as np

from common.base import ISegmentationFactory
from common.processors import SimpleResizePreprocessor, SemanticSegmentationPostprocessor
from common.visualizers import SemanticSegmentationVisualizer

# UNet-MobileNetV2: 3-class Oxford Pets segmentation
# Model output classes: 0=foreground(pet), 1=background, 2=boundary
# Colors in BGR format
_UNET_3CLASS_PALETTE = np.array([
    [255, 128,   0],    # 0: foreground/pet (blue in BGR)
    [0,     0,   0],    # 1: background     (skip)
    [0,   128, 255],    # 2: boundary       (orange in BGR)
], dtype=np.uint8)


class Unet_mobilenet_v2Factory(ISegmentationFactory):
    """Factory for creating UNet-MobileNetV2 semantic segmentation components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        config = {**self.config, 'num_classes': 3}
        return SemanticSegmentationPostprocessor(input_width, input_height, config,
                                                 resize_to_original=False)
    
    def create_visualizer(self):
        return SemanticSegmentationVisualizer(custom_palette=_UNET_3CLASS_PALETTE,
                                             background_class=1)
    
    def get_model_name(self) -> str:
        return "unet_mobilenet_v2"
    
    def get_task_type(self) -> str:
        return "semantic_segmentation"
