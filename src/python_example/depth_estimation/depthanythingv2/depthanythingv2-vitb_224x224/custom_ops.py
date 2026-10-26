"""Family-specific processor construction that a variant config cannot express.

These variants pass COMPUTED arguments -- an ``imagenet_mean`` constant, a
``[m * 255.0 for m in mean]`` comprehension, a PPU anchor table -- so they cannot be
reduced to literal JSON. Their original factory modules are carried over verbatim
below and the family factory delegates to them, which preserves behaviour by
construction rather than by a re-derivation that could drift.
"""

# ---- carried over verbatim from src/python_example/depth_estimation/depth_anything_v2_vitb/factory/depth_anything_v2_vitb_factory.py ----
"""
Depth_anything_v2_vitb Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IDepthEstimationFactory
from common.processors import DepthEstimationPostprocessor, SimpleResizePreprocessor
from common.visualizers import DepthVisualizer


class Depth_anything_v2_vitbFactory(IDepthEstimationFactory):
    """Factory for creating Depth_anything_v2_vitb components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # Depth Anything V2 expects ImageNet-normalized float input:
        #   (pixel/255 - mean) / std, mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] (RGB)
        # SimpleResizePreprocessor applies mean/std on the [0,255] range, so scale by 255.
        imagenet_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        imagenet_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        return SimpleResizePreprocessor(
            input_width, input_height, mean=imagenet_mean, std=imagenet_std)

    def create_postprocessor(self, input_width: int, input_height: int):
        return DepthEstimationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DepthVisualizer()

    def get_model_name(self) -> str:
        return "depth_anything_v2_vitb"

    def get_task_type(self) -> str:
        return "depth_estimation"


# ---- carried over verbatim from src/python_example/depth_estimation/depth_anything_v2_vitl/factory/depth_anything_v2_vitl_factory.py ----
"""
Depth_anything_v2_vitl Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IDepthEstimationFactory
from common.processors import DepthEstimationPostprocessor, SimpleResizePreprocessor
from common.visualizers import DepthVisualizer


class Depth_anything_v2_vitlFactory(IDepthEstimationFactory):
    """Factory for creating Depth_anything_v2_vitl components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # Depth Anything V2 expects ImageNet-normalized float input:
        #   (pixel/255 - mean) / std, mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] (RGB)
        # SimpleResizePreprocessor applies mean/std on the [0,255] range, so scale by 255.
        imagenet_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        imagenet_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        return SimpleResizePreprocessor(
            input_width, input_height, mean=imagenet_mean, std=imagenet_std)

    def create_postprocessor(self, input_width: int, input_height: int):
        return DepthEstimationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DepthVisualizer()

    def get_model_name(self) -> str:
        return "depth_anything_v2_vitl"

    def get_task_type(self) -> str:
        return "depth_estimation"


# ---- carried over verbatim from src/python_example/depth_estimation/depth_anything_v2_vits/factory/depth_anything_v2_vits_factory.py ----
"""
Depth_anything_v2_vits Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IDepthEstimationFactory
from common.processors import DepthEstimationPostprocessor, SimpleResizePreprocessor
from common.visualizers import DepthVisualizer


class Depth_anything_v2_vitsFactory(IDepthEstimationFactory):
    """Factory for creating Depth_anything_v2_vits components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # Depth Anything V2 expects ImageNet-normalized float input:
        #   (pixel/255 - mean) / std, mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] (RGB)
        # SimpleResizePreprocessor applies mean/std on the [0,255] range, so scale by 255.
        imagenet_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        imagenet_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        return SimpleResizePreprocessor(
            input_width, input_height, mean=imagenet_mean, std=imagenet_std)

    def create_postprocessor(self, input_width: int, input_height: int):
        return DepthEstimationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DepthVisualizer()

    def get_model_name(self) -> str:
        return "depth_anything_v2_vits"

    def get_task_type(self) -> str:
        return "depth_estimation"


# variant -> the factory class that builds it
FACTORIES = {
    "depthanythingv2-vitb_224x224": Depth_anything_v2_vitbFactory,
    "depthanythingv2-vitl_224x224": Depth_anything_v2_vitlFactory,
    "depthanythingv2-vits_224x224": Depth_anything_v2_vitsFactory,
}
