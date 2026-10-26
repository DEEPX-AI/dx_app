"""Family-specific processor construction that a variant config cannot express.

These variants pass COMPUTED arguments -- an ``imagenet_mean`` constant, a
``[m * 255.0 for m in mean]`` comprehension, a PPU anchor table -- so they cannot be
reduced to literal JSON. Their original factory modules are carried over verbatim
below and the family factory delegates to them, which preserves behaviour by
construction rather than by a re-derivation that could drift.
"""

# ---- carried over verbatim from src/python_example/object_pose_estimation/dope_hope_ketchup/factory/dope_hope_ketchup_factory.py ----
"""
DOPE Object Pose Estimation Factory (Hope-Ketchup)
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base.i_factory import _FactoryConfigMixin
from common.processors import SimpleResizePreprocessor, DOPEPostprocessor
from common.visualizers import DOPEVisualizer


class Dope_hope_ketchupFactory(_FactoryConfigMixin):
    """Factory for creating DOPE Hope-Ketchup 6DoF pose estimation components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return DOPEPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        # Pass the same intrinsic/object tuning the postprocessor uses so the
        # drawn cuboid matches DopeResult.pose (see example README).
        return DOPEVisualizer(
            object_size_cm=self.config.get("object_size_cm"),
            focal_length=self.config.get("focal_length"),
        )

    def get_model_name(self) -> str:
        return "dope_hope_ketchup"

    def get_task_type(self) -> str:
        return "object_pose_estimation"


# variant -> the factory class that builds it
FACTORIES = {
    "dope-hope-ketchup_480x640": Dope_hope_ketchupFactory,
}
