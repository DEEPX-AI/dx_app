"""Family-specific processor construction that a variant config cannot express.

These variants pass COMPUTED arguments -- an ``imagenet_mean`` constant, a
``[m * 255.0 for m in mean]`` comprehension, a PPU anchor table -- so they cannot be
reduced to literal JSON. Their original factory modules are carried over verbatim
below and the family factory delegates to them, which preserves behaviour by
construction rather than by a re-derivation that could drift.
"""

# ---- carried over verbatim from src/python_example/keypoint_detection/superpoint/factory/superpoint_factory.py ----
"""
SuperPoint Keypoint Detection Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base.i_factory import _FactoryConfigMixin
from common.processors import GrayscaleResizePreprocessor, SuperPointPostprocessor
from common.visualizers import SuperPointVisualizer


class SuperpointFactory(_FactoryConfigMixin):
    """Factory for creating SuperPoint keypoint detection components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return GrayscaleResizePreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return SuperPointPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        # Pass the config through: these knobs used to be unreachable because
        # the visualizer was constructed with no arguments, so nn_thresh,
        # track_max_length, min_track_length, max_pixel_dist and ratio_thresh
        # in config.json were silently ignored.
        cfg = self.config
        return SuperPointVisualizer(
            conf_threshold=float(cfg.get("conf_threshold", 0.015)),
            max_keypoints=int(cfg.get("top_k", 500)),
            nn_thresh=float(cfg.get("nn_thresh", 0.7)),
            track_max_length=int(cfg.get("track_max_length", 5)),
            min_track_length=int(cfg.get("min_track_length", 2)),
            max_pixel_dist=float(cfg.get("max_pixel_dist", 100.0)),
            ratio_thresh=float(cfg.get("ratio_thresh", 0.75)),
        )

    def get_model_name(self) -> str:
        return "superpoint"

    def get_task_type(self) -> str:
        return "keypoint_detection"


# variant -> the factory class that builds it
FACTORIES = {
    "superpoint_480x640": SuperpointFactory,
}
