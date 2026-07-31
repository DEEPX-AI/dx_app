"""
Yolo26nDepth Factory — YOLO26n-Depth monocular depth estimation (DX-M1 NPU).

Source model: Ultralytics PR #25065 (branch `depth_anything`), `yolo26n-depth.pt`,
exported to ONNX (single dense output `output0` [1,1,640,640]) and compiled to
`yolo26n-depth_640x640.dxnn`.

Preprocessing: YOLO normalization (`pixel / 255`, no ImageNet mean/std) — this
MUST match the INT8 calibration domain used at compile time. SimpleResizePreprocessor
computes `(pixel - mean) / std` over the [0,255] range, so mean=0, std=255 yields
exactly `pixel/255`.
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IDepthEstimationFactory
from common.processors import DepthEstimationPostprocessor, SimpleResizePreprocessor
from common.visualizers import DepthVisualizer


class Yolo26nDepthFactory(IDepthEstimationFactory):
    """Factory for YOLO26n-Depth components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # YOLO26-Depth is YOLO-normalized: pixel/255, no mean subtraction.
        return SimpleResizePreprocessor(
            input_width, input_height,
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )

    def create_postprocessor(self, input_width: int, input_height: int):
        return DepthEstimationPostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DepthVisualizer()

    def get_model_name(self) -> str:
        return "yolo26n_depth"

    def get_task_type(self) -> str:
        return "depth_estimation"
