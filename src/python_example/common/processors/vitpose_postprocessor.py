"""
VitPose Heatmap-based Pose Postprocessor

Model: vit_pose_small_bn
Input:  FLOAT [1, 3, 256, 192] CHW normalized
Output: FLOAT [1, 17, 64, 48] — heatmap for 17 COCO keypoints

Each channel [k, :, :] is the activation map for keypoint k.
"""

import numpy as np
from typing import List

from ..base import IPostprocessor, PreprocessContext, Keypoint, PoseResult


class VitPosePostprocessor(IPostprocessor):
    """Postprocessor for ViT-based top-down pose estimation."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext):
        heatmap = np.squeeze(outputs[0], axis=0)
        num_keypoints, hm_h, hm_w = heatmap.shape
        orig_w = ctx.original_width
        orig_h = ctx.original_height

        keypoints = []
        for k in range(num_keypoints):
            hm = heatmap[k]
            flat_idx = np.argmax(hm)
            y_hm, x_hm = divmod(int(flat_idx), hm_w)
            confidence = float(hm[y_hm, x_hm])
            x_orig = float(x_hm) / hm_w * orig_w
            y_orig = float(y_hm) / hm_h * orig_h
            keypoints.append(Keypoint(x=x_orig, y=y_orig, confidence=confidence))

        return [PoseResult(keypoints=keypoints, confidence=1.0)]

    def get_model_name(self) -> str:
        return "vit_pose_small_bn"
