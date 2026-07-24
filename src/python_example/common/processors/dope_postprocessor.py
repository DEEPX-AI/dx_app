"""
DOPE (Deep Object Pose Estimation) Postprocessor

Model: dope-hope-ketchup
Input:  UINT8 [1, 480, 640, 3]
Output tensors (order may vary):
  conv2d_84: FLOAT [1, 9, 60, 80]  — 9 belief maps (8 vertices + centroid)
  conv2d_91: FLOAT [1, 16, 60, 80] — 16 affinity fields (not used here)

Reference: dx-modelzoo dope_decode custom op
  - Sub-pixel refinement: 3x3 belief-weighted centroid around argmax peak
  - Normalized [0,1] keypoint coords  (cx + 0.5) / hm_w
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from ..base import IPostprocessor, PreprocessContext
from ._dope_geometry import (
    KETCHUP_SIZE_CM,
    build_cuboid_3d,
    build_camera_matrix,
    solve_pose,
)


@dataclass
class DopeResult:
    """Result from DOPE 6DoF pose estimation.

    keypoints: (9, 2) normalized [0,1] coords — 8 vertices + centroid (ch8)
    all_conf:  (9,)   belief map peak values
    pose:      {'R': (3,3), 't': (3,)} or None when PnP not requested / failed
    """
    keypoints: np.ndarray             # (9, 2) normalized [0,1]
    centroid: np.ndarray              # (2,) normalized [0,1]
    confidence: float
    all_conf: np.ndarray              # (9,)
    pose: Optional[Dict[str, Any]]    # {'R': ..., 't': ...} or None
    # Image-space coords for visualization (multiply keypoints by image WxH)
    image_width: int = 0
    image_height: int = 0


class DOPEPostprocessor(IPostprocessor):
    """Postprocessor for DOPE 6DoF object pose estimation.

    Matches dx-modelzoo dope_decode op:
    - argmax peak on each of 9 belief-map channels
    - 3x3 belief-weighted sub-pixel centroid refinement
    - (cx + 0.5) / hm_w normalization → [0,1] coords
    """

    NUM_BELIEFS = 9

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        # Suppress a detection when the centroid belief peak is below this value.
        # Default 0.0 keeps the demo's "always detect" behavior (backward compat).
        self.conf_threshold = float(self.config.get("conf_threshold", 0.0))
        self.refine = bool(self.config.get("subpixel_refine", True))
        # PnP (demo): see example README for camera-intrinsic / object-size tuning.
        self.solve_pnp = bool(self.config.get("solve_pnp", True))
        self.object_size_cm = self.config.get("object_size_cm", list(KETCHUP_SIZE_CM))
        self.focal_length = self.config.get("focal_length", None)
        self._obj_3d = build_cuboid_3d(self.object_size_cm)

    @staticmethod
    def _subpixel(m: np.ndarray, iy: int, ix: int) -> Tuple[float, float]:
        """3x3 belief-weighted centroid refinement around (iy, ix).

        Mirrors DopeDecode._subpixel() from dx-modelzoo dope/custom_ops.py.
        """
        h, w = m.shape
        y0, y1 = max(iy - 1, 0), min(iy + 2, h)
        x0, x1 = max(ix - 1, 0), min(ix + 2, w)
        patch = np.clip(m[y0:y1, x0:x1], 0.0, None)
        s = float(patch.sum())
        if s <= 1e-9:
            return float(ix), float(iy)
        ys = np.arange(y0, y1, dtype=np.float64)
        xs = np.arange(x0, x1, dtype=np.float64)
        cy = float((patch.sum(axis=1) * ys).sum() / s)
        cx = float((patch.sum(axis=0) * xs).sum() / s)
        return cx, cy

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext):
        # dxrt returns a single merged tensor [1, 25, H, W]:
        #   channels  0–8:  9 belief maps
        #   channels  9–24: 16 affinity fields (unused)
        raw = np.squeeze(outputs[0], axis=0) if outputs[0].ndim == 4 else outputs[0]
        beliefs = raw[:self.NUM_BELIEFS]   # (9, H, W)

        hm_h, hm_w = beliefs.shape[1], beliefs.shape[2]

        kps = np.zeros((self.NUM_BELIEFS, 2), dtype=np.float64)
        confs = np.zeros(self.NUM_BELIEFS, dtype=np.float64)

        for c in range(self.NUM_BELIEFS):
            m = beliefs[c]
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            confs[c] = float(m[iy, ix])
            if self.refine:
                cx, cy = self._subpixel(m, int(iy), int(ix))
            else:
                cx, cy = float(ix), float(iy)
            # +0.5 pixel-center offset → normalize to [0,1]
            kps[c, 0] = (cx + 0.5) / hm_w
            kps[c, 1] = (cy + 0.5) / hm_h

        centroid_conf = float(confs[self.NUM_BELIEFS - 1])  # channel 8 = centroid
        # Confidence gating: drop the detection when the centroid peak is weak.
        # (No affinity/thresh_map gating in this demo — this is the single knob
        #  to suppress spurious detections when no object is present.)
        if centroid_conf < self.conf_threshold:
            return []

        # 6DoF pose via PnP on the 9 keypoints (demo intrinsic — see README).
        pose = None
        img_w, img_h = ctx.original_width, ctx.original_height
        if self.solve_pnp and img_w > 0 and img_h > 0:
            kps_px = kps.copy()
            kps_px[:, 0] *= img_w
            kps_px[:, 1] *= img_h
            K = build_camera_matrix(img_w, img_h, self.focal_length)
            pose = solve_pose(kps_px, self._obj_3d, K)

        return [DopeResult(
            keypoints=kps,
            centroid=kps[self.NUM_BELIEFS - 1].copy(),   # channel 8 = centroid
            confidence=centroid_conf,
            all_conf=confs,
            pose=pose,
            image_width=img_w,
            image_height=img_h,
        )]

    def get_model_name(self) -> str:
        return "dope"
