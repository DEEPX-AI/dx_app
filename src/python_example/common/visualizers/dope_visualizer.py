"""
DOPE 6DoF Object Pose Visualizer — PnP-based 3D cuboid overlay

Workflow:
  1. Extract 8 belief-map peak coordinates (2D image space)
  2. Define 3D cuboid model vertices (real-world dimensions)
  3. cv2.solvePnP → rvec, tvec
  4. cv2.projectPoints → project all 9 3D corners to image
  5. Draw 12 cuboid edges + X on top face

Fallback: if < 4 valid peaks, draw belief peak 2D coords directly.
"""

import numpy as np
import cv2
from typing import List

from ..base import IVisualizer
from ..processors._dope_geometry import (
    KETCHUP_SIZE_CM,
    build_cuboid_3d,
    build_camera_matrix,
)


CUBOID_EDGES = [
    (1, 0), (0, 3), (3, 2), (2, 1),   # front face
    (5, 4), (4, 7), (7, 6), (6, 5),   # rear face
    (2, 6), (1, 5), (3, 7), (0, 4),   # connecting
]
TOP_X_EDGES = [(0, 5), (1, 4)]

VERTEX_COLORS = [
    (255,   0,   0),   # 0
    (  0, 255,   0),   # 1
    (  0,   0, 255),   # 2
    (255, 255,   0),   # 3
    (  0, 255, 255),   # 4
    (255,   0, 255),   # 5
    (128, 255,   0),   # 6
    (  0, 128, 255),   # 7
]


class DOPEVisualizer(IVisualizer):
    """
    Visualizer for DOPE 6DoF object pose estimation.

    Runs cv2.solvePnP on belief-map peaks and overlays the projected 3-D
    cuboid onto the image. Falls back to direct peak connections when fewer
    than 4 valid peaks are available.
    """

    def __init__(self, object_size_cm=None, focal_length=None):
        size = object_size_cm if object_size_cm is not None else KETCHUP_SIZE_CM
        self._obj_pts_3d = build_cuboid_3d(size)  # (9, 3)
        self._focal_length = focal_length

    def visualize(self, image: np.ndarray, results: list) -> np.ndarray:
        out = image.copy()
        img_h, img_w = out.shape[:2]

        # Demo pinhole camera matrix (focal = image width unless overridden).
        # Must match the intrinsic used by DOPEPostprocessor so a reused pose
        # projects consistently — see example README for tuning.
        K = build_camera_matrix(img_w, img_h, self._focal_length)
        dist = np.zeros((4, 1), dtype=np.float64)

        for result in results:
            # Keypoints are normalized [0,1] — scale to image pixels
            w = result.image_width if result.image_width > 0 else img_w
            h = result.image_height if result.image_height > 0 else img_h
            kps_px = result.keypoints.copy()
            kps_px[:, 0] *= w
            kps_px[:, 1] *= h

            all_pts = kps_px.tolist()   # 9 x [x, y]

            # Draw belief peak dots
            for i, (x, y) in enumerate(all_pts[:8]):
                px, py = int(x), int(y)
                if 0 <= px < img_w and 0 <= py < img_h:
                    cv2.circle(out, (px, py), 6, VERTEX_COLORS[i], -1, cv2.LINE_AA)
                    cv2.putText(out, str(i), (px + 6, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, VERTEX_COLORS[i], 1)

            # Draw centroid
            cx, cy = int(all_pts[8][0]), int(all_pts[8][1])
            if 0 <= cx < img_w and 0 <= cy < img_h:
                cv2.circle(out, (cx, cy), 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 8, (0, 0, 0), 2, cv2.LINE_AA)

            # Prefer the pose already solved by the postprocessor so the drawn
            # cuboid matches DopeResult.pose exactly; otherwise solve here.
            pose = getattr(result, "pose", None)
            rvec = tvec = None
            if pose is not None and pose.get("rvec") is not None:
                rvec, tvec = pose["rvec"], pose["tvec"]
            else:
                pts_2d, pts_3d = [], []
                for i, (x, y) in enumerate(all_pts):
                    if 0 <= x < img_w and 0 <= y < img_h:
                        pts_2d.append([x, y])
                        pts_3d.append(self._obj_pts_3d[i])
                if len(pts_2d) >= 4:
                    ret, rvec, tvec = cv2.solvePnP(
                        np.array(pts_3d, dtype=np.float64),
                        np.array(pts_2d, dtype=np.float64),
                        K, dist, flags=cv2.SOLVEPNP_EPNP)
                    if not ret:
                        rvec = tvec = None

            if rvec is not None:
                proj, _ = cv2.projectPoints(
                    self._obj_pts_3d, rvec, tvec, K, dist)
                proj = np.int32(proj).reshape(-1, 2)

                for a, b in CUBOID_EDGES:
                    cv2.line(out, tuple(proj[a]), tuple(proj[b]),
                             (0, 255, 0), 2, cv2.LINE_AA)
                for a, b in TOP_X_EDGES:
                    cv2.line(out, tuple(proj[a]), tuple(proj[b]),
                             (0, 255, 255), 2, cv2.LINE_AA)
            else:
                self._draw_fallback(out, all_pts, img_w, img_h)

        cv2.putText(out, "DOPE 6DoF", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return out

    @staticmethod
    def _draw_fallback(out, all_pts, w, h):
        """Directly connect belief peaks when PnP is not available."""
        def valid(i):
            x, y = all_pts[i]
            return 0 <= x < w and 0 <= y < h

        for a, b in CUBOID_EDGES:
            if valid(a) and valid(b):
                cv2.line(out,
                         (int(all_pts[a][0]), int(all_pts[a][1])),
                         (int(all_pts[b][0]), int(all_pts[b][1])),
                         (0, 255, 0), 2, cv2.LINE_AA)
