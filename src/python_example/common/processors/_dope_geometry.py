"""Shared geometry helpers for the DOPE 6DoF demo.

Single source of truth for the 3D cuboid model, the (demo) pinhole camera
matrix, and the PnP solve, so the postprocessor (which fills ``DopeResult.pose``)
and the visualizer (which draws the cuboid) always agree on the same pose.

NOTE (demo-only): the camera matrix here is a *guess* (focal = image width,
principal point = image center). For a metrically accurate pose you must supply
a real calibrated intrinsic via ``focal_length`` in config.json. See the example
README for tuning parameters.
"""

import numpy as np

# Hope-Ketchup default object size [width, height, depth] in cm
# (from DOPE config_pose.yaml).
KETCHUP_SIZE_CM = (14.860799789428711, 4.3368000984191895, 6.4513998031616211)


def build_cuboid_3d(size_cm=KETCHUP_SIZE_CM) -> np.ndarray:
    """Return (9, 3) cuboid vertices in the OpenCV frame (X right, Y down, Z fwd).

    Vertex order matches the DOPE belief-map channel order:
      0 FrontTopRight 1 FrontTopLeft  2 FrontBottomLeft 3 FrontBottomRight
      4 RearTopRight  5 RearTopLeft   6 RearBottomLeft  7 RearBottomRight
      8 Center
    """
    w, h, d = size_cm[0] / 2.0, size_cm[1] / 2.0, size_cm[2] / 2.0
    return np.array([
        [ w, -h,  d],   # 0: FrontTopRight
        [-w, -h,  d],   # 1: FrontTopLeft
        [-w,  h,  d],   # 2: FrontBottomLeft
        [ w,  h,  d],   # 3: FrontBottomRight
        [ w, -h, -d],   # 4: RearTopRight
        [-w, -h, -d],   # 5: RearTopLeft
        [-w,  h, -d],   # 6: RearBottomLeft
        [ w,  h, -d],   # 7: RearBottomRight
        [ 0,  0,  0],   # 8: Center
    ], dtype=np.float64)


def build_camera_matrix(img_w: int, img_h: int, focal=None) -> np.ndarray:
    """Demo pinhole intrinsic. ``focal`` defaults to the image width (px)."""
    f = float(focal) if focal else float(img_w)
    return np.array([
        [f, 0, img_w / 2.0],
        [0, f, img_h / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)


def solve_pose(keypoints_px: np.ndarray, obj_3d: np.ndarray,
               K: np.ndarray, dist: np.ndarray = None):
    """Solve 6DoF pose from 2D-3D vertex correspondences.

    keypoints_px : (N, 2) image-space keypoints (vertex order = obj_3d order)
    Returns ``{'R': (3,3), 't': (3,), 'rvec': (3,1), 'tvec': (3,1)}`` or None
    when there are fewer than 4 points or solvePnP fails.
    """
    import cv2  # lazy: keeps `import common.processors` cv2-free until used
    if dist is None:
        dist = np.zeros((4, 1), dtype=np.float64)

    pts_2d = np.asarray(keypoints_px, dtype=np.float64)
    pts_3d = np.asarray(obj_3d, dtype=np.float64)
    if pts_2d.shape[0] < 4 or pts_2d.shape[0] != pts_3d.shape[0]:
        return None

    ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, dist,
                                  flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.reshape(3), "rvec": rvec, "tvec": tvec}
