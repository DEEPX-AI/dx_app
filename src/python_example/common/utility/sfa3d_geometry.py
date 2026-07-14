"""3D box geometry, camera projection, and point-cloud rendering for SFA3D."""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np

from .kitti_calib import KittiCalib, default_camera_canvas_size

# Wireframe edges for 8-corner 3D boxes
_BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_CLASS_COLORS_BGR = [
    (255, 128, 0),  # Pedestrian
    (0, 255, 0),    # Car
    (0, 128, 255),  # Cyclist
]

# KITTI velodyne ROI used by SFA3D BEV preprocessor
_X_MIN, _X_MAX = 0.0, 50.0
_Y_MIN, _Y_MAX = -25.0, 25.0
_Z_MIN, _Z_MAX = -2.5, 1.5


def _det_color(det: Any) -> Tuple[int, int, int]:
    cls_id = int(getattr(det, "class_id", 0))
    return _CLASS_COLORS_BGR[cls_id % len(_CLASS_COLORS_BGR)]


def box_corners_velo(det: Any) -> np.ndarray:
    """Build 8 box corners in velodyne coordinates (KITTI bottom-center z)."""
    cx = float(getattr(det, "x3d", 0.0))
    cy = float(getattr(det, "y3d", 0.0))
    cz = float(getattr(det, "z3d", 0.0))
    h = float(getattr(det, "dim_h", 1.0))
    w = float(getattr(det, "dim_w", 1.0))
    l = float(getattr(det, "dim_l", 1.0))
    yaw = float(getattr(det, "yaw", 0.0))

    # z3d is bottom-center height in KITTI / SFA3D convention
    template = np.array([
        [l / 2, w / 2, 0.0],
        [l / 2, -w / 2, 0.0],
        [-l / 2, -w / 2, 0.0],
        [-l / 2, w / 2, 0.0],
        [l / 2, w / 2, h],
        [l / 2, -w / 2, h],
        [-l / 2, -w / 2, h],
        [-l / 2, w / 2, h],
    ], dtype=np.float64)

    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)
    rot = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    corners = template @ rot.T
    corners += np.array([cx, cy, cz], dtype=np.float64)
    return corners


def world_xy_to_bev_px(
    x: float,
    y: float,
    *,
    width: int = 608,
    height: int = 608,
    display: bool = True,
) -> Tuple[float, float]:
    """Map velodyne XY (metres) to BEV pixel coordinates.

    Preprocessor grid uses col = (y - y_min) for model input. For display on a
    horizontally mirrored BEV (KITTI +y = left aligned with camera), set
    display=True to map lateral axis as col = (y_max - y).
    """
    y_span = max(_Y_MAX - _Y_MIN, 1e-3)
    if display:
        col = (_Y_MAX - y) / y_span * width
    else:
        col = (y - _Y_MIN) / y_span * width
    row = (_X_MAX - x) / max(_X_MAX - _X_MIN, 1e-3) * height
    return col, row


def bev_box_corners(
    det: Any,
    *,
    width: int = 608,
    height: int = 608,
) -> np.ndarray:
    """Return 4 BEV polygon corners from 3D pose (x, y, l, w, yaw)."""
    cx = float(getattr(det, "x3d", 0.0))
    cy = float(getattr(det, "y3d", 0.0))
    l = float(getattr(det, "dim_l", 1.0))
    w = float(getattr(det, "dim_w", 1.0))
    yaw = float(getattr(det, "yaw", 0.0))
    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)

    corners: List[Tuple[float, float]] = []
    for lx, ly in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
        wx = cx + lx * cos_a - ly * sin_a
        wy = cy + lx * sin_a + ly * cos_a
        corners.append(world_xy_to_bev_px(wx, wy, width=width, height=height))
    return np.asarray(corners, dtype=np.float32)


def draw_bev_boxes(
    canvas: np.ndarray,
    detections: Sequence[Any],
    *,
    width: int | None = None,
    height: int | None = None,
    class_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw class-colored oriented boxes and labels on a BEV image."""
    if canvas is None or canvas.size == 0:
        return canvas
    output = canvas.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    h, w = output.shape[:2]
    panel_w = width or w
    panel_h = height or h
    fill_layer = output.copy()

    for det in detections:
        color = _det_color(det)
        corners = bev_box_corners(det, width=panel_w, height=panel_h)
        pts = np.round(corners).astype(np.int32)
        cv2.fillPoly(fill_layer, [pts], color, lineType=cv2.LINE_AA)
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        cx = float(getattr(det, "x3d", 0.0))
        cy = float(getattr(det, "y3d", 0.0))
        length = float(getattr(det, "dim_l", 1.0))
        yaw = float(getattr(det, "yaw", 0.0))
        front_x = cx + (length / 2.0) * math.cos(yaw)
        front_y = cy + (length / 2.0) * math.sin(yaw)
        center = world_xy_to_bev_px(cx, cy, width=panel_w, height=panel_h)
        front = world_xy_to_bev_px(front_x, front_y, width=panel_w, height=panel_h)
        cv2.arrowedLine(
            output,
            (int(round(center[0])), int(round(center[1]))),
            (int(round(front[0])), int(round(front[1]))),
            color, 2, tipLength=0.25, line_type=cv2.LINE_AA,
        )

        name = getattr(det, "class_name", "Obj")
        score = float(getattr(det, "confidence", 0.0))
        label = f"{name} {score:.2f}"
        lx = int(np.clip(pts[:, 0].min(), 0, w - 1))
        ly = int(np.clip(pts[:, 1].min() - 4, 15, h - 1))
        cv2.putText(output, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(output, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.addWeighted(fill_layer, 0.20, output, 0.80, 0.0, output)

    if class_names:
        y0 = 48
        for i, cn in enumerate(class_names):
            cv2.putText(output, cn, (10, y0 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        _CLASS_COLORS_BGR[i % len(_CLASS_COLORS_BGR)], 1, cv2.LINE_AA)
    return output


def project_velo_points(
    points_xyz: np.ndarray,
    calib: KittiCalib,
    *,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project Nx3 velodyne points to pixel coordinates."""
    if points_xyz.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

    pts = np.asarray(points_xyz, dtype=np.float64)
    hom = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float64)])
    proj = (calib.velo_to_image @ hom.T).T
    depth = proj[:, 2]
    valid = depth > min_depth
    uv = np.zeros((pts.shape[0], 2), dtype=np.float64)
    uv[valid, 0] = proj[valid, 0] / depth[valid]
    uv[valid, 1] = proj[valid, 1] / depth[valid]
    return uv, valid


def _camera_box_is_visible(
    corners_uv: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_in_frame_corners: int = 2,
    margin: int = 8,
) -> bool:
    """Return True when enough projected corners fall inside the camera image."""
    h, w = image_shape[:2]
    valid = np.all(np.isfinite(corners_uv), axis=1)
    if valid.sum() < min_in_frame_corners:
        return False
    pts = corners_uv[valid]
    inside = (
        (pts[:, 0] >= margin) & (pts[:, 0] < w - margin)
        & (pts[:, 1] >= margin) & (pts[:, 1] < h - margin)
    )
    if inside.sum() >= min_in_frame_corners:
        return True
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    return (-margin <= cx < w + margin) and (-margin <= cy < h + margin) and inside.any()


def _draw_box_wireframe(
    canvas: np.ndarray,
    corners_uv: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = canvas.shape[:2]
    for i0, i1 in _BOX_EDGES:
        p0 = corners_uv[i0]
        p1 = corners_uv[i1]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        pt0 = (int(round(p0[0])), int(round(p0[1])))
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        ok, c0, c1 = cv2.clipLine((0, 0, w - 1, h - 1), pt0, pt1)
        if ok:
            cv2.line(canvas, c0, c1, color, thickness, cv2.LINE_AA)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox-resize image into a fixed panel size."""
    if image is None or image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    frame = image.copy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    scale = min(width / max(frame.shape[1], 1), height / max(frame.shape[0], 1))
    new_w = max(1, int(round(frame.shape[1] * scale)))
    new_h = max(1, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (16, 16, 20)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def make_camera_canvas(calib: KittiCalib) -> np.ndarray:
    """Create a dark camera canvas sized from calibration intrinsics."""
    w, h = default_camera_canvas_size(calib)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 16)
    return canvas


def render_synthetic_camera_image(
    points: np.ndarray,
    calib: KittiCalib,
) -> np.ndarray:
    """Build a grayscale camera-like image by projecting LiDAR intensity."""
    w, h = default_camera_canvas_size(calib)
    intensity_map = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64)
    uv, valid = project_velo_points(pts[:, :3], calib)
    for idx in np.where(valid)[0]:
        xi = int(round(uv[idx, 0]))
        yi = int(round(uv[idx, 1]))
        if 0 <= xi < w and 0 <= yi < h:
            val = float(pts[idx, 3]) if pts.shape[1] > 3 else 0.5
            intensity_map[yi, xi] = max(intensity_map[yi, xi], val)
    if intensity_map.max() > 0:
        intensity_u8 = np.clip(intensity_map / intensity_map.max() * 255.0, 0, 255).astype(np.uint8)
    else:
        intensity_u8 = intensity_map.astype(np.uint8)
    return cv2.cvtColor(intensity_u8, cv2.COLOR_GRAY2BGR)


def render_camera_overlay(
    image: np.ndarray,
    detections: Sequence[Any],
    calib: KittiCalib,
) -> np.ndarray:
    """Draw 3D detection wireframes on a camera image (no LiDAR scatter)."""
    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for det in detections:
        color = _det_color(det)
        corners = box_corners_velo(det)
        corners_uv, corner_valid = project_velo_points(corners, calib)
        corners_uv[~corner_valid] = np.nan
        if not _camera_box_is_visible(corners_uv, output.shape):
            continue
        if not corner_valid.any():
            continue
        _draw_box_wireframe(output, corners_uv, color, thickness=2)

        # Emphasise ground-contact face for readability
        bottom = corners_uv[:4]
        if np.isfinite(bottom).all():
            pts = np.round(bottom).astype(np.int32)
            cv2.polylines(output, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

        name = getattr(det, "class_name", "Obj")
        score = float(getattr(det, "confidence", 0.0))
        valid_uv = corners_uv[np.all(np.isfinite(corners_uv), axis=1)]
        if valid_uv.size >= 2:
            lx = int(np.clip(valid_uv[:, 0].min(), 0, output.shape[1] - 1))
            ly = int(np.clip(valid_uv[:, 1].min() - 4, 15, output.shape[0] - 1))
            label = f"{name} {score:.2f}"
            cv2.putText(output, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def _side_view_project(
    points_xyz: np.ndarray,
    width: int,
    height: int,
    *,
    margin: int = 28,
) -> np.ndarray:
    """Map velodyne XYZ to side-view pixels (x forward, z up)."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    usable_w = max(width - 2 * margin, 1)
    usable_h = max(height - 2 * margin, 1)
    px = margin + (pts[:, 0] - _X_MIN) / max(_X_MAX - _X_MIN, 1e-3) * usable_w
    py = margin + (1.0 - (pts[:, 2] - _Z_MIN) / max(_Z_MAX - _Z_MIN, 1e-3)) * usable_h
    return np.stack([px, py], axis=1)


def render_pointcloud_3d_view(
    points: np.ndarray,
    detections: Sequence[Any],
    *,
    width: int = 608,
    height: int = 608,
    max_points: int = 30000,
) -> np.ndarray:
    """Render a KITTI-style side view (X forward, Z up) with 3D box wireframes."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)

    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return canvas

    mask = (
        (pts[:, 0] >= _X_MIN) & (pts[:, 0] < _X_MAX)
        & (pts[:, 1] >= _Y_MIN) & (pts[:, 1] < _Y_MAX)
        & (pts[:, 2] >= _Z_MIN) & (pts[:, 2] < _Z_MAX)
    )
    pts = pts[mask]
    if pts.shape[0] > max_points:
        step = max(1, pts.shape[0] // max_points)
        pts = pts[::step]

    px = _side_view_project(pts[:, :3], width, height)
    y_vals = pts[:, 1]
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    span = max(y_max - y_min, 1e-3)
    for (xi, yi), y in zip(px, y_vals):
        ix, iy = int(round(xi)), int(round(yi))
        if 0 <= ix < width and 0 <= iy < height:
            t = (float(y) - y_min) / span
            color = (int(255 * (1 - t)), int(80 + 120 * t), int(255 * t))
            cv2.circle(canvas, (ix, iy), 1, color, -1, cv2.LINE_AA)

    for det in detections:
        corners = box_corners_velo(det)
        corner_px = _side_view_project(corners, width, height)
        _draw_box_wireframe(canvas, corner_px, _det_color(det), thickness=2)

    # Ground line reference
    ground_y = int(round(
        28 + (1.0 - (0.0 - _Z_MIN) / max(_Z_MAX - _Z_MIN, 1e-3)) * max(height - 56, 1)))
    cv2.line(canvas, (28, ground_y), (width - 28, ground_y), (60, 60, 80), 1, cv2.LINE_AA)
    return canvas


def compose_panels(
    panels: Sequence[Tuple[np.ndarray, str]],
    *,
    panel_width: int = 608,
    panel_height: int = 608,
    gap: int = 4,
) -> np.ndarray:
    """Compose equal-sized panels in a single horizontal row."""
    if not panels:
        raise ValueError("panels must not be empty")

    tiles: List[np.ndarray] = []
    for img, label in panels:
        tile = _fit_panel(img, panel_width, panel_height)
        cv2.putText(tile, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    if len(tiles) == 1:
        return tiles[0]

    sep_v = np.full((panel_height, gap, 3), 32, dtype=np.uint8)
    out = tiles[0]
    for tile in tiles[1:]:
        out = np.hstack([out, sep_v, tile])
    return out


def load_camera_image_bundle(
    source_bin: str,
    points: np.ndarray,
    calib: KittiCalib,
    image_path: str | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (raw_image_panel, overlay_base_image)."""
    raw = None
    if image_path is not None:
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw is None:
        raw = render_synthetic_camera_image(points, calib)
    overlay_base = raw.copy()
    return raw, overlay_base
