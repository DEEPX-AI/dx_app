"""
SFA3D BEV (Bird's Eye View) Preprocessor

Converts KITTI-format LiDAR point cloud (.bin) files into a 3-channel
BEV (Bird's Eye View) image tensor suitable for SFA3D model input.

BEV encoding channels:
  0 — intensity (max projection)
  1 — height (max projection, normalised)
  2 — density (log-normalised point count)

Reference: SFA3D / Complex-YOLO BEV map conventions.
"""

import numpy as np
from typing import Tuple

from ..base import IPreprocessor, PreprocessContext

# KITTI front-view crop defaults (metres)
_X_MIN, _X_MAX = 0.0, 50.0
_Y_MIN, _Y_MAX = -25.0, 25.0
_Z_MIN, _Z_MAX = -2.5, 1.0

# Discretisation — match model input resolution
_BEV_HEIGHT = 608
_BEV_WIDTH = 608


def pointcloud_to_bev(
    pointcloud: np.ndarray,
    input_height: int = _BEV_HEIGHT,
    input_width: int = _BEV_WIDTH,
    *,
    x_range: Tuple[float, float] = (_X_MIN, _X_MAX),
    y_range: Tuple[float, float] = (_Y_MIN, _Y_MAX),
    z_range: Tuple[float, float] = (_Z_MIN, _Z_MAX),
) -> np.ndarray:
    """Convert an Nx4 point cloud to a 3-channel BEV uint8 image (HWC).

    Args:
        pointcloud: (N, >=4) float32 array [x, y, z, intensity, ...].
        input_height / input_width: output spatial resolution.
        x_range / y_range / z_range: crop limits in metres.

    Returns:
        (input_height, input_width, 3) uint8 BEV image (BGR-like for cv2).
    """
    if pointcloud.ndim != 2 or pointcloud.shape[1] < 4:
        raise ValueError(
            f"Expected Nx4+ point cloud, got shape {pointcloud.shape}")

    x, y, z, intensity = (
        pointcloud[:, 0], pointcloud[:, 1],
        pointcloud[:, 2], pointcloud[:, 3],
    )

    # Crop to region of interest
    mask = (
        (x >= x_range[0]) & (x < x_range[1]) &
        (y >= y_range[0]) & (y < y_range[1]) &
        (z >= z_range[0]) & (z < z_range[1])
    )
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    if len(x) == 0:
        return np.zeros((input_height, input_width, 3), dtype=np.uint8)

    # Discretise into grid cells
    x_res = (x_range[1] - x_range[0]) / input_height
    y_res = (y_range[1] - y_range[0]) / input_width

    # Map to pixel coordinates (row = x-forward, col = y-lateral, flipped)
    row = np.clip(
        ((x_range[1] - x) / x_res).astype(np.int32), 0, input_height - 1)
    col = np.clip(
        ((y - y_range[0]) / y_res).astype(np.int32), 0, input_width - 1)

    # Allocate BEV channels
    intensity_map = np.zeros((input_height, input_width), dtype=np.float32)
    height_map = np.full((input_height, input_width), z_range[0], dtype=np.float32)
    density_map = np.zeros((input_height, input_width), dtype=np.float32)

    # Max-pool intensity and height; accumulate density
    for i in range(len(x)):
        r, c = row[i], col[i]
        if intensity[i] > intensity_map[r, c]:
            intensity_map[r, c] = intensity[i]
        if z[i] > height_map[r, c]:
            height_map[r, c] = z[i]
        density_map[r, c] += 1.0

    # Normalise to [0, 255]
    int_max = intensity_map.max()
    if int_max > 0:
        intensity_map = (intensity_map / int_max * 255).astype(np.uint8)
    else:
        intensity_map = intensity_map.astype(np.uint8)

    h_range = z_range[1] - z_range[0]
    height_map = np.clip(
        ((height_map - z_range[0]) / h_range * 255), 0, 255
    ).astype(np.uint8)

    dens_max = density_map.max()
    if dens_max > 0:
        density_map = np.clip(
            np.log1p(density_map) / np.log1p(dens_max) * 255, 0, 255
        ).astype(np.uint8)
    else:
        density_map = density_map.astype(np.uint8)

    bev = np.stack([intensity_map, height_map, density_map], axis=-1)
    return bev


def load_kitti_pointcloud(bin_path: str) -> np.ndarray:
    """Load a KITTI-format .bin point cloud file.

    Returns:
        (N, 4) float32 array of [x, y, z, intensity].
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


class SFA3DBEVPreprocessor(IPreprocessor):
    """Preprocessor that converts a point-cloud BEV image to model input."""

    def __init__(self, input_width: int, input_height: int):
        self._input_width = input_width
        self._input_height = input_height

    def process(self, input_image: np.ndarray) -> Tuple[np.ndarray, PreprocessContext]:
        """Process a BEV image (already HxWx3 uint8) into model input tensor.

        If the caller loaded a .bin and converted via ``pointcloud_to_bev``,
        ``input_image`` is already the BEV map. If an ordinary BGR image is
        provided (e.g. a pre-rendered BEV saved as PNG), we use it as-is after
        resizing.
        """
        import cv2

        h, w = input_image.shape[:2]
        ctx = PreprocessContext(
            original_width=w,
            original_height=h,
            input_width=self._input_width,
            input_height=self._input_height,
            scale_x=self._input_width / max(w, 1),
            scale_y=self._input_height / max(h, 1),
        )

        if h != self._input_height or w != self._input_width:
            resized = cv2.resize(
                input_image,
                (self._input_width, self._input_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = input_image

        # SFA3D dxnn expects raw uint8 BEV pixels (0-255), not float-normalised input.
        if resized.dtype != np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
        tensor = resized
        return tensor, ctx

    def get_input_width(self) -> int:
        return self._input_width

    def get_input_height(self) -> int:
        return self._input_height
