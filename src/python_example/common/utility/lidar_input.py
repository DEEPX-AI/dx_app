"""LiDAR / KITTI point-cloud input helpers for SFA3D examples."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Bundled KITTI demo frames (velodyne/calib/image_2/label_2 share the same stem)
SFA3D_SAMPLE_FRAME_IDS = ("000049", "000535")

_LIDAR_EXTENSIONS = {".bin"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_lidar_input(path: str) -> bool:
    """Return True when *path* is a KITTI velodyne .bin file."""
    return Path(path).suffix.lower() in _LIDAR_EXTENSIONS


def is_bev_image_input(path: str) -> bool:
    """Return True when *path* is a raster image usable as a pre-rendered BEV."""
    return Path(path).suffix.lower() in _IMAGE_EXTENSIONS


def load_lidar_bev_frame(
    path: str,
    *,
    input_height: int = 608,
    input_width: int = 608,
) -> np.ndarray:
    """Load a KITTI .bin file and convert it to a 3-channel BEV image (HWC uint8)."""
    from ..processors.sfa3d_bev_preprocessor import (
        load_kitti_pointcloud,
        pointcloud_to_bev,
    )

    points = load_kitti_pointcloud(path)
    return pointcloud_to_bev(
        points,
        input_height=input_height,
        input_width=input_width,
    )


def load_display_frame(
    path: str,
    *,
    input_height: int = 608,
    input_width: int = 608,
) -> np.ndarray:
    """Load a display/preprocess frame from LiDAR (.bin) or image path."""
    if is_lidar_input(path):
        return load_lidar_bev_frame(
            path,
            input_height=input_height,
            input_width=input_width,
        )

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load input: {path}")
    return img


def collect_media_inputs(dir_path: str) -> List[str]:
    """Collect supported image and LiDAR inputs from a directory."""
    files: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp", "*.bin"):
        files.extend(glob.glob(os.path.join(dir_path, ext)))
        files.extend(glob.glob(os.path.join(dir_path, ext.upper())))
    return sorted(set(files))


def resolve_default_lidar_sample(start: Optional[Path] = None) -> Optional[str]:
    """Locate bundled sample/kitti/velodyne/*.bin relative to dx_app root."""
    preferred = f"{SFA3D_SAMPLE_FRAME_IDS[0]}.bin"
    if start is not None:
        for _ in range(8):
            sample_dir = start / "sample" / "kitti" / "velodyne"
            if sample_dir.is_dir():
                preferred_path = sample_dir / preferred
                if preferred_path.is_file():
                    return str(preferred_path)
                for frame_id in SFA3D_SAMPLE_FRAME_IDS:
                    candidate = sample_dir / f"{frame_id}.bin"
                    if candidate.is_file():
                        return str(candidate)
            start = start.parent
    return None
