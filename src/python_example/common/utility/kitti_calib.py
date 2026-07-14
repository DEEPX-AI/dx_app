"""KITTI calibration parsing and companion-path resolution for SFA3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class KittiCalib:
    """Rectified left-color camera calibration (P2) and velodyne extrinsics."""
    p2: np.ndarray          # 3x4
    r0_rect: np.ndarray     # 3x3
    tr_velo_to_cam: np.ndarray  # 3x4

    @property
    def velo_to_image(self) -> np.ndarray:
        """Combined 3x4 projection: velodyne homogeneous -> image homogeneous."""
        r0_4 = np.eye(4, dtype=np.float64)
        r0_4[:3, :3] = self.r0_rect
        tr_4 = np.eye(4, dtype=np.float64)
        tr_4[:3, :] = self.tr_velo_to_cam
        p_4 = np.eye(4, dtype=np.float64)
        p_4[:3, :] = self.p2
        return p_4 @ r0_4 @ tr_4


def _parse_matrix_line(line: str) -> np.ndarray:
    key, values = line.split(":", 1)
    del key
    nums = np.array([float(x) for x in values.split()], dtype=np.float64)
    return nums


def load_kitti_calib(path: str | Path) -> KittiCalib:
    """Load KITTI calib txt (P2, R0_rect, Tr_velo_to_cam)."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or ":" not in raw:
                continue
            key = raw.split(":", 1)[0].strip()
            data[key] = _parse_matrix_line(raw)

    if "P2" not in data or "R0_rect" not in data or "Tr_velo_to_cam" not in data:
        raise ValueError(f"Incomplete KITTI calib: {path}")

    return KittiCalib(
        p2=data["P2"].reshape(3, 4),
        r0_rect=data["R0_rect"].reshape(3, 3),
        tr_velo_to_cam=data["Tr_velo_to_cam"].reshape(3, 4),
    )


_KITTI_CALIB_DIR: Optional[Path] = None
_KITTI_IMAGE2_DIR: Optional[Path] = None


def configure_kitti_companion_dirs(
    *,
    calib_dir: str | Path | None = None,
    image2_dir: str | Path | None = None,
) -> None:
    """Override KITTI calib/image_2 lookup roots ({stem}.txt / {stem}.png)."""
    global _KITTI_CALIB_DIR, _KITTI_IMAGE2_DIR
    _KITTI_CALIB_DIR = Path(calib_dir) if calib_dir else None
    _KITTI_IMAGE2_DIR = Path(image2_dir) if image2_dir else None


def apply_kitti_companion_dirs_from_args(args) -> None:
    configure_kitti_companion_dirs(
        calib_dir=getattr(args, "calib_dir", None),
        image2_dir=getattr(args, "image2_dir", None),
    )


def candidate_calib_paths(bin_path: Path) -> List[Path]:
    """Return likely calib file paths for a velodyne .bin frame."""
    stem = bin_path.stem
    parent = bin_path.parent
    candidates: List[Path] = []
    if _KITTI_CALIB_DIR is not None:
        candidates.append(_KITTI_CALIB_DIR / f"{stem}.txt")
    candidates.extend([
        parent / f"{stem}.txt",
        parent.parent / "calib" / f"{stem}.txt",
        parent.parent / "calib" / f"{stem}.TXT",
    ])
    if parent.name == "velodyne":
        candidates.extend([
            parent.parent / "calib" / f"{stem}.txt",
            parent.parent / "calib" / f"{stem}.TXT",
        ])
    return candidates


def find_calib_path(bin_path: str | Path) -> Optional[Path]:
    path = Path(bin_path)
    for candidate in candidate_calib_paths(path):
        if candidate.is_file():
            return candidate
    return None


def candidate_image_paths(bin_path: Path) -> Iterable[Path]:
    """Return likely left-color image paths for a velodyne frame."""
    stem = bin_path.stem
    root = bin_path.parent.parent if bin_path.parent.name == "velodyne" else bin_path.parent
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    if _KITTI_IMAGE2_DIR is not None:
        for ext in exts:
            yield _KITTI_IMAGE2_DIR / f"{stem}{ext}"
    subdirs = ("image_2", "image_02", "image", "images")
    for sub in subdirs:
        folder = root / sub
        for ext in exts:
            yield folder / f"{stem}{ext}"


def find_image_path(bin_path: str | Path) -> Optional[Path]:
    path = Path(bin_path)
    for candidate in candidate_image_paths(path):
        if candidate.is_file():
            return candidate
    return None


def default_camera_canvas_size(calib: KittiCalib) -> tuple[int, int]:
    """Infer a reasonable camera canvas size from P2 intrinsics."""
    fx = float(calib.p2[0, 0])
    cx = float(calib.p2[0, 2])
    cy = float(calib.p2[1, 2])
    width = int(max(1242, round(cx * 2)))
    height = int(max(375, round(cy * 2)))
    return width, height
