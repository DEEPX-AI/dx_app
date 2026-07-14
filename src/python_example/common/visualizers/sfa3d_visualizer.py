"""
SFA3D 3D Detection Visualizer

Horizontal mosaic when LiDAR input is used:
  BEV | Image | Cam+Box
"""

from typing import Any, List, Optional

import cv2
import numpy as np

from ..base import IVisualizer
from ..utility.kitti_calib import find_calib_path, find_image_path, load_kitti_calib
from ..utility.lidar_input import is_lidar_input
from ..utility.sfa3d_geometry import (
    compose_panels,
    draw_bev_boxes,
    load_camera_image_bundle,
    render_camera_overlay,
)


class SFA3DVisualizer(IVisualizer):
    """Visualise 3D detections on BEV and camera views."""

    def __init__(self, class_names: List[str] = None):
        self._class_names = class_names or ["Pedestrian", "Car", "Cyclist"]
        self._source_path: Optional[str] = None
        self._panel_size = 608

    def set_source_path(self, path: Optional[str]) -> None:
        self._source_path = path

    def visualize(self, frame: np.ndarray, results: List[Any]) -> np.ndarray:
        bev = self._prepare_bev_frame(frame)
        # Display-only mirror: model BEV keeps +y on the right; flip raster before overlays.
        bev = cv2.flip(bev, 1)
        bev = draw_bev_boxes(
            bev, results,
            width=self._panel_size, height=self._panel_size,
            class_names=self._class_names,
        )

        source = self._source_path
        if not source or not is_lidar_input(source):
            return compose_panels([(bev, "BEV")])

        from ..processors.sfa3d_bev_preprocessor import load_kitti_pointcloud

        points = load_kitti_pointcloud(source)
        calib_path = find_calib_path(source)
        if calib_path is None:
            placeholder = np.zeros((self._panel_size, self._panel_size, 3), dtype=np.uint8)
            placeholder[:] = (24, 24, 28)
            cv2.putText(placeholder, "No calib", (10, self._panel_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            return compose_panels([
                (bev, "BEV"),
                (placeholder, "Image"),
                (placeholder, "Cam+Box"),
            ])

        calib = load_kitti_calib(calib_path)
        image_path = find_image_path(source)
        raw_img, overlay_base = load_camera_image_bundle(
            source, points, calib, str(image_path) if image_path else None)
        cam_overlay = render_camera_overlay(overlay_base, results, calib)

        return compose_panels([
            (bev, "BEV"),
            (raw_img, "Image"),
            (cam_overlay, "Cam+Box"),
        ])

    def _prepare_bev_frame(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        if output.dtype != np.uint8:
            if output.max() <= 1.0:
                output = (output * 255).clip(0, 255).astype(np.uint8)
            else:
                output = output.clip(0, 255).astype(np.uint8)
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        return output
