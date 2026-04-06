"""
Embedding / Enhancement / Face Alignment / Hand Landmark Visualizers

Visualizers for new model types:
- EmbeddingVisualizer: CLIP, ArcFace (displays embedding info as text)
- SuperResolutionVisualizer: ESPCN (side-by-side low-res vs upscaled)
- EnhancementVisualizer: Zero-DCE (side-by-side dark vs enhanced)
- FaceAlignmentVisualizer: 3DDFA (draws 68-point landmarks + pose axes)
- HandLandmarkVisualizer: MediaPipe (draws 21-point hand skeleton)
"""

import cv2
import numpy as np
from typing import List, Any

from ..base import IVisualizer


class EmbeddingVisualizer(IVisualizer):
    """Visualizer for embedding models (CLIP, ArcFace)."""

    def visualize(self, image: np.ndarray, results: List[Any]) -> np.ndarray:
        output = image.copy()
        if not results:
            return output

        result = results[0]
        emb = result.embedding
        model_type = result.model_type

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        cv2.putText(output, f"Model: {model_type}", (10, y), font, 0.7, (0, 255, 0), 2)
        y += 30
        cv2.putText(output, f"Embedding dim: {len(emb)}", (10, y), font, 0.7, (0, 255, 0), 2)
        y += 30
        norm = np.linalg.norm(emb)
        cv2.putText(output, f"L2 norm: {norm:.4f}", (10, y), font, 0.7, (0, 255, 0), 2)
        y += 30
        # Show first few values
        vals = ", ".join([f"{v:.3f}" for v in emb[:5]])
        cv2.putText(output, f"Values: [{vals}, ...]", (10, y), font, 0.5, (200, 200, 200), 1)

        return output


class SuperResolutionVisualizer(IVisualizer):
    """Visualizer for super-resolution models (ESPCN)."""

    def visualize(self, image: np.ndarray, results: list) -> np.ndarray:
        result = results[0]
        sr_img = result.output_image

        if sr_img.ndim == 2:
            sr_bgr = cv2.cvtColor(sr_img, cv2.COLOR_GRAY2BGR)
        else:
            sr_bgr = sr_img

        # Side-by-side: input (upscaled to match SR) | SR output
        sr_h, sr_w = sr_bgr.shape[:2]
        input_upscaled = cv2.resize(image, (sr_w, sr_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.hstack([input_upscaled, sr_bgr])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Input (upscaled)", (10, 30), font, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, "Super-Resolved", (sr_w + 10, 30), font, 1.0, (0, 255, 0), 2)
        return canvas


class EnhancementVisualizer(IVisualizer):
    """Visualizer for image enhancement models (Zero-DCE)."""

    def visualize(self, image: np.ndarray, results: list) -> np.ndarray:
        result = results[0]
        enhanced = result.output_image
        h, w = image.shape[:2]

        if enhanced.ndim == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        enhanced_resized = cv2.resize(enhanced, (w, h))

        # Side-by-side: input | output
        canvas = np.hstack([image, enhanced_resized])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Input", (10, 30), font, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, "Enhanced", (w + 10, 30), font, 1.0, (0, 255, 0), 2)
        return canvas


class FaceAlignmentVisualizer(IVisualizer):
    """Visualizer for 3D face alignment (3DDFA)."""

    # 68-point face landmark connectivity for drawing
    _FACE_PARTS = {
        'contour': list(range(0, 17)),
        'left_eyebrow': list(range(17, 22)),
        'right_eyebrow': list(range(22, 27)),
        'nose_bridge': list(range(27, 31)),
        'nose_bottom': list(range(31, 36)),
        'left_eye': list(range(36, 42)) + [36],  # close the loop
        'right_eye': list(range(42, 48)) + [42],
        'outer_lip': list(range(48, 60)) + [48],
        'inner_lip': list(range(60, 68)) + [60],
    }

    def visualize(self, image: np.ndarray, results: List[Any]) -> np.ndarray:
        output = image.copy()
        if not results:
            return output

        result = results[0]
        lmks = result.landmarks_2d  # [68, 2]
        pose = result.pose  # [yaw, pitch, roll]

        # Draw landmarks
        colors = {
            'contour': (200, 200, 200),
            'left_eyebrow': (0, 255, 0),
            'right_eyebrow': (0, 255, 0),
            'nose_bridge': (255, 200, 0),
            'nose_bottom': (255, 200, 0),
            'left_eye': (255, 0, 0),
            'right_eye': (255, 0, 0),
            'outer_lip': (0, 0, 255),
            'inner_lip': (0, 100, 255),
        }

        for part_name, indices in self._FACE_PARTS.items():
            color = colors.get(part_name, (255, 255, 255))
            for i in range(len(indices) - 1):
                if indices[i] < len(lmks) and indices[i + 1] < len(lmks):
                    pt1 = (int(lmks[indices[i], 0]), int(lmks[indices[i], 1]))
                    pt2 = (int(lmks[indices[i + 1], 0]), int(lmks[indices[i + 1], 1]))
                    cv2.line(output, pt1, pt2, color, 1, cv2.LINE_AA)

        # Draw landmark points
        for i in range(min(68, len(lmks))):
            pt = (int(lmks[i, 0]), int(lmks[i, 1]))
            cv2.circle(output, pt, 2, (0, 255, 255), -1)

        # Display pose text
        if pose and len(pose) >= 3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(output,
                        f"Yaw: {pose[0]:.1f} Pitch: {pose[1]:.1f} Roll: {pose[2]:.1f}",
                        (10, 30), font, 0.6, (0, 255, 0), 2)

        return output


class HandLandmarkVisualizer(IVisualizer):
    """Visualizer for hand landmark models (MediaPipe HandLandmark)."""

    # MediaPipe hand connections: (start_idx, end_idx)
    _CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),             # Palm
    ]

    _FINGER_COLORS = {
        'thumb': (0, 255, 255),    # Yellow
        'index': (0, 255, 0),      # Green
        'middle': (255, 0, 0),     # Blue
        'ring': (0, 165, 255),     # Orange
        'pinky': (255, 0, 255),    # Magenta
        'palm': (200, 200, 200),   # Gray
    }

    def visualize(self, image: np.ndarray, results: List[Any]) -> np.ndarray:
        output = image.copy()
        if not results:
            return output

        for result in results:
            lmks = result.landmarks  # [21, 3]
            if len(lmks) == 0:
                continue

            # Draw connections
            for start, end in self._CONNECTIONS:
                if start < len(lmks) and end < len(lmks):
                    pt1 = (int(lmks[start, 0]), int(lmks[start, 1]))
                    pt2 = (int(lmks[end, 0]), int(lmks[end, 1]))
                    color = self._get_connection_color(start, end)
                    cv2.line(output, pt1, pt2, color, 2, cv2.LINE_AA)

            # Draw keypoints
            for i in range(min(21, len(lmks))):
                pt = (int(lmks[i, 0]), int(lmks[i, 1]))
                # Wrist is larger
                radius = 5 if i == 0 else 3
                cv2.circle(output, pt, radius, (0, 0, 255), -1)
                cv2.circle(output, pt, radius, (255, 255, 255), 1)

            # Show handedness
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = f"{result.handedness} ({result.confidence:.2f})"
            wrist = (int(lmks[0, 0]), int(lmks[0, 1]) - 15)
            cv2.putText(output, label, wrist, font, 0.6, (0, 255, 0), 2)

        return output

    def _get_connection_color(self, start, end):
        """Get color based on which finger the connection belongs to."""
        if start <= 4 and end <= 4:
            return self._FINGER_COLORS['thumb']
        elif 5 <= end <= 8 or (start == 0 and end == 5):
            return self._FINGER_COLORS['index']
        elif 9 <= end <= 12 or (start == 0 and end == 9):
            return self._FINGER_COLORS['middle']
        elif 13 <= end <= 16 or (start == 0 and end == 13):
            return self._FINGER_COLORS['ring']
        elif 17 <= end <= 20 or (start == 0 and end == 17):
            return self._FINGER_COLORS['pinky']
        else:
            return self._FINGER_COLORS['palm']
