"""
SuperPoint Keypoint Visualizer with inter-frame sparse optical flow tracking.

Ports the PointTracker + draw_tracks from the original Magic Leap SuperPoint
demo (DeTone & Malisiewicz, 2018) to work with our SuperPointResult format.
"""

import numpy as np
import cv2
from typing import List, Optional

from ..base import IVisualizer


# Jet colormap (10 colour stops, RGB normalised) — same as original demo
_MYJET = np.array([
    [0., 0., 0.5],
    [0., 0., 0.99910873],
    [0., 0.37843137, 1.],
    [0., 0.83333333, 1.],
    [0.30044276, 1., 0.66729918],
    [0.66729918, 1., 0.30044276],
    [1., 0.90123457, 0.],
    [1., 0.48002905, 0.],
    [0.99910873, 0.07334786, 0.],
    [0.5, 0., 0.],
])


class PointTracker:
    """Manages sparse keypoint tracks across frames via descriptor matching.

    Directly ported from the original Magic Leap SuperPoint demo:
      https://github.com/magicleap/SuperPointPretrainedNetwork

    Internally stores a tracks matrix of shape (M, 2+L) where:
      tracks[m] = [track_id, avg_desc_distance, pt_id_frame_0, ..., pt_id_frame_{L-1}]
    """

    def __init__(self, max_length: int = 5, nn_thresh: float = 0.7,
                 max_pixel_dist: float = 100.0):
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.max_pixel_dist = max_pixel_dist
        # all_pts[i]: (2, N_i) xy-coords of keypoints in frame i
        self.all_pts: List[np.ndarray] = [np.zeros((2, 0)) for _ in range(max_length)]
        self.last_desc: Optional[np.ndarray] = None  # (256, N_last)
        self.tracks: np.ndarray = np.zeros((0, max_length + 2))
        self.track_count: int = 0
        self.max_score: float = 9999.0

    def nn_match_two_way(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """Two-way nearest-neighbour descriptor matching.

        Args:
            desc1: (D, N1) unit-normalised descriptors
            desc2: (D, N2) unit-normalised descriptors
        Returns:
            (3, M) array of [idx1, idx2, l2_distance]
        """
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2.0 - 2.0 * np.clip(dmat, -1.0, 1.0))
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        keep = scores < self.nn_thresh
        idx2 = np.argmin(dmat, axis=0)
        keep = keep & (np.arange(len(idx)) == idx2[idx])
        m_idx1 = np.where(keep)[0]
        m_idx2 = idx[keep]
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores[keep]
        return matches

    def get_offsets(self) -> np.ndarray:
        sizes = [pts.shape[1] for pts in self.all_pts[:-1]]
        return np.cumsum([0] + sizes)

    def update(self, pts: np.ndarray, desc: np.ndarray) -> None:
        """Add a new frame's observations and update track links.

        Args:
            pts:  (3, N) array [x, y, confidence]
            desc: (256, N) unit-normalised descriptor matrix
        """
        if pts is None or desc is None:
            return
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))

        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts[:2, :].copy())

        # Shift track columns left (drop oldest)
        self.tracks = np.delete(self.tracks, 2, axis=1)
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1

        offsets = self.get_offsets()
        self.tracks = np.hstack([self.tracks, -np.ones((self.tracks.shape[0], 1))])

        matched = np.zeros(pts.shape[1], dtype=bool)
        matches = self.nn_match_two_way(self.last_desc, desc)
        prev_pts = self.all_pts[-2] if self.all_pts[-2].shape[1] > 0 else None
        for match in matches.T:
            # Skip match if pixel distance is too large
            if prev_pts is not None:
                px, py = prev_pts[:, int(match[0])]
                cx, cy = pts[0, int(match[1])], pts[1, int(match[1])]
                if (cx - px) ** 2 + (cy - py) ** 2 > self.max_pixel_dist ** 2:
                    continue
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    self.tracks[row, 1] = match[2]
                else:
                    track_len = float((self.tracks[row, 2:] != -1).sum() - 1)
                    frac = 1.0 / track_len if track_len > 0 else 1.0
                    self.tracks[row, 1] = (1 - frac) * self.tracks[row, 1] + frac * match[2]

        new_ids = np.arange(pts.shape[1], dtype=float) + offsets[-1]
        new_ids = new_ids[~matched]
        if new_ids.size > 0:
            new_tracks = -np.ones((new_ids.size, self.maxl + 2))
            new_tracks[:, 0] = self.track_count + np.arange(new_ids.size)
            new_tracks[:, 1] = self.max_score
            new_tracks[:, -1] = new_ids
            self.tracks = np.vstack([self.tracks, new_tracks])
            self.track_count += int(new_ids.size)

        self.tracks = self.tracks[np.any(self.tracks[:, 2:] >= 0, axis=1), :]
        self.last_desc = desc.copy()

    def get_tracks(self, min_length: int) -> np.ndarray:
        """Return tracks with >= min_length observations that include the latest frame."""
        good_len = (self.tracks[:, 2:] != -1).sum(axis=1) >= min_length
        not_headless = self.tracks[:, -1] != -1
        return self.tracks[good_len & not_headless, :].copy()

    def draw_tracks(self, out: np.ndarray, tracks: np.ndarray) -> None:
        """Overlay coloured track lines on a BGR image (in-place).

        Track colour encodes matching quality via jet colormap (blue=bad, red=good).
        The most recent endpoint is drawn as a red dot.
        """
        N = len(self.all_pts)
        offsets = self.get_offsets()
        for track in tracks:
            clr_rgb = _MYJET[int(np.clip(track[1] * 10, 0, 9)), :] * 255
            color = (int(clr_rgb[2]), int(clr_rgb[1]), int(clr_rgb[0]))  # RGB → BGR
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                idx1 = int(track[i + 2] - offsets[i])
                idx2 = int(track[i + 3] - offsets[i + 1])
                if not (0 <= idx1 < self.all_pts[i].shape[1]):
                    continue
                if not (0 <= idx2 < self.all_pts[i + 1].shape[1]):
                    continue
                pt1 = self.all_pts[i][:, idx1]
                pt2 = self.all_pts[i + 1][:, idx2]
                p1 = (int(round(float(pt1[0]))), int(round(float(pt1[1]))))
                p2 = (int(round(float(pt2[0]))), int(round(float(pt2[1]))))
                cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
                if i == N - 2:
                    cv2.circle(out, p2, 2, (0, 0, 255), -1, cv2.LINE_AA)


class SuperPointVisualizer(IVisualizer):
    """SuperPoint keypoint visualizer with inter-frame sparse optical flow.

    Draws keypoints as confidence-scaled dots and overlays coloured track
    lines between matched keypoints across consecutive frames.
    """

    def __init__(
        self,
        radius: int = 2,
        color: tuple = (0, 255, 0),
        conf_threshold: float = 0.015,
        max_keypoints: int = 500,
        nn_thresh: float = 0.7,
        track_max_length: int = 5,
        min_track_length: int = 2,
    ):
        self.radius = radius
        self.color = color
        self.conf_threshold = conf_threshold
        self.max_keypoints = max_keypoints
        self.min_track_length = min_track_length
        self._tracker = PointTracker(max_length=track_max_length, nn_thresh=nn_thresh)

    def visualize(self, image: np.ndarray, results: list) -> np.ndarray:
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        h, w = output.shape[:2]

        # Scale radius to maintain consistent visual appearance across image sizes.
        # Reference: 960×540 (diagonal ≈ 1100 px). Images smaller than the
        # reference get a proportionally smaller radius so circles don't appear
        # oversized when the display window scales the image up.
        _ref_diag = (960**2 + 540**2) ** 0.5
        _img_diag = (w**2 + h**2) ** 0.5
        _r_scale = min(1.0, _img_diag / _ref_diag)
        _adj_radius = max(1, round(self.radius * _r_scale))
        _adj_max = max(_adj_radius, round((self.radius + 2) * _r_scale))

        for result in results:
            kpts = result.keypoints
            scores = result.scores
            descs = result.descriptors  # (N, 256)

            if len(scores) > 0:
                order = np.argsort(scores)[::-1][: self.max_keypoints]
            else:
                order = []

            # Draw keypoints as confidence-scaled dots
            for idx in order:
                score = scores[idx]
                if score < self.conf_threshold:
                    continue
                x, y = kpts[idx]
                cx, cy = int(x), int(y)
                if 0 <= cx < w and 0 <= cy < h:
                    r = max(1, min(int(_adj_radius * score * 3), _adj_max))
                    cv2.circle(output, (cx, cy), r, self.color, -1, cv2.LINE_AA)

            # Update tracker and draw inter-frame tracks
            if len(kpts) > 0 and descs is not None and descs.shape[0] > 0:
                pts_np = np.array(
                    [[x for x, y in kpts], [y for x, y in kpts], list(scores)],
                    dtype=np.float32,
                )  # (3, N)
                desc_np = descs.T.astype(np.float32)  # (256, N)

                self._tracker.update(pts_np, desc_np)
                tracks = self._tracker.get_tracks(self.min_track_length)
                if tracks.shape[0] > 0:
                    # Normalise score to [0, 1] for jet colormap
                    tracks[:, 1] /= max(float(self._tracker.nn_thresh), 1e-6)
                    self._tracker.draw_tracks(output, tracks)

        total = sum(len(r.keypoints) for r in results)
        cv2.putText(
            output,
            f"Keypoints: {total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        return output
