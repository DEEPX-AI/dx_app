"""
SuperPoint Keypoint Detection & Description Postprocessor

Model: superpoint
Input:  UINT8 [1, 480, 640, 1] — grayscale
Outputs:
  semi: FLOAT [1, 65, 60, 80]
  desc: FLOAT [1, 256, 60, 80]

Reference: Magic Leap SuperPoint (DeTone & Malisiewicz, 2018)
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from ..base import IPostprocessor, PreprocessContext


@dataclass
class SuperPointResult:
    """Result from SuperPoint keypoint detection."""
    keypoints: List[Tuple[float, float]]
    scores: List[float]
    descriptors: np.ndarray


class SuperPointPostprocessor(IPostprocessor):
    """Postprocessor for SuperPoint homographic adaptation keypoint detector."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.conf_threshold = float(self.config.get("conf_threshold", 0.015))
        self.top_k = int(self.config.get("top_k", 500))
        self.nms_dist = int(self.config.get("nms_dist", 4))
        self._cell = 8
        # One full cell, not the reference implementation's 4 px. The model
        # collapses each border cell's 64-way softmax onto its first sub-pixel,
        # so the last cell row/column carries a ridge ~7x the image mean while
        # the 7 sub-rows behind it are dead. A 4 px margin trims only the dead
        # part and leaves the ridge, which was filling 15-22% of the top-k with
        # points along the image edge rather than real corners.
        #
        # The trailing/right edges lose nothing real: behind the ridge the model
        # emits no candidates at all. The leading/top-left rows 1..7 do carry
        # signal, but recovering them is a net loss because top_k is a fixed
        # budget -- measured at top_k=500, border_remove of 1/2/4 admitted 72-93
        # edge points and evicted the same number of better interior ones,
        # taking corner-response quality from 10.8x a random-position control
        # down to 9.1x and 5.0x on two scenes. The cost of keeping one cell is a
        # 24x18 px band (5.8% of a 1920x1080 frame); that is accepted for
        # visualisation. Genuine edge coverage needs overlapping tiles, not a
        # smaller margin.
        self.border_remove = int(self.config.get("border_remove", self._cell))

    def _nms_fast(self, pts: np.ndarray, H: int, W: int, dist_thresh: int):
        """
        Fast approximate NMS on 3×N corners array [x, y, conf].
        Iterates from highest to lowest confidence; suppresses an (2d+1)×(2d+1)
        neighbourhood around each kept point (infinity-norm distance d).
        Returns surviving corners (3×N) and their original indices, ordered by
        descending confidence.

        Ties are broken deterministically: ``pts`` arrives in row-major order
        (``np.where`` over the heatmap), so a stable sort orders equal scores by
        ascending y then x -- the same rule the C++ postprocessor applies, which
        keeps the two implementations byte-comparable.
        """
        inds1 = np.argsort(-pts[2, :], kind="stable")
        corners = pts[:, inds1]
        n = corners.shape[1]

        if n == 0:
            return np.zeros((3, 0)), np.zeros(0, dtype=int)
        if n == 1:
            out = np.vstack((corners[:2, :].round().astype(int), corners[2])).reshape(3, 1)
            return out, np.zeros(1, dtype=int)

        rx = np.clip(corners[0, :].round().astype(np.intp), 0, W - 1)
        ry = np.clip(corners[1, :].round().astype(np.intp), 0, H - 1)

        pad = dist_thresh
        # Occupancy grid, pre-padded so a suppression box never needs clipping.
        # int8 keeps the (2d+1)² zero-fill cheap. Cells: 0 = empty/suppressed,
        # 1 = candidate alive, -1 = kept.
        grid = np.zeros((H + 2 * pad, W + 2 * pad), dtype=np.int8)
        # Candidates come from np.where() over the heatmap, so every (x, y) is
        # unique -- this scatter is exactly equivalent to a per-point loop.
        grid[ry + pad, rx + pad] = 1

        # The greedy pass is order-dependent and cannot be vectorised, so keep
        # its body free of numpy scalar calls: index Python lists, not arrays.
        gx = (rx + pad).tolist()
        gy = (ry + pad).tolist()
        rxl = rx.tolist()
        ryl = ry.tolist()

        # The caller applies border removal then top-k straight after this NMS,
        # and both this scan and the output are ordered by descending
        # confidence. So once top_k border-passing points are kept, no later
        # (lower-confidence) candidate can still reach the output -- stop there.
        bord = self.border_remove
        top_k = self.top_k
        hi_x, hi_y = W - bord, H - bord

        keep = []
        n_inside = 0
        for i in range(n):
            y = gy[i]
            x = gx[i]
            if grid[y, x] == 1:
                # grid is pre-padded, and (rx, ry) are clipped to the image, so
                # y is in [pad, H-1+pad]: the box spans [y-pad, y+pad] inclusive,
                # which is always inside the (H+2*pad, W+2*pad) grid.
                grid[y - pad:y + pad + 1, x - pad:x + pad + 1] = 0
                grid[y, x] = -1
                keep.append(i)
                if bord <= rxl[i] < hi_x and bord <= ryl[i] < hi_y:
                    n_inside += 1
                    if top_k > 0 and n_inside >= top_k:
                        break

        inds_keep = np.asarray(keep, dtype=np.intp)
        out = corners[:, inds_keep]
        inds2 = np.argsort(-out[-1, :], kind="stable")
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def _sample_desc_bilinear(self, desc: np.ndarray, pts: np.ndarray,
                              H: int, W: int) -> np.ndarray:
        """
        Bilinear interpolation of descriptor map at keypoint locations.
        Equivalent to torch.nn.functional.grid_sample with align_corners=False.

        Args:
            desc: (256, Hc, Wc) coarse descriptor map
            pts:  (2, N) keypoint positions [x, y] in full-res pixel space
            H, W: full image height and width
        Returns:
            (256, N) interpolated descriptors (not yet L2-normalised)
        """
        D, Hc, Wc = desc.shape

        # Map pixel coords → normalised [-1, 1]
        sx = pts[0, :] / (W * 0.5) - 1.0
        sy = pts[1, :] / (H * 0.5) - 1.0

        # Map normalised → descriptor-map coords [0, Wc-1] x [0, Hc-1]
        xd = (sx + 1.0) * 0.5 * (Wc - 1)
        yd = (sy + 1.0) * 0.5 * (Hc - 1)

        x0 = np.clip(np.floor(xd).astype(int), 0, Wc - 1)
        y0 = np.clip(np.floor(yd).astype(int), 0, Hc - 1)
        x1 = np.clip(x0 + 1, 0, Wc - 1)
        y1 = np.clip(y0 + 1, 0, Hc - 1)

        wa = ((x1 - xd) * (y1 - yd))[np.newaxis, :]
        wb = ((x1 - xd) * (yd - y0))[np.newaxis, :]
        wc = ((xd - x0) * (y1 - yd))[np.newaxis, :]
        wd = ((xd - x0) * (yd - y0))[np.newaxis, :]

        return (wa * desc[:, y0, x0] +
                wb * desc[:, y1, x0] +
                wc * desc[:, y0, x1] +
                wd * desc[:, y1, x1])

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext):
        semi, desc = None, None
        for output in outputs:
            if output.shape[1] == 65:
                semi = output
            elif output.shape[1] == 256:
                desc = output

        if semi is None or desc is None:
            return [SuperPointResult(keypoints=[], scores=[], descriptors=np.zeros((0, 256), np.float32))]

        semi = np.squeeze(semi, 0)   # (65, Hc, Wc)
        desc = np.squeeze(desc, 0)   # (256, Hc, Wc)
        hc, wc = semi.shape[1], semi.shape[2]
        H = hc * self._cell
        W = wc * self._cell

        # Heatmap: softmax with numerical stability (max subtraction), dustbin excluded
        nodust = semi[:-1, :, :]
        nodust_exp = np.exp(nodust - nodust.max(axis=0, keepdims=True))
        probs = nodust_exp / (nodust_exp.sum(axis=0, keepdims=True) + 1e-6)
        probs = probs.transpose(1, 2, 0)
        probs = probs.reshape(hc, wc, self._cell, self._cell)
        heatmap = probs.transpose(0, 2, 1, 3).reshape(H, W)

        # Confidence threshold → pts: 3×N [x, y, conf]
        row_ids, col_ids = np.where(heatmap >= self.conf_threshold)
        if len(row_ids) == 0:
            return [SuperPointResult(keypoints=[], scores=[], descriptors=np.zeros((0, 256), np.float32))]

        pts = np.zeros((3, len(row_ids)))
        pts[0, :] = col_ids               # x = column index
        pts[1, :] = row_ids               # y = row index
        pts[2, :] = heatmap[row_ids, col_ids]

        # NMS (grid-based, dist_thresh=4)
        pts, _ = self._nms_fast(pts, H, W, dist_thresh=self.nms_dist)

        # _nms_fast already returns points ordered by descending confidence with
        # a deterministic tie-break; a stable re-sort preserves that order.
        pts = pts[:, np.argsort(-pts[2, :], kind="stable")]

        # Border removal (4-px margin)
        bord = self.border_remove
        to_remove = (
            (pts[0, :] < bord) | (pts[0, :] >= W - bord) |
            (pts[1, :] < bord) | (pts[1, :] >= H - bord)
        )
        pts = pts[:, ~to_remove]

        # Top-K
        if self.top_k > 0 and pts.shape[1] > self.top_k:
            pts = pts[:, :self.top_k]

        if pts.shape[1] == 0:
            return [SuperPointResult(keypoints=[], scores=[], descriptors=np.zeros((0, 256), np.float32))]

        # Bilinear descriptor interpolation + L2 normalisation
        desc_interp = self._sample_desc_bilinear(desc, pts[:2, :], H, W)
        desc_norm = desc_interp / (np.linalg.norm(desc_interp, axis=0, keepdims=True) + 1e-6)
        descriptors = desc_norm.T.astype(np.float32)   # (N, 256)

        scale_x = ctx.original_width / W
        scale_y = ctx.original_height / H
        keypoints = [(float(x) * scale_x, float(y) * scale_y)
                     for x, y in zip(pts[0, :], pts[1, :])]
        scores = pts[2, :].tolist()

        return [SuperPointResult(keypoints=keypoints, scores=scores, descriptors=descriptors)]

    def get_model_name(self) -> str:
        return "superpoint"
