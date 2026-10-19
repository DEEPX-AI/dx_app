# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Query beside its top-k gallery hits -- image retrieval, VPR and person ReID.

Modelled on the panel ``gmberton/VPR-methods-evaluation`` produces for its toy set:
the query on the left, the ranked database images to its right, each captioned with
its cosine similarity. That layout is the only way to *see* whether a retrieval model
works; a bare descriptor tells the viewer nothing.

Unlike :class:`EmbeddingVisualizer` this renders on the FIRST frame, because the
comparison set is the committed gallery rather than a previous frame. So these tasks
produce an output image from a single ``--image`` run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from ..base import IVisualizer
from ..processors.gallery_retrieval_postprocessor import resolve_gallery_path

_BG = (24, 24, 28)
_FG = (235, 235, 235)
_DIM = (150, 150, 155)
_ACCENT = (80, 200, 120)      # best match
_QUERY = (90, 170, 245)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class RetrievalVisualizer(IVisualizer):
    """``RetrievalResult`` -> a query/top-k contact sheet."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.max_display = int(cfg.get("max_display", 3))
        self.tile = int(cfg.get("tile_size", 260))
        self.title = str(cfg.get("title", "Retrieval"))
        self._missing: set[str] = set()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _fit(img: np.ndarray, box: int) -> np.ndarray:
        """Letterbox into a square tile, so aspect ratio is never misrepresented."""
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.full((box, box, 3), _BG, np.uint8)
        s = min(box / w, box / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((box, box, 3), _BG, np.uint8)
        y, x = (box - nh) // 2, (box - nw) // 2
        canvas[y:y + nh, x:x + nw] = small
        return canvas

    def _load(self, path: str, box: int) -> np.ndarray:
        """A gallery tile, or a labelled placeholder when the file is gone.

        The gallery stores paths, not pixels, so a moved sample directory must show up
        as a visible gap rather than a crash mid-render.
        """
        img = cv2.imread(str(resolve_gallery_path(path)), cv2.IMREAD_COLOR)
        if img is None:
            if path not in self._missing:
                self._missing.add(path)
            tile = np.full((box, box, 3), _BG, np.uint8)
            cv2.putText(tile, "missing", (10, box // 2), _FONT, 0.6, _DIM, 1,
                        cv2.LINE_AA)
            cv2.putText(tile, Path(path).name[:26], (10, box // 2 + 24), _FONT, 0.45,
                        _DIM, 1, cv2.LINE_AA)
            return tile
        return self._fit(img, box)

    @staticmethod
    def _frame(tile: np.ndarray, colour: tuple, width: int = 3) -> np.ndarray:
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), colour,
                      width)
        return tile

    # ----------------------------------------------------------- visualize
    def visualize(self, frame: np.ndarray, results: List[Any]) -> np.ndarray:
        if not results:
            return frame.copy()
        r = results[0]
        matches = list(getattr(r, "matches", []) or [])[:self.max_display]

        box = self.tile
        pad, head, cap, foot = 14, 40, 46, 30
        cols = 1 + max(1, len(matches))
        W = pad + cols * (box + pad)
        H = head + box + cap + foot
        canvas = np.full((H, W, 3), _BG, np.uint8)

        cv2.putText(canvas, self.title, (pad, 27), _FONT, 0.72, _FG, 2, cv2.LINE_AA)

        # -- query tile
        x = pad
        canvas[head:head + box, x:x + box] = self._frame(self._fit(frame, box), _QUERY)
        cv2.putText(canvas, "QUERY", (x + 4, head + box + 22), _FONT, 0.58, _QUERY, 2,
                    cv2.LINE_AA)
        dim = int(np.asarray(getattr(r, "embedding", [])).size)
        cv2.putText(canvas, f"{dim}-d descriptor", (x + 4, head + box + 40), _FONT,
                    0.46, _DIM, 1, cv2.LINE_AA)

        # -- ranked tiles
        if not matches:
            x = pad + box + pad
            tile = np.full((box, box, 3), _BG, np.uint8)
            cv2.putText(tile, "no gallery", (12, box // 2 - 10), _FONT, 0.7, _DIM, 2,
                        cv2.LINE_AA)
            for i, line in enumerate((
                    "build one with:", "scripts/build_gallery_", "database.py")):
                cv2.putText(tile, line, (12, box // 2 + 22 + i * 22), _FONT, 0.46,
                            _DIM, 1, cv2.LINE_AA)
            canvas[head:head + box, x:x + box] = self._frame(tile, _DIM, 1)
        else:
            for i, m in enumerate(matches):
                x = pad + (i + 1) * (box + pad)
                colour = _ACCENT if i == 0 else _FG
                canvas[head:head + box, x:x + box] = self._frame(
                    self._load(m.path, box), colour, 3 if i == 0 else 1)
                cv2.putText(canvas, f"#{m.rank}  cos {m.score:.4f}",
                            (x + 4, head + box + 22), _FONT, 0.56, colour, 2,
                            cv2.LINE_AA)
                sub = m.label or Path(m.path).name
                cv2.putText(canvas, sub[:30], (x + 4, head + box + 40), _FONT, 0.46,
                            _DIM, 1, cv2.LINE_AA)

        # -- footer: which gallery answered, so a stale one is visible in the picture
        gallery = getattr(r, "gallery_name", "") or "-"
        size = int(getattr(r, "gallery_size", 0))
        cv2.putText(canvas, f"gallery: {gallery}  ({size} images)  |  "
                            f"{getattr(r, 'model_type', '')}",
                    (pad, H - 10), _FONT, 0.46, _DIM, 1, cv2.LINE_AA)
        return canvas
