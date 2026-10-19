# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Rank a query descriptor against a prebuilt gallery: retrieval, VPR and ReID.

An embedding model on its own cannot answer "which place is this?". It returns a
descriptor; the answer needs a *gallery* of known descriptors to rank against. The
zoo's Image Retrieval, Visual Place Recognition and Person ReID categories are the
same operation over different galleries, so they share this class.

The gallery is built ONCE, on the NPU, by ``scripts/build_gallery_database.py`` and
committed as a ``.bin`` in the single format :mod:`common.processors.gallery_format`
defines -- the same file the C++ tree reads, so the two cannot rank against different
data. At runtime this is pure numpy: L2-normalise the query, one ``(N, D) @ (D,)``
product, argsort.

Two refusals, both because the failure is otherwise invisible -- a mismatched gallery
does not error, it just ranks wrongly and returns confident numbers:

``embedding_dim``  a gallery of a different width cannot be multiplied at all.
``model``          a gallery built by a *different* model is a different embedding
                   space. The product still computes. Set
                   ``allow_model_mismatch`` to compare two spaces deliberately.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..base import GalleryMatch, IPostprocessor, PreprocessContext, RetrievalResult
from .gallery_format import read_gallery


def _repo_root() -> Path:
    """The dx_app checkout, found by the directory that holds ``sample/``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "sample").is_dir() and (parent / "config").is_dir():
            return parent
    return Path.cwd()


def resolve_gallery_path(raw: str) -> Path:
    """An absolute path, a cwd-relative path, or a repo-relative one -- in that order."""
    p = Path(raw)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return _repo_root() / p


class GalleryRetrievalPostprocessor(IPostprocessor):
    """Descriptor -> cosine top-k against a committed gallery."""

    def __init__(self, input_width: int = 224, input_height: int = 224,
                 config: dict = None, model_type: str = "retrieval_embedding",
                 model_name: str = "gallery_retrieval"):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self._model_type = model_type
        self._model_name = model_name
        self.top_k = int(self.config.get("top_k", 5))
        self.gallery_path = self.config.get("gallery")
        self.allow_model_mismatch = bool(
            self.config.get("allow_model_mismatch", False))
        self._gallery: dict | None = None
        self._load_failed: str | None = None

    # --------------------------------------------------------------- gallery
    def _load(self) -> dict | None:
        """The gallery, or ``None`` when none is configured/present.

        A missing gallery is NOT fatal: the descriptor is still correct and worth
        returning, and the visualizer reports the absence. A *corrupt* or mismatched
        gallery is fatal, because that is a wrong answer rather than a missing one.
        """
        if self._gallery is not None or self._load_failed is not None:
            return self._gallery
        if not self.gallery_path:
            self._load_failed = "no gallery configured"
            return None

        path = resolve_gallery_path(str(self.gallery_path))
        if not path.is_file():
            self._load_failed = f"gallery not found: {path}"
            return None

        raw = read_gallery(path)
        vectors = np.asarray(raw["embeddings"], dtype=np.float32)
        paths, labels, model = raw["paths"], raw["labels"], raw["model"]

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._gallery = {
            "vectors": vectors / np.maximum(norms, 1e-12),
            "paths": paths,
            "labels": labels,
            "model": model,
            "name": path.name,
        }
        return self._gallery

    def _check_space(self, gallery: dict, dim: int) -> None:
        gdim = gallery["vectors"].shape[1]
        if gdim != dim:
            raise ValueError(
                f"[DXAPP] [ERROR] {type(self).__name__} - gallery "
                f"'{gallery['name']}' is {gdim}-d and this model emits {dim}-d. "
                f"A gallery belongs to the model that built it; rebuild it with:\n"
                f"    python scripts/build_gallery_database.py --model {self._model_name}"
            )
        model = gallery["model"]
        if model and model != self._model_name and not self.allow_model_mismatch:
            raise ValueError(
                f"[DXAPP] [ERROR] {type(self).__name__} - gallery "
                f"'{gallery['name']}' was built by '{model}' but this is "
                f"'{self._model_name}'. Two encoders of the same width still occupy "
                f"different embedding spaces, so the ranking would be meaningless "
                f"while still looking plausible. Rebuild the gallery for this model, "
                f"or set config allow_model_mismatch=true to compare deliberately."
            )

    # --------------------------------------------------------------- process
    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[RetrievalResult]:
        out = outputs[0]
        if out.ndim == 3:            # [1, seq, D] -> pooled last position
            out = out[0, -1, :]
        query = out.flatten().astype(np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 1e-8:
            query = query / norm

        gallery = self._load()
        matches: List[GalleryMatch] = []
        size, name = 0, ""
        if gallery is not None:
            self._check_space(gallery, query.size)
            scores = gallery["vectors"] @ query
            k = min(self.top_k, scores.size)
            # argsort ascending then reverse: ties keep a stable, reproducible order.
            order = np.argsort(-scores, kind="stable")[:k]
            matches = [
                GalleryMatch(rank=i + 1, score=float(scores[j]),
                             path=gallery["paths"][j], label=gallery["labels"][j])
                for i, j in enumerate(order)
            ]
            size, name = len(gallery["paths"]), gallery["name"]

        return [RetrievalResult(embedding=query, model_type=self._model_type,
                                matches=matches, gallery_size=size,
                                gallery_name=name)]

    def get_model_name(self) -> str:
        return self._model_name

    @property
    def gallery_unavailable(self) -> str | None:
        """Why no gallery is loaded, for the visualizer to display."""
        self._load()
        return self._load_failed
