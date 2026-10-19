# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The on-disk retrieval gallery: ONE format, read by both example trees.

A gallery is a set of descriptors encoded once on the NPU, committed, and ranked
against at runtime. It is deliberately a single file in a single format rather than a
.npz for Python and something else for C++: two files mean two writers and two
readers, and the moment they disagree the symptom is not an error but a plausible,
wrong ranking. ``scripts/build_gallery_database.py`` writes it with
:func:`write_gallery`; :class:`GalleryRetrievalPostprocessor` and the C++
``loadRetrievalGallery`` in ``common/processors/gallery_retrieval_postprocessor.hpp``
read it.

The format is flat and little-endian so that forty lines of C++ suffice -- no zip, no
JSON, and float32 that round-trips bit-for-bit::

    char[8]        "DXGAL1\\0\\0"   magic + format version
    uint32         n              entries
    uint32         dim            descriptor width
    float32[n*dim] embeddings     row-major, L2-normalised
    uint32 + bytes model          producing .dxnn stem
    uint32 + bytes paths          n lines, '\\n'-joined, repo-relative
    uint32 + bytes labels         n lines, '\\n'-joined (may be blank)

``model`` is part of the payload, not a convention, because two encoders of the same
width still occupy different embedding spaces: without it a swapped gallery cannot be
detected at all.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

import numpy as np

MAGIC = b"DXGAL1\0\0"
HEADER = struct.Struct("<II")
LENGTH = struct.Struct("<I")


def write_gallery(path: str | Path, embeddings: np.ndarray, paths: Sequence[str],
                  labels: Sequence[str], model: str) -> Path:
    """Write one gallery. ``embeddings`` must be (n, dim) and L2-normalised."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be (n, dim), got {matrix.shape}")
    n, dim = matrix.shape
    if len(paths) != n or len(labels) != n:
        raise ValueError(
            f"{n} embeddings but {len(paths)} paths and {len(labels)} labels")
    if any("\n" in p for p in paths) or any("\n" in l for l in labels):
        raise ValueError("paths and labels are newline-joined, so they cannot "
                         "contain a newline")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(HEADER.pack(n, dim))
        fh.write(np.ascontiguousarray(matrix, dtype="<f4").tobytes())
        for blob in (model.encode("utf-8"),
                     "\n".join(paths).encode("utf-8"),
                     "\n".join(labels).encode("utf-8")):
            fh.write(LENGTH.pack(len(blob)))
            fh.write(blob)
    return out


def read_gallery(path: str | Path) -> dict:
    """``{embeddings, paths, labels, model, dim}``. Raises on a malformed file."""
    p = Path(path)
    raw = p.read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(
            f"[DXAPP] [ERROR] {p} is not a DXGAL1 gallery (magic {raw[:8]!r}). "
            f"Rebuild it with scripts/build_gallery_database.py.")
    if len(raw) < 8 + HEADER.size:
        raise ValueError(f"[DXAPP] [ERROR] {p} is truncated: {len(raw)} bytes, "
                         f"the header alone needs {8 + HEADER.size}")
    n, dim = HEADER.unpack_from(raw, 8)
    if n == 0 or dim == 0:
        raise ValueError(f"[DXAPP] [ERROR] {p} declares {n}x{dim} entries")

    offset = 8 + HEADER.size
    count = n * dim
    need = offset + count * 4
    if len(raw) < need:
        raise ValueError(f"[DXAPP] [ERROR] {p} is truncated: {len(raw)} bytes, "
                         f"embeddings alone need {need}")
    vectors = np.frombuffer(raw, dtype="<f4", count=count,
                            offset=offset).reshape(n, dim)
    offset = need

    blobs = []
    for field in ("model", "paths", "labels"):
        if len(raw) < offset + LENGTH.size:
            raise ValueError(f"[DXAPP] [ERROR] {p} is missing its {field} field")
        (length,) = LENGTH.unpack_from(raw, offset)
        offset += LENGTH.size
        if len(raw) < offset + length:
            raise ValueError(f"[DXAPP] [ERROR] {p} has a truncated {field} field")
        blobs.append(raw[offset:offset + length].decode("utf-8"))
        offset += length
    model, paths_blob, labels_blob = blobs

    paths = paths_blob.split("\n") if paths_blob else []
    labels = labels_blob.split("\n") if labels_blob else [""] * n
    if len(paths) != n:
        raise ValueError(f"[DXAPP] [ERROR] {p} declares {n} entries but carries "
                         f"{len(paths)} paths")
    labels = (labels + [""] * n)[:n]
    return {"embeddings": vectors, "paths": paths, "labels": labels,
            "model": model, "dim": int(dim)}
