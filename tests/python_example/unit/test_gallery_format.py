# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The gallery file format, and the one place its two readers could drift apart.

A retrieval gallery is read by BOTH example trees -- numpy in
``common/processors/gallery_format.py`` and forty lines of C++ in
``common/processors/gallery_retrieval_postprocessor.hpp``. There is one file and one
format precisely so they cannot rank against different data, and these tests hold the
format still: a silent change to the header, the field order or the endianness would
leave the Python side working and the C++ side reading noise.

Measured agreement on the real galleries, DX-RT 3.5.0, five variants: identical ranking
order, identical gallery name and size, worst absolute score difference 4.47e-07
(float32 rounding). That end-to-end check lives in the project's parity run; what is
pinned here is the format itself.
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.processors.gallery_format import (  # noqa: E402
    MAGIC, read_gallery, write_gallery,
)

CPP_READER = (PROJECT_ROOT / "src" / "cpp_example" / "common" / "processors"
              / "gallery_retrieval_postprocessor.hpp")


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_round_trip_preserves_everything(tmp_path):
    rng = np.random.default_rng(7)
    vectors = _unit(rng.normal(size=(5, 12)).astype(np.float32))
    paths = [f"sample/img/{i}.jpg" for i in range(5)]
    labels = ["a", "b", "", "d", "e"]
    out = write_gallery(tmp_path / "g.bin", vectors, paths, labels, "my-model")

    got = read_gallery(out)
    assert np.array_equal(got["embeddings"], vectors)   # float32, bit-for-bit
    assert got["paths"] == paths
    assert got["labels"] == labels
    assert got["model"] == "my-model"
    assert got["dim"] == 12


def test_header_is_the_documented_little_endian_layout(tmp_path):
    """The C++ reader hardcodes this layout, so it is a contract, not an internal."""
    vectors = _unit(np.ones((3, 4), dtype=np.float32))
    out = write_gallery(tmp_path / "g.bin", vectors, ["a", "b", "c"], ["", "", ""], "m")
    raw = out.read_bytes()
    assert raw[:8] == MAGIC == b"DXGAL1\0\0"
    n, dim = struct.unpack_from("<II", raw, 8)
    assert (n, dim) == (3, 4)
    # embeddings start immediately after the 16-byte header
    body = np.frombuffer(raw, dtype="<f4", count=12, offset=16).reshape(3, 4)
    assert np.array_equal(body, vectors)


def test_all_blank_labels_round_trip_as_blanks(tmp_path):
    """The unlabelled case: '\\n'.join of empty strings must not collapse to one entry."""
    vectors = _unit(np.eye(4, dtype=np.float32))
    out = write_gallery(tmp_path / "g.bin", vectors, list("abcd"), [""] * 4, "m")
    got = read_gallery(out)
    assert got["labels"] == ["", "", "", ""]
    assert len(got["paths"]) == 4


def test_newline_in_a_path_is_refused_at_write_time(tmp_path):
    """Paths are newline-joined, so a newline would silently split one entry into two."""
    with pytest.raises(ValueError, match="newline"):
        write_gallery(tmp_path / "g.bin", np.eye(2, dtype=np.float32),
                      ["ok.jpg", "bad\nname.jpg"], ["", ""], "m")


def test_length_mismatch_is_refused_at_write_time(tmp_path):
    with pytest.raises(ValueError, match="2 embeddings but"):
        write_gallery(tmp_path / "g.bin", np.eye(2, dtype=np.float32),
                      ["only-one.jpg"], ["", ""], "m")


@pytest.mark.parametrize("cut", [4, 8, 15, 20])
def test_truncation_is_detected_rather_than_read_as_a_smaller_gallery(tmp_path, cut):
    good = write_gallery(tmp_path / "g.bin", _unit(np.eye(6, dtype=np.float32)),
                         [f"{i}.jpg" for i in range(6)], [""] * 6, "m")
    raw = good.read_bytes()
    bad = tmp_path / "bad.bin"
    bad.write_bytes(raw[:cut])
    with pytest.raises(ValueError):
        read_gallery(bad)


def test_foreign_file_is_rejected_by_magic(tmp_path):
    p = tmp_path / "g.bin"
    p.write_bytes(b"PK\x03\x04 this is a zip, i.e. the old .npz")
    with pytest.raises(ValueError, match="not a DXGAL1 gallery"):
        read_gallery(p)


def test_the_cpp_reader_declares_the_same_magic_and_field_order():
    """A cheap guard against the two readers drifting: they must agree on the header.

    Not a substitute for the end-to-end parity run, but it fails fast and locally if
    someone renames a field or changes the magic on one side only.
    """
    assert CPP_READER.is_file(), CPP_READER
    src = CPP_READER.read_text(encoding="utf-8")
    assert 'std::memcmp(magic, "DXGAL1\\0", 7)' in src, "C++ magic check changed"
    # The three trailing blobs must be READ in this order. Matched on the readBlob
    # calls, not on first mention: the locals are declared before they are filled.
    calls = re.findall(r"readBlob\(fh,\s*(?:g\.)?(\w+)\)", src)
    assert calls == ["model", "paths_blob", "labels_blob"], (
        f"C++ reads the trailing blobs as {calls}")
    assert "readU32" in src and "<< 8" in src, "C++ reader is no longer little-endian"
