# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Ranking a descriptor against a committed gallery, and the two refusals.

These tests exist because the failure mode of a retrieval example is silence. A
gallery built by a different encoder, or at a different width, still multiplies: the
scores stay in a plausible 0.3-0.7 band and the ranking is simply wrong. Nothing
raises, nothing looks odd in the picture. So the postprocessor refuses both cases and
these tests pin the refusals as behaviour rather than as an implementation detail.

MEASURED context for the numbers below (DX-RT 3.5.0):

    eigenplaces-resnet18_512x512   out descriptor [1, 512], already unit norm
    clip-img_resnet50_224x224_openai out image_embedding [1, 1024]
    repvgg-a0-reid_256x128         out test_output [1, 512]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext, RetrievalResult  # noqa: E402
from common.processors import GalleryRetrievalPostprocessor  # noqa: E402
from common.processors.gallery_format import write_gallery  # noqa: E402


def _ctx() -> PreprocessContext:
    return PreprocessContext(original_width=640, original_height=480,
                             input_width=512, input_height=512, scale=1.0)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _gallery(tmp_path: Path, vectors: np.ndarray, model: str,
             labels: list[str] | None = None, name: str = "g.bin") -> Path:
    """A gallery written by the SAME writer the builder uses -- a test that hand-rolls
    the format would pass while the real file was unreadable."""
    paths = [f"sample/x/{i}.jpg" for i in range(len(vectors))]
    return write_gallery(tmp_path / name, vectors,
                         paths, labels if labels else [""] * len(paths), model)


def _post(gallery: Path | None, model_name: str = "m", **extra):
    cfg = {"top_k": 3, **extra}
    if gallery is not None:
        cfg["gallery"] = str(gallery)
    return GalleryRetrievalPostprocessor(512, 512, config=cfg,
                                         model_name=model_name)


# ----------------------------------------------------------------- ranking
def test_ranks_by_cosine_best_first(tmp_path):
    """The gallery row nearest the query comes first, and scores descend."""
    g = np.eye(4, dtype=np.float32)          # four orthogonal directions
    post = _post(_gallery(tmp_path, g, "m"))
    # A query leaning mostly on row 2, then row 0.
    q = np.array([[0.5, 0.0, 0.8, 0.1]], dtype=np.float32)
    out = post.process([q], _ctx())

    assert len(out) == 1 and isinstance(out[0], RetrievalResult)
    r = out[0]
    assert [m.rank for m in r.matches] == [1, 2, 3]
    assert r.matches[0].path.endswith("2.jpg")
    assert r.matches[1].path.endswith("0.jpg")
    scores = [m.score for m in r.matches]
    assert scores == sorted(scores, reverse=True)
    # cos against a unit basis vector is just the normalised component.
    assert r.matches[0].score == pytest.approx(0.8 / np.linalg.norm(q), abs=1e-6)


def test_query_is_l2_normalised_so_scores_are_cosines(tmp_path):
    """Scaling the query must not change the ranking or the scores."""
    g = _unit(np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32))
    post_a, post_b = _post(_gallery(tmp_path, g, "m")), _post(_gallery(tmp_path, g, "m"))
    q = np.random.default_rng(1).normal(size=(1, 8)).astype(np.float32)

    a = post_a.process([q], _ctx())[0]
    b = post_b.process([q * 37.0], _ctx())[0]
    assert [m.path for m in a.matches] == [m.path for m in b.matches]
    for x, y in zip(a.matches, b.matches):
        assert x.score == pytest.approx(y.score, abs=1e-6)
    assert np.linalg.norm(a.embedding) == pytest.approx(1.0, abs=1e-6)


def test_top_k_clamps_to_gallery_size(tmp_path):
    g = np.eye(2, dtype=np.float32)
    post = _post(_gallery(tmp_path, g, "m"))     # top_k=3, gallery of 2
    r = post.process([np.array([[1.0, 0.2]], dtype=np.float32)], _ctx())[0]
    assert len(r.matches) == 2
    assert r.gallery_size == 2


def test_labels_and_gallery_identity_travel_with_the_result(tmp_path):
    """The visualizer prints these; a stale gallery has to be visible in the output."""
    g = np.eye(3, dtype=np.float32)
    path = _gallery(tmp_path, g, "m", labels=["alpha", "beta", "gamma"],
                    name="places.bin")
    r = _post(path).process([np.array([[0.1, 1.0, 0.1]], np.float32)], _ctx())[0]
    assert r.matches[0].label == "beta"
    assert r.gallery_name == "places.bin"
    assert r.gallery_size == 3


def test_three_dimensional_output_is_pooled_not_flattened(tmp_path):
    """A [1, seq, D] tower must yield a D-wide descriptor, not seq*D."""
    g = np.eye(4, dtype=np.float32)
    post = _post(_gallery(tmp_path, g, "m"))
    seq = np.zeros((1, 7, 4), dtype=np.float32)
    seq[0, -1] = [0.0, 0.0, 1.0, 0.0]
    r = post.process([seq], _ctx())[0]
    assert r.embedding.size == 4
    assert r.matches[0].path.endswith("2.jpg")


# ----------------------------------------------------------------- refusals
def test_width_mismatch_is_refused(tmp_path):
    """A 512-d gallery cannot answer a 1024-d query -- and must not try."""
    post = _post(_gallery(tmp_path, np.eye(512, dtype=np.float32), "m"))
    with pytest.raises(ValueError, match="512-d"):
        post.process([np.ones((1, 1024), dtype=np.float32)], _ctx())


def test_foreign_encoder_is_refused_even_at_the_same_width(tmp_path):
    """Same width, different embedding space: the product computes and is meaningless.

    This is the dangerous case -- eigenplaces-resnet18, pp-shituv2 and repvgg-a0-reid
    all emit 512-d, so nothing about the shapes catches a swapped gallery.
    """
    g = _gallery(tmp_path, np.eye(512, dtype=np.float32),
                 "eigenplaces-resnet18_512x512")
    post = _post(g, model_name="repvgg-a0-reid_256x128")
    with pytest.raises(ValueError, match="different embedding spaces"):
        post.process([np.ones((1, 512), dtype=np.float32)], _ctx())


def test_foreign_encoder_allowed_when_asked_explicitly(tmp_path):
    g = _gallery(tmp_path, np.eye(4, dtype=np.float32), "some-other-model")
    post = _post(g, model_name="mine", allow_model_mismatch=True)
    r = post.process([np.array([[1.0, 0, 0, 0]], np.float32)], _ctx())[0]
    assert len(r.matches) == 3


# ------------------------------------------------- a missing gallery is not fatal
def test_no_gallery_configured_still_returns_the_descriptor(tmp_path):
    """The descriptor is correct on its own; only the comparison is unavailable."""
    r = _post(None).process([np.ones((1, 16), dtype=np.float32)], _ctx())[0]
    assert r.embedding.size == 16
    assert r.matches == []
    assert r.gallery_size == 0
    assert "no gallery configured" in _post(None).gallery_unavailable


def test_absent_gallery_file_reports_why_without_raising(tmp_path):
    post = _post(tmp_path / "not-built-yet.bin")
    r = post.process([np.ones((1, 16), dtype=np.float32)], _ctx())[0]
    assert r.matches == []
    assert "not-built-yet.bin" in post.gallery_unavailable


def test_corrupt_gallery_is_fatal(tmp_path):
    """A missing gallery is a missing answer; a malformed one is a WRONG answer."""
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not a gallery at all")
    with pytest.raises(ValueError, match="not a DXGAL1 gallery"):
        _post(bad).process([np.ones((1, 3), dtype=np.float32)], _ctx())


def test_truncated_gallery_is_fatal(tmp_path):
    """Half a descriptor matrix would otherwise be read as a smaller gallery."""
    good = _gallery(tmp_path, np.eye(4, dtype=np.float32), "m", name="cut.bin")
    raw = good.read_bytes()
    good.write_bytes(raw[:len(raw) // 2])
    with pytest.raises(ValueError, match="truncated"):
        _post(good).process([np.ones((1, 4), dtype=np.float32)], _ctx())
