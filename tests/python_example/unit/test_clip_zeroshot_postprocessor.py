# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""CLIP zero-shot classification against a precomputed prompt bank.

The zoo ships no usable text tower: ``clip-text_resnet50_77x512_openai`` emits the
text transformer's 77x512 HIDDEN STATES, not a joint embedding -- turning those into
one needs `ln_final` and `text_projection`, which are not in the .dxnn -- and it is an
OpenAI RN50 checkpoint, a different space from every image tower we ship. So text is
encoded ONCE at build time by ``scripts/build_clip_prompt_bank.py`` and shipped as
``prompt_bank.json``; at runtime this postprocessor does one (N,D) @ (D,) dot product
in numpy.

The bank and the model must come from the SAME checkpoint. A mismatch is not
detectable from the numbers -- cosine similarities stay in a plausible 0.1..0.4 band
and the ranking is simply wrong -- so the dimension check is the only guard that can
fire, and it must.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import CLIPZeroShotPostprocessor  # noqa: E402

DIM = 8


def _ctx() -> PreprocessContext:
    return PreprocessContext(original_width=640, original_height=480,
                             input_width=256, input_height=256, scale=1.0,
                             pad_x=0.0, pad_y=0.0)


def _bank(tmp_path: Path, labels, vectors, dim: int = DIM) -> str:
    path = tmp_path / "prompt_bank.json"
    path.write_text(json.dumps({
        "checkpoint": "test-checkpoint",
        "embed_dim": dim,
        "labels": list(labels),
        "prompts": [f"a photo of a {l}" for l in labels],
        "embeddings": [list(map(float, v)) for v in vectors],
    }), encoding="utf-8")
    return str(path)


def _onehot(i: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_the_nearest_prompt_wins(tmp_path):
    bank = _bank(tmp_path, ["dog", "horse", "kitchen"], [_onehot(0), _onehot(1), _onehot(2)])
    post = CLIPZeroShotPostprocessor(256, 256, {"prompt_bank": bank})
    results = post.process([_onehot(1).reshape(1, DIM)], _ctx())
    assert results[0].class_name == "horse"
    assert results[0].class_id == 1


def test_results_are_ordered_by_score_and_limited_to_top_k(tmp_path):
    # Three prompts at decreasing angle from the query, so the order is known.
    q = np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    near = np.array([0.9, 0.436, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    far = np.array([0.5, 0.866, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    bank = _bank(tmp_path, ["far", "near", "exact"], [far, near, q])
    post = CLIPZeroShotPostprocessor(256, 256, {"prompt_bank": bank, "top_k": 2})
    results = post.process([q.reshape(1, DIM)], _ctx())
    assert [r.class_name for r in results] == ["exact", "near"]
    assert results[0].confidence >= results[1].confidence


def test_probabilities_sum_to_one_over_the_whole_bank(tmp_path):
    bank = _bank(tmp_path, ["a", "b", "c", "d"], [_onehot(i) for i in range(4)])
    post = CLIPZeroShotPostprocessor(256, 256, {"prompt_bank": bank, "top_k": 4})
    results = post.process([_onehot(0).reshape(1, DIM)], _ctx())
    assert sum(r.confidence for r in results) == pytest.approx(1.0, abs=1e-5)


def test_a_bank_from_another_checkpoint_is_refused_by_its_dimension(tmp_path):
    """The failure this guard exists for is silent: plausible numbers, wrong ranking."""
    bank = _bank(tmp_path, ["dog"], [np.ones(4, dtype=np.float32) / 2.0], dim=4)
    post = CLIPZeroShotPostprocessor(256, 256, {"prompt_bank": bank})
    with pytest.raises(ValueError) as exc:
        post.process([np.ones((1, DIM), dtype=np.float32)], _ctx())
    message = str(exc.value)
    assert "4" in message and str(DIM) in message


def test_an_unnormalised_image_embedding_still_ranks_correctly(tmp_path):
    """The engine's raw output is not L2-normalised; the postprocessor must do it."""
    bank = _bank(tmp_path, ["dog", "horse"], [_onehot(0), _onehot(1)])
    post = CLIPZeroShotPostprocessor(256, 256, {"prompt_bank": bank})
    scaled = (_onehot(0) * 17.3).reshape(1, DIM)
    assert post.process([scaled], _ctx())[0].class_name == "dog"


def test_a_missing_bank_names_the_path(tmp_path):
    post = CLIPZeroShotPostprocessor(256, 256, {"prompt_bank": str(tmp_path / "nope.json")})
    with pytest.raises(FileNotFoundError) as exc:
        post.process([np.ones((1, DIM), dtype=np.float32)], _ctx())
    assert "nope.json" in str(exc.value)


def test_labels_are_exposed_for_the_visualizer(tmp_path):
    """ClassificationVisualizer indexes its own label list by class_id."""
    bank = _bank(tmp_path, ["dog", "horse"], [_onehot(0), _onehot(1)])
    post = CLIPZeroShotPostprocessor(256, 256, {"prompt_bank": bank})
    assert post.labels == ["dog", "horse"]
