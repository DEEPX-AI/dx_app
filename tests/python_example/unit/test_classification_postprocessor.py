"""Python ClassificationPostprocessor matches the C++ EfficientNetPostprocessor.

C++ (src/cpp_example/common/processors/classification_postprocessor.hpp):
stable softmax over the logits, skipped when the output already is a
distribution (every value >= -1e-4 and |sum - 1| <= 1e-3); ImageNet class
names when num_classes == 1000; the first result carries the ranked top-k.
The on-NPU comparison of both on the same image is
tests/python_example/test_classification_parity.py.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.base import PreprocessContext  # noqa: E402
from common.processors.classification_postprocessor import (  # noqa: E402
    ClassificationPostprocessor,
)
from common.utility.labels import IMAGENET_1000  # noqa: E402


def _ctx():
    return PreprocessContext(original_width=224, original_height=224)


def _run(values, num_classes=1000, top_k=5):
    pp = ClassificationPostprocessor(config={"num_classes": num_classes, "top_k": top_k})
    return pp.process([np.asarray(values, dtype=np.float32).reshape(1, -1)], _ctx())


def _reference_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def test_logits_become_probabilities_in_logit_order():
    rng = np.random.default_rng(3)
    logits = rng.permutation(1000).astype(np.float32) / 50.0  # distinct values
    results = _run(logits, top_k=1000)

    confs = np.array([r.confidence for r in results])
    assert np.all((confs >= 0.0) & (confs <= 1.0))
    assert abs(confs.sum() - 1.0) < 1e-4
    assert [r.class_id for r in results] == list(np.argsort(-logits))
    ref = _reference_softmax(logits)
    for r in results[:5]:
        assert r.confidence == pytest.approx(ref[r.class_id], rel=1e-5)


def test_softmax_is_stable_for_large_logits():
    logits = np.zeros(1000, dtype=np.float32)
    logits[7] = 1000.0
    logits[8] = 999.0
    results = _run(logits)
    assert np.isfinite([r.confidence for r in results]).all()
    assert results[0].class_id == 7
    assert results[0].confidence == pytest.approx(1.0 / (1.0 + np.exp(-1.0)), rel=1e-5)


def test_a_distribution_is_not_softmaxed_again():
    dist = np.full(1000, 0.4 / 999, dtype=np.float32)
    dist[10] = 0.6
    results = _run(dist)
    assert results[0].class_id == 10
    assert results[0].confidence == pytest.approx(0.6, abs=1e-6)
    double = _reference_softmax(dist)[10]
    assert abs(results[0].confidence - double) > 1e-3


def test_distribution_tolerances_match_cpp():
    near = np.full(1000, 0.4 / 999, dtype=np.float32)
    near[10] = 0.6 + 5e-4          # |sum - 1| = 5e-4 <= 1e-3
    near[11] = -5e-5               # >= -1e-4
    assert _run(near)[0].confidence == pytest.approx(0.6 + 5e-4, abs=1e-6)

    off = near.copy()
    off[10] = 0.6 + 5e-3           # sum off by 5e-3: logits, softmax applies
    assert _run(off)[0].confidence == pytest.approx(_reference_softmax(off)[10], rel=1e-5)

    neg = near.copy()
    neg[11] = -1e-3                # clearly negative: logits
    assert _run(neg)[0].confidence == pytest.approx(_reference_softmax(neg)[10], rel=1e-5)


def test_imagenet_names_for_1000_classes():
    logits = np.zeros(1000, dtype=np.float32)
    logits[260] = 5.0
    logits[248] = 4.0
    results = _run(logits)
    assert (results[0].class_id, results[0].class_name) == (260, "Chow Chow")
    assert results[1].class_name == IMAGENET_1000[248]
    assert all(r.class_name == IMAGENET_1000[r.class_id] for r in results)


def test_no_names_for_other_class_counts():
    results = _run(np.arange(10, dtype=np.float32), num_classes=10)
    assert all(r.class_name == "" for r in results)


def test_single_value_output_is_a_class_id():
    pp = ClassificationPostprocessor(config={"num_classes": 1000})
    results = pp.process([np.array([260], dtype=np.float32)], _ctx())
    assert (results[0].class_id, results[0].confidence, results[0].class_name) == (260, 1.0, "Chow Chow")


def test_first_result_carries_top_k():
    logits = np.zeros(1000, dtype=np.float32)
    logits[[3, 7, 1, 9, 5, 2]] = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    results = _run(logits, top_k=5)
    assert len(results) == 5
    assert [cid for cid, _ in results[0].top_k] == [3, 7, 1, 9, 5]
    assert [conf for _, conf in results[0].top_k] == [r.confidence for r in results]
    assert all(r.top_k == [] for r in results[1:])


def test_python_imagenet_names_equal_cpp_names():
    """Both languages name classes from their own copy of the ImageNet list."""
    src = (_ROOT / "src/cpp_example/common/utility/labels.hpp").read_text()
    body = re.search(r"getImageNetClassName.*?\{(.*?)\};", src, re.S).group(1)
    cpp = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    assert cpp == IMAGENET_1000
