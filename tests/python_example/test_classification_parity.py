"""
Python and C++ classification postprocessing agree on a real NPU output.

``bin/resnet50_224x224_sync`` classifies an image with ``--dump-tensors`` and
``DXAPP_VERIFY=1``: the dumped output tensor is exactly what its C++
EfficientNetPostprocessor turned into the verify JSON. The Python
ClassificationPostprocessor gets the same tensor and must give the same top
class, ImageNet name, probability and ranked top-k probabilities.

The logic itself is pinned without hardware by
``unit/test_classification_postprocessor.py``. (The two whole examples are not
compared end to end: the C++ classification factories letterbox the input
while the Python ones resize it, so they feed the NPU different pixels.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.proc import run_bounded  # noqa: E402
from test_helpers.constants import BIN_DIR, MODELS_DIR, PROJECT_ROOT  # noqa: E402
from test_helpers.utils import setup_environment  # noqa: E402

from common.base import PreprocessContext  # noqa: E402
from common.processors.classification_postprocessor import (  # noqa: E402
    ClassificationPostprocessor,
)

MODEL = MODELS_DIR / "resnet50_224x224.dxnn"
IMAGE = PROJECT_ROOT / "sample" / "img" / "sample_dog.jpg"
EXE = BIN_DIR / "resnet50_224x224_sync"

# C++ computes in float32, Python in float64.
CONF_TOL = 1e-6


@pytest.mark.e2e
@pytest.mark.verify
def test_python_postprocessor_matches_cpp_on_the_same_npu_output(tmp_path):
    if not MODEL.exists():
        pytest.skip("{} absent - run ./setup.sh --models resnet50_224x224".format(MODEL.name))
    if not EXE.exists():
        pytest.skip("Binary not found: {}".format(EXE.name))

    env = setup_environment()
    env["DXAPP_VERIFY"] = "1"
    env["DXAPP_VERIFY_DIR"] = str(tmp_path / "verify")
    cmd = [str(EXE), "-m", str(MODEL), "-i", str(IMAGE), "--no-display", "-l", "1",
           "--dump-tensors", "--save-dir", str(tmp_path / "run")]
    result = run_bounded(cmd, capture_output=True, text=True, timeout=180,
                         env=env, cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]

    tensors = list((tmp_path / "run").rglob("output_tensor_0.bin"))
    assert len(tensors) == 1, tensors
    logits = np.fromfile(str(tensors[0]), dtype=np.float32)
    assert logits.size == 1000
    cpp = json.loads((tmp_path / "verify" / "resnet50_224x224.json").read_text())

    py = ClassificationPostprocessor(config={"top_k": len(cpp["top_k_confs"])}).process(
        [logits.reshape(1, -1)], PreprocessContext())

    c = cpp["classifications"][0]
    assert (py[0].class_id, py[0].class_name) == (c["class_id"], c["class_name"])
    assert py[0].class_name, "no ImageNet name"
    assert py[0].confidence == pytest.approx(c["conf"], abs=CONF_TOL)
    assert [conf for _, conf in py[0].top_k] == pytest.approx(cpp["top_k_confs"], abs=CONF_TOL)
    print("\n  C++ {} {!r} {:.7f} | Python {} {!r} {:.7f}".format(
        c["class_id"], c["class_name"], c["conf"],
        py[0].class_id, py[0].class_name, py[0].confidence))
