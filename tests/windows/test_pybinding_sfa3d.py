"""SFA3D dx_postprocess pybinding tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if sys.platform == "win32":
    _DXRT_DIR = os.environ.get("DEEPX_SDK_DIR", r"C:\Program Files\DEEPX\DXNN\sdk")
    _DLL_SEARCH_DIRS = [
        os.path.join(_DXRT_DIR, "csharp"),
        os.path.join(_DXRT_DIR, "bin"),
        str(PROJECT_ROOT / "bin" / "Release"),
        str(PROJECT_ROOT / "lib"),
    ]
    for _dll_dir in _DLL_SEARCH_DIRS:
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)


def _try_import_dx_postprocess():
    try:
        import dx_postprocess
        return dx_postprocess
    except ImportError:
        return None


dx_postprocess = _try_import_dx_postprocess()

skip_if_no_binding = pytest.mark.skipif(
    dx_postprocess is None,
    reason="dx_postprocess not installed (run build first)",
)


@skip_if_no_binding
class TestSFA3DPostProcess:
    @pytest.mark.pybinding
    def test_constructor_and_getters(self):
        proc = dx_postprocess.SFA3DPostProcess(608, 608)
        assert proc.get_input_width() == 608
        assert proc.get_input_height() == 608
        assert proc.get_conf_threshold() == pytest.approx(0.3)
        assert proc.get_topk() == 50

    @pytest.mark.pybinding
    def test_zero_outputs_return_empty_n_by_nine(self):
        proc = dx_postprocess.SFA3DPostProcess(608, 608)
        outputs = [
            np.zeros((1, 3, 152, 152), np.float32),
            np.zeros((1, 2, 152, 152), np.float32),
            np.zeros((1, 2, 152, 152), np.float32),
            np.zeros((1, 1, 152, 152), np.float32),
            np.zeros((1, 3, 152, 152), np.float32),
        ]

        result = proc.postprocess(outputs)

        assert result.shape == (0, 9)

    @pytest.mark.pybinding
    def test_negative_logit_peak_decodes(self):
        proc = dx_postprocess.SFA3DPostProcess(608, 608)
        outputs = [
            np.full((1, 3, 152, 152), -10.0, np.float32),
            np.zeros((1, 2, 152, 152), np.float32),
            np.zeros((1, 2, 152, 152), np.float32),
            np.zeros((1, 1, 152, 152), np.float32),
            np.zeros((1, 3, 152, 152), np.float32),
        ]
        y, x = 10, 20
        outputs[0][0, 2, y, x] = -0.5
        outputs[4][0, 1, y, x] = 8.0
        outputs[4][0, 2, y, x] = 16.0

        result = proc.postprocess(outputs)

        assert result.shape == (1, 9)
        np.testing.assert_allclose(result[0, :2], [82.0, 42.0], atol=1e-4)
        np.testing.assert_allclose(result[0, 4:6], [8.0, 16.0], atol=1e-4)
        assert result[0, 7] == pytest.approx(1.0 / (1.0 + np.exp(0.5)), rel=1e-6)
        assert result[0, 8] == pytest.approx(2.0)
