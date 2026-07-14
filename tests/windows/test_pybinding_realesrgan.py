"""
RealESRGAN dx_postprocess pybinding tests.
"""
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
class TestRealESRGANPostProcess:
    @pytest.mark.pybinding
    def test_constructor_and_getters(self):
        proc = dx_postprocess.RealESRGANPostProcess(64, 32, 4)
        assert proc.get_input_width() == 64
        assert proc.get_input_height() == 32
        assert proc.get_scale_factor() == 4

    @pytest.mark.pybinding
    def test_postprocess_returns_hwc_and_clips(self):
        proc = dx_postprocess.RealESRGANPostProcess(2, 2, 2)
        output = np.array(
            [[[
                [-0.5, 0.25, 1.1, 0.75],
                [0.2, 0.4, 0.6, 0.8],
                [0.9, 1.2, -0.1, 0.0],
                ]]],
            dtype=np.float32,
        ).reshape(1, 3, 2, 2)
        result = proc.postprocess([output])

        assert result.shape == (2, 2, 3)
        expected = np.array(
            [
                [[0.0, 0.2, 0.9], [0.25, 0.4, 1.0]],
                [[1.0, 0.6, 0.0], [0.75, 0.8, 0.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-6)
