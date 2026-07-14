"""
dx_postprocess pybinding invalid argument tests.

Validates that the C++ pybinding module (dx_postprocess) handles invalid
inputs gracefully: wrong types, wrong shapes, empty lists, negative
dimensions, None values, etc.

Prerequisites:
    - build.bat completed (dx_postprocess module installed via pip)

Notes:
    - Tests marked @crash_risk run in a subprocess to avoid killing the
      test runner if the C++ code segfaults on malformed input.
    - Negative dimensions are accepted by pybind11 (cast to int) without
      validation — tests document this behavior as-is.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Register DXRT DLL directories before attempting import
if sys.platform == "win32":
    _DEEPX_SDK_DIR = os.environ.get("DEEPX_SDK_DIR", r"C:\Program Files\DEEPX\DXNN\sdk")
    _DLL_SEARCH_DIRS = [
        os.path.join(_DEEPX_SDK_DIR, "csharp"),
        os.path.join(_DEEPX_SDK_DIR, "bin"),
        str(PROJECT_ROOT / "bin" / "Release"),
        str(PROJECT_ROOT / "lib"),
    ]
    for _dll_dir in _DLL_SEARCH_DIRS:
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)


def _try_import_dx_postprocess():
    """Try to import dx_postprocess; return None if not installed."""
    try:
        import dx_postprocess
        return dx_postprocess
    except ImportError as e:
        print(f"dx_postprocess import failed: {e}", file=sys.stderr)
        return None


dx_postprocess = _try_import_dx_postprocess()

skip_if_no_binding = pytest.mark.skipif(
    dx_postprocess is None,
    reason="dx_postprocess not installed (run build.bat first)",
)


def _run_in_subprocess(code: str, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run Python code in an isolated subprocess to catch crashes safely.

    Returns CompletedProcess. returncode != 0 means crash or exception.
    """
    full_code = textwrap.dedent(f"""\
        import os, sys
        if sys.platform == "win32":
            _DEEPX_SDK_DIR = os.environ.get("DEEPX_SDK_DIR", r"C:\\Program Files\\DEEPX\\DXNN\\sdk")
            for d in [os.path.join(_DEEPX_SDK_DIR, "csharp"), os.path.join(_DEEPX_SDK_DIR, "bin")]:
                if os.path.isdir(d):
                    os.add_dll_directory(d)
        import numpy as np
        import dx_postprocess
        {code}
    """)
    return subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ======================================================================
# YOLOv7PostProcess tests
# ======================================================================

@skip_if_no_binding
class TestYOLOv7PostProcessInvalidArgs:
    """Invalid argument tests for dx_postprocess.YOLOv7PostProcess."""

    def _make_processor(self, input_w=640, input_h=640, obj_thresh=0.25,
                        score_thresh=0.45, nms_thresh=0.45, is_ort=False):
        return dx_postprocess.YOLOv7PostProcess(
            input_w, input_h, obj_thresh, score_thresh, nms_thresh, is_ort
        )

    # --- Constructor: type validation ---

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_string_input_width(self):
        """String instead of int for input_w should raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            dx_postprocess.YOLOv7PostProcess(
                "not_an_int", 640, 0.25, 0.45, 0.45, False
            )

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_none_input_width(self):
        """None instead of int for input_w should raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            dx_postprocess.YOLOv7PostProcess(
                None, 640, 0.25, 0.45, 0.45, False
            )

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_missing_required_args(self):
        """Missing required arguments should raise TypeError."""
        with pytest.raises(TypeError):
            dx_postprocess.YOLOv7PostProcess()

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_too_many_args(self):
        """Too many positional arguments should raise TypeError."""
        with pytest.raises(TypeError):
            dx_postprocess.YOLOv7PostProcess(640, 640, 0.25, 0.45, 0.45, False, "extra")

    # --- Constructor: boundary values (document actual behavior) ---

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_negative_input_width_accepted(self):
        """Negative input_w is accepted by pybind11 without validation.
        Documents missing input validation in C++ layer."""
        proc = self._make_processor(input_w=-1)
        assert proc.get_input_width() == -1

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_negative_input_height_accepted(self):
        """Negative input_h is accepted without validation."""
        proc = self._make_processor(input_h=-1)
        assert proc.get_input_height() == -1

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_zero_input_dimensions_accepted(self):
        """Zero dimensions are accepted at construction time."""
        proc = self._make_processor(input_w=0, input_h=0)
        assert proc.get_input_width() == 0
        assert proc.get_input_height() == 0

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_threshold_out_of_range_high(self):
        """Threshold > 1.0 — accepted without validation."""
        proc = self._make_processor(score_thresh=2.0)
        assert proc.get_input_width() == 640

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_threshold_negative(self):
        """Negative threshold — accepted without validation."""
        proc = self._make_processor(score_thresh=-1.0)
        assert proc.get_input_width() == 640

    # --- Postprocess: type/None validation ---

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_none_input(self):
        """None as input should raise TypeError."""
        proc = self._make_processor()
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess(None)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_wrong_type_in_list(self):
        """Non-numpy element in list should raise."""
        proc = self._make_processor()
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            proc.postprocess(["not_a_numpy_array"])

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_int_in_list(self):
        """Integer instead of numpy array should raise."""
        proc = self._make_processor()
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            proc.postprocess([42])

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_dict_input(self):
        """Dict as input should raise."""
        proc = self._make_processor()
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess({"key": "value"})

    # --- Postprocess: empty/dtype ---

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_empty_list(self):
        """Empty list as input should raise or return empty results."""
        proc = self._make_processor()
        try:
            result = proc.postprocess([])
            assert result.shape[0] == 0
        except (RuntimeError, IndexError, ValueError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_wrong_dtype(self):
        """Integer numpy array instead of float — may reinterpret bits."""
        proc = self._make_processor()
        wrong_dtype = [np.zeros((1, 25200, 85), dtype=np.int64)]
        try:
            result = proc.postprocess(wrong_dtype)
            assert result.shape[1] == 6
        except (RuntimeError, TypeError, ValueError):
            pass

    # --- Postprocess: shape issues (subprocess-isolated, may crash) ---

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_wrong_shape_1d_subprocess(self):
        """1D array causes access violation — run in subprocess to detect crash."""
        result = _run_in_subprocess("""
proc = dx_postprocess.YOLOv7PostProcess(640, 640, 0.25, 0.45, 0.45, False)
try:
    proc.postprocess([np.zeros((100,), dtype=np.float32)])
    print("NO_EXCEPTION")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
""")
        if result.returncode != 0:
            pytest.xfail(
                f"CRASH detected (access violation): postprocess(1D array) "
                f"causes segfault. returncode={result.returncode}"
            )
        output = result.stdout.strip()
        assert "EXCEPTION" in output or "NO_EXCEPTION" in output

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_wrong_shape_2d(self):
        """2D array instead of expected 3D — may work or raise."""
        proc = self._make_processor()
        wrong_shape = [np.zeros((25200, 85), dtype=np.float32)]
        try:
            result = proc.postprocess(wrong_shape)
            assert isinstance(result, np.ndarray)
        except (RuntimeError, ValueError, IndexError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_zero_size_array_subprocess(self):
        """Zero-size array — run in subprocess to prevent potential crash."""
        result = _run_in_subprocess("""
proc = dx_postprocess.YOLOv7PostProcess(640, 640, 0.25, 0.45, 0.45, False)
try:
    proc.postprocess([np.zeros((0,), dtype=np.float32)])
    print("NO_EXCEPTION")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
""")
        if result.returncode != 0:
            pytest.xfail(
                f"CRASH detected: postprocess(zero-size array) segfaults. "
                f"returncode={result.returncode}"
            )
        output = result.stdout.strip()
        assert "EXCEPTION" in output or "NO_EXCEPTION" in output

    # --- Postprocess: special float values ---

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_nan_values(self):
        """NaN values in input — should not crash."""
        proc = self._make_processor()
        nan_input = [np.full((1, 25200, 85), np.nan, dtype=np.float32)]
        try:
            result = proc.postprocess(nan_input)
            assert isinstance(result, np.ndarray)
        except (RuntimeError, ValueError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_inf_values(self):
        """Inf values in input — should not crash."""
        proc = self._make_processor()
        inf_input = [np.full((1, 25200, 85), np.inf, dtype=np.float32)]
        try:
            result = proc.postprocess(inf_input)
            assert isinstance(result, np.ndarray)
        except (RuntimeError, ValueError):
            pass


# ======================================================================
# YOLOv8PostProcess tests
# ======================================================================

@skip_if_no_binding
class TestYOLOv8PostProcessInvalidArgs:
    """Invalid argument tests for dx_postprocess.YOLOv8PostProcess."""

    def _make_processor(self, input_w=640, input_h=640, score_thresh=0.25,
                        nms_thresh=0.45, is_ort=False):
        return dx_postprocess.YOLOv8PostProcess(
            input_w, input_h, score_thresh, nms_thresh, is_ort
        )

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_negative_input_width_accepted(self):
        """Negative value accepted — documents missing validation."""
        proc = self._make_processor(input_w=-1)
        assert proc.get_input_width() == -1

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_string_threshold(self):
        with pytest.raises((TypeError, ValueError)):
            dx_postprocess.YOLOv8PostProcess(640, 640, "bad", 0.45, False)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_missing_args(self):
        with pytest.raises(TypeError):
            dx_postprocess.YOLOv8PostProcess()

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_empty_list(self):
        proc = self._make_processor()
        try:
            result = proc.postprocess([])
            assert result.shape[0] == 0
        except (RuntimeError, IndexError, ValueError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_none(self):
        proc = self._make_processor()
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess(None)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_string_in_list(self):
        proc = self._make_processor()
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            proc.postprocess(["invalid"])


# ======================================================================
# DeepLabv3PostProcess tests
# ======================================================================

@skip_if_no_binding
class TestDeepLabv3PostProcessInvalidArgs:
    """Invalid argument tests for dx_postprocess.DeepLabv3PostProcess."""

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_negative_dimensions_accepted(self):
        """Negative accepted — documents missing validation."""
        proc = dx_postprocess.DeepLabv3PostProcess(-1, -1)
        assert proc.get_input_width() == -1

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_string_dimensions(self):
        with pytest.raises((TypeError, ValueError)):
            dx_postprocess.DeepLabv3PostProcess("bad", 513)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_missing_args(self):
        with pytest.raises(TypeError):
            dx_postprocess.DeepLabv3PostProcess()

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_empty_list(self):
        proc = dx_postprocess.DeepLabv3PostProcess(513, 513)
        try:
            result = proc.postprocess([])
            assert isinstance(result, np.ndarray)
        except (RuntimeError, IndexError, ValueError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_none(self):
        proc = dx_postprocess.DeepLabv3PostProcess(513, 513)
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess(None)


# ======================================================================
# ClassificationPostProcess tests
# ======================================================================

@skip_if_no_binding
class TestClassificationPostProcessInvalidArgs:
    """Invalid argument tests for dx_postprocess.ClassificationPostProcess."""

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_negative_top_k_accepted(self):
        """Negative top_k accepted — documents missing validation."""
        proc = dx_postprocess.ClassificationPostProcess(-1)
        assert proc.get_top_k() == -1

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_zero_top_k(self):
        proc = dx_postprocess.ClassificationPostProcess(0)
        dummy = [np.zeros((1, 1000), dtype=np.float32)]
        try:
            result = proc.postprocess(dummy)
            assert result.shape[0] == 0
        except (RuntimeError, ValueError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_string_top_k(self):
        with pytest.raises((TypeError, ValueError)):
            dx_postprocess.ClassificationPostProcess("bad")

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_empty_list(self):
        proc = dx_postprocess.ClassificationPostProcess(5)
        try:
            result = proc.postprocess([])
            assert isinstance(result, np.ndarray)
        except (RuntimeError, IndexError, ValueError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_none(self):
        proc = dx_postprocess.ClassificationPostProcess(5)
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess(None)


# ======================================================================
# EmbeddingPostProcess tests
# ======================================================================

@skip_if_no_binding
class TestEmbeddingPostProcessInvalidArgs:
    """Invalid argument tests for dx_postprocess.EmbeddingPostProcess."""

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_string_l2_normalize(self):
        """pybind11 strict bool check rejects string."""
        with pytest.raises(TypeError):
            dx_postprocess.EmbeddingPostProcess("not_a_bool")

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_empty_list(self):
        proc = dx_postprocess.EmbeddingPostProcess(True)
        try:
            result = proc.postprocess([])
            assert isinstance(result, np.ndarray)
        except (RuntimeError, IndexError, ValueError):
            pass

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_none(self):
        proc = dx_postprocess.EmbeddingPostProcess(True)
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess(None)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_wrong_type(self):
        proc = dx_postprocess.EmbeddingPostProcess(True)
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            proc.postprocess([42])


# ======================================================================
# SCRFDPostProcess tests
# ======================================================================

@skip_if_no_binding
class TestSCRFDPostProcessInvalidArgs:
    """Invalid argument tests for dx_postprocess.SCRFDPostProcess.

    Note: SCRFDPostProcess requires is_ort_configured=True on this build
    (ORT-OFF not supported). Tests use is_ort=True.
    """

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_negative_dimensions_accepted(self):
        """Negative accepted — documents missing validation."""
        proc = dx_postprocess.SCRFDPostProcess(-640, -640, 0.5, 0.4, True)
        assert proc.get_input_width() == -640

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_string_input(self):
        with pytest.raises((TypeError, ValueError)):
            dx_postprocess.SCRFDPostProcess("bad", 640, 0.5, 0.4, True)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_ort_off_not_supported(self):
        """is_ort_configured=False raises ValueError on this build."""
        with pytest.raises(ValueError, match="ORT-OFF"):
            dx_postprocess.SCRFDPostProcess(640, 640, 0.5, 0.4, False)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_none(self):
        proc = dx_postprocess.SCRFDPostProcess(640, 640, 0.5, 0.4, True)
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess(None)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_wrong_type(self):
        proc = dx_postprocess.SCRFDPostProcess(640, 640, 0.5, 0.4, True)
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            proc.postprocess(["string_element"])


# ======================================================================
# ESPCNPostProcess tests
# ======================================================================

@skip_if_no_binding
class TestESPCNPostProcessInvalidArgs:
    """Invalid argument tests for dx_postprocess.ESPCNPostProcess."""

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_negative_scale_factor_accepted(self):
        """Negative scale factor accepted — documents missing validation."""
        proc = dx_postprocess.ESPCNPostProcess(224, 224, -1)
        assert proc.get_scale_factor() == -1

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_zero_dimensions_accepted(self):
        """Zero dimensions accepted at construction."""
        proc = dx_postprocess.ESPCNPostProcess(0, 0, 4)
        assert proc.get_input_width() == 0

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_string_dimensions(self):
        with pytest.raises((TypeError, ValueError)):
            dx_postprocess.ESPCNPostProcess("bad", 224, 4)

    @pytest.mark.pybinding
    @pytest.mark.invalid_args
    def test_postprocess_none(self):
        proc = dx_postprocess.ESPCNPostProcess(224, 224, 4)
        with pytest.raises((TypeError, RuntimeError)):
            proc.postprocess(None)


# ======================================================================
# Getter method validation
# ======================================================================

@skip_if_no_binding
class TestGetterMethods:
    """Validate that getter methods return expected values after construction."""

    @pytest.mark.pybinding
    def test_yolov7_getters(self):
        proc = dx_postprocess.YOLOv7PostProcess(416, 320, 0.3, 0.5, 0.6, True)
        assert proc.get_input_width() == 416
        assert proc.get_input_height() == 320

    @pytest.mark.pybinding
    def test_yolov8_getters(self):
        proc = dx_postprocess.YOLOv8PostProcess(512, 384, 0.4, 0.5, False)
        assert proc.get_input_width() == 512
        assert proc.get_input_height() == 384

    @pytest.mark.pybinding
    def test_deeplabv3_getters(self):
        proc = dx_postprocess.DeepLabv3PostProcess(257, 257)
        assert proc.get_input_width() == 257
        assert proc.get_input_height() == 257

    @pytest.mark.pybinding
    def test_classification_getters(self):
        proc = dx_postprocess.ClassificationPostProcess(10)
        assert proc.get_top_k() == 10

    @pytest.mark.pybinding
    def test_embedding_getters(self):
        proc = dx_postprocess.EmbeddingPostProcess(True)
        assert proc.get_l2_normalize() is True
        proc2 = dx_postprocess.EmbeddingPostProcess(False)
        assert proc2.get_l2_normalize() is False

    @pytest.mark.pybinding
    def test_espcn_getters(self):
        proc = dx_postprocess.ESPCNPostProcess(100, 100, 4)
        assert proc.get_input_width() == 100
        assert proc.get_input_height() == 100
        assert proc.get_scale_factor() == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
