"""Super-resolution output must keep the source aspect ratio.

RealESRGAN has a fixed square model input (192x192), and the preprocessor
stretches any frame into it. Without restoring the geometry afterwards a
1200x900 (4:3) input came out as 768x768 (1:1) — the output ratio was the
*model input* ratio, not the source ratio.

The restore maps the model output back to ``original_size * upscale_factor``,
which is what the tiled ESPCN path already produces. Same-size restoration
models (DnCNN denoising, upscale factor 1) are deliberately left untouched.
"""

import numpy as np
import pytest

from common.base import PreprocessContext
from common.processors.restoration_postprocessor import (
    RealESRGANPostprocessor,
    restore_source_geometry,
)
from common.utility import convert_cpp_restoration


MODEL_IN = 8          # stand-in for 192
UPSCALE = 4           # stand-in for x4
MODEL_OUT = MODEL_IN * UPSCALE


def _ctx(orig_w, orig_h, model_in=MODEL_IN):
    return PreprocessContext(original_width=orig_w, original_height=orig_h,
                             input_width=model_in, input_height=model_in)


def _nchw_output(size=MODEL_OUT):
    """[1, 3, size, size] float output in [0, 1]."""
    return np.zeros((1, 3, size, size), dtype=np.float32)


class TestRestoreSourceGeometry:
    def test_landscape_source_keeps_ratio(self):
        img = np.zeros((MODEL_OUT, MODEL_OUT, 3), np.uint8)
        out = restore_source_geometry(img, _ctx(20, 10), MODEL_IN, MODEL_IN)
        assert (out.shape[1], out.shape[0]) == (20 * UPSCALE, 10 * UPSCALE)

    def test_portrait_source_keeps_ratio(self):
        img = np.zeros((MODEL_OUT, MODEL_OUT, 3), np.uint8)
        out = restore_source_geometry(img, _ctx(9, 16), MODEL_IN, MODEL_IN)
        assert (out.shape[1], out.shape[0]) == (9 * UPSCALE, 16 * UPSCALE)

    def test_same_size_model_is_untouched(self):
        """Denoising (upscale factor 1) must keep the model-space output."""
        img = np.zeros((MODEL_IN, MODEL_IN, 3), np.uint8)
        out = restore_source_geometry(img, _ctx(20, 10), MODEL_IN, MODEL_IN)
        assert out.shape == img.shape

    def test_unknown_original_size_is_untouched(self):
        img = np.zeros((MODEL_OUT, MODEL_OUT, 3), np.uint8)
        out = restore_source_geometry(img, PreprocessContext(), MODEL_IN, MODEL_IN)
        assert out.shape == img.shape

    def test_no_ctx_is_untouched(self):
        img = np.zeros((MODEL_OUT, MODEL_OUT, 3), np.uint8)
        assert restore_source_geometry(img, None, MODEL_IN, MODEL_IN).shape == img.shape


class TestRealESRGANPostprocessor:
    def test_output_matches_source_ratio(self):
        post = RealESRGANPostprocessor(MODEL_IN, MODEL_IN)
        results = post.process([_nchw_output()], _ctx(20, 10))
        img = results[0].output_image
        assert (img.shape[1], img.shape[0]) == (80, 40)
        assert round(img.shape[1] / img.shape[0], 3) == 2.0

    def test_square_source_is_also_scaled_by_the_upscale_factor(self):
        """A square source keeps its 1:1 ratio but still gets original * scale,
        so the output size depends on the input size, not on the model input."""
        post = RealESRGANPostprocessor(MODEL_IN, MODEL_IN)
        results = post.process([_nchw_output()], _ctx(10, 10))
        img = results[0].output_image
        assert (img.shape[1], img.shape[0]) == (10 * UPSCALE, 10 * UPSCALE)


class TestCppPostprocessConversion:
    """The *_cpp_postprocess variants must agree with the Python path."""

    def test_cpp_conversion_restores_ratio(self):
        chw = np.zeros((3, MODEL_OUT, MODEL_OUT), dtype=np.float32)
        results = convert_cpp_restoration(chw, _ctx(20, 10))
        img = results[0].output_image
        assert (img.shape[1], img.shape[0]) == (80, 40)

    def test_cpp_conversion_without_ctx_is_untouched(self):
        chw = np.zeros((3, MODEL_OUT, MODEL_OUT), dtype=np.float32)
        results = convert_cpp_restoration(chw, None)
        img = results[0].output_image
        assert (img.shape[1], img.shape[0]) == (MODEL_OUT, MODEL_OUT)
