"""Every CLIP image-encoder factory must match its own .dxnn's input contract.

The four CLIP encoders in this tree do NOT share one contract -- they split in two, and
the correct preprocessing is the opposite in each group:

**float32 NCHW, normalization NOT baked in** (ViT-B/32-256, ViT-L/14 datacomp_xl,
ViT-L/14-quickgelu DFN2B). The app must apply open_clip's transform itself. A plain
``x/255`` stretch silently places image embeddings in a different space from
host-encoded text: retrieval degrades without any error. Measured on the ViT-B/32
sibling against host FP32 over 21 sample images, p5 cosine was 0.4724 without these
constants and 0.8350 with them.

**uint8 NHWC, normalization baked into the compiled graph** (RN50x16-openai). The app
must hand over the RAW resized image. Emitting float32 [0,1] here is worse than merely
redundant: ``SyncRunner._prep_input`` casts it back with ``astype(np.uint8)``, which
floors almost every pixel to 0 (measured 0.0018% non-zero), so every image produces the
same embedding -- cross-image cosine 0.9991 over 8 sample images, versus 0.3890 once the
raw image is fed.

These tests need no ``.dxnn`` and no NPU: the factory classes are imported directly and
only their preprocessors' arithmetic is checked on a synthetic frame. The per-model
input contract is therefore restated here as a table rather than read back from the
model -- that duplication is the point, since it is what pins the factories down.

Two conventions this file pins down, because getting either wrong is silent:

**Units.** ``SimpleResizePreprocessor`` applies mean/std to the RAW 0-255 resized image,
while these constants (and any config override) are in open_clip's [0,1] units, so a
factory rescales by 255 on the way in. Asserting on the [0,1] form keeps that honest.

**Channel order: the emitted tensor is RGB, channel 0 = R.** Determined by reading
``SimpleResizePreprocessor.process``: with the default ``bgr=False`` it runs
``cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)`` before resizing, then subtracts
``mean[0]`` from that first (red) channel; verified empirically against a BGR frame with
three distinct channel values. The frames below therefore use DISTINCT per-channel
values and compare per channel WITHOUT sorting -- sorting would make these tests blind
to a channel-order bug, which is exactly the sibling of the defect they exist to catch.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# open_clip's published constants, in channel order R, G, B. Confirmed per checkpoint
# from its own open_clip_config.json "preprocess_cfg" -- all three float32 encoders
# carry the same standard OpenAI CLIP pair. Hard-coded rather than imported so that
# editing a factory's constants fails here.
OPEN_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
OPEN_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# (model directory name, model input edge length). Input edge is square for all four.
FLOAT32_ENCODERS = [
    ("vit_b_32_256_datacomp_s34b_b86k", 256),
    ("vit_l_14_datacomp_xl_s13b_b90k", 224),
    ("vit_l_14_quickgelu_dfn2b", 224),
]
UINT8_ENCODER_MODEL, UINT8_ENCODER_EDGE = "rn50x16_openai", 384

# A BGR frame whose three channels are distinct, so a swapped channel is detectable.
FRAME_B, FRAME_G, FRAME_R = 40, 128, 200


def _bgr_frame(blue: int = FRAME_B, green: int = FRAME_G, red: int = FRAME_R,
               edge: int = 256) -> np.ndarray:
    frame = np.zeros((edge, edge, 3), dtype=np.uint8)
    frame[:, :, 0] = blue
    frame[:, :, 1] = green
    frame[:, :, 2] = red
    return frame


def _channel_values(tensor: np.ndarray) -> np.ndarray:
    """Per-channel value at one pixel of a CHW or HWC tensor.

    One pixel rather than a mean: every pixel is identical here, and averaging 65536
    float32 values accumulates ~2e-4 of error, which is larger than the tolerance these
    assertions want to use.
    """
    return tensor[:, 0, 0] if tensor.shape[0] == 3 else tensor[0, 0, :]


# CLIP image encoders live under the task/family/variant tree
# (``src/python_example/<task>/<family>/<variant>/``), each variant with its own
# ``factory/<variant>_factory.py`` and ``custom_ops.py``. ``model`` here is the legacy
# example-dir name the parametrisation still uses; the registry maps it to its row,
# whose ``task``, ``family`` and ``variant`` give the directory.
_PY_EXAMPLES = PROJECT_ROOT / "src/python_example"


def _variant_for(model: str) -> str:
    from common.variants import resolve_variant

    return resolve_variant(model).variant


def _variant_dir(model: str) -> Path:
    from common.variants import resolve_variant

    row = resolve_variant(model)
    return _PY_EXAMPLES / row.task / row.family / row.variant


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _custom_ops(model: str):
    """Load the variant's custom_ops.py by path."""
    variant = _variant_for(model)
    return _load(f"clip_custom_ops_{variant}", _variant_dir(model) / "custom_ops.py")


def _factory_cls(model: str):
    """Load the variant factory by path -- the variant dir is not an importable package."""
    variant = _variant_for(model)
    path = _variant_dir(model) / "factory" / f"{variant}_factory.py"
    module = _load(f"clip_variant_factory_{variant}", path)
    return next(o for n, o in vars(module).items()
                if isinstance(o, type) and n.endswith("Factory") and n != "Factory"
                and o.__module__ == module.__name__)


def _preprocessor(model: str, edge: int, config=None):
    return _factory_cls(model)(config or {}, variant=_variant_for(model)) \
        .create_preprocessor(edge, edge)


def _process(model: str, edge: int, config=None) -> np.ndarray:
    tensor, _ = _preprocessor(model, edge, config).process(_bgr_frame(edge=edge))
    return tensor


# --------------------------------------------------------------------------------
# float32 NCHW encoders: the app owns the open_clip transform
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_factory_constants_are_the_open_clip_canonical_values(model, edge):
    # The CLIP constants are computed rather than literal, so they stay as code in the
    # variant's custom_ops.py -- the escape hatch a variant config cannot cover.
    module = _custom_ops(model)
    np.testing.assert_allclose(
        np.array(module.CLIP_MEAN, dtype=np.float32), OPEN_CLIP_MEAN, rtol=0, atol=1e-7,
    )
    np.testing.assert_allclose(
        np.array(module.CLIP_STD, dtype=np.float32), OPEN_CLIP_STD, rtol=0, atol=1e-7,
    )


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_preprocessor_applies_canonical_mean_std_per_channel(model, edge):
    """Distinct channels + unsorted comparison: catches wrong values AND wrong order."""
    tensor = _process(model, edge)

    # The tensor is RGB, so the R channel of the BGR input comes first.
    rgb = np.array([FRAME_R, FRAME_G, FRAME_B], dtype=np.float32) / 255.0
    expected = (rgb - OPEN_CLIP_MEAN) / OPEN_CLIP_STD

    assert tensor.dtype == np.float32
    np.testing.assert_allclose(_channel_values(tensor), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_float32_encoder_emits_chw_matching_the_model_nchw_input(model, edge):
    """The .dxnn declares [1, 3, edge, edge]; a HWC tensor would be silently misread."""
    tensor = _process(model, edge)
    assert tensor.shape == (3, edge, edge)


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_channel_order_is_rgb_not_bgr(model, edge):
    """Pins the cvtColor: if the RGB conversion is dropped, these values swap."""
    actual = _channel_values(_process(model, edge))

    bgr = np.array([FRAME_B, FRAME_G, FRAME_R], dtype=np.float32) / 255.0
    if_bgr_were_emitted = (bgr - OPEN_CLIP_MEAN) / OPEN_CLIP_STD

    assert not np.allclose(actual, if_bgr_were_emitted, rtol=1e-3, atol=1e-3), (
        "preprocessor emitted BGR - the BGR->RGB conversion was lost"
    )
    # Red is the brightest channel here, so after normalization it must stay the largest.
    assert int(np.argmax(actual)) == 0, f"channel 0 should be R (brightest), got {actual}"


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_a_plain_zero_one_stretch_is_not_what_the_factory_produces(model, edge):
    """Guards the specific regression: reverting to normalize_float=True."""
    stretch = np.array([FRAME_R, FRAME_G, FRAME_B], dtype=np.float32) / 255.0
    assert not np.allclose(_channel_values(_process(model, edge)), stretch,
                           rtol=1e-3, atol=1e-3), (
        "preprocessor returned a plain [0,1] stretch - the canonical mean/std was lost"
    )


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_config_overrides_mean_and_std_in_zero_one_units(model, edge):
    tensor = _process(model, edge, {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]})
    # mean=0 / std=1 in [0,1] units is exactly the identity x/255, so the channels must
    # come back as the plain stretch - not 0-255 values, which forgetting the rescale
    # would give.
    expected = np.array([FRAME_R, FRAME_G, FRAME_B], dtype=np.float32) / 255.0
    np.testing.assert_allclose(_channel_values(tensor), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
@pytest.mark.parametrize(
    "config",
    [
        # The exact shape a caller would copy from retinaface_mobilenet_v1_*_factory.py,
        # which legitimately passes raw 0-255 means.
        {"mean": [104.0, 117.0, 123.0]},
        {"mean": [0.485, 0.456, 0.406], "std": [58.395, 57.12, 57.375]},
        {"mean": [-0.1, 0.5, 0.5]},
        {"std": [0.0, 0.5, 0.5]},
    ],
)
def test_out_of_range_override_is_rejected(model, edge, config):
    """0-255 units must raise, not silently produce a wrongly normalized tensor."""
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _preprocessor(model, edge, config)


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_the_rejection_message_names_the_value_and_the_convention(model, edge):
    with pytest.raises(ValueError) as excinfo:
        _preprocessor(model, edge, {"mean": [104.0, 117.0, 123.0]})
    message = str(excinfo.value)
    assert "104.0" in message, message
    assert "255" in message, message


@pytest.mark.parametrize("model,edge", FLOAT32_ENCODERS)
def test_a_wrong_length_override_is_rejected(model, edge):
    with pytest.raises(ValueError, match="3 per-channel values"):
        _preprocessor(model, edge, {"mean": [0.5, 0.5]})


# --------------------------------------------------------------------------------
# uint8 NHWC encoder: the .dxnn owns the transform, the app must not touch it
# --------------------------------------------------------------------------------


def test_uint8_encoder_emits_the_raw_resized_rgb_image():
    """rn50x16's .dxnn takes uint8 NHWC with normalization compiled in.

    So the contract is the plain resized RGB image: uint8, HWC, untouched values.
    """
    tensor = _process(UINT8_ENCODER_MODEL, UINT8_ENCODER_EDGE)

    assert tensor.dtype == np.uint8
    assert tensor.shape == (UINT8_ENCODER_EDGE, UINT8_ENCODER_EDGE, 3)
    np.testing.assert_array_equal(
        _channel_values(tensor), np.array([FRAME_R, FRAME_G, FRAME_B], dtype=np.uint8),
    )


def test_uint8_encoder_does_not_normalize():
    """Guards the specific regression: normalize_float=True on a uint8-input model.

    That produced float32 [0,1], which the runner cast straight back to uint8 - flooring
    the frame to zeros and making every image yield the same embedding. Any float output
    here, or any value squeezed into [0,1], means the defect is back.
    """
    tensor = _process(UINT8_ENCODER_MODEL, UINT8_ENCODER_EDGE)

    assert not np.issubdtype(tensor.dtype, np.floating), (
        "uint8-input model received a float tensor - SyncRunner._prep_input will cast "
        "it back to uint8 and floor it to zeros"
    )
    assert int(tensor.max()) > 1, (
        f"tensor collapsed into [0,1] (max={tensor.max()}) - the raw 0-255 image was lost"
    )
