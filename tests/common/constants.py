"""
Shared constants for C++ and Python visualization / E2E tests.

Single source of truth — never duplicate these maps in individual test modules.
"""

from __future__ import annotations

from pathlib import Path

# ======================================================================
# Paths (relative to ``dx_app/`` project root)
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # dx_app/
BIN_DIR = PROJECT_ROOT / "bin"
LIB_DIR = PROJECT_ROOT / "lib"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
SAMPLE_DIR = PROJECT_ROOT / "sample"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.json"

# Visualization result root — ``tests/test_visualization_result/``
VIS_RESULT_DIR = PROJECT_ROOT / "tests" / "test_visualization_result"

# ======================================================================
# Common sample image paths  (relative to PROJECT_ROOT)
# ======================================================================
_IMG = "sample/img"
_SAMPLE_DOG       = f"{_IMG}/sample_dog.jpg"
_SAMPLE_KITCHEN   = f"{_IMG}/sample_kitchen.jpg"
_SAMPLE_FACE      = f"{_IMG}/sample_face.jpg"
_SAMPLE_PEOPLE    = f"{_IMG}/sample_people.jpg"
_SAMPLE_STREET    = f"{_IMG}/sample_street.jpg"
_SAMPLE_HAND      = f"{_IMG}/sample_hand.jpg"
_SAMPLE_DENOISING = f"{_IMG}/sample_denoising.jpg"
_SAMPLE_LOWLIGHT  = f"{_IMG}/sample_lowlight.jpg"
_SAMPLE_SUPERRES  = f"{_IMG}/sample_superresolution.png"
_SAMPLE_DOTA      = "sample/dota8_test/P0177.png"

# ======================================================================
# Task → sample image mapping  (relative to PROJECT_ROOT)
# ======================================================================
TASK_IMAGE_MAP: dict[str, str] = {
    "object_detection":       _SAMPLE_DOG,
    "classification":         _SAMPLE_KITCHEN,
    "face_detection":         _SAMPLE_FACE,
    "pose_estimation":        _SAMPLE_PEOPLE,
    "instance_segmentation":  _SAMPLE_STREET,
    "semantic_segmentation":  _SAMPLE_STREET,
    "depth_estimation":       _SAMPLE_KITCHEN,
    "hand_landmark":          _SAMPLE_HAND,
    "embedding":              _SAMPLE_FACE,
    "obb_detection":          _SAMPLE_DOTA,
    "image_denoising":        _SAMPLE_DENOISING,
    "image_enhancement":      _SAMPLE_LOWLIGHT,
    "super_resolution":       _SAMPLE_SUPERRES,
    "ppu":                    _SAMPLE_DOG,
    # aliases used in some model_registry entries
    "face_alignment":         _SAMPLE_FACE,
    "detection":              _SAMPLE_DOG,
    "pose":                   _SAMPLE_PEOPLE,
}

# ======================================================================
# Per-model image overrides  (model_name → relative image path)
# ======================================================================
MODEL_IMAGE_OVERRIDE: dict[str, str] = {
    "yolov5pose_ppu":             _SAMPLE_PEOPLE,
    "yolov5pose":                 _SAMPLE_PEOPLE,
    "yolov8m_pose":               _SAMPLE_PEOPLE,
    "yolov8s_pose":               _SAMPLE_PEOPLE,
    "centerpose_regnetx_800mf":   _SAMPLE_PEOPLE,
}

# ======================================================================
# Multi-model executables  (base_name → [(flag, dxnn_filename), ...])
# The actual binary names are ``{base}_sync`` / ``{base}_async``.
# ======================================================================
MULTI_MODEL_EXECUTABLES: dict[str, list[tuple[str, str]]] = {
    "yolov7_x_deeplabv3": [("-y", "YoloV7.dxnn"), ("-d", "DeepLabV3PlusMobilenet.dxnn")],
}

# ======================================================================
# Models to skip in visualization (no image input, etc.)
# ======================================================================
SKIP_MODELS: set[str] = {
    "clip_resnet50_text_encoder_77x512",
}
