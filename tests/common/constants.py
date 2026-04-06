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

# ======================================================================
# E2E short-list models  (used by ``--e2e-short`` / ``-m e2e_short``)
#
# Format: model directory names under src/{python,cpp}_example/<task>/
#
# Display name              → model_dir_name
# --------------------------------------------------------
# Object Detection
#   YoloV5S-1               → yolov5s
#   YoloV6n_0.2.1-1         → yolov6n_0_2_1
#   YoloV7-1                → yolov7
#   YoloV8S-1               → yolov8s
#   YoloV9S-1               → yolov9s
#   YOLOV10S-1              → yolov10s
#   YOLOV11S-1              → yolov11s
#   YoloXTiny-1             → yoloxtiny
#   yolo26x-1               → yolo26x
#   yolo26n-1               → yolo26n
#   SSDMV2Lite-1            → ssdmv2lite
#   NanoDet_RepVGG-2        → nanodet_repvgg
# PPU
#   YoloV5S_PPU-1           → yolov5s_ppu
#   YoloV7_PPU-1            → yolov7_ppu
#   YoloV8_PPU-1            → yolov8_ppu
# Image Classification
#   ResNet50-1              → resnet50
#   MobileNetV3L-1          → mobilenetv3large
#   EfficientNetV2S-1       → efficientnetv2s
#   ViTBaseP32-1            → vitbasep32_384_hug
#   RegNetY800MF-1          → regnety800mf
#   HarDNet39DS-1           → hardnet39ds
# Face Detection
#   YOLOV7_Face-1           → yolov7_face
#   SCRFD10G-1              → scrfd10g
#   SCRFD500M-1             → scrfd500m
# Segmentation
#   DeepLabV3Plus-1         → deeplabv3plusmobilenet
#   BiSeNetV2-1             → bisenetv2
# Instance Segmentation
#   YOLOV8M_SEG-1           → yolov8m_seg
# Pose Estimation
#   YOLOV8M_POSE-1          → yolov8m_pose
# Image De-noising
#   DnCNN-2                 → dncnn_25
# Super Resolution
#   ESPCN_x4-1              → espcn_x4
# Depth Estimation
#   FastDepth-1             → fastdepth_1
# ======================================================================
E2E_SHORT_MODELS: set[str] = {
    # Object Detection
    "yolov5s",
    "yolov6n_0_2_1",
    "yolov7",
    "yolov8s",
    "yolov9s",
    "yolov10s",
    "yolov11s",
    "yolo26n",
    "yoloxtiny",
    "yolo26x",
    "ssdmv2lite",
    "nanodet_repvgg",
    # PPU
    "yolov5s_ppu",
    "yolov7_ppu",
    "yolov8_ppu",
    # Image Classification
    "resnet50",
    "mobilenetv3large",
    "efficientnetv2s",
    "vitbasep32_384_hug",
    "regnety800mf",
    "hardnet39ds",
    # Face Detection
    "yolov7_face",
    "scrfd10g",
    "scrfd500m",
    # Segmentation
    "deeplabv3plusmobilenet",
    "bisenetv2",
    # Instance Segmentation
    "yolov8m_seg",
    # Pose Estimation
    "yolov8m_pose",
    # Image De-noising
    "dncnn_25",
    # Super Resolution
    "espcn_x4",
    # Depth Estimation
    "fastdepth_1",
}
