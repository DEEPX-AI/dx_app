"""
Shared constants for C++ and Python visualization / E2E tests.

Single source of truth — never duplicate these maps in individual test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
# Super-resolution needs a genuinely low-resolution input, and the right size
# depends on the model's scale factor — see MODEL_IMAGE_OVERRIDE below.
_SAMPLE_LOWRES_275x150 = f"{_IMG}/sample_lowres275x150.png"
_SAMPLE_LOWRES_165x90  = f"{_IMG}/sample_lowres165x90.png"
_SAMPLE_DOTA      = f"{_IMG}/sample_airport_satellite_view.png"

# Non-image sample inputs — these live directly under ``sample/``, NOT under
# ``sample/img/``, so they must NOT carry the ``_IMG`` prefix. SFA3D takes a
# KITTI LiDAR ``.bin``; DOPE takes a static-object frame under ``sample/dope/``.
# (Source of truth: scripts/run_examples.sh CATEGORY_IMAGE.)
_SAMPLE_LIDAR_KITTI = "sample/kitti/velodyne/000049.bin"
_SAMPLE_DOPE        = "sample/dope/000000.png"
_SAMPLE_FACE_PAIR_REF = f"{_IMG}/face_pair/1_reference.jpg"
_SAMPLE_PERSON_A2     = f"{_IMG}/sample_person_a2.jpg"

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
    "hand_detection":         _SAMPLE_HAND,   # MediaPipe palm — needs a hand in frame
    "embedding":              _SAMPLE_FACE,
    "obb_detection":          _SAMPLE_DOTA,
    "image_denoising":        _SAMPLE_DENOISING,
    "image_enhancement":      _SAMPLE_LOWLIGHT,
    "super_resolution":       _SAMPLE_LOWRES_275x150,   # ESPCN default; see MODEL_IMAGE_OVERRIDE
    "ppu":                    _SAMPLE_DOG,
    # aliases used in some model_registry entries
    "face_alignment":         _SAMPLE_FACE_PAIR_REF,
    "detection":              _SAMPLE_DOG,
    "pose":                   _SAMPLE_PEOPLE,
    "keypoint_detection":     _SAMPLE_STREET,
    "object_pose_estimation": _SAMPLE_DOPE,          # sample/dope/*.png (NOT sample/img)
    "panoptic_driving_perception": _SAMPLE_STREET,
    "3d_object_detection":    _SAMPLE_LIDAR_KITTI,   # sample/kitti/velodyne/*.bin (NOT sample/img)
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
    "unet_mobilenet_v2":          _SAMPLE_DOG,
    "mediapipe_hand_detector":    _SAMPLE_PERSON_A2,
    "scrfd500m_ppu":              _SAMPLE_PERSON_A2,
    # Super-resolution: ESPCN upscales a 275x150 crop; Real-ESRGAN takes a
    # smaller 165x90 one so the x8 output stays a sane size.
    "espcn_x2":                   _SAMPLE_LOWRES_275x150,
    "espcn_x3":                   _SAMPLE_LOWRES_275x150,
    "espcn_x4":                   _SAMPLE_LOWRES_275x150,
    "realesrgan_x2":              _SAMPLE_LOWRES_165x90,
    "realesrgan_x4":              _SAMPLE_LOWRES_165x90,
    "realesrgan_x8":              _SAMPLE_LOWRES_165x90,
}

# ======================================================================
# Explicit model-dir → .dxnn filename aliases.
# Last-resort mapping for the rare cases where the example directory name
# cannot be derived from the .dxnn filename by normalisation / suffix rules
# (an arbitrary rename). Key: normalised model-dir name. Value: exact .dxnn
# filename under assets/models/.
# ======================================================================
MODEL_DXNN_ALIAS: dict[str, str] = {
    "deitbase384": "deit-b_384x384.dxnn",
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
# Image-only task categories — examples take image input ONLY and do NOT
# support video/camera/rtsp stream input. Stream (video) E2E tests must skip
# these. Single source of truth; keep in sync with scripts/run_examples.sh
# IMAGE_ONLY_CATEGORIES.
# ======================================================================
IMAGE_ONLY_TASKS: frozenset = frozenset({
    "embedding",
    "reid",
    "attribute_recognition",
    "object_pose_estimation",
    "3d_object_detection",
    # hand_detection / hand_landmark are NOT image-only: the single model runs
    # per frame on video/camera/RTSP (no palm-detector crop stage), so they are
    # exercised by the stream E2E tests like any other video-capable task.
    # super_resolution: a video/stream path exists (see SR-stream regression
    # tests), but the harness exercises it image-only — upscaled stream output
    # is large/slow and not meaningful for E2E regression.
    "super_resolution",
})

# ======================================================================
# Tasks whose SINGLE-MODEL example runners HARD-REJECT stream input
# (-v/--video, -c/--camera, -r/--rtsp) with a fatal "image input only" error.
# Keyed by example DIRECTORY name (the C++/Python runners map the
# 3d_object_detection directory to the internal task name "3d_detection").
#
# This is a *stricter* subset of IMAGE_ONLY_TASKS used by NEGATIVE tests that
# verify the SDKREQ-517 exclusion is actually ENFORCED (not merely skipped).
# It is NOT the same as IMAGE_ONLY_TASKS and it differs by language, because:
#   - super_resolution is image-only in the harness, but BOTH runners still
#     accept a video path — it is in NEITHER reject set.
#   - hand_detection / hand_landmark now SUPPORT stream input in both the C++
#     and Python runners (single model per frame), so they are in NEITHER reject
#     set — they are exercised by the positive stream tests instead.
# Source of truth: src/cpp_example/common/runner/*_runner.hpp guards and
# src/python_example/common/runner/sync_runner.py::_IMAGE_ONLY_TASKS.
# ======================================================================
STREAM_REJECTING_TASKS_CPP: frozenset = frozenset({
    "3d_object_detection",
    "embedding",
    "reid",
    "attribute_recognition",
    "object_pose_estimation",
})
STREAM_REJECTING_TASKS_PY: frozenset = frozenset({
    "3d_object_detection",
    "embedding",
    "reid",
    "attribute_recognition",
    "object_pose_estimation",
})

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
#   ViTBaseP32-1            → vitbasep32_384
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
    "yolov8s_ppu",
    # Image Classification
    "resnet50",
    "mobilenetv3large",
    "efficientnetv2s",
    "vitb32",
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
    "espcn_x2",
    "realesrgan_x2",
    # Depth Estimation
    "fastdepth_1",
    # OBB Detection
    "yolo26n_obb",
    # 3D Object Detection
    "sfa3d_608x608",
    # Object Pose Estimation
    "dope_hope_ketchup",
    # Panoptic Driving Perception
    "yolopv2",
    # Keypoint Detection
    "superpoint",
    # Image Enhancement
    "zero_dce",
    # Face Alignment
    "3ddfa_v2_mobilnetv1_120x120",
    # Hand Detection
    "mediapipe_hand_detector",
    # Hand Landmark
    "handlandmarklite_1",
    # Embedding
    "arcface_mobilefacenet",
    # ReID
    "casvit_t",
    # Attribute Recognition
    "deepmar_resnet50",
}


# ======================================================================
# E2E heavy-model loop cap
# ======================================================================
E2E_HEAVY_MODELS: frozenset = frozenset({
    "casvit-t-fpn-resnet50_512x512",
    "realesrgan-x8_192x192",
    "realesrgan-x4_192x192",
    "stdc2-seg50_512x1024",
    "retinaface_mobilenetv1_736x1280",
    "zerodce_400x600",
    "fcn8_resnet50_512x512",
    "yolov5-x6_1280x1280",
    "yolov7-e6e_1280x1280",
    "yolov7-d6_1280x1280",
    "yolov7-e6_1280x1280",
    "efficientdet-d4_1024x1024",
    "yolov5-l6_1280x1280",
    "yolov7-w6_1280x1280_nodecode",
    "yolov7-w6_1280x1280",
    "yolov5-m6_1280x1280_v6.1",
    "yolov5-m6_1280x1280",
    "realesrgan-x2_192x192",
    "yolov5-s6_1280x1280_v6.1",
    "yolov5-s6_1280x1280",
    "yolov3-gluon_608x608",
    "yolov6-l6_1280x1280",
    "yolov5-n6_1280x1280",
    "depthanythingv2-vitl_224x224",
    "yolov5-n6_1280x1280_v6.1",
    "yolo26-x-obb_1024x1024",
    "zerodce-pp_400x600",
    "bisenetv1_1024x2048",
    "deit-b_384x384",
    "beit-l-p16_224x224",
    "fastsam-s_1024x1024",
    "yolo26-x-seg_640x640",
    "clip-img_vit-l14_224x224_datacomp-xl-s13b-b90k",
    "clip-img_vit-l14-quickgelu_224x224_dfn2b",
    "yolov7-w6-face_1280x1280_tta",
    "yolopv2_384x640",
    "yolo11-x-seg_640x640",
    "yolov5-x-seg_640x640",
    "yolov8-x-seg_640x640",
    "dope-hope-ketchup_480x640",
    "deeplabv3plus-drn_512x512",
    "regnet-y32gf_384x384",
    "yolo26-l-obb_1024x1024",
    "yolact_regnet-x1.6gf_512x512",
    "deit-b_384x384_distilled",
    "yolov3-gluon_416x416",
    "yolact_regnet-x800mf_512x512",
    "dncnn-color_512x512",
    "bisenetv2_1024x2048",
    "dncnn-gray_512x512",
    "pidnet-s_1024x2048",
    "centerpose_regnet-x800mf_640x640",
    "yolov5-l-seg_640x640",
    "depthanythingv2-vitb_224x224",
    "dncnn-15_512x512",
    "dncnn-50_512x512",
    "regnet-y16gf_384x384",
    "dncnn-25_512x512",
    "yolov8-l-seg_640x640",
    "yolo26-m-obb_1024x1024",
    "yolo26-l-seg_640x640",
    "yolo26-x_640x640",
    "yolov8-x_640x640",
    "yolo11-x_640x640",
    "yolo11-l-seg_640x640",
    "yolo26-x-pose_640x640",
    "yolov5-x_640x640",
    "yolov5-m-seg_640x640",
    "yolo11-x-pose_640x640",
    "yolox-x_640x640",
    "yolov8-x-pose_640x640",
    "yolov10-x_640x640",
    "yolov7-x_640x640",
    "yolo26-m-seg_640x640",
    "efficientdet-d2_768x768",
    "yolov5-n-seg_640x640",
    "unet_mobilenetv2_256x256",
    "yolov8-m-seg_640x640",
    "yolov5-s-seg_640x640",
    "yolo11-m-seg_640x640",
    "densenet161_224x224",
    "yolov3_640x640",
    "yolov7-w6-face_960x960",
    "densenet201_224x224",
    "espcn_x2", 
    "espcn_x3", 
    "espcn_x4", 
    "realesrgan_x2",
    "realesrgan_x4",
    "realesrgan_x8",
    "dope_hope_ketchup",
    "sfa3d_608x608",
})

# Loop-count cap applied to every model in :data:`E2E_HEAVY_MODELS`.
E2E_HEAVY_MODEL_LOOP_CAP: int = 2


def e2e_effective_loop(model_stem: Optional[str], loop_count: int) -> int:
    """Cap ``loop_count`` for heavy models so E2E runs don't time out.

    Models absent from :data:`E2E_HEAVY_MODELS` run the full requested
    ``loop_count``. Heavy models -- those measured at or below the 20 FPS
    threshold -- are capped at :data:`E2E_HEAVY_MODEL_LOOP_CAP`. Never
    *raises* the loop count.
    """
    if not model_stem:
        return loop_count
    if model_stem in E2E_HEAVY_MODELS:
        return min(loop_count, E2E_HEAVY_MODEL_LOOP_CAP)
    return loop_count
