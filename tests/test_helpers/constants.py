"""
Shared constants for C++ and Python visualization / E2E tests.

Single source of truth — never duplicate these maps in individual test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .platform_paths import resolve_bin_dir

# ======================================================================
# Paths (relative to ``dx_app/`` project root)
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # dx_app/
BIN_DIR = resolve_bin_dir(PROJECT_ROOT)  # bin/ on Linux; see platform_paths
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
    "image_classification":   _SAMPLE_DOG,
    "oriented_object_detection": _SAMPLE_DOTA,
    "face_recognition":       _SAMPLE_FACE_PAIR_REF,
    "face_attribute":         f"{_IMG}/sample_person_a1.jpg",
    "face_landmark":          _SAMPLE_FACE_PAIR_REF,
    "person_attribute":       f"{_IMG}/sample_person_a1.jpg",
    "person_reid":            "sample/reid/queries/sample_person_a2.jpg",
    "image_retrieval":        _SAMPLE_PERSON_A2,
    "visual_place_recognition": "sample/vpr/queries/q1.jpg",
    "image_matting":          f"{_IMG}/sample_person_b.jpg",
    "low_light_enhancement":  _SAMPLE_LOWLIGHT,
    "zero_shot_image_classification": _SAMPLE_DOG,
    "zero_shot_instance_segmentation": _SAMPLE_STREET,
    "anomaly_detection":      _SAMPLE_STREET,
}

# ======================================================================
# Per-model image overrides  (model_name → relative image path)
# ======================================================================
# Example sources kept on disk, excluded from the build, the registry, and tests.
EXCLUDED_EXAMPLE_VARIANTS = frozenset({"SCRFD500M_PPU", "YOLOV5Pose_PPU"})

MODEL_IMAGE_OVERRIDE: dict[str, str] = {
    # Keyed by variant name (the binary / script stem), which is what
    # resolve_image_for_model() is called with.
    "yolov5-s6-pose_640x640":           _SAMPLE_PEOPLE,
    "yolov8-m-pose_640x640":            _SAMPLE_PEOPLE,
    "yolov8-s-pose_640x640":            _SAMPLE_PEOPLE,
    "centerpose_regnet-x800mf_640x640": _SAMPLE_PEOPLE,
    "unet_mobilenetv2_256x256":         _SAMPLE_DOG,
    "mediapipe-hand-detector_192x192":  _SAMPLE_PERSON_A2,
    # Super-resolution: ESPCN upscales a 275x150 crop; Real-ESRGAN takes a
    # smaller 165x90 one so the x8 output stays a sane size.
    "espcn-x2_17x17":                   _SAMPLE_LOWRES_275x150,
    "espcn-x3_17x17":                   _SAMPLE_LOWRES_275x150,
    "espcn-x4_17x17":                   _SAMPLE_LOWRES_275x150,
    "realesrgan-x2_192x192":            _SAMPLE_LOWRES_165x90,
    "realesrgan-x4_192x192":            _SAMPLE_LOWRES_165x90,
    "realesrgan-x8_192x192":            _SAMPLE_LOWRES_165x90,
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
    "face_recognition",
    "person_attribute",
    "face_attribute",
    "person_reid",
    "image_retrieval",
    "visual_place_recognition",
    "object_pose_estimation",
    "3d_object_detection",
    "anomaly_detection",
    # hand_detection / hand_landmark are NOT image-only: the single model runs
    # per frame on video/camera/RTSP (no palm-detector crop stage), so they are
    # exercised by the stream E2E tests like any other video-capable task.
    # super_resolution: a video/stream path exists (see SR-stream regression
    # tests), but the harness exercises it image-only — upscaled stream output
    # is large/slow and not meaningful for E2E regression. The stream path
    # itself is exercised on a small clip by test_super_resolution_stream_e2e
    # (C++ and Python test_e2e.py).
    "super_resolution",
})

# ======================================================================
# Video E2E skips only these models (U-32): the W6 face detectors run at
# ~3-20 s/frame (TTA ~125 s/frame) on aarch64, so the 478-frame clip cannot
# finish inside VIDEO_RUN_TIMEOUT_S (2000 s). Every other face model runs the video
# tests like any detector.
# ======================================================================
VIDEO_TOO_SLOW_MODELS: frozenset = frozenset({"yolov7-w6-face_960x960", "yolov7-w6-face_1280x1280_tta"})

# Bound for one C++ example run over the whole 478-frame test video (video E2E
# and video save). The heaviest video-capable models run near 0.6 frame/s on
# x86_64 (ppmatting-hrnet-w48 sync: about 800 s for the clip), so the old 600 s
# bound of the video save tests ended a run that was still making progress.
VIDEO_RUN_TIMEOUT_S = 2000


def video_too_slow(name: str) -> bool:
    """True for an executable, script stem or model name of a model in
    VIDEO_TOO_SLOW_MODELS (a _sync / _async suffix, and the Python
    _cpp_postprocess variant's suffix after it, is ignored)."""
    if name.endswith("_cpp_postprocess"):
        name = name[: -len("_cpp_postprocess")]
    for suffix in ("_sync", "_async"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name in VIDEO_TOO_SLOW_MODELS

# ======================================================================
# Tasks whose SINGLE-MODEL example runners HARD-REJECT stream input
# (-v/--video, -c/--camera, -r/--rtsp) with a fatal "image input only" error.
# Keyed by example DIRECTORY name, which is also the task name both runners report.
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
    "face_recognition",
    "person_attribute",
    "face_attribute",
    "person_reid",
    "image_retrieval",
    "visual_place_recognition",
    "object_pose_estimation",
})
STREAM_REJECTING_TASKS_PY: frozenset = frozenset({
    "3d_object_detection",
    "embedding",
    "reid",
    "attribute_recognition",
    "face_recognition",
    "person_attribute",
    "face_attribute",
    "person_reid",
    "image_retrieval",
    "visual_place_recognition",
    "object_pose_estimation",
})

# ======================================================================
# E2E short-list models  (used by ``--e2e-short`` / ``-m e2e_short``)
#
# Format: variant directory names under src/{python,cpp}_example/<task>/<family>/
#
# One representative per task at least, and one per postprocessor that no other
# short model exercises, so the short suite touches every decode path. The
# smallest published variant of a family is preferred. ``unit/test_e2e_short_coverage.py``
# fails when a registry task has no short model, so a new task cannot be added
# without a representative here.
E2E_SHORT_MODELS: set[str] = {
    "yolov5-s_640x640",
    "yolov6-n0_640x640_v0.2.1",
    "yolov7_640x640",
    "yolov8-s_640x640",
    "yolov9-s_640x640",
    "yolov10-s_640x640",
    "yolo11-s_640x640",
    "yolo26-n_640x640",
    "yolox-t_416x416",
    "yolo26-x_640x640",
    "ssd-mobilenetv2-lite_300x300",
    "nanodet_224x224",
    "yolov5-s_640x640_ppu",
    "yolov7_640x640_ppu",
    "yolov8-s_640x640_ppu",
    "resnet50_224x224",
    "mobilenetv3-large_224x224",
    "efficientnetv2-s_384x384",
    "vit-b-p32_224x224",
    "regnet-y800mf_224x224",
    "hardnet39ds_224x224",
    "yolov7-face_640x640",
    "scrfd-10g_640x640",
    "scrfd-500m_640x640",
    "deeplabv3plus_mobilenetv1_512x512",
    "bisenetv2_1024x2048",
    "yolov8-m-seg_640x640",
    "yolov8-m-pose_640x640",
    "dncnn-25_512x512",
    "espcn-x2_17x17",
    "realesrgan-x2_192x192",
    "fastdepth_224x224",
    "yolo26-n-obb_1024x1024",
    "sfa3d_608x608",
    "dope-hope-ketchup_480x640",
    "yolopv2_384x640",
    "superpoint_480x640",
    "zerodce_400x600",
    "3ddfa-v2_mobilenetv1_120x120",
    "mediapipe-hand-detector_192x192",
    "mediapipe-hands-lite_224x224",
    "arcface_mobilefacenet_112x112",
    "casvit-t_224x224",
    "deepmar_resnet50_224x224",
    # --- v3.3.0 tasks and postprocessors -----------------------------------
    # anomaly_detection: EfficientAD runs its teacher/student/autoencoder trio;
    # PatchCore is the single-network feature response.
    "efficientad-m-student_256x256",
    "patchcore_224x224",
    # face_attribute (image-only)
    "faceattr_resnetv1-18_218x178",
    # image_matting: PP-Matting alpha matte
    "ppmatting-hrnet-w48-composition_512x512",
    # image_retrieval: CLIP image tower + gallery, and the text tower
    "clip-img_resnet50_224x224_openai",
    "clip-text_resnet50_77x512_openai",
    # mainbody_detection: PP-PicoDet per-level GFL decode
    "pp-shituv2-mainbody-detection_640x640",
    # person_reid and visual_place_recognition: gallery retrieval
    "repvgg-a0-reid_256x128",
    "eigenplaces-resnet18_512x512",
    "pp-shituv2-feature-extraction_224x224",
    # zero_shot_image_classification: CLIP prompt bank
    "clip-img_vit-b32_256x256_datacomp-s34b-b86k",
    # zero_shot_instance_segmentation
    "fastsam-s_1024x1024",
    # object_detection: RT-DETR (NMS-free table), the pre-optimized top-k table,
    # and the YOLOv12/v13 heads decoded with the YOLOv8 postprocessor
    "rtdetr-r18vd-6x_640x640",
    "yolo26-n_640x640_pre-optimized",
    "yolov12-n_640x640",
    "yolov13-n_640x640",
    # instance_segmentation: mask-RT-DETR per-query masks, YOLOv12 prototypes
    "mask-rtdetr-hgnetv2-s-6x_640x640",
    "yolov12-seg-n_640x640",
    # pose_estimation: pre-optimized pose table, CenterPose, ViTPose heatmaps
    "yolo26-n-pose_640x640_pre-optimized",
    "centerpose_repvgg-a0_416x416",
    "vitpose-s_256x192",
    # image_classification: a network that already ends in softmax
    "pplcnetv2-small_224x224",
    # semantic_segmentation: PP-LiteSeg (non-square input)
    "pp-liteseg-stdc1-camvid-10k_960x720",
}


# ======================================================================
# E2E heavy-model loop cap
# ======================================================================
E2E_HEAVY_MODELS: frozenset = frozenset({
    "casvit-t-fpn-resnet50_512x512",
    "realesrgan-x8_192x192",   # postprocess-bound (see note at end of set)
    "realesrgan-x4_192x192",   # postprocess-bound
    "stdc2-seg50_512x1024",
    # Larger spatial size than stdc2-seg50_512x1024 (768x1536, seg75).
    "stdc1-seg75_768x1536",
    "stdc2-seg75_768x1536",
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
    "realesrgan-x2_192x192",   # postprocess-bound
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
    # Same network as beit-l-p16_224x224 with a larger input (384 vs 224).
    "beit-l-p16_384x384",
    "fastsam-s_1024x1024",
    "yolo26-x-seg_640x640",
    "clip-img_vit-l14_224x224_datacomp-xl-s13b-b90k",
    "clip-img_vit-l14-quickgelu_224x224_dfn2b",
    "yolov7-w6-face_1280x1280_tta",
    "yolopv2_384x640",
    "yolo11-x-seg_640x640",
    "yolov5-x-seg_640x640",
    "yolov8-x-seg_640x640",
    "dope-hope-ketchup_480x640",   # postprocess-bound
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
    # Postprocess-bound entries: NPU inference itself is fast (these are not
    # <=20 FPS), but the host-side postprocess is heavy enough to blow the E2E
    # subprocess timeout at a full loop count -- super-resolution writes a
    # large upscaled frame per iteration, DOPE runs belief-map peak extraction
    # + PnP per iteration, SuperPoint runs a per-keypoint Python NMS loop
    # (~2.7 s/frame measured on DX-M1: infer 31.6 ms vs postprocess 2691 ms,
    # 0.4 FPS overall for the Python-postprocess variants). Capped
    # deliberately, not by FPS measurement.
    "espcn-x2_17x17",
    "espcn-x3_17x17",
    "espcn-x4_17x17",
    "sfa3d_608x608",
    "superpoint_480x640",
    # semantic_segmentation
    "deeplabv3-resnet101_512x512",
    "deeplabv3-resnet50_512x512",
    "deeplabv3_mobilenetv2_512x512",
    "deeplabv3_mobilenetv2_513x513_nodilation",
    "deeplabv3plus-resnet101_512x512",
    "deeplabv3plus-resnet50_512x512",
    "deeplabv3plus_mobilenetv1_512x512",
    "deeplabv3plus_mobilenetv2_512x512",
    "fcn8_resnet18_1024x1920",
    "segformer_mit-b0_512x1024",
    # instance_segmentation
    "yolo11-n-seg_640x640",
    "yolo11-s-seg_640x640",
    "yolo26-n-seg_640x640",
    "yolo26-s-seg_640x640",
    "yolov8-n-seg_640x640",
    "yolov8-s-seg_640x640",
    # obb_detection
    "yolo26-n-obb_1024x1024",
    "yolo26-s-obb_1024x1024",
    # image_matting: a full C++ -l 50 does not finish quickly. Cap at 2.
    "ppmatting-hrnet-w48-composition_512x512",
    "ppmatting-hrnet-w48-distinctions_512x512",
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
