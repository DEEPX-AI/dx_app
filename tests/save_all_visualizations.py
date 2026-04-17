#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Full-model inference + visualization save script.

Usage:
  export LD_LIBRARY_PATH=$HOME/test_dx_app/dx-all-suite/dx-runtime/dx_rt/build_x86_64/lib:$LD_LIBRARY_PATH
  ~/dx-venv/bin/python3 tests/save_all_visualizations.py

Output: artifacts/visualization_check/<task>/<model>.jpg
"""
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# ======================================================================
# Paths
# ======================================================================
ROOT = Path(__file__).resolve().parent.parent          # dx_app/
SRC  = ROOT / "src" / "python_example"
MODELS_DIR = ROOT / "assets" / "models"
SAMPLE_DIR = ROOT / "sample"
OUTPUT_DIR = ROOT / "artifacts" / "visualization_check"

sys.path.insert(0, str(SRC))

from dx_engine import InferenceEngine, InferenceOption  # noqa: E402

# ======================================================================
# Factory directory to .dxnn file mapping
# key = "task/model"   value = dxnn filename (without extension) or None
# ======================================================================
FACTORY_TO_DXNN = {
    # ── classification (39) ─────────────────────────────────────────
    "classification/alexnet":              "AlexNet",
    "classification/densenet121":          "DenseNet121",
    "classification/densenet161":          "DenseNet161",
    "classification/efficientnetb2":       "EfficientNetB2",
    "classification/efficientnet_lite0":   "EfficientNet_Lite0",
    "classification/efficientnet_lite1":   "EfficientNet_Lite1",
    "classification/efficientnet_lite2":   "EfficientNet_Lite2",
    "classification/efficientnet_lite3":   "EfficientNet_Lite3",
    "classification/efficientnet_lite4":   "EfficientNet_Lite4",
    "classification/efficientnetv2s":      "EfficientNetV2S",
    "classification/hardnet39ds":          "HarDNet39DS",
    "classification/hardnet68":            "HarDNet68",
    "classification/inceptionv1":          "InceptionV1",
    "classification/mobilenetv1":          "MobileNetV1",
    "classification/mobilenetv2":          "MobileNetV2",
    "classification/mobilenetv3large":     "MobileNetV3Large",
    "classification/regnetx400mf":         "RegNetX400MF",
    "classification/regnetx800mf":         "RegNetX800MF",
    "classification/regnety200mf":         "RegNetY200MF",
    "classification/regnety400mf":         "RegNetY400MF",
    "classification/regnety800mf":         "RegNetY800MF",
    "classification/repvgga1":             "RepVGGA1",
    "classification/repvgga2":             "RepVGGA2",
    "classification/resnet101":            "ResNet101",
    "classification/resnet18":             "ResNet18",
    "classification/resnet34":             "ResNet34",
    "classification/resnet50":             "ResNet50",
    "classification/resnext26_32x4d":      "ResNeXt26_32x4d",
    "classification/resnext50_32x4d":      "ResNeXt50_32x4d",
    "classification/resnext50_32x4d_h":"ResNeXt50_32x4d_h",
    "classification/squeezenet1_0":        "SqueezeNet1_0",
    "classification/squeezenet1_1":        "SqueezeNet1_1",
    "classification/vgg11":                "VGG11",
    "classification/vgg11bn":              "VGG11BN",
    "classification/vgg13":                "VGG13",
    "classification/vgg13bn":              "VGG13BN",
    "classification/vgg19bn":              "VGG19BN",
    "classification/wideresnet101_2":      "WideResNet101_2",
    "classification/wideresnet50_2":       "WideResNet50_2",
    # ── depth_estimation (1) ────────────────────────────────────────
    "depth_estimation/fastdepth_1":        "FastDepth_1",
    # ── classification (continued: osnet) ────────────────────────────
    "classification/osnet0_25":                         "OSNet0_25",
    "classification/osnet0_5":                          "OSNet0_5",
    # ── embedding (3) ───────────────────────────────────────────────
    "embedding/arcface_mobilefacenet":                  None,
    "embedding/clip_resnet50_image_encoder_224x224":    None,
    "embedding/clip_resnet50_text_encoder_77x512":      None,
    # ── face_detection (10) ─────────────────────────────────────────
    "face_detection/retinaface_mobilenet0_25_640":  None,
    "face_detection/scrfd10g":                      "SCRFD10G",
    "face_detection/scrfd2_5g":                     "SCRFD2_5G",
    "face_detection/scrfd500m":                     "SCRFD500M",
    "face_detection/yolov5m_face":                  "YOLOv5m_Face",
    "face_detection/yolov5s_face":                  "YOLOv5s_Face",
    "face_detection/yolov7_face":                   "YOLOv7_Face",
    "face_detection/yolov7s_face":                  "YOLOv7s_Face",
    "face_detection/yolov7_w6_face":                "YOLOv7_W6_Face",
    "face_detection/yolov7_w6_tta_face":            "YOLOv7_W6_TTA_Face",
    # ── hand_landmark (1) ───────────────────────────────────────────
    "hand_landmark/handlandmarklite_1":    "HandLandmarkLite_1",
    # ── image_denoising (3) ─────────────────────────────────────────
    "image_denoising/dncnn_15":            "DnCNN_15",
    "image_denoising/dncnn_25":            "DnCNN_25",
    "image_denoising/dncnn_50":            "DnCNN_50",
    # ── image_enhancement (1) ───────────────────────────────────────
    "image_enhancement/zero_dce":          None,
    # ── instance_segmentation (8) ───────────────────────────────────
    "instance_segmentation/yolact_regnetx_800mf":  None,
    "instance_segmentation/yolov5l_seg":           "yolov5l_seg",
    "instance_segmentation/yolov5m_seg":           "yolov5m_seg",
    "instance_segmentation/yolov5n_seg":           "yolov5n_seg",
    "instance_segmentation/yolov5s_seg":           "yolov5s_seg",
    "instance_segmentation/yolov8m_seg":           "yolov8m_seg",
    "instance_segmentation/yolov8n_seg":           "yolov8n_seg",
    "instance_segmentation/yolov8s_seg":           "yolov8s_seg",
    # ── obb_detection (1) ───────────────────────────────────────────
    "obb_detection/yolo26n_obb":           "yolo26n-obb",
    # ── object_detection (43) ───────────────────────────────────────
    "object_detection/centernet_resnet18":          None,
    "object_detection/damoyolol":                   "DamoYoloL",
    "object_detection/damoyolom":                   "DamoYoloM",
    "object_detection/damoyolos":                   "DamoYoloS",
    "object_detection/damoyolot":                   "DamoYoloT",
    "object_detection/damoyolo_tinynasl20_t":       "DamoYolo_tinynasL20_T",
    "object_detection/damoyolo_tinynasl25_s":       "DamoYolo_tinynasL25_S",
    "object_detection/damoyolo_tinynasl35_m":       "DamoYolo_tinynasL35_M",
    "object_detection/nanodet_repvgg":              "NanoDet_RepVGG",
    "object_detection/nanodet_repvgga1":            "NanoDet_RepVGGA1",
    "object_detection/ssdmv1":                      "SSDMV1",
    "object_detection/ssdmv2lite":                  "SSDMV2Lite",
    "object_detection/yolo26l":                     "yolo26l",
    "object_detection/yolo26m":                     "yolo26m",
    "object_detection/yolo26n":                     "yolo26n",
    "object_detection/yolo26s":                     "yolo26s",
    "object_detection/yolo26x":                     "yolo26x",
    "object_detection/yolov10b":                    "YOLOV10B",
    "object_detection/yolov10l":                    "YOLOV10L",
    "object_detection/yolov10m":                    "YOLOV10M",
    "object_detection/yolov10n":                    "YOLOV10N",
    "object_detection/yolov10s":                    "YOLOV10S",
    "object_detection/yolov10x":                    "YOLOV10X",
    "object_detection/yolov11l":                    "YOLOV11L",
    "object_detection/yolov11m":                    "YOLOV11M",
    "object_detection/yolov11n":                    "YOLOV11N",
    "object_detection/yolov11s":                    "YOLOV11S",
    "object_detection/yolov11x":                    "YOLOV11X",
    "object_detection/yolov12n":                    "YOLOV12N-1",
    "object_detection/yolov3":                      "YoloV3",
    "object_detection/yolov5l":                     "YoloV5L",
    "object_detection/yolov5m":                     "YoloV5M",
    "object_detection/yolov5m_6_1":                 "YoloV5M_6.1",
    "object_detection/yolov5n":                     "YoloV5N",
    "object_detection/yolov5s":                     "YoloV5S",
    "object_detection/yolov6n_0_1_0":               "YoloV6N_0.1.0",
    "object_detection/yolov6n_0_2_1":               "YoloV6n_0.2.1",
    "object_detection/yolov7":                      "YoloV7",
    "object_detection/yolov7e6":                    "YoloV7E6",
    "object_detection/yolov7tiny":                  "YoloV7Tiny",
    "object_detection/yolov8l":                     "YoloV8L",
    "object_detection/yolov8m":                     "YoloV8M",
    "object_detection/yolov8n":                     "YoloV8N",
    "object_detection/yolov8s":                     "YoloV8S",
    "object_detection/yolov8x":                     "YoloV8X",
    "object_detection/yolov9c":                     "YoloV9C",
    "object_detection/yolov9s":                     "YoloV9S",
    "object_detection/yolov9t":                     "YoloV9T",
    "object_detection/yolox_l_leaky":               "YoloX_L_Leaky",
    "object_detection/yoloxs":                      "YoloXS",
    "object_detection/yolox_s_leaky":               "YoloX_S_Leaky",
    "object_detection/yolox_s_wide_leaky":          "YoloX_S_Wide_Leaky",
    "object_detection/yoloxtiny":                   "YoloXTiny",
    # ── pose_estimation (3) ─────────────────────────────────────────
    "pose_estimation/yolov5pose":          "YOLOV5Pose640_1",
    "pose_estimation/yolov8m_pose":        "yolov8m_pose",
    "pose_estimation/yolov8s_pose":        "yolov8s_pose",
    # ── ppu (3) ─────────────────────────────────────────────────────
    "ppu/yolov5pose_ppu":                  "YOLOV5Pose_PPU",
    "ppu/yolov5s_ppu":                     "YOLOV5S_PPU",
    "ppu/yolov7_ppu":                      None,
    # ── semantic_segmentation (4) ───────────────────────────────────
    "semantic_segmentation/bisenetv1":               "BiSeNetV1",
    "semantic_segmentation/bisenetv2":               "BiSeNetV2",
    "semantic_segmentation/deeplabv3plusmobilenet":   "DeepLabV3PlusMobilenet",
    "semantic_segmentation/segformer_b0_512x1024":   None,
    # ── super_resolution (1) ────────────────────────────────────────
    "super_resolution/espcn_x4":           None,
}

# ======================================================================
# Task → sample image mapping
# ======================================================================
TASK_IMAGE_MAP = {
    "object_detection":       "sample/img/sample_dog.jpg",
    "classification":         "sample/ILSVRC2012/0.jpeg",
    "face_detection":         "sample/img/sample_face.jpg",
    "pose_estimation":        "sample/img/sample_people.jpg",
    "instance_segmentation":  "sample/img/sample_street.jpg",
    "semantic_segmentation":  "sample/img/sample_street.jpg",
    "depth_estimation":       "sample/img/sample_kitchen.jpg",
    "hand_landmark":          "sample/img/sample_hand.jpg",
    "embedding":              "sample/img/sample_face.jpg",
    "obb_detection":          "sample/dota8_test/P0177.png",
    "image_denoising":        "sample/img/sample_denoising.jpg",
    "image_enhancement":      "sample/img/sample_lowlight.jpg",
    "super_resolution":       "sample/img/sample_superresolution.png",
    "ppu":                    "sample/img/sample_dog.jpg",
}

# Per-model image overrides (used instead of task defaults)
MODEL_IMAGE_OVERRIDE = {
    "ppu/yolov5pose_ppu":  "sample/img/sample_people.jpg",
}


# ======================================================================
# Utilities
# ======================================================================
def _resolve_input_shape(shape):
    """Replicate sync_runner._resolve_input_shape logic."""
    if len(shape) >= 4:
        if shape[-1] in (1, 3, 4):
            return shape[1], shape[2]   # NHWC → (H, W)
        return shape[2], shape[3]       # NCHW → (H, W)
    if len(shape) == 3:
        return shape[1], shape[2]
    if len(shape) == 2:
        return 1, shape[1]
    return 1, 1


def _load_factory(factory_key: str):
    """Load factory instance from factory_key='task/model'."""
    module_path = f"{factory_key.replace('/', '.')}.factory"
    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (isinstance(obj, type)
                and attr_name.endswith("Factory")
                and attr_name != "Factory"):
            try:
                return obj()
            except TypeError:
                continue
    raise RuntimeError(f"No Factory class in {module_path}")


def _load_config(factory_key: str, factory):
    """Auto-apply config.json if present."""
    cfg_path = SRC / factory_key / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg and hasattr(factory, "load_config"):
            factory.load_config(cfg)


# ======================================================================
# Single model execution
# ======================================================================
def run_single_model(factory_key: str, dxnn_name: str) -> dict:
    t0 = time.perf_counter()
    task = factory_key.split("/")[0]
    model_name = factory_key.split("/")[1]

    # Locate .dxnn model file
    dxnn_path = MODELS_DIR / f"{dxnn_name}.dxnn"
    if not dxnn_path.exists():
        return {"status": "SKIP", "msg": f"dxnn missing: {dxnn_name}"}

    # Sample image (per-model override takes priority)
    img_rel = MODEL_IMAGE_OVERRIDE.get(factory_key) or TASK_IMAGE_MAP.get(task)
    if not img_rel:
        return {"status": "SKIP", "msg": f"no image for task={task}"}
    img_path = ROOT / img_rel
    img = cv2.imread(str(img_path))
    if img is None:
        return {"status": "FAIL", "msg": f"imread fail: {img_rel}"}

    # InferenceEngine
    ie = InferenceEngine(str(dxnn_path))
    info = ie.get_input_tensors_info()
    shape = info[0]["shape"]
    if len(shape) < 3:
        return {"status": "SKIP", "msg": f"non-image shape={shape}"}
    input_h, input_w = _resolve_input_shape(shape)

    # Factory
    factory = _load_factory(factory_key)
    _load_config(factory_key, factory)
    preprocessor  = factory.create_preprocessor(input_w, input_h)
    postprocessor = factory.create_postprocessor(input_w, input_h)
    visualizer    = factory.create_visualizer()

    # Inference pipeline
    tensor, ctx = preprocessor.process(img)
    outputs = ie.run([tensor])
    results = postprocessor.process(outputs, ctx)
    vis = visualizer.visualize(img, results)

    # Save result
    elapsed = time.perf_counter() - t0
    if vis is not None:
        out = OUTPUT_DIR / task / f"{model_name}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        return {"status": "OK", "msg": f"{elapsed:.2f}s", "elapsed": elapsed}

    return {"status": "FAIL", "msg": "visualize returned None"}


# ======================================================================
# Main
# ======================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = [(k, v) for k, v in FACTORY_TO_DXNN.items() if v is not None]
    n = len(entries)
    ok = skip = fail = 0
    failures = []

    print(f"\n{'='*70}")
    print(f"  Full-model inference visualization test  ({n} models)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    for i, (key, dxnn) in enumerate(entries, 1):
        label = f"[{i:3d}/{n}] {key}"
        try:
            r = run_single_model(key, dxnn)
            s = r["status"]
            if s == "OK":
                ok += 1
                print(f"  OK   {label}  ({r['msg']})")
            elif s == "SKIP":
                skip += 1
                print(f"  SKIP {label}  — {r['msg']}")
            else:
                fail += 1
                print(f"  FAIL {label}  — {r['msg']}")
                failures.append((key, r["msg"]))
        except Exception as e:
            fail += 1
            msg = f"{type(e).__name__}: {e}"
            print(f"  FAIL {label}  — {msg}")
            failures.append((key, msg))
            traceback.print_exc()
            print()

    # Summary
    print(f"\n{'='*70}")
    print(f"  OK={ok}  SKIP={skip}  FAIL={fail}  TOTAL={ok+skip+fail}")
    print(f"{'='*70}")
    if failures:
        print("\n  Failures:")
        for k, m in failures:
            print(f"    ✗ {k}: {m}")

    skipped_no_dxnn = [k for k, v in FACTORY_TO_DXNN.items() if v is None]
    print(f"\n  Excluded (no .dxnn file) ({len(skipped_no_dxnn)}):")
    for k in skipped_no_dxnn:
        print(f"    - {k}")

    print()
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
