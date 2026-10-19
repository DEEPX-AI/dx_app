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

for _stream in (sys.__stdout__, sys.__stderr__, sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, OSError, ValueError):
        pass  # None (pythonw), or a stream without reconfigure()

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
def _registry_entries():
    """``[(task/family, variant, dxnn stem)]`` straight from the registry.

    Replaces a hard-coded task/model -> dxnn table. Under the dx-modelzoo family layout
    one directory serves every variant of its family, so the table would have had to
    list each variant against a shared directory anyway -- and the old one had already
    drifted onto pre-rename .dxnn names like "AlexNet" that are no longer on disk.
    """
    reg = json.loads((ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))
    return [(f"{e['task']}/{e['family']}", e["variant"], e["dxnn_file"][: -len(".dxnn")])
            for e in reg]

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
    "obb_detection":          "sample/img/sample_airport_satellite_view.png",
    "image_denoising":        "sample/img/sample_denoising.jpg",
    "image_enhancement":      "sample/img/sample_lowlight.jpg",
    "super_resolution":       "sample/img/sample_lowres275x150.png",   # ESPCN default
    "ppu":                    "sample/img/sample_dog.jpg",
}

# Per-model image overrides (used instead of task defaults)
MODEL_IMAGE_OVERRIDE = {
    # Super-resolution: ESPCN upscales a 275x150 crop; Real-ESRGAN takes a
    # smaller 165x90 one so the x8 output stays a sane size.
    "super_resolution/espcn_x2":      "sample/img/sample_lowres275x150.png",
    "super_resolution/espcn_x3":      "sample/img/sample_lowres275x150.png",
    "super_resolution/espcn_x4":      "sample/img/sample_lowres275x150.png",
    "super_resolution/realesrgan_x2": "sample/img/sample_lowres165x90.png",
    "super_resolution/realesrgan_x4": "sample/img/sample_lowres165x90.png",
    "super_resolution/realesrgan_x8": "sample/img/sample_lowres165x90.png",
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


def _load_factory(factory_key: str, variant: str):
    """Load the family factory BY PATH and bind it to *variant*.

    Path-based rather than ``import_module``: a family directory is not an importable
    package under every task name (``3d_object_detection`` starts with a digit) and the
    factory module of a digit-leading family carries an ``n_`` prefix.
    """
    import importlib.util

    fdir = SRC / factory_key / "factory"
    cands = sorted(fdir.glob("*_factory.py"))
    if not cands:
        raise RuntimeError(f"no factory module in {fdir}")
    key = f"fam_{factory_key.replace('/', '_')}"
    pkg = sys.modules.get(key)
    if pkg is None:
        import importlib.machinery
        pkg = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(key, None, is_package=True))
        pkg.__path__ = [str(SRC / factory_key)]
        sys.modules[key] = pkg
    spec = importlib.util.spec_from_file_location(f"{key}.factory", cands[0])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (isinstance(obj, type) and attr_name.endswith("Factory")
                and attr_name != "Factory" and obj.__module__ == mod.__name__):
            return obj(variant=variant)
    raise RuntimeError(f"No Factory class in {cands[0]}")


def _variant_config(factory_key: str, variant: str) -> dict:
    p = SRC / factory_key / variant / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


# ======================================================================
# Single model execution
# ======================================================================
def run_single_model(factory_key: str, dxnn_name: str, variant: str) -> dict:
    t0 = time.perf_counter()
    task = factory_key.split("/")[0]
    model_name = variant

    # Locate .dxnn model file
    dxnn_path = MODELS_DIR / f"{dxnn_name}.dxnn"
    if not dxnn_path.exists():
        return {"status": "SKIP", "msg": f"dxnn missing: {dxnn_name}"}

    # Sample image (per-model override takes priority)
    # The variant config's default_image was baked from the legacy task tables during
    # the restructure, so it is the per-variant answer those tables used to give.
    vcfg = _variant_config(factory_key, variant)
    img_rel = (vcfg.get("default_image")
               or MODEL_IMAGE_OVERRIDE.get(factory_key)
               or TASK_IMAGE_MAP.get(task))
    if not img_rel:
        return {"status": "SKIP", "msg": f"no image for task={task}"}
    img_path = ROOT / img_rel
    # An image-pair task (face_recognition, person re-id) points at a DIRECTORY; this
    # script visualises one frame, so take its first image rather than failing imread.
    if img_path.is_dir():
        members = sorted(q for q in img_path.iterdir()
                         if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not members:
            return {"status": "SKIP", "msg": f"no image in dir: {img_rel}"}
        img_path = members[0]
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
    factory = _load_factory(factory_key, variant)
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

    # An image-pair task (face_recognition, re-id) compares two images and has no
    # single-frame visualisation, so a None here is expected rather than a failure.
    if vcfg.get("image_only"):
        return {"status": "SKIP", "msg": "pair task: no single-frame visualization"}
    return {"status": "FAIL", "msg": "visualize returned None"}


# ======================================================================
# Main
# ======================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = _registry_entries()
    n = len(entries)
    ok = skip = fail = 0
    failures = []

    print(f"\n{'='*70}")
    print(f"  Full-model inference visualization test  ({n} models)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    for i, (key, variant, dxnn) in enumerate(entries, 1):
        label = f"[{i:3d}/{n}] {key}"
        try:
            r = run_single_model(key, dxnn, variant)
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

    skipped_no_dxnn = []
    print(f"\n  Excluded (no .dxnn file) ({len(skipped_no_dxnn)}):")
    for k in skipped_no_dxnn:
        print(f"    - {k}")

    print()
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
