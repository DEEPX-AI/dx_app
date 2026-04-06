#!/usr/bin/env python3
"""
verify_inference_output.py — Numerical verification for DX-APP model outputs.

Reads a JSON dump produced by SyncRunner (DXAPP_VERIFY=1) and validates
the postprocess results against task-specific rules.

Exit codes:
    0 = PASS
    1 = FAIL (domain violation)
    2 = WARN (soft violation, e.g. low detections on unknown image)
    3 = ERROR (file not found, bad JSON, etc.)

Usage:
    python3 verify_inference_output.py <verify_result.json>
    python3 verify_inference_output.py <verify_result.json> --rules inference_verify_rules.json
    python3 verify_inference_output.py <dir_of_jsons>  --summary
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

_MSG_CONTAINS_NAN = "contains NaN"

# ============================================================================
# Validators per task
# ============================================================================

class ValidationError:
    """Single validation failure."""
    def __init__(self, field: str, message: str, severity: str = "FAIL"):
        self.field = field
        self.message = message
        self.severity = severity  # FAIL or WARN

    def __repr__(self):
        return f"[{self.severity}] {self.field}: {self.message}"


def _check_bbox(bbox: List[float], img_w: int, img_h: int) -> List[ValidationError]:
    """Validate [x1, y1, x2, y2] bounding box."""
    errors = []
    if len(bbox) != 4:
        errors.append(ValidationError("bbox", f"expected 4 values, got {len(bbox)}"))
        return errors
    x1, y1, x2, y2 = bbox
    if x2 <= x1:
        errors.append(ValidationError("bbox", f"x2 ({x2:.1f}) <= x1 ({x1:.1f})"))
    if y2 <= y1:
        errors.append(ValidationError("bbox", f"y2 ({y2:.1f}) <= y1 ({y1:.1f})"))
    # Allow small margin (5%) outside image due to letterbox mapping
    margin_x = img_w * 0.05
    margin_y = img_h * 0.05
    if x1 < -margin_x or y1 < -margin_y or x2 > img_w + margin_x or y2 > img_h + margin_y:
        errors.append(ValidationError(
            "bbox",
            f"[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] out of image "
            f"({img_w}x{img_h})",
            severity="WARN"))
    return errors


def _check_conf(conf: float) -> List[ValidationError]:
    """Validate confidence score."""
    errors = []
    if not (0.0 <= conf <= 1.0):
        errors.append(ValidationError("conf", f"{conf:.4f} not in [0, 1]"))
    return errors


def _check_class_id(cls_id: int, num_classes: Optional[int]) -> List[ValidationError]:
    errors = []
    if cls_id < 0:
        errors.append(ValidationError("class_id", f"negative: {cls_id}"))
    if num_classes is not None and cls_id >= num_classes:
        errors.append(ValidationError("class_id", f"{cls_id} >= num_classes ({num_classes})"))
    return errors


def validate_detection(data: dict, rules: dict) -> List[ValidationError]:
    """Validate object_detection / face_detection results."""
    errors = []
    detections = data.get("detections", [])
    img_w = data.get("image_width", 0)
    img_h = data.get("image_height", 0)
    num_classes = data.get("num_classes")
    min_det = rules.get("min_detections", 0)

    if len(detections) < min_det:
        errors.append(ValidationError(
            "count", f"detections={len(detections)} < min_detections={min_det}"))

    for i, det in enumerate(detections):
        bbox = det.get("bbox", [])
        errors.extend(_check_bbox(bbox, img_w, img_h))
        errors.extend(_check_conf(det.get("conf", -1)))
        errors.extend(_check_class_id(det.get("class_id", -1), num_classes))

    return errors


def validate_pose(data: dict, rules: dict) -> List[ValidationError]:
    """Validate pose_estimation results."""
    errors = []
    detections = data.get("detections", [])
    img_w = data.get("image_width", 0)
    img_h = data.get("image_height", 0)
    min_det = rules.get("min_detections", 0)

    if len(detections) < min_det:
        errors.append(ValidationError(
            "count", f"detections={len(detections)} < min_detections={min_det}"))

    for i, det in enumerate(detections):
        errors.extend(_check_bbox(det.get("bbox", []), img_w, img_h))
        errors.extend(_check_conf(det.get("conf", -1)))
        kpts = det.get("keypoints", [])
        if kpts:
            for ki, kpt in enumerate(kpts):
                _, _ = kpt.get("x", 0), kpt.get("y", 0)
                c = kpt.get("conf", 0)
                if not (0 <= c <= 1):
                    errors.append(ValidationError(
                        f"det[{i}].kpt[{ki}].conf", f"{c:.3f} not in [0,1]"))

    return errors


def validate_obb(data: dict, rules: dict) -> List[ValidationError]:
    """Validate obb_detection results."""
    errors = []
    detections = data.get("detections", [])
    min_det = rules.get("min_detections", 0)

    if len(detections) < min_det:
        errors.append(ValidationError(
            "count", f"detections={len(detections)} < min_detections={min_det}"))

    for i, det in enumerate(detections):
        errors.extend(_check_conf(det.get("conf", -1)))
        angle = det.get("angle", 0)
        if not (-math.pi <= angle <= math.pi):
            errors.append(ValidationError(
                f"det[{i}].angle", f"{angle:.3f} not in [-π, π]"))

    return errors


def validate_classification(data: dict, rules: dict) -> List[ValidationError]:
    """Validate classification results."""
    errors = []
    results = data.get("classifications", [])
    if not results:
        errors.append(ValidationError("count", "no classification results"))
        return errors

    top = results[0]
    conf = top.get("conf", -1)
    if not (0.0 <= conf <= 1.0):
        errors.append(ValidationError("conf", f"top-1 conf={conf:.4f} not in [0,1]"))
    cls_id = top.get("class_id", -1)
    if cls_id < 0:
        errors.append(ValidationError("class_id", f"negative: {cls_id}"))

    # Check softmax sum if top_k provided
    top_k = data.get("top_k_confs")
    if top_k and isinstance(top_k, list):
        s = sum(top_k)
        if not (0.9 <= s <= 1.1):
            errors.append(ValidationError(
                "softmax", f"top_k sum={s:.4f}, expected ~1.0", severity="WARN"))

    return errors


def validate_segmentation(data: dict, rules: dict) -> List[ValidationError]:
    """Validate semantic_segmentation results."""
    errors = []
    mask_shape = data.get("mask_shape", [])
    unique_classes = data.get("unique_classes", 0)
    min_classes = rules.get("min_classes", 2)

    if len(mask_shape) < 2:
        errors.append(ValidationError("mask_shape", f"invalid: {mask_shape}"))
    elif mask_shape[0] <= 0 or mask_shape[1] <= 0:
        errors.append(ValidationError("mask_shape", f"zero dim: {mask_shape}"))

    if unique_classes < min_classes:
        errors.append(ValidationError(
            "unique_classes",
            f"{unique_classes} < min_classes={min_classes}"))

    return errors


def validate_instance_seg(data: dict, rules: dict) -> List[ValidationError]:
    """Validate instance_segmentation results."""
    errors = validate_detection(data, rules)
    # Additional: mask exists per detection
    detections = data.get("detections", [])
    for i, det in enumerate(detections):
        if not det.get("has_mask", False):
            errors.append(ValidationError(
                f"det[{i}].mask", "missing instance mask", severity="WARN"))
    return errors


def validate_depth(data: dict, rules: dict) -> List[ValidationError]:
    """Validate depth_estimation results."""
    errors = []
    stats = data.get("output_stats", {})
    if stats.get("has_nan", False):
        errors.append(ValidationError("output", _MSG_CONTAINS_NAN))
    if stats.get("has_inf", False):
        errors.append(ValidationError("output", "contains Inf"))
    if stats.get("std", 0) <= 0:
        errors.append(ValidationError("output", "std=0 (flat output)"))
    if stats.get("min", 0) < 0:
        errors.append(ValidationError("output", f"min={stats['min']:.3f} < 0",
                                      severity="WARN"))
    return errors


def validate_image_output(data: dict, rules: dict) -> List[ValidationError]:
    """Validate image-to-image tasks (denoising, SR, enhancement)."""
    errors = []
    stats = data.get("output_stats", {})
    if stats.get("has_nan", False):
        errors.append(ValidationError("output", _MSG_CONTAINS_NAN))
    if stats.get("has_inf", False):
        errors.append(ValidationError("output", "contains Inf"))

    out_shape = data.get("output_shape", [])
    in_shape = data.get("input_image_shape", [])

    task = data.get("task", "")
    if task == "super_resolution" and out_shape and in_shape:
        # output should be larger than input
        if len(out_shape) >= 2 and len(in_shape) >= 2:
            if out_shape[0] <= in_shape[0] and out_shape[1] <= in_shape[1]:
                errors.append(ValidationError(
                    "output_shape",
                    f"SR output {out_shape} not larger than input {in_shape}"))

    return errors


def validate_embedding(data: dict, rules: dict) -> List[ValidationError]:
    """Validate embedding results."""
    errors = []
    vec = data.get("embedding", {})
    dim = vec.get("dim", 0)
    norm = vec.get("l2_norm", 0)
    has_nan = vec.get("has_nan", False)

    if dim <= 0:
        errors.append(ValidationError("embedding", f"dim={dim}"))
    if has_nan:
        errors.append(ValidationError("embedding", _MSG_CONTAINS_NAN))
    if norm <= 0:
        errors.append(ValidationError("embedding", f"l2_norm={norm:.4f} <= 0"))
    return errors


def validate_hand_landmark(data: dict, rules: dict) -> List[ValidationError]:
    """Validate hand_landmark results."""
    errors = []
    detections = data.get("detections", [])
    min_det = rules.get("min_detections", 0)

    if len(detections) < min_det:
        errors.append(ValidationError(
            "count", f"detections={len(detections)} < min_detections={min_det}"))

    for i, det in enumerate(detections):
        landmarks = det.get("landmarks", [])
        if landmarks:
            expected = 21
            if len(landmarks) != expected:
                errors.append(ValidationError(
                    f"det[{i}].landmarks",
                    f"count={len(landmarks)}, expected={expected}"))
    return errors


# Dispatcher
VALIDATORS = {
    "object_detection": validate_detection,
    "face_detection": validate_detection,
    "pose_estimation": validate_pose,
    "obb_detection": validate_obb,
    "classification": validate_classification,
    "semantic_segmentation": validate_segmentation,
    "instance_segmentation": validate_instance_seg,
    "depth_estimation": validate_depth,
    "image_denoising": validate_image_output,
    "super_resolution": validate_image_output,
    "image_enhancement": validate_image_output,
    "embedding": validate_embedding,
    "ppu": validate_detection,
    "hand_landmark": validate_hand_landmark,
}


# ============================================================================
# Main
# ============================================================================

def verify_single(json_path: str, rules_path: Optional[str] = None) -> Tuple[str, List[ValidationError]]:
    """Verify a single model output JSON. Returns (model_name, errors)."""
    with open(json_path) as f:
        data = json.load(f)

    task = data.get("task", "")
    model = data.get("model", os.path.basename(json_path))
    image = data.get("input_image", "")

    # Load rules
    rules = {}
    if rules_path and os.path.isfile(rules_path):
        with open(rules_path) as f:
            all_rules = json.load(f)
        # Match image path
        for rule_img, task_rules in all_rules.items():
            if rule_img in image or image.endswith(rule_img):
                rules = task_rules.get(task, {})
                break

    validator = VALIDATORS.get(task)
    if validator is None:
        return model, [ValidationError("task", f"unknown task: {task}")]

    errors = validator(data, rules)
    return model, errors


def _resolve_input_files(path_str: str) -> List[Path]:
    """Resolve input path to a list of JSON files, or exit on error."""
    p = Path(path_str)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if files:
            return files
        print(f"[ERROR] No .json files found in: {path_str}", file=sys.stderr)
        sys.exit(3)
    print(f"[ERROR] Not found: {path_str}", file=sys.stderr)
    sys.exit(3)


def _classify_errors(errors: List[ValidationError], strict: bool) -> str:
    """Return 'FAIL', 'WARN', or 'PASS' for a list of validation errors."""
    has_fail = any(e.severity == "FAIL" for e in errors)
    has_warn = any(e.severity == "WARN" for e in errors)
    if has_fail or (strict and has_warn):
        return "FAIL"
    if has_warn:
        return "WARN"
    return "PASS"


_STATUS_SYM   = {"PASS": "\u2713", "WARN": "\u26a0", "FAIL": "\u2717"}
_STATUS_COLOR = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}


def _print_result_line(model: str, status: str, errors: List[ValidationError]):
    """Print a single model's verification result."""
    sym = _STATUS_SYM[status]
    color = _STATUS_COLOR[status]
    print(f"{color}{sym}\033[0m {model}: {status}")
    for e in errors:
        print(f"    {e}")


def _print_summary(total: int, passed: int, warned: int, failed: int,
                   results: List[Tuple[str, str, List[ValidationError]]]):
    """Print final summary table and failed-model details."""
    print(f"\n{'='*50}")
    print("  Verification Summary")
    print(f"  Total: {total}  PASS: {passed}  WARN: {warned}  FAIL: {failed}")
    print(f"{'='*50}")
    for model, status, errors in results:
        if status != "FAIL":
            continue
        print(f"    \u2717 {model}")
        for e in errors:
            if e.severity == "FAIL":
                print(f"        {e}")


def main():
    parser = argparse.ArgumentParser(description="DX-APP numerical output verifier")
    parser.add_argument("path", help="Single .json file or directory of .json files")
    parser.add_argument("--rules", default=None, help="inference_verify_rules.json path")
    parser.add_argument("--summary", action="store_true", help="Print summary table")
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN as FAIL")
    args = parser.parse_args()

    files = _resolve_input_files(args.path)

    total = 0
    passed = 0
    warned = 0
    failed = 0
    results = []

    for f in files:
        total += 1
        model, errors = verify_single(str(f), args.rules)
        status = _classify_errors(errors, args.strict)

        if status == "FAIL":
            failed += 1
        elif status == "WARN":
            warned += 1
        else:
            passed += 1

        results.append((model, status, errors))

        if not args.summary:
            _print_result_line(model, status, errors)

    if args.summary or total > 1:
        _print_summary(total, passed, warned, failed, results)

    # Exit code
    if failed > 0:
        sys.exit(1)
    if warned > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
