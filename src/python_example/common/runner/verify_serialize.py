#!/usr/bin/env python3
"""
Serialize postprocess results to JSON for numerical verification.

Activated by DXAPP_VERIFY=1 environment variable.
Writes the last frame to ``{model}.json`` and every frame to
``{model}.frames.jsonl`` (one compact record per line with ``"frame": n``,
0-based per process) in ``$DXAPP_VERIFY_DIR`` (default ``logs/verify``).

Supported result types:
  - DetectionResult, FaceResult  → detections[]
  - PoseResult                   → detections[] + keypoints[]
  - InstanceSegResult            → detections[] + has_mask
  - SegmentationResult           → mask_shape, unique_classes
  - ClassificationResult         → classifications[]
  - OBBResult                    → detections[] + angle
  - EmbeddingResult              → embedding{dim, l2_norm, has_nan}
  - RetrievalResult              → embedding{...} + matches[]
  - HandLandmarkResult           → detections[] + landmarks[]
  - SuperResolutionResult        → output_shape, output_stats
  - EnhancedImageResult          → output_shape, output_stats
  - RestorationResult            → output_shape, output_stats
  - YOLOPv2Result                → detections[] + drivable_stats, lane_stats
  - Detection3DResult            → detections[] {bev, center, dims, yaw}
  - SuperPointResult             → detections[0].keypoints[] + descriptor_dim
  - DopeResult                   → detections[] + keypoints[] (pixels), has_pose
  - FaceAlignmentResult          → detections[] {landmarks_2d, pose, params_size}
  - raw numpy array              → output_stats
  - list/tuple of 2+ raw arrays  → output_stats_list
"""

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def is_verify_enabled() -> bool:
    """Check if DXAPP_VERIFY=1 is set."""
    return os.environ.get("DXAPP_VERIFY", "0") == "1"


def _get_verify_dir() -> Path:
    """Return and create the verify output directory."""
    d = Path(os.environ.get("DXAPP_VERIFY_DIR", "logs/verify"))
    d.mkdir(parents=True, exist_ok=True)
    return d


#: Every verify write in the process goes through this lock (async workers,
#: tiled SR helpers); <stem>.frames.jsonl gets its frame numbers under it.
_WRITE_LOCK = threading.Lock()
_NEXT_FRAME: Dict[str, int] = {}


def _write_atomically(path: Path, text: str) -> None:
    """Write through a sibling temp file and os.replace(): a reader never sees a torn file."""
    tmp = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(str(tmp), str(path))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _np_stats(arr: np.ndarray) -> dict:
    """Safe statistics for a numpy array."""
    flat = arr.astype(np.float64).ravel()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(flat)) if flat.size > 0 else 0.0,
        "max": float(np.max(flat)) if flat.size > 0 else 0.0,
        "mean": float(np.mean(flat)) if flat.size > 0 else 0.0,
        "std": float(np.std(flat)) if flat.size > 0 else 0.0,
        "has_nan": bool(np.isnan(flat).any()),
        "has_inf": bool(np.isinf(flat).any()),
    }


def _det_entry(d):
    return {"bbox": list(map(float, d.box)), "conf": float(d.confidence),
            "class_id": int(d.class_id), "class_name": str(d.class_name)}


def _ser_detection(items, img_h, img_w):
    return {"image_height": img_h, "image_width": img_w,
            "detections": [_det_entry(d) for d in items]}


def _ser_face(items, img_h, img_w):
    return {
        "image_height": img_h, "image_width": img_w,
        "detections": [
            {"bbox": list(map(float, d.box)), "conf": float(d.confidence),
             "class_id": int(d.class_id),
             "keypoints": [{"x": float(kp.x), "y": float(kp.y), "conf": float(kp.confidence)}
                           for kp in (d.keypoints or [])]}
            for d in items
        ],
    }


def _ser_pose(items, img_h, img_w):
    return {
        "image_height": img_h, "image_width": img_w,
        "detections": [
            {"bbox": list(map(float, d.box)), "conf": float(d.confidence),
             "class_id": int(d.class_id),
             "keypoints": [{"x": float(kp.x), "y": float(kp.y), "conf": float(kp.confidence)}
                           for kp in d.keypoints]}
            for d in items
        ],
    }


def _ser_instance_seg(items, img_h, img_w):
    return {
        "image_height": img_h, "image_width": img_w,
        "detections": [
            {"bbox": list(map(float, d.box)), "conf": float(d.confidence),
             "class_id": int(d.class_id), "class_name": str(d.class_name),
             "has_mask": d.mask is not None and d.mask.size > 0,
             "mask_shape": list(d.mask.shape) if d.mask is not None and d.mask.size > 0 else []}
            for d in items
        ],
    }


def _ser_depth(items, img_h, img_w):
    first = items[0]
    return {
        "image_height": img_h, "image_width": img_w,
        "output_stats": _np_stats(first.depth_map)
        if hasattr(first, "depth_map") and first.depth_map is not None else {},
    }


def _ser_anomaly(items, img_h, img_w):
    """Heatmap stats plus the two scalars, so the C++ peer can be diffed against it.

    Neither tree serialised this result type: the family was added after the verify
    harness, and the two implementations then diverged unnoticed -- Python moved to the
    three-network EfficientAD combination while C++ kept one network's magnitude.
    """
    first = items[0]
    return {
        "image_height": img_h, "image_width": img_w,
        "output_stats": _np_stats(first.heatmap)
        if getattr(first, "heatmap", None) is not None else {},
        "score": float(first.score),
        "channels": int(first.channels),
    }


def _ser_segmentation(items, img_h, img_w):
    first = items[0]
    unique = int(len(np.unique(first.mask))) if first.mask.size > 0 else 0
    return {
        "image_height": img_h, "image_width": img_w,
        "mask_shape": [int(first.height), int(first.width)],
        "unique_classes": unique,
        "class_ids": [int(c) for c in first.class_ids],
    }


def _ser_classification(items, img_h, img_w):
    first = items[0]
    top_k_confs = [float(conf) for _, conf in first.top_k] if first.top_k else []
    return {
        "image_height": img_h, "image_width": img_w,
        "classifications": [
            {"class_id": int(first.class_id), "class_name": str(first.class_name),
             "conf": float(first.confidence)}
        ],
        "top_k_confs": top_k_confs,
    }


def _ser_obb(items, img_h, img_w):
    return {
        "image_height": img_h, "image_width": img_w,
        "detections": [
            {"cx": float(d.cx), "cy": float(d.cy),
             "width": float(d.width), "height": float(d.height),
             "angle": float(d.angle), "conf": float(d.confidence),
             "class_id": int(d.class_id), "class_name": str(d.class_name)}
            for d in items
        ],
    }


def _ser_embedding(items, img_h, img_w):
    first = items[0]
    vec = first.embedding
    return {
        "image_height": img_h, "image_width": img_w,
        "embedding": {
            "dim": int(vec.size) if vec is not None else 0,
            "l2_norm": float(np.linalg.norm(vec))
            if vec is not None and vec.size > 0 else 0.0,
            "has_nan": bool(np.isnan(vec).any())
            if vec is not None and vec.size > 0 else False,
            "model_type": str(first.model_type),
        },
    }


def _ser_retrieval(items, img_h, img_w):
    """Descriptor plus its ranking. Both are compared across trees: a matching
    descriptor with a different order means the gallery differs, not the model."""
    first = items[0]
    vec = np.asarray(first.embedding) if first.embedding is not None else np.array([])
    return {
        "image_height": img_h, "image_width": img_w,
        "embedding": {
            "dim": int(vec.size),
            "l2_norm": float(np.linalg.norm(vec)) if vec.size > 0 else 0.0,
            "has_nan": bool(np.isnan(vec).any()) if vec.size > 0 else False,
            "model_type": str(first.model_type),
        },
        "gallery": {"name": str(first.gallery_name),
                    "size": int(first.gallery_size)},
        "matches": [
            {"rank": int(m.rank), "score": float(m.score),
             "path": str(m.path), "label": str(m.label)}
            for m in (first.matches or [])
        ],
    }


def _ser_hand_landmark(items, img_h, img_w):
    return {
        "image_height": img_h, "image_width": img_w,
        "detections": [
            {"confidence": float(d.confidence), "handedness": str(d.handedness),
             "landmarks": [{"x": float(d.landmarks[i, 0]),
                            "y": float(d.landmarks[i, 1]),
                            "z": float(d.landmarks[i, 2])}
                           for i in range(d.landmarks.shape[0])]
             if d.landmarks is not None and d.landmarks.size > 0 else []}
            for d in items
        ],
    }


def _ser_image_output(items, img_h, img_w):
    """SuperResolutionResult / EnhancedImageResult — identical schema."""
    first = items[0]
    return {
        "image_height": img_h, "image_width": img_w,
        "output_shape": list(first.output_image.shape)
        if first.output_image.size > 0 else [],
        "input_image_shape": [img_h, img_w],
        "output_stats": _np_stats(first.output_image)
        if first.output_image.size > 0 else {},
    }


def _ser_panoptic(items, img_h, img_w):
    first = items[0]
    return {"image_height": img_h, "image_width": img_w,
            "detections": [_det_entry(d) for d in first.detections],
            "drivable_stats": _np_stats(np.asarray(first.drivable_mask)),
            "lane_stats": _np_stats(np.asarray(first.lane_mask))}


def _ser_detection3d(items, img_h, img_w):
    return {"image_height": img_h, "image_width": img_w, "detections": [
        {"class_id": int(d.class_id), "class_name": str(d.class_name), "conf": float(d.confidence),
         "bev": [float(d.bev_x), float(d.bev_y), float(d.bev_w), float(d.bev_h)],
         "center": [float(d.x3d), float(d.y3d), float(d.z3d)],
         "dims": [float(d.dim_h), float(d.dim_w), float(d.dim_l)],
         "yaw": float(d.yaw)} for d in items]}


def _ser_superpoint(items, img_h, img_w):
    first = items[0]
    desc = np.asarray(first.descriptors)
    return {"image_height": img_h, "image_width": img_w,
            "detections": [{"bbox": [], "conf": 1.0, "keypoints": [
                {"x": float(x), "y": float(y), "conf": float(s)}
                for (x, y), s in zip(first.keypoints, first.scores)]}],
            "descriptor_dim": int(desc.shape[1]) if desc.ndim == 2 else 0}


def _ser_dope(items, img_h, img_w):
    return {"image_height": img_h, "image_width": img_w, "detections": [
        {"bbox": [], "conf": float(d.confidence),
         "keypoints": [{"x": float(k[0]) * img_w, "y": float(k[1]) * img_h, "conf": float(c)}
                       for k, c in zip(np.asarray(d.keypoints), np.asarray(d.all_conf))],
         "has_pose": d.pose is not None} for d in items]}


def _ser_face_alignment(items, img_h, img_w):
    return {"image_height": img_h, "image_width": img_w, "detections": [
        {"landmarks_2d": [{"x": float(p[0]), "y": float(p[1])} for p in np.asarray(d.landmarks_2d)],
         "pose": [float(v) for v in d.pose],
         "params_size": int(np.asarray(d.params).size)} for d in items]}


_SERIALIZER_MAP = {
    "DetectionResult":        _ser_detection,
    "FaceResult":             _ser_face,
    "PoseResult":             _ser_pose,
    "InstanceSegResult":      _ser_instance_seg,
    "DepthResult":            _ser_depth,
    "AnomalyResult":          _ser_anomaly,
    "SegmentationResult":     _ser_segmentation,
    "ClassificationResult":   _ser_classification,
    "OBBResult":              _ser_obb,
    "EmbeddingResult":        _ser_embedding,
    "RetrievalResult":        _ser_retrieval,
    "HandLandmarkResult":     _ser_hand_landmark,
    "SuperResolutionResult":  _ser_image_output,
    "EnhancedImageResult":    _ser_image_output,
    "RestorationResult":      _ser_image_output,
    "YOLOPv2Result":          _ser_panoptic,
    "Detection3DResult":      _ser_detection3d,
    "SuperPointResult":       _ser_superpoint,
    "DopeResult":             _ser_dope,
    "FaceAlignmentResult":    _ser_face_alignment,
}


def _serialize_results(results: Any, image_hw: tuple) -> dict:
    """
    Convert postprocess results to a JSON-serializable dictionary.

    Parameters
    ----------
    results : list or single result
        Output from postprocessor.process()
    image_hw : tuple (height, width)
        Original image dimensions for metadata.

    Returns
    -------
    dict with standardized fields depending on result type.
    """
    img_h, img_w = image_hw

    # Handle empty results
    if isinstance(results, (list, tuple)) and len(results) == 0:
        return {"image_height": img_h, "image_width": img_w, "detections": []}

    items = results if isinstance(results, (list, tuple)) else [results]
    first = items[0]
    cls_name = type(first).__name__

    # Dispatch by result type name
    serializer = _SERIALIZER_MAP.get(cls_name)
    if serializer is not None:
        return serializer(items, img_h, img_w)

    # Raw C++ postprocess output (``*_cpp_postprocess`` without a convert function)
    if isinstance(first, np.ndarray):
        if len(items) > 1 and all(isinstance(x, np.ndarray) for x in items):
            return {"image_height": img_h, "image_width": img_w,
                    "output_stats_list": [_np_stats(x) for x in items]}
        return {"image_height": img_h, "image_width": img_w,
                "output_stats": _np_stats(first)}

    # Truly unknown
    return {"image_height": img_h, "image_width": img_w,
            "result_type": cls_name, "repr": str(results)[:500]}


def dump_verify_json(
    results: Any,
    image_path: str,
    model_path: str,
    task: str,
    image_hw: tuple,
    verbose: bool = False,
) -> Optional[str]:
    """
    Serialize results; write them to ``<model>.json`` (the last frame) and
    append them to ``<model>.frames.jsonl`` (every frame, with ``"frame"``).

    Thread-safe; the JSON is replaced atomically.

    Parameters
    ----------
    results : postprocess output
    image_path : path to input image
    model_path : path to .dxnn file
    task : task category string
    image_hw : (height, width)

    Returns
    -------
    Path to written JSON file, or None on error.
    """
    try:
        data = _serialize_results(results, image_hw)
        data["task"] = task
        data["model"] = os.path.basename(model_path)
        data["model_path"] = model_path
        data["input_image"] = image_path

        with _WRITE_LOCK:
            verify_dir = _get_verify_dir()
            model_stem = Path(model_path).stem
            json_path = verify_dir / f"{model_stem}.json"
            frames_path = verify_dir / f"{model_stem}.frames.jsonl"
            _write_atomically(json_path, json.dumps(data, indent=2, ensure_ascii=False))
            frame = _NEXT_FRAME.get(str(frames_path), 0)
            record = dict(data)
            record["frame"] = frame
            with open(frames_path, "w" if frame == 0 else "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Only a written record uses up its number: until the first write
            # succeeds, the next dump still truncates (never appends to an old run).
            _NEXT_FRAME[str(frames_path)] = frame + 1

        if verbose:
            print(f"[VERIFY] Dumped → {json_path}")
        return str(json_path)

    except Exception as e:
        print(f"[DXAPP] [WARN] verify_serialize failed: {e}")
        return None
