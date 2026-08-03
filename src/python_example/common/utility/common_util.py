"""
Common utility functions for neural network operations.
"""

import os
import subprocess
from typing import Any, List, Tuple
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Screen-aware display window
# ---------------------------------------------------------------------------
_window_initialized = False


def _get_screen_resolution():
    """Return (width, height) of the primary monitor.

    Detection order:
      1. DXAPP_SCREEN_W / DXAPP_SCREEN_H environment variables
      2. xdpyinfo (X11)
      3. Fallback 1920×1080
    """
    env_w = os.environ.get("DXAPP_SCREEN_W")
    env_h = os.environ.get("DXAPP_SCREEN_H")
    if env_w and env_h:
        try:
            w, h = int(env_w), int(env_h)
            if w > 0 and h > 0:
                return w, h
        except ValueError:
            pass

    try:
        result = subprocess.run(
            ["xdpyinfo"], capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.splitlines():
            if "dimensions" in line:
                # "  dimensions:    2560x1440 pixels ..."
                part = line.split(":")[1].strip().split()[0]
                w, h = part.split("x")
                return int(w), int(h)
    except Exception:
        pass

    return 1920, 1080


def show_output(img, winname: str = "Output"):
    """Display *img* in a resizable OpenCV window.

    On the first call the window is created and sized to half the screen
    width and half the screen height (≈ 1/4 screen area), preserving the
    frame's aspect ratio.  Subsequent calls just update the image.
    """
    if img is None:
        return
    global _window_initialized
    if not _window_initialized:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        screen_w, screen_h = _get_screen_resolution()
        target_w = screen_w // 2
        target_h = screen_h // 2
        h, w = img.shape[:2]
        if h > 0 and w > 0:
            scale = min(target_w / w, target_h / h)
            cv2.resizeWindow(winname, int(w * scale), int(h * scale))
        _window_initialized = True
    cv2.imshow(winname, img)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid activation.
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid-activated array
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax activation.
    
    Args:
        x: Input array
        axis: Axis to apply softmax over
        
    Returns:
        Softmax-activated array
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def argmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Get argmax indices.
    
    Args:
        x: Input array
        axis: Axis to find argmax over
        
    Returns:
        Array of indices
    """
    return np.argmax(x, axis=axis)


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union for two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, 
        iou_threshold: float = 0.45) -> List[int]:
    """
    Apply Non-Maximum Suppression.
    
    Args:
        boxes: Array of boxes [N, 4] in [x1, y1, x2, y2] format
        scores: Array of confidence scores [N]
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(int(current))
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        remaining = indices[1:]
        ious = np.array([iou(boxes[current], boxes[i]) for i in remaining])
        
        # Keep boxes with IoU below threshold
        indices = remaining[ious < iou_threshold]
    
    return keep


def nms_by_class(boxes: np.ndarray, scores: np.ndarray, 
                 class_ids: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Apply NMS per class.
    
    Args:
        boxes: Array of boxes [N, 4]
        scores: Array of confidence scores [N]
        class_ids: Array of class IDs [N]
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    keep = []
    unique_classes = np.unique(class_ids)
    
    for cls in unique_classes:
        mask = class_ids == cls
        cls_indices = np.where(mask)[0]
        cls_keep = nms(boxes[mask], scores[mask], iou_threshold)
        keep.extend([cls_indices[i] for i in cls_keep])
    
    return sorted(keep)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2].
    
    Args:
        boxes: Array of boxes [N, 4] in xywh format
        
    Returns:
        Array of boxes [N, 4] in xyxy format
    """
    result = np.zeros_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return result


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height].
    
    Args:
        boxes: Array of boxes [N, 4] in xyxy format
        
    Returns:
        Array of boxes [N, 4] in xywh format
    """
    result = np.zeros_like(boxes)
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
    result[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    result[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return result


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Clip boxes to image boundaries.
    
    Args:
        boxes: Array of boxes [N, 4] in xyxy format
        width: Image width
        height: Image height
        
    Returns:
        Clipped boxes
    """
    result = boxes.copy()
    result[:, 0] = np.clip(result[:, 0], 0, width)
    result[:, 1] = np.clip(result[:, 1], 0, height)
    result[:, 2] = np.clip(result[:, 2], 0, width)
    result[:, 3] = np.clip(result[:, 3], 0, height)
    return result


def convert_cpp_detections(detections: np.ndarray) -> List:
    """
    Convert C++ postprocessor output (numpy array) to DetectionResult objects.
    
    C++ postprocessors return numpy arrays of shape [N, 6] where each row is:
    [x1, y1, x2, y2, confidence, class_id]
    
    Args:
        detections: numpy array of shape [N, 6] from C++ postprocessor
        
    Returns:
        List of DetectionResult-compatible objects
    """
    from ..base import DetectionResult
    
    results = []
    if detections is None or len(detections) == 0:
        return results
    
    for det in detections:
        result = DetectionResult(
            box=[float(det[0]), float(det[1]), float(det[2]), float(det[3])],
            confidence=float(det[4]),
            class_id=int(det[5])
        )
        results.append(result)
    
    return results


def convert_cpp_face_detections(detections: np.ndarray) -> List:
    """
    Convert C++ face postprocessor output (numpy array) to FaceResult with keypoints.
    
    Args:
        detections: numpy array [N, 6+keypoints] from C++ postprocessor
                   [x1, y1, x2, y2, confidence, class_id, kp1_x, kp1_y, ...]
    
    Returns:
        List of FaceResult objects with Keypoint objects
    """
    from ..processors.face_postprocessor import FaceResult
    from ..base import Keypoint
    
    results = []
    if detections is None or len(detections) == 0:
        return results
    
    NUM_FACE_KEYPOINTS = 5  # SCRFD / YOLOv5Face: 5 landmarks
    for det in detections:
        # Parse keypoints as Keypoint objects (x, y pairs)
        kp_data = det[6:].flatten() if hasattr(det[6:], 'flatten') else list(det[6:])
        # Limit to expected number of keypoints to ignore trailing garbage data
        max_values = NUM_FACE_KEYPOINTS * 2
        kp_data = kp_data[:max_values]
        keypoints = []
        for i in range(0, len(kp_data) - 1, 2):
            keypoints.append(Keypoint(
                x=float(kp_data[i]),
                y=float(kp_data[i + 1]),
                confidence=1.0
            ))
        
        result = FaceResult(
            box=[float(det[0]), float(det[1]), float(det[2]), float(det[3])],
            confidence=float(det[4]),
            class_id=int(det[5]),
            keypoints=keypoints
        )
        results.append(result)
    
    return results


def convert_cpp_pose_detections(detections: np.ndarray) -> List:
    """
    Convert C++ pose postprocessor output (numpy array) to PoseResult objects.
    
    Args:
        detections: numpy array [N, 6+keypoints] from C++ postprocessor
                   [x1, y1, x2, y2, confidence, class_id, kp1_x, kp1_y, kp1_conf, ...]
    
    Returns:
        List of PoseResult-compatible objects with keypoints
    """
    from ..base import PoseResult, Keypoint
    
    results = []
    if detections is None or len(detections) == 0:
        return results
    
    for det in detections:
        keypoint_data = det[6:] if len(det) > 6 else []
        keypoints = []
        
        # Parse keypoints (x, y, conf) triplets
        for i in range(0, len(keypoint_data) - 2, 3):
            kp = Keypoint(
                x=float(keypoint_data[i]),
                y=float(keypoint_data[i + 1]),
                confidence=float(keypoint_data[i + 2])
            )
            keypoints.append(kp)
        
        result = PoseResult(
            box=[float(det[0]), float(det[1]), float(det[2]), float(det[3])],
            confidence=float(det[4]),
            class_id=int(det[5]),
            keypoints=keypoints
        )
        results.append(result)
    
    return results


def convert_cpp_classification(predictions: np.ndarray) -> List:
    """
    Convert C++ classification postprocessor output to ClassificationResult objects.

    Args:
        predictions: numpy array [K, 2] where each row is [class_id, confidence]

    Returns:
        List of ClassificationResult
    """
    from ..base import ClassificationResult

    results = []
    if predictions is None or len(predictions) == 0:
        return results

    for pred in predictions:
        results.append(ClassificationResult(
            class_id=int(pred[0]),
            confidence=float(pred[1]),
            class_name=""
        ))

    return results


def convert_cpp_attribute(predictions: np.ndarray, labels=None) -> List:
    """
    Convert C++ AttributePostProcess output to ClassificationResult objects.

    The C++ binding performs the sigmoid/softmax + threshold + sort; this
    converter only attaches the human-readable attribute label for each
    activated index, mirroring the Python AttributePostprocessor output.

    Args:
        predictions: numpy array [K, 2] where each row is [attr_index, prob].
        labels: optional list of attribute names indexed by attr_index.

    Returns:
        List of ClassificationResult (already sorted by descending confidence).
    """
    from ..base import ClassificationResult

    results = []
    if predictions is None or len(predictions) == 0:
        return results

    for pred in predictions:
        idx = int(pred[0])
        name = labels[idx] if labels is not None and idx < len(labels) else f"attr_{idx}"
        results.append(ClassificationResult(
            class_id=idx,
            confidence=float(pred[1]),
            class_name=name,
        ))

    return results



def convert_cpp_obb_detections(detections: np.ndarray) -> List:
    """
    Convert C++ OBB postprocessor output to OBBResult objects.

    Args:
        detections: numpy array [N, 7] where each row is
                   [cx, cy, w, h, confidence, class_id, angle]

    Returns:
        List of OBBResult
    """
    from ..base import OBBResult
    from ..processors.obb_postprocessor import DOTAV1_LABELS

    results = []
    if detections is None or len(detections) == 0:
        return results

    for det in detections:
        cid = int(det[5])
        class_name = DOTAV1_LABELS[cid] if 0 <= cid < len(DOTAV1_LABELS) else f"class_{cid}"
        results.append(OBBResult(
            cx=float(det[0]),
            cy=float(det[1]),
            width=float(det[2]),
            height=float(det[3]),
            confidence=float(det[4]),
            class_id=cid,
            angle=float(det[6]),
            class_name=class_name
        ))

    return results


def convert_cpp_embedding(embedding: np.ndarray) -> List:
    """
    Convert C++ embedding postprocessor output to EmbeddingResult.

    Args:
        embedding: numpy array [D] - L2-normalized embedding vector

    Returns:
        List containing a single EmbeddingResult
    """
    from ..base import EmbeddingResult

    if embedding is None or len(embedding) == 0:
        return []

    return [EmbeddingResult(embedding=embedding)]


def convert_cpp_hand_landmark(result_tuple, ctx=None) -> list:
    """
    Convert C++ HandLandmarkPostProcess output to HandLandmarkResult list.

    Args:
        result_tuple: tuple(landmarks[21,3], confidence, handedness) from dx_postprocess.
        ctx: Optional PreprocessContext for coordinate scaling.

    Returns:
        List containing a single HandLandmarkResult.
    """
    from ..base import HandLandmarkResult

    landmarks, confidence, handedness = result_tuple
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.size == 0:
        return []

    # Scale normalized [0,1] coords to original image space if context available
    if ctx is not None:
        landmarks[:, 0] *= ctx.original_width
        landmarks[:, 1] *= ctx.original_height

    result = HandLandmarkResult(
        landmarks=landmarks,
        confidence=float(confidence),
        handedness=str(handedness)
    )
    return [result]


def convert_cpp_restoration(image: np.ndarray, ctx=None) -> list:
    """
    Convert C++ DnCNNPostProcess output to RestorationResult list.

    The C++ binding returns the clipped restored image as [H, W] (grayscale)
    or [C, H, W] (color, CHW). This replicates the Python DnCNNPostprocessor
    finalization: CHW->HWC and range-based uint8 normalization.

    Args:
        image: numpy array [H, W] or [C, H, W] float, values in [0, 1].
        ctx: Optional PreprocessContext. For upscaling models it carries the
             source geometry used to undo the square-input stretch, so the
             *_cpp_postprocess variants match the Python path.

    Returns:
        List containing a single RestorationResult.
    """
    from ..processors.restoration_postprocessor import (
        RestorationResult, restore_source_geometry,
    )

    if image is None:
        return []

    raw = np.asarray(image)

    # CHW -> HWC for color models; leave HWC/grayscale untouched.
    if raw.ndim == 3:
        if raw.shape[2] <= 4 and raw.shape[0] > 4:
            pass  # already HWC
        else:
            raw = np.transpose(raw, (1, 2, 0))  # CHW -> HWC

    dmin, dmax = float(raw.min()), float(raw.max())
    if dmax - dmin < 1e-6:
        out = np.zeros_like(raw, dtype=np.uint8)
    elif dmin >= -0.1 and dmax <= 1.1:
        out = (np.clip(raw, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif dmin >= -1.0 and dmax <= 256.0:
        out = np.clip(raw, 0.0, 255.0).astype(np.uint8)
    else:
        out = ((raw - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

    out = restore_source_geometry(out, ctx,
                                  int(getattr(ctx, "input_width", 0) or 0),
                                  int(getattr(ctx, "input_height", 0) or 0))
    return [RestorationResult(output_image=out)]


def convert_cpp_super_resolution(image: np.ndarray, ctx=None) -> list:
    """
    Convert C++ ESPCNPostProcess output to SuperResolutionResult list.

    The C++ binding returns the clipped upscaled Y channel as [H, W]. This
    replicates the Python ESPCNPostprocessor finalization, including YCbCr
    color restoration from the original image when available.

    Args:
        image: numpy array [H, W] (single-channel Y) or [C, H, W], values in [0, 1].
        ctx: Optional PreprocessContext with original_image and input_height.

    Returns:
        List containing a single SuperResolutionResult.
    """
    from ..base import SuperResolutionResult

    if image is None:
        return []

    output = np.asarray(image)
    if output.ndim == 3:
        output = np.transpose(output, (1, 2, 0))  # CHW -> HWC

    input_height = getattr(ctx, "input_height", None) or getattr(ctx, "model_height", 0)
    if output.ndim >= 2 and input_height:
        scale = max(1, round(output.shape[0] / input_height))
    else:
        scale = 2

    output = np.clip(output, 0.0, 1.0)

    original_image = getattr(ctx, "original_image", None) if ctx is not None else None
    if output.ndim == 2 and original_image is not None:
        from .colorspace import bgr_to_ycrcb_limited, ycrcb_limited_to_bgr

        sr_y = (output * 255.0).astype(np.uint8)
        out_h, out_w = sr_y.shape
        # ESPCN's Y is limited range (MATLAB rgb2ycbcr) — keep the chroma
        # planes and the inverse matrix on the same convention.
        original_ycrcb = bgr_to_ycrcb_limited(original_image)
        cr_up = cv2.resize(original_ycrcb[:, :, 1], (out_w, out_h),
                           interpolation=cv2.INTER_CUBIC)
        cb_up = cv2.resize(original_ycrcb[:, :, 2], (out_w, out_h),
                           interpolation=cv2.INTER_CUBIC)
        merged = np.stack([sr_y, cr_up, cb_up], axis=2)
        out = ycrcb_limited_to_bgr(merged)
    else:
        out = (output * 255.0).astype(np.uint8)

    return [SuperResolutionResult(output_image=out, scale_factor=scale)]


def convert_cpp_zero_dce(image: np.ndarray, ctx=None) -> list:
    """
    Convert C++ Zero-DCE postprocessor output to EnhancedImageResult list.

    Args:
        image: numpy array - raw curve params [24, H, W] or enhanced image
               [C, H, W] float32 / [H, W, C] uint8.
        ctx: Optional PreprocessContext with normalized_input for LE curve application.

    Returns:
        List containing a single EnhancedImageResult
    """
    from ..base import EnhancedImageResult

    if image is None:
        return []

    img = np.asarray(image)

    if img.ndim == 3 and img.shape[0] not in (1, 3):
        return _apply_le_curves_from_params(img, ctx)

    if img.ndim == 3 and img.shape[0] in (1, 3) and img.dtype in (np.float32, np.float64):
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        else:
            img = img.squeeze(0)  # 1HW → HW
        img = np.clip(img, 0.0, 1.0)
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]  # RGB → BGR
        img = (img * 255.0).astype(np.uint8)

    return [EnhancedImageResult(output_image=img)]


def convert_cpp_face3d(params: np.ndarray, ctx=None) -> list:
    """
    Convert C++ Face3DPostProcess output to FaceAlignmentResult list.

    Uses BFM (Basel Face Model) data to reconstruct 68 facial landmarks
    from the 62 raw 3DMM parameters output by 3DDFA v2.

    Args:
        params: numpy array of raw 3DMM parameters (62 floats)
        ctx: Optional PreprocessContext with original image dimensions.

    Returns:
        List containing a single FaceAlignmentResult
    """
    from ..base import FaceAlignmentResult
    from ..processors._bfm_data import load_bfm as _load_bfm
    import math

    if params is None or len(params) == 0:
        return []

    raw = np.asarray(params, dtype=np.float32).flatten()
    result = FaceAlignmentResult()
    result.params = raw

    INPUT_W, INPUT_H = 120, 120

    bfm = _load_bfm()

    # Denormalize
    p = raw * bfm['param_std'] + bfm['param_mean']

    # Parse affine
    R_ = p[:12].reshape(3, 4)
    R = R_[:, :3]
    offset = R_[:, 3:].reshape(3, 1)
    alpha_shp = p[12:52].reshape(-1, 1)
    alpha_exp = p[52:62].reshape(-1, 1)

    # Euler angles
    sy = float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-6:
        yaw = math.atan2(float(R[2, 1]) / cp, float(R[2, 2]) / cp)
        roll = math.atan2(float(R[1, 0]) / cp, float(R[0, 0]) / cp)
    else:
        yaw = 0.0
        roll = math.atan2(float(-R[0, 1]), float(R[1, 1]))
    result.pose = [math.degrees(yaw), math.degrees(pitch), math.degrees(roll)]

    # Reconstruct 68 landmarks: pts3d = R @ (u + W_shp@a_shp + W_exp@a_exp) + offset
    shp_deform = np.einsum('ijk,kl->ij', bfm['w_shp_base'], alpha_shp)
    exp_deform = np.einsum('ijk,kl->ij', bfm['w_exp_base'], alpha_exp)
    vertices = bfm['u_base'] + shp_deform + exp_deform
    pts3d = R @ vertices + offset  # (3, 68)

    # y-flip: BFM y-up → image y-down
    pts3d[0, :] -= 1
    pts3d[1, :] = INPUT_H - pts3d[1, :]

    ow = ctx.original_width if ctx and hasattr(ctx, 'original_width') and ctx.original_width > 0 else INPUT_W
    oh = ctx.original_height if ctx and hasattr(ctx, 'original_height') and ctx.original_height > 0 else INPUT_H
    sx, sy_s = ow / INPUT_W, oh / INPUT_H

    lmks = np.zeros((68, 2), dtype=np.float32)
    lmks[:, 0] = pts3d[0, :] * sx
    lmks[:, 1] = pts3d[1, :] * sy_s

    result.landmarks_2d = lmks
    result.landmarks_3d = np.column_stack([lmks, np.zeros(len(lmks))])
    return [result]


def convert_cpp_semantic_seg(class_map: np.ndarray, ctx=None,
                             resize_to_original: bool = True) -> list:
    """
    Convert C++ SemanticSegPostProcess output (a [H, W] class map) to a
    SegmentationResult list.

    The C++ binding performs the argmax / pre-argmax passthrough; this
    converter applies the context-dependent resize back to the original
    image size, mirroring SemanticSegmentationPostprocessor.

    Args:
        class_map: numpy array [H, W] of integer class indices.
        ctx: Optional PreprocessContext with original image dimensions.
        resize_to_original: When True, resize the class map to the original
            image size (matches the pre-argmaxed path of the Python golden,
            and SegFormer's resize_to_original=True). When False, the class
            map is returned at model resolution (e.g. U-Net).

    Returns:
        List containing a single SegmentationResult.
    """
    from ..base import SegmentationResult

    if class_map is None or class_map.size == 0:
        return []

    cm = np.asarray(class_map).astype(np.int32)
    if cm.ndim != 2:
        cm = cm.squeeze()

    if resize_to_original and ctx is not None:
        ow = getattr(ctx, 'original_width', 0)
        oh = getattr(ctx, 'original_height', 0)
        if ow > 0 and oh > 0:
            pad_x = getattr(ctx, 'pad_x', 0)
            pad_y = getattr(ctx, 'pad_y', 0)
            if pad_x == 0 and pad_y == 0:
                if cm.shape[1] != ow or cm.shape[0] != oh:
                    cm = cv2.resize(cm.astype(np.float32), (ow, oh),
                                    interpolation=cv2.INTER_NEAREST).astype(np.int32)
            else:
                gain = max(getattr(ctx, 'scale', 1.0), 1e-6)
                unpad_h = int(round(oh * gain))
                unpad_w = int(round(ow * gain))
                top, left = int(pad_y), int(pad_x)
                cropped = cm[top:top + unpad_h, left:left + unpad_w]
                if cropped.size > 0:
                    cm = cv2.resize(cropped.astype(np.float32), (ow, oh),
                                    interpolation=cv2.INTER_NEAREST).astype(np.int32)

    h, w = cm.shape
    return [SegmentationResult(
        mask=cm,
        width=w,
        height=h,
        class_ids=np.unique(cm).tolist(),
        class_names=[],
    )]


def _apply_le_curves_from_params(params: np.ndarray, ctx) -> list:
    """Apply iterative LE curves from raw curve parameters [24, H, W]."""
    from ..base import EnhancedImageResult

    h, w = params.shape[1], params.shape[2]
    n_iters = params.shape[0] // 3

    if ctx is not None and hasattr(ctx, 'normalized_input') and ctx.normalized_input is not None:
        enhanced = ctx.normalized_input.copy().astype(np.float32)
        if enhanced.ndim == 3 and enhanced.shape[0] != 3 and enhanced.shape[2] == 3:
            enhanced = np.transpose(enhanced, (2, 0, 1))
        if enhanced.shape[1] != h or enhanced.shape[2] != w:
            resized = np.zeros((3, h, w), dtype=np.float32)
            for c in range(3):
                resized[c] = cv2.resize(enhanced[c], (w, h), interpolation=cv2.INTER_LINEAR)
            enhanced = resized
    else:
        enhanced = np.full((3, h, w), 0.5, dtype=np.float32)

    for i in range(n_iters):
        alpha = params[i * 3:(i + 1) * 3]
        enhanced = enhanced + alpha * enhanced * (1.0 - enhanced)
    enhanced = np.clip(enhanced, 0.0, 1.0)

    img_out = np.transpose(enhanced, (1, 2, 0))[:, :, ::-1]  # CHW RGB → HWC BGR

    if ctx is not None and hasattr(ctx, 'original_width') and ctx.original_width > 0 and ctx.original_height > 0:
        if img_out.shape[1] != ctx.original_width or img_out.shape[0] != ctx.original_height:
            img_out = cv2.resize(img_out, (ctx.original_width, ctx.original_height),
                                 interpolation=cv2.INTER_LINEAR)

    return [EnhancedImageResult(output_image=(img_out * 255.0).astype(np.uint8))]


# ──────────────────────────────────────────────────────────────────────────────
# New C++ postprocess converters (VitPose, DOPE, SuperPoint, YOLOPv2, MediaPipe)
# ──────────────────────────────────────────────────────────────────────────────

def _scale_xy(x: float, y: float, ctx) -> tuple:
    """Scale (x, y) from model-input space to original image space using ctx."""
    if ctx is None:
        return x, y
    try:
        from .preprocessing import scale_to_original
        return scale_to_original(x, y, ctx)
    except Exception:
        return x, y


def convert_cpp_vitpose(keypoints_arr: np.ndarray, ctx=None) -> list:
    """Convert C++ VitPosePostProcess output [17, 3] to List[PoseResult].

    The runner's _scale_cpp_results_to_original will handle coordinate
    conversion via PoseResult.keypoints, so we return model-space pixels.
    """
    from ..base import PoseResult, Keypoint

    arr = np.asarray(keypoints_arr)
    if arr.size == 0:
        return []

    keypoints = [
        Keypoint(x=float(arr[i, 0]), y=float(arr[i, 1]), confidence=float(arr[i, 2]))
        for i in range(arr.shape[0])
    ]
    return [PoseResult(keypoints=keypoints, confidence=1.0)]


def convert_cpp_dope(peaks_arr: np.ndarray, ctx=None) -> list:
    """Convert C++ DOPEPostProcess output [9, 3] (heatmap-space peaks) to
    List[DopeResult], matching the Python DOPEPostprocessor schema.

    The C++ peaks are (x, y, conf) in heatmap pixels (already +0.5 centred).
    DOPEVisualizer expects keypoints normalized to [0,1] (it multiplies by the
    original image size), so divide by the heatmap dims (= model input / 8).
    Pose is left None — the visualizer solves PnP from the 2D points itself.
    """
    from ..processors.dope_postprocessor import DopeResult

    arr = np.asarray(peaks_arr, dtype=np.float64)
    if arr.size == 0:
        return []

    iw = int(getattr(ctx, "input_width", 0) or 0)
    ih = int(getattr(ctx, "input_height", 0) or 0)
    hm_w = (iw // 8) or 80   # dope-hope-ketchup: 640/8 = 80
    hm_h = (ih // 8) or 60   #                    480/8 = 60

    n = arr.shape[0]
    kps = np.zeros((n, 2), dtype=np.float64)
    confs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        kps[i, 0] = arr[i, 0] / hm_w
        kps[i, 1] = arr[i, 1] / hm_h
        confs[i] = arr[i, 2]

    return [DopeResult(
        keypoints=kps,
        centroid=kps[8].copy() if n > 8 else np.zeros(2, dtype=np.float64),
        confidence=float(confs[8]) if n > 8 else 0.0,
        all_conf=confs,
        pose=None,
        image_width=int(getattr(ctx, "original_width", 0) or 0),
        image_height=int(getattr(ctx, "original_height", 0) or 0),
    )]


def convert_cpp_superpoint(result_tuple, ctx=None) -> list:
    """Convert C++ SuperPointPostProcess output (kps[N,3], descs[N,256]) to List[SuperPointResult].

    Coordinate conversion is done here because SuperPointResult.keypoints is a
    list of plain tuples (not Keypoint objects with .x/.y attributes).
    """
    from ..processors.superpoint_postprocessor import SuperPointResult

    kps_arr, desc_arr = result_tuple
    kps_arr = np.asarray(kps_arr)
    desc_arr = np.asarray(desc_arr)

    if kps_arr.shape[0] == 0:
        return [SuperPointResult(keypoints=[], scores=[],
                                 descriptors=np.zeros((0, 256), np.float32))]

    keypoints = []
    scores = []
    for i in range(kps_arr.shape[0]):
        x, y, score = float(kps_arr[i, 0]), float(kps_arr[i, 1]), float(kps_arr[i, 2])
        x, y = _scale_xy(x, y, ctx)
        keypoints.append((x, y))
        scores.append(score)

    return [SuperPointResult(keypoints=keypoints, scores=scores, descriptors=desc_arr)]


def convert_cpp_yolopv2(result_tuple, ctx=None) -> list:
    """Convert C++ YOLOPv2PostProcess output (dets[M,6], driv[H,W], lane[H,W])
    to List[YOLOPv2Result].

    Coordinate conversion is done here because YOLOPv2Result wraps DetectionResult
    inside .detections rather than exposing .box directly on the result object.
    """
    from ..processors.yolopv2_postprocessor import YOLOPv2Result
    from ..base import DetectionResult

    dets_arr, driv_arr, lane_arr = result_tuple
    dets_arr = np.asarray(dets_arr)

    detections = []
    for i in range(dets_arr.shape[0]):
        x1, y1, x2, y2, conf, cls_id = (float(dets_arr[i, j]) for j in range(6))
        x1, y1 = _scale_xy(x1, y1, ctx)
        x2, y2 = _scale_xy(x2, y2, ctx)
        detections.append(DetectionResult(
            box=[x1, y1, x2, y2],
            confidence=conf,
            class_id=int(cls_id),
            class_name="",
        ))

    return [YOLOPv2Result(
        detections=detections,
        drivable_mask=np.asarray(driv_arr),
        lane_mask=np.asarray(lane_arr),
    )]


def convert_cpp_mediapipe_hand(detections_arr: np.ndarray, ctx=None) -> list:
    """Convert C++ MediaPipeHandPostProcess output [K, 5] (normalized [0,1]) to
    List[FaceResult].

    Boxes are returned in normalized coordinates so the runner's
    _scale_cpp_results_to_original can apply the full letterbox→original
    transform via the standard .box path.
    """
    from ..processors.face_postprocessor import FaceResult

    arr = np.asarray(detections_arr)
    if arr.shape[0] == 0:
        return []

    return [
        FaceResult(
            box=[float(arr[i, 0]), float(arr[i, 1]),
                 float(arr[i, 2]), float(arr[i, 3])],
            confidence=float(arr[i, 4]),
            class_id=0,
            keypoints=[],
        )
        for i in range(arr.shape[0])
    ]


def convert_cpp_sfa3d(detections_arr: np.ndarray, ctx=None) -> list:
    """Convert C++ SFA3DPostProcess output [N, 9] to DetectionResult list."""
    from ..base import DetectionResult

    arr = np.asarray(detections_arr, dtype=np.float32)
    if arr.size == 0:
        return []

    class_names = ["Car", "Pedestrian", "Cyclist"]

    results = []
    for det in arr:
        cx, cy, z, h3d, w3d, l3d, yaw, conf, cls_id = [float(v) for v in det[:9]]
        box = [cx - l3d * 0.5, cy - w3d * 0.5, cx + l3d * 0.5, cy + w3d * 0.5]
        cid = int(cls_id)
        result = DetectionResult(
            box=box,
            confidence=float(conf),
            class_id=cid,
            class_name=class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}",
        )
        result.cx = cx
        result.cy = cy
        result.z = z
        result.height_3d = h3d
        result.width_3d = w3d
        result.length_3d = l3d
        result.yaw = yaw
        results.append(result)
    return results


def convert_cpp_sfa3d_detections(detections: Any, ctx=None) -> List:
    """Convert SFA3D C++ postprocess output to Detection3DResult list.

    Supported input shapes:
    - list[Detection3DResult-like objects] (returned as-is)
    - ndarray/list with row layouts:
      1) [cls, score, x, y, z, h, w, l, yaw, bev_x, bev_y, bev_w, bev_h]
      2) [score, cls, x, y, z, h, w, l, yaw, bev_x, bev_y, bev_w, bev_h]
      3) [cls, score, x, y, z, h, w, l, yaw]
    """
    from ..processors.sfa3d_postprocessor import Detection3DResult, SFA3D_CLASSES

    def _cls_name(class_id: int) -> str:
        if 0 <= class_id < len(SFA3D_CLASSES):
            return SFA3D_CLASSES[class_id]
        return f"cls_{class_id}"

    if detections is None:
        return []

    if isinstance(detections, list) and detections:
        first = detections[0]
        if hasattr(first, "x3d") and hasattr(first, "yaw"):
            return detections

    arr = np.asarray(detections)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    results: List[Detection3DResult] = []
    for row in arr:
        vals = np.asarray(row).flatten()
        if vals.size < 9:
            continue

        first_is_int = bool(np.isfinite(vals[0]) and abs(vals[0] - round(vals[0])) < 1e-3)
        second_is_int = bool(np.isfinite(vals[1]) and abs(vals[1] - round(vals[1])) < 1e-3)
        class_first = int(round(vals[0])) if np.isfinite(vals[0]) else -1
        class_second = int(round(vals[1])) if np.isfinite(vals[1]) else -1
        if first_is_int and 0 <= class_first < len(SFA3D_CLASSES):
            class_id = class_first
            score = float(vals[1])
        elif second_is_int and 0 <= class_second < len(SFA3D_CLASSES):
            class_id = class_second
            score = float(vals[0])
        else:
            class_id = max(0, class_first)
            score = float(vals[1]) if vals.size > 1 else 0.0

        if not np.isfinite(score):
            score = 0.0

        x3d = float(vals[2]) if vals.size > 2 else 0.0
        y3d = float(vals[3]) if vals.size > 3 else 0.0
        z3d = float(vals[4]) if vals.size > 4 else 0.0
        dim_h = float(vals[5]) if vals.size > 5 else 0.0
        dim_w = float(vals[6]) if vals.size > 6 else 0.0
        dim_l = float(vals[7]) if vals.size > 7 else 0.0
        yaw = float(vals[8]) if vals.size > 8 else 0.0

        bev_x = float(vals[9]) if vals.size > 9 else 0.0
        bev_y = float(vals[10]) if vals.size > 10 else 0.0
        bev_w = float(vals[11]) if vals.size > 11 else 0.0
        bev_h = float(vals[12]) if vals.size > 12 else 0.0

        results.append(
            Detection3DResult(
                class_id=class_id,
                class_name=_cls_name(class_id),
                confidence=score,
                bev_x=bev_x,
                bev_y=bev_y,
                bev_w=bev_w,
                bev_h=bev_h,
                x3d=x3d,
                y3d=y3d,
                z3d=z3d,
                dim_h=dim_h,
                dim_w=dim_w,
                dim_l=dim_l,
                yaw=yaw,
            )
        )
    return results
