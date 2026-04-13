#!/usr/bin/env python3
"""
Embedding Comparison Demo

Demonstrates face embedding comparison with proper face alignment:
  1. SCRFD face detection → 5-point landmarks
  2. Similarity transform → 112×112 aligned face crop
  3. ArcFace embedding extraction
  4. Cosine similarity comparison

Usage:
    python3 scripts/demo_embedding_compare.py \
        --model assets/models/arcface_mobilefacenet.dxnn \
        --image1 sample/img/sample_face_a1.jpg \
        --image2 sample/img/sample_face_b.jpg \
        [--detector assets/models/SCRFD500M.dxnn] \
        [--no-display]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project paths
_script_dir = Path(__file__).resolve().parent
_app_dir = _script_dir.parent
_py_dir = _app_dir / "src" / "python_example"
for p in [str(_py_dir), str(_app_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dx_engine import InferenceEngine
from common.processors.face_postprocessor import SCRFDPostprocessor
from common.base import PreprocessContext

# ArcFace canonical 5-point template for 112×112 alignment
ARCFACE_REF_POINTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def detect_face_landmarks(det_ie: InferenceEngine, postprocessor: SCRFDPostprocessor,
                          img: np.ndarray) -> np.ndarray:
    """Detect the largest face and return 5-point landmarks in original image coords.

    Returns:
        np.ndarray of shape (5, 2) — landmark coordinates, or None if no face found.
    """
    h, w = img.shape[:2]
    # Letterbox resize to 640×640
    scale = min(640 / w, 640 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x = (640 - new_w) // 2
    pad_y = (640 - new_h) // 2

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    outputs = det_ie.run([rgb])
    if not outputs:
        return None

    ctx = PreprocessContext()
    ctx.original_width = w
    ctx.original_height = h
    ctx.input_width = 640
    ctx.input_height = 640
    ctx.scale = scale
    ctx.scale_x = scale
    ctx.scale_y = scale
    ctx.pad_x = pad_x
    ctx.pad_y = pad_y

    results = postprocessor.process(outputs, ctx)
    if not results:
        return None

    # Pick the largest face
    best = max(results, key=lambda r: (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]))
    if len(best.keypoints) < 5:
        return None

    landmarks = np.array([[kp.x, kp.y] for kp in best.keypoints[:5]], dtype=np.float32)
    return landmarks


def align_face(img: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Warp face to canonical 112×112 template using similarity transform from 5 landmarks."""
    # Estimate similarity transform (no shear) from landmarks → reference
    tform, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_REF_POINTS, method=cv2.LMEDS)
    if tform is None:
        # Fallback: simple crop around landmarks center
        cx, cy = landmarks.mean(axis=0)
        half = output_size // 2
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        crop = img[y1:y1 + output_size, x1:x1 + output_size]
        return cv2.resize(crop, (output_size, output_size))

    aligned = cv2.warpAffine(img, tform, (output_size, output_size),
                             borderValue=(0, 0, 0))
    return aligned


def extract_embedding(embed_ie: InferenceEngine, face_rgb: np.ndarray,
                      input_w: int, input_h: int) -> np.ndarray:
    """Run ArcFace on an aligned RGB face crop, return L2-normalized embedding."""
    resized = cv2.resize(face_rgb, (input_w, input_h))
    outputs = embed_ie.run([resized])
    if not outputs:
        return None

    embedding = np.array(outputs[0], dtype=np.float32).flatten()
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    return embedding


def make_comparison_canvas(img1: np.ndarray, img2: np.ndarray,
                           aligned1: np.ndarray, aligned2: np.ndarray,
                           similarity: float, name1: str, name2: str,
                           display_h: int = 300) -> np.ndarray:
    """Create comparison canvas: original faces on top, aligned faces below, similarity bar."""
    # Resize originals to same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    nw1 = int(w1 * display_h / h1)
    nw2 = int(w2 * display_h / h2)
    img1_r = cv2.resize(img1, (nw1, display_h))
    img2_r = cv2.resize(img2, (nw2, display_h))

    # Aligned faces (112×112 → display size)
    align_sz = display_h // 2
    a1_r = cv2.resize(aligned1, (align_sz, align_sz))
    a2_r = cv2.resize(aligned2, (align_sz, align_sz))

    gap = 10
    bar_h = 60
    top_w = nw1 + gap + nw2
    bottom_w = align_sz + gap + align_sz
    canvas_w = max(top_w, bottom_w)
    canvas_h = display_h + gap + align_sz + bar_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place original images
    canvas[:display_h, :nw1] = img1_r
    canvas[:display_h, nw1 + gap:nw1 + gap + nw2] = img2_r

    # Place aligned faces centered
    ax = (canvas_w - (align_sz * 2 + gap)) // 2
    ay = display_h + gap
    canvas[ay:ay + align_sz, ax:ax + align_sz] = a1_r
    canvas[ay:ay + align_sz, ax + align_sz + gap:ax + align_sz * 2 + gap] = a2_r

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, name1, (5, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, name2, (nw1 + gap + 5, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, "Aligned", (ax, ay - 5), font, 0.5, (200, 200, 200), 1)

    # Similarity bar
    match = similarity > 0.4
    color = (0, 200, 0) if match else (0, 0, 220)
    label = "SAME" if match else "DIFFERENT"
    text = f"Cosine Similarity: {similarity:.4f}  [{label}]"
    cv2.putText(canvas, text, (10, canvas_h - 15), font, 0.7, color, 2)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Embedding Comparison Demo")
    parser.add_argument("-m", "--model", required=True, help="ArcFace model (.dxnn)")
    parser.add_argument("--detector", default=None,
                        help="SCRFD face detector model (.dxnn). "
                             "Auto-detected from assets/models/ if not specified.")
    parser.add_argument("--image1", required=True, help="First image path")
    parser.add_argument("--image2", required=True, help="Second image path")
    parser.add_argument("--no-display", action="store_true", help="Skip GUI display")
    args = parser.parse_args()

    # Auto-find SCRFD detector
    detector_path = args.detector
    if detector_path is None:
        for name in ["SCRFD500M.dxnn", "SCRFD2_5G.dxnn", "SCRFD10G.dxnn"]:
            candidate = _app_dir / "assets" / "models" / name
            if candidate.exists():
                detector_path = str(candidate)
                break
    if detector_path is None:
        print("[ERROR] No SCRFD detector model found. Use --detector to specify.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"   EMBEDDING COMPARISON DEMO (with Face Alignment)")
    print(f"{'='*55}")
    print(f" Detector : {Path(detector_path).name}")
    print(f" Embedder : {Path(args.model).name}")
    print(f" Image1   : {args.image1}")
    print(f" Image2   : {args.image2}")
    print(f"{'='*55}\n")

    # Initialize SCRFD detector
    det_ie = InferenceEngine(detector_path)
    scrfd_post = SCRFDPostprocessor(640, 640, {'score_threshold': 0.15})

    # Initialize ArcFace embedder
    embed_ie = InferenceEngine(args.model)
    input_info = embed_ie.get_input_tensors_info()
    shape = input_info[0]["shape"]
    if len(shape) == 4:
        if shape[1] <= 4:  # NCHW
            input_h, input_w = shape[2], shape[3]
        else:  # NHWC
            input_h, input_w = shape[1], shape[2]
    else:
        input_h, input_w = shape[-2], shape[-1]
    # Process image 1
    img1 = cv2.imread(args.image1)
    if img1 is None:
        print(f"[ERROR] Cannot read image: {args.image1}")
        sys.exit(1)

    lmk1 = detect_face_landmarks(det_ie, scrfd_post, img1)
    if lmk1 is None:
        print(f"[ERROR] No face detected in: {args.image1}")
        sys.exit(1)

    aligned1 = align_face(img1, lmk1, output_size=112)
    aligned1_rgb = cv2.cvtColor(aligned1, cv2.COLOR_BGR2RGB)
    emb1 = extract_embedding(embed_ie, aligned1_rgb, input_w, input_h)

    # Process image 2
    img2 = cv2.imread(args.image2)
    if img2 is None:
        print(f"[ERROR] Cannot read image: {args.image2}")
        sys.exit(1)

    lmk2 = detect_face_landmarks(det_ie, scrfd_post, img2)
    if lmk2 is None:
        print(f"[ERROR] No face detected in: {args.image2}")
        sys.exit(1)

    aligned2 = align_face(img2, lmk2, output_size=112)
    aligned2_rgb = cv2.cvtColor(aligned2, cv2.COLOR_BGR2RGB)
    emb2 = extract_embedding(embed_ie, aligned2_rgb, input_w, input_h)

    # Compare
    sim = cosine_similarity(emb1, emb2)
    match = sim > 0.4  # Lowered from 0.5: better recall for same-person pairs

    print(f"\n{'='*55}")
    print(f"          COMPARISON RESULT")
    print(f"{'='*55}")
    print(f" Cosine Similarity : {sim:.4f}")
    print(f" Verdict           : {'SAME PERSON' if match else 'DIFFERENT PERSON'}")
    print(f"{'='*55}\n")

    # Display
    if not args.no_display:
        canvas = make_comparison_canvas(
            img1, img2, aligned1, aligned2, sim,
            Path(args.image1).stem, Path(args.image2).stem)
        cv2.imshow("Embedding Comparison", canvas)
        print("[INFO] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
