#!/usr/bin/env python3
"""
Person Re-ID Comparison Demo

Demonstrates person re-identification via embedding comparison:
  1. Simple resize to model input size (no alignment needed)
  2. CasViT embedding extraction
  3. Cosine similarity comparison

Pair 1 — same person, different poses  : sample_person_a1 vs sample_person_a2  → SAME
Pair 2 — different persons             : sample_person_a1 vs sample_person_b   → DIFFERENT

Usage:
    python3 scripts/demo_reid_compare.py \
        --model assets/models/casvit_t.dxnn \
        --image1 sample/img/sample_person_a1.jpg \
        --image2 sample/img/sample_person_a2.jpg \
        --image3 sample/img/sample_person_b.jpg \
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_embedding(embed_ie: InferenceEngine, img_bgr: np.ndarray,
                      input_w: int, input_h: int) -> np.ndarray:
    """Resize person image to model input size and extract L2-normalized embedding."""
    resized = cv2.resize(img_bgr, (input_w, input_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    outputs = embed_ie.run([rgb])
    if not outputs:
        return None

    embedding = np.array(outputs[0], dtype=np.float32).flatten()
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    return embedding


def make_comparison_canvas(img1: np.ndarray, img2: np.ndarray,
                           input_w: int, input_h: int,
                           similarity: float, name1: str, name2: str,
                           display_h: int = 320) -> np.ndarray:
    """Create side-by-side comparison canvas with similarity result bar."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    nw1 = int(w1 * display_h / h1)
    nw2 = int(w2 * display_h / h2)
    img1_r = cv2.resize(img1, (nw1, display_h))
    img2_r = cv2.resize(img2, (nw2, display_h))

    # Resized crops (what the model actually sees)
    crop_h = display_h // 2
    crop_w = int(input_w * crop_h / input_h)
    c1 = cv2.resize(img1, (crop_w, crop_h))
    c2 = cv2.resize(img2, (crop_w, crop_h))

    gap = 10
    bar_h = 70
    top_w = nw1 + gap + nw2
    bottom_w = crop_w + gap + crop_w
    canvas_w = max(top_w, bottom_w, 500)
    canvas_h = display_h + gap + crop_h + bar_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Original images (top row)
    canvas[:display_h, :nw1] = img1_r
    canvas[:display_h, nw1 + gap:nw1 + gap + nw2] = img2_r

    # Model input crops (bottom row, centered)
    ax = (canvas_w - (crop_w * 2 + gap)) // 2
    ay = display_h + gap
    canvas[ay:ay + crop_h, ax:ax + crop_w] = c1
    canvas[ay:ay + crop_h, ax + crop_w + gap:ax + crop_w * 2 + gap] = c2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, name1, (5, 25), font, 0.65, (0, 255, 0), 2)
    cv2.putText(canvas, name2, (nw1 + gap + 5, 25), font, 0.65, (0, 255, 0), 2)
    cv2.putText(canvas, f"Input {input_w}x{input_h}",
                (ax, ay - 5), font, 0.45, (180, 180, 180), 1)

    # Similarity result bar
    match = similarity > 0.5
    color = (0, 200, 0) if match else (0, 0, 220)
    label = "SAME PERSON" if match else "DIFFERENT PERSON"
    text = f"Cosine Similarity: {similarity:.4f}  [{label}]"
    cv2.putText(canvas, text, (10, canvas_h - 15), font, 0.7, color, 2)

    return canvas


def compare_pair(embed_ie: InferenceEngine, input_w: int, input_h: int,
                 path1: str, path2: str,
                 label: str, no_display: bool) -> None:
    """Load two images, extract embeddings, compare, and optionally display."""
    img1 = cv2.imread(path1)
    if img1 is None:
        print(f"[ERROR] Cannot read image: {path1}")
        return
    img2 = cv2.imread(path2)
    if img2 is None:
        print(f"[ERROR] Cannot read image: {path2}")
        return

    emb1 = extract_embedding(embed_ie, img1, input_w, input_h)
    emb2 = extract_embedding(embed_ie, img2, input_w, input_h)
    if emb1 is None or emb2 is None:
        print("[ERROR] Embedding extraction failed.")
        return

    sim = cosine_similarity(emb1, emb2)
    match = sim > 0.5

    print(f"\n{'='*55}")
    print(f"   {label}")
    print(f"{'='*55}")
    print(f" Image1            : {Path(path1).name}")
    print(f" Image2            : {Path(path2).name}")
    print(f" Cosine Similarity : {sim:.4f}")
    print(f" Verdict           : {'SAME PERSON' if match else 'DIFFERENT PERSON'}")
    print(f"{'='*55}\n")

    if not no_display:
        canvas = make_comparison_canvas(
            img1, img2, input_w, input_h, sim,
            Path(path1).stem, Path(path2).stem)
        win_title = f"Re-ID: {Path(path1).stem} vs {Path(path2).stem}"
        cv2.imshow(win_title, canvas)
        print("[INFO] Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)


def main():
    parser = argparse.ArgumentParser(description="Person Re-ID Comparison Demo")
    parser.add_argument("-m", "--model", required=True,
                        help="CasViT Re-ID model (.dxnn)")
    parser.add_argument("--image1", required=True, help="Person A image (pose 1)")
    parser.add_argument("--image2", required=True, help="Person A image (pose 2) — expected SAME")
    parser.add_argument("--image3", default=None,
                        help="Person B image — expected DIFFERENT from image1")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip GUI display")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"     PERSON Re-ID COMPARISON DEMO")
    print(f"{'='*55}")
    print(f" Model  : {Path(args.model).name}")
    print(f" Image1 : {args.image1}")
    print(f" Image2 : {args.image2}")
    if args.image3:
        print(f" Image3 : {args.image3}")
    print(f"{'='*55}\n")

    # Initialize embedder and resolve input shape
    embed_ie = InferenceEngine(args.model)
    input_info = embed_ie.get_input_tensors_info()
    shape = input_info[0]["shape"]
    if len(shape) == 4:
        if shape[1] <= 4:   # NCHW
            input_h, input_w = shape[2], shape[3]
        else:               # NHWC
            input_h, input_w = shape[1], shape[2]
    else:
        input_h, input_w = shape[-2], shape[-1]

    print(f"[INFO] Model input size: {input_w}x{input_h}")

    compare_pair(embed_ie, input_w, input_h,
                 args.image1, args.image2,
                 "Pair 1 — Same person, different poses", args.no_display)

    if args.image3:
        compare_pair(embed_ie, input_w, input_h,
                     args.image1, args.image3,
                     "Pair 2 — Different persons", args.no_display)


if __name__ == "__main__":
    main()
