#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Cut person crops out of sample photos, so the ReID gallery is what ReID expects.

MEASURED, and the reason this script exists: fed whole scene photographs,
``repvgg-a0-reid_256x128`` ranked an unrelated group-of-people picture (cos 0.7031)
ABOVE a second photograph of the query's own subject, which did not even reach the top
three. RepVGG-A0 is trained on Market-1501-style crops -- one person, tight box, 2:1
portrait aspect. Squashing a 16:9 garden scene into 128x256 destroys exactly the
geometry the descriptor encodes, and the failure is silent: the numbers stay in a
plausible 0.3-0.7 band and the ranking is simply wrong.

So the gallery is built from detector crops. A person detector runs first (the
detect-then-describe pipeline every ReID system uses), each crop is padded slightly and
written out, and ``build_gallery_database.py`` then encodes those.

Usage::

    python scripts/build_reid_crops.py --images sample/img/sample_person_a1.jpg ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python_example"))

DETECTOR = "yolov8-n_640x640"
PERSON_CLASS_ID = 0          # COCO


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, nargs="+")
    ap.add_argument("--out", default="sample/reid/gallery")
    ap.add_argument("--detector", default=DETECTOR)
    ap.add_argument("--models-dir", default=str(ROOT / "assets" / "models"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=0.06,
                    help="fraction of box size added on each side")
    ap.add_argument("--max-per-image", type=int, default=1,
                    help="keep the N largest person boxes per image")
    ap.add_argument("--min-height", type=int, default=64,
                    help="skip boxes shorter than this many pixels")
    args = ap.parse_args()

    import cv2
    from dx_engine import InferenceEngine
    from common.variant_config import build_processor, load_variant_config

    hits = sorted((ROOT / "src" / "python_example").glob(
        f"*/*/{args.detector}/config.json"))
    if not hits:
        raise SystemExit(f"[DXAPP] [ERROR] no example for detector {args.detector}")
    spec = load_variant_config(str(hits[0].parent.parent), args.detector)
    w, h = int(spec["input_width"]), int(spec["input_height"])
    pre = build_processor(spec["preprocessor"], input_width=w, input_height=h,
                          config=spec.get("config") or {})
    post = build_processor(spec["postprocessor"], input_width=w, input_height=h,
                           config={**(spec.get("config") or {}),
                                   "conf_threshold": args.conf})

    dxnn = Path(args.models_dir) / f"{args.detector}.dxnn"
    if not dxnn.is_file():
        raise SystemExit(
            f"[DXAPP] [ERROR] detector not found: {dxnn}\n"
            f"  python scripts/download_models.py --models YoloV8N")
    ie = InferenceEngine(str(dxnn))

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for raw in args.images:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / p
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [WARN] unreadable: {p}")
            continue
        tensor, ctx = pre.process(img)
        dets = post.process(ie.run([tensor]), ctx)
        people = [d for d in dets
                  if int(getattr(d, "class_id", -1)) == PERSON_CLASS_ID
                  and float(getattr(d, "confidence", 0.0)) >= args.conf]
        people.sort(key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
                    reverse=True)
        if not people:
            print(f"  [WARN] no person found in {p.name}")
            continue

        H, W = img.shape[:2]
        for i, d in enumerate(people[:args.max_per_image]):
            x1, y1, x2, y2 = (float(v) for v in d.box[:4])
            bw, bh = x2 - x1, y2 - y1
            if bh < args.min_height:
                continue
            mx, my = bw * args.margin, bh * args.margin
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(W, int(x2 + mx)), min(H, int(y2 + my))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            name = f"{p.stem}_p{i}.jpg" if args.max_per_image > 1 else f"{p.stem}.jpg"
            dest = out_dir / name
            cv2.imwrite(str(dest), crop)
            written += 1
            print(f"  [OK] {dest.relative_to(ROOT)}  {x2-x1}x{y2-y1}  "
                  f"conf {float(d.confidence):.3f}")

    print(f"[DXAPP] [INFO] {written} crop(s) -> {out_dir.relative_to(ROOT)}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
