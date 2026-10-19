#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Encode an image folder on the NPU into a retrieval gallery ``.npz``.

Image retrieval, visual place recognition and person ReID all need a set of KNOWN
descriptors to rank a query against. That set is built once, here, and committed --
so the example is a single engine at runtime and needs no dataset on the user's disk.

The gallery is built through the variant's OWN factory, not a reimplementation of its
preprocessing. That is the whole point: a gallery encoded with different resizing or
normalisation than the query path lands in a slightly different place in embedding
space, and the symptom is not an error -- it is a ranking that looks reasonable and is
wrong. Driving `config.json` guarantees the two sides cannot drift, and a later fix to
a variant's preprocessing is inherited by simply rebuilding.

The written file is the ONE format both example trees read (see
``common/processors/gallery_format``), and it records ``model`` and its width so
:class:`GalleryRetrievalPostprocessor` can refuse a gallery that belongs to another
encoder instead of silently ranking against a foreign space.

Usage::

    python scripts/build_gallery_database.py \
        --variant eigenplaces-resnet18_512x512 \
        --images sample/vpr/database \
        --out sample/gallery/vpr_eigenplaces-resnet18_512x512.bin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python_example"))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_variant_dir(variant: str) -> Path:
    """``src/python_example/<task>/<family>/<variant>/`` -- searched, not hardcoded.

    The task a variant belongs to is exactly what this change set moves around, so
    locating it by name keeps the builder working across the re-classification.
    """
    hits = sorted((ROOT / "src" / "python_example").glob(f"*/*/{variant}/config.json"))
    if not hits:
        raise SystemExit(
            f"[DXAPP] [ERROR] no example found for variant '{variant}'.\n"
            f"  Looked for src/python_example/*/*/{variant}/config.json"
        )
    if len(hits) > 1:
        raise SystemExit(
            f"[DXAPP] [ERROR] variant '{variant}' exists in {len(hits)} places: "
            + ", ".join(str(h.parent.relative_to(ROOT)) for h in hits)
        )
    return hits[0].parent


def collect_images(roots: list[str], exclude: list[str] | None = None) -> list[Path]:
    """Every image under the given folders/files, sorted for a reproducible order.

    ``exclude`` drops files by CONTENT, not by name. The repo ships the same photo
    under two names (``sample/img/person_pair/2_same.jpg`` is byte-identical to
    ``sample/img/sample_person_a2.jpg``), so excluding a query by path alone leaves its
    twin in the gallery and every demo then reports a perfect 1.0000 self-match -- a
    result that looks excellent and proves nothing.
    """
    out: list[Path] = []
    for raw in roots:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(q for q in sorted(p.rglob("*"))
                       if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise SystemExit(f"[DXAPP] [ERROR] not found: {p}")
    if exclude:
        import hashlib
        banned = set()
        for raw in exclude:
            q = Path(raw)
            if not q.is_absolute():
                q = ROOT / q
            if q.is_file():
                banned.add(hashlib.sha256(q.read_bytes()).hexdigest())
            else:
                raise SystemExit(f"[DXAPP] [ERROR] --exclude not found: {q}")
        kept = []
        for q in out:
            if hashlib.sha256(q.read_bytes()).hexdigest() in banned:
                print(f"  [excluded] {q.relative_to(ROOT) if q.is_relative_to(ROOT) else q}")
            else:
                kept.append(q)
        out = kept
    # Sorted by repo-relative path: the gallery order is then stable across machines.
    return sorted(set(out), key=lambda q: str(q))


def label_for(path: Path, images: list[Path], mode: str) -> str:
    """``folder`` uses the immediate parent name, unless every image shares it."""
    if mode == "none":
        return ""
    parents = {q.parent for q in images}
    if len(parents) <= 1:
        return ""
    return path.parent.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True,
                    help="the .dxnn stem, e.g. eigenplaces-resnet18_512x512")
    ap.add_argument("--images", required=True, nargs="+",
                    help="folder(s) or file(s) to encode")
    ap.add_argument("--out", required=True, help="destination .npz")
    ap.add_argument("--models-dir", default=str(ROOT / "assets" / "models"))
    ap.add_argument("--labels", choices=("folder", "none"), default="folder")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="held-out query image(s): dropped from the gallery by "
                         "content hash, so a duplicate under another name goes too")
    ap.add_argument("--limit", type=int, default=0,
                    help="encode at most N images (0 = all)")
    args = ap.parse_args()

    import cv2
    from dx_engine import InferenceEngine
    from common.processors import GalleryRetrievalPostprocessor
    from common.processors.gallery_format import write_gallery
    from common.variant_config import build_processor, load_variant_config

    variant_dir = find_variant_dir(args.variant)
    family_dir = variant_dir.parent
    spec = load_variant_config(str(family_dir), args.variant)

    dxnn = Path(args.models_dir) / f"{args.variant}.dxnn"
    if not dxnn.is_file():
        raise SystemExit(
            f"[DXAPP] [ERROR] model not found: {dxnn}\n"
            f"  Download it with:  python scripts/download_models.py --models {args.variant}"
        )

    images = collect_images(args.images, args.exclude)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit("[DXAPP] [ERROR] no images to encode")

    ie = InferenceEngine(str(dxnn))
    # Width/height come from the variant config -- the same values the runner builds
    # the preprocessor with -- so the gallery cannot be encoded at another scale.
    iw, ih = int(spec["input_width"]), int(spec["input_height"])

    pre = build_processor(spec["preprocessor"], input_width=iw, input_height=ih,
                         config=spec.get("config") or {})
    # No gallery configured: the postprocessor then returns the descriptor and an
    # empty ranking, which is precisely the runtime path minus the comparison.
    post = GalleryRetrievalPostprocessor(iw, ih, config={},
                                         model_name=args.variant)

    print(f"[DXAPP] [INFO] variant  : {args.variant}")
    print(f"[DXAPP] [INFO] example  : {variant_dir.relative_to(ROOT)}")
    print(f"[DXAPP] [INFO] preproc  : {spec['preprocessor']['class']} "
          f"{iw}x{ih}  kwargs={spec['preprocessor'].get('kwargs') or {}}")
    print(f"[DXAPP] [INFO] images   : {len(images)}")

    vectors: list[np.ndarray] = []
    kept: list[Path] = []
    for i, path in enumerate(images, 1):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [WARN] unreadable, skipped: {path}")
            continue
        tensor, ctx = pre.process(img)
        outputs = ie.run([tensor])
        result = post.process(outputs, ctx)[0]
        vectors.append(result.embedding.astype(np.float32))
        kept.append(path)
        if i % 10 == 0 or i == len(images):
            print(f"  encoded {i}/{len(images)}")

    if not vectors:
        raise SystemExit("[DXAPP] [ERROR] nothing encoded")

    dims = {v.size for v in vectors}
    if len(dims) != 1:
        raise SystemExit(f"[DXAPP] [ERROR] inconsistent descriptor widths: {dims}")
    matrix = np.stack(vectors).astype(np.float32)
    rel = [str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
           for p in kept]
    labels = [label_for(p, kept, args.labels) for p in kept]

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_gallery(out, matrix, rel, labels, args.variant)

    norms = np.linalg.norm(matrix, axis=1)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"[DXAPP] [INFO] wrote {shown}: "
          f"{matrix.shape[0]} x {matrix.shape[1]}-d, "
          f"L2 norm min/max {norms.min():.6f}/{norms.max():.6f}, "
          f"{out.stat().st_size:,d} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
