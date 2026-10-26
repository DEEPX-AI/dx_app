#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Encode a prompt set with a CLIP text tower, once, into ``prompt_bank.json``.

This is a BUILD-TIME tool. It needs ``torch`` and ``open_clip``, which the runtime
venv deliberately does not carry: the generated app must stay numpy-only and
self-contained, so the text side is frozen here instead of being computed on device.

Why not the text tower .dxnn: ``clip-text_resnet50_77x512_openai`` emits the text
transformer's ``[1,77,512]`` hidden states, not a joint embedding -- `ln_final` and
`text_projection` are outside the graph -- and it is an OpenAI RN50 checkpoint, a
different embedding space from every image tower in the zoo.

Run it with an interpreter that has torch + open_clip, e.g.::

    pip install --target /tmp/oc open_clip_torch
    PYTHONPATH=/tmp/oc <venv-with-torch>/bin/python scripts/build_clip_prompt_bank.py \
        --variant clip-img_vit-b32_256x256_datacomp-s34b-b86k

A bank belongs to ONE checkpoint. ``checkpoint`` and ``embed_dim`` are written into
the file so the runtime can refuse a mismatch instead of silently ranking wrongly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The open_clip identifier behind each image tower we ship. A bank is only valid for
# the checkpoint that produced it, so this mapping is the whole correctness story.
CHECKPOINTS = {
    "clip-img_vit-b32_256x256_datacomp-s34b-b86k":
        "hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K",
    "clip-img_vit-l14_224x224_datacomp-xl-s13b-b90k":
        "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "clip-img_vit-l14-quickgelu_224x224_dfn2b":
        "hf-hub:apple/DFN2B-CLIP-ViT-L-14",
    "clip-img_vit-b16-quickgelu_224x224_metaclip-fullcc":
        "hf-hub:facebook/metaclip-b16-fullcc2.5b",
}

# Chosen to cover what the repo's own sample media actually contains -- the demo is
# only as convincing as its label set, and a set that cannot name the picture makes
# the model look wrong.
DEFAULT_LABELS = [
    "dog", "horse", "cat", "bird",
    "person", "crowd of people", "people dancing", "snowboarder",
    "hand", "human face",
    "city street", "parking lot", "road at night", "highway traffic",
    "kitchen", "dark room", "office desk",
    "boat on water", "beach", "mountain",
    "airport seen from above", "aerial view of a city",
    "food on a plate", "flower",
]
TEMPLATE = "a photo of a {}"

VARIANTS_DIR = ROOT / "src" / "python_example" / "zero_shot_image_classification" / "clip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="clip-img_vit-b32_256x256_datacomp-s34b-b86k",
                   help="dxnn stem of the IMAGE tower the bank is built for")
    p.add_argument("--labels", type=Path,
                   help="file with one label per line (default: the built-in set)")
    p.add_argument("--template", default=TEMPLATE,
                   help="prompt template, '{}' is replaced by the label")
    p.add_argument("--out", type=Path,
                   help="output path (default: <example dir>/prompt_bank.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.variant not in CHECKPOINTS:
        raise SystemExit(
            f"no open_clip checkpoint recorded for {args.variant!r}.\n"
            f"Known: {sorted(CHECKPOINTS)}"
        )
    tag = CHECKPOINTS[args.variant]

    try:
        import torch
        import open_clip
    except ImportError as exc:                      # pragma: no cover - build-time
        raise SystemExit(
            f"{exc}. This is a build-time tool: install torch and open_clip_torch "
            "(the runtime venv stays numpy-only on purpose)."
        ) from exc

    labels = ([l.strip() for l in args.labels.read_text(encoding="utf-8").splitlines()
               if l.strip()] if args.labels else list(DEFAULT_LABELS))
    prompts = [args.template.format(l) for l in labels]

    model, _, _ = open_clip.create_model_and_transforms(tag)
    model.eval()
    tokenizer = open_clip.get_tokenizer(tag)
    with torch.no_grad():
        vectors = model.encode_text(tokenizer(prompts)).float()
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    vectors = vectors.cpu().numpy()

    out = args.out or (VARIANTS_DIR / "prompt_bank.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "variant": args.variant,
        "checkpoint": tag,
        "embed_dim": int(vectors.shape[1]),
        "template": args.template,
        "labels": labels,
        "prompts": prompts,
        # Rounded to 6 decimals: the file is read by the demo at every start, and the
        # extra digits cost ~40% of the size while moving no cosine by 1e-5.
        "embeddings": [[round(float(v), 6) for v in row] for row in vectors],
    }, indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(labels)} prompts x {vectors.shape[1]}d, {tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
