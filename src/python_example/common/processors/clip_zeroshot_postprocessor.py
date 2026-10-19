# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""CLIP zero-shot classification: image embedding on the NPU, text encoded offline.

The zoo's only text tower, ``clip-text_resnet50_77x512_openai``, cannot close this
loop. Measured, it takes ``[1,77,512]`` and returns ``[1,77,512]`` -- the text
transformer's hidden states. The joint embedding needs `token_embedding` and
`positional_embedding` on the way in and `ln_final` + `text_projection` on the way
out, none of which is in the .dxnn; and it is an OpenAI RN50 checkpoint, a different
embedding space from every image tower here. Pairing it with a ViT-B/32 image tower
would compare two unrelated spaces and still produce numbers.

So the text side is computed ONCE, at build time, by
``scripts/build_clip_prompt_bank.py`` (torch + open_clip) and shipped next to the
example as ``prompt_bank.json``. At runtime this class is pure numpy: L2-normalise
the image embedding, one ``(N, D) @ (D,)`` product, softmax. The app stays
self-contained -- no torch, no tokenizer, no network.

The consequence, stated here because it is a real limitation rather than an
oversight: the prompt set is FIXED at build time. New prompts mean re-running the
generator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..base import ClassificationResult, IPostprocessor, PreprocessContext


class CLIPZeroShotPostprocessor(IPostprocessor):
    """Cosine similarity against a precomputed prompt bank -> top-k labels."""

    def __init__(self, input_width: int = 256, input_height: int = 256,
                 config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.bank_path = str(self.config.get("prompt_bank", "prompt_bank.json"))
        self.top_k = int(self.config.get("top_k", 5))
        # CLIP's trained logit scale is exp(4.6052) = 100. It is what turns cosine
        # similarities -- which live in a narrow 0.1..0.4 band -- into a usable
        # distribution, so it is not a cosmetic constant.
        self.logit_scale = float(self.config.get("logit_scale", 100.0))
        self._bank: dict | None = None

    # ------------------------------------------------------------------ bank
    def _load(self) -> dict:
        if self._bank is not None:
            return self._bank
        path = Path(self.bank_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"[DXAPP] [ERROR] CLIPZeroShotPostprocessor - prompt bank not found: "
                f"{path}\n  Build it with:\n"
                f"    python scripts/build_clip_prompt_bank.py --variant <dxnn-stem>"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        vectors = np.asarray(raw["embeddings"], dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"[DXAPP] [ERROR] prompt bank {path} has embeddings of shape "
                f"{vectors.shape}; expected (num_prompts, embed_dim)."
            )
        # Normalise defensively: the generator writes unit vectors, but a bank edited
        # by hand would otherwise skew every score without any visible symptom.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._bank = {
            "labels": [str(x) for x in raw["labels"]],
            "vectors": vectors / np.maximum(norms, 1e-12),
            "checkpoint": raw.get("checkpoint", "?"),
        }
        return self._bank

    @property
    def labels(self) -> List[str]:
        return self._load()["labels"]

    # ------------------------------------------------------------------ decode
    def process(self, outputs: Sequence[np.ndarray],
                ctx: PreprocessContext) -> List[ClassificationResult]:
        bank = self._load()
        vectors = bank["vectors"]

        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if embedding.size != vectors.shape[1]:
            raise ValueError(
                "[DXAPP] [ERROR] CLIPZeroShotPostprocessor - the prompt bank and the "
                "model come from different checkpoints.\n"
                f"  bank {self.bank_path!r} ({bank['checkpoint']}): embed_dim "
                f"{vectors.shape[1]}\n"
                f"  model output: {embedding.size}\n"
                "  Rebuild the bank for THIS model: "
                "python scripts/build_clip_prompt_bank.py --variant <dxnn-stem>"
            )
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)

        logits = self.logit_scale * (vectors @ embedding)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        order = np.argsort(-probs)[:max(1, self.top_k)]
        top_k = [(int(i), float(probs[i])) for i in order]
        return [
            ClassificationResult(class_id=int(i), class_name=bank["labels"][int(i)],
                                 confidence=float(probs[i]), top_k=top_k)
            for i in order
        ]

    def get_model_name(self) -> str:
        return "clip_zeroshot"
