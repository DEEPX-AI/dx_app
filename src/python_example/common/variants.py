# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Variant resolution -- the one place that reads ``config/model_registry.json``.

A *variant* is identified by its ``.dxnn`` stem (e.g. ``yolov5-s_640x640``), which the
dx-modelzoo alignment made the registry's primary key. Resolution is therefore a plain
dict lookup: there is deliberately no normalisation, prefix matching or suffix
stripping here. The 4-pass fuzzy matcher this replaces existed only because example
directory names did not match their ``.dxnn`` -- with the family/variant layout they do,
and a typo must fail loudly instead of resolving to a neighbouring model.

Legacy example-directory names (``yolov5s``, ``deitbase384``) stay accepted as secondary
keys so scripts written against the old per-variant layout keep working.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY = _PROJECT_ROOT / "config" / "model_registry.json"
_MODELS_DIR = _PROJECT_ROOT / "assets" / "models"


class VariantNotFound(KeyError):
    """Raised when a key matches neither a variant nor a legacy model name."""

    def __str__(self) -> str:                      # KeyError quotes its arg otherwise
        return self.args[0] if self.args else ""


@dataclass(frozen=True)
class Variant:
    """One compiled model: its identity, geometry, processors and defaults."""

    variant: str
    dxnn_file: str
    task: str
    family: str
    postprocessor: str
    input_width: int
    input_height: int
    config: dict
    image_only: bool
    model_name: str
    task_legacy: str
    alias_of: Optional[str]

    @property
    def model_path(self) -> Path:
        """Absolute path to the ``.dxnn`` under ``assets/models/``."""
        return _MODELS_DIR / self.dxnn_file


def _to_variant(e: dict[str, Any]) -> Variant:
    return Variant(
        variant=e["variant"],
        dxnn_file=e["dxnn_file"],
        task=e["task"],
        family=e["family"],
        postprocessor=e["postprocessor"],
        input_width=e["input_width"],
        input_height=e["input_height"],
        config=dict(e.get("config") or {}),
        image_only=e["image_only"],
        model_name=e["model_name"],
        task_legacy=e["task_legacy"],
        alias_of=e["alias_of"],
    )


@lru_cache(maxsize=1)
def _index() -> dict[str, Variant]:
    """``{variant | legacy model_name -> Variant}``.

    Legacy names go in first and canonical variants overwrite them, so a legacy name
    that happens to equal some other model's variant never shadows that variant. An
    alias contributes its own ``model_name`` as a key but does not claim the variant
    key -- the canonical entry owns it.
    """
    entries = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    index: dict[str, Variant] = {}
    for e in entries:
        v = _to_variant(e)
        index[e["model_name"]] = v
        # 53 registry names carry a re-publish or quantisation suffix while their old
        # example directory did not (rn50x16_openai_1 -> rn50x16_openai,
        # deeplabv3plus_drn_512x512_q_lite -> deeplabv3plus_drn_512x512). Scripts and
        # tests refer to the DIRECTORY name, so accept that spelling as well.
        stripped = re.sub(r"_q_lite$", "", re.sub(r"_\d+$", "", e["model_name"]))
        if stripped != e["model_name"]:
            index.setdefault(stripped, v)
    for e in entries:
        if e["alias_of"] is None:
            index[e["variant"]] = _to_variant(e)
    return index


@lru_cache(maxsize=1)
def _canonical_variants() -> tuple[Variant, ...]:
    entries = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    seen: set[str] = set()
    out: list[Variant] = []
    for e in entries:
        if e["variant"] in seen:
            continue
        seen.add(e["variant"])
        out.append(_to_variant(e))
    return tuple(out)


def resolve_variant(key: str) -> Variant:
    """Look up *key* as a variant, then as a legacy example-dir / model name."""
    try:
        return _index()[key]
    except KeyError:
        raise VariantNotFound(
            f"unknown variant {key!r}: expected a .dxnn stem such as "
            f"'yolov5-s_640x640', or a legacy model name such as 'yolov5s'. "
            f"{len(_canonical_variants())} variants are registered in "
            f"config/model_registry.json."
        ) from None


def variant_from_model_path(path: str) -> str | None:
    """The variant key a ``.dxnn`` path denotes, or None when it is not a known one.

    A variant key IS the ``.dxnn`` stem, so a caller that only passes ``--model`` has
    already named the variant. Honouring that keeps every pre-existing ``-m``-based
    caller -- run_examples.sh, bench_models.sh, the .vscode launch configs -- correct
    under the family layout without changing any of them, and makes the python entry
    behave like the C++ one, which derives the variant the same way.

    A path outside assets/models, or a locally compiled .dxnn, simply returns None and
    the family default applies.
    """
    if not path:
        return None
    stem = Path(path).name
    if stem.endswith(".dxnn"):
        stem = stem[: -len(".dxnn")]
    v = _index().get(stem)
    return v.variant if v is not None else None


def list_variants(
    *, task: str | None = None, family: str | None = None
) -> Iterator[Variant]:
    """Yield each distinct variant, optionally filtered by task and/or family."""
    for v in _canonical_variants():
        if task is not None and v.task != task:
            continue
        if family is not None and v.family != family:
            continue
        yield v
