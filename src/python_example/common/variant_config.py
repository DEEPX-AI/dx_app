# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Build a variant's processors from its ``<family>/<variant>/config.json``.

Under the dx-modelzoo family/variant layout one factory serves every variant of its
family, so the per-variant differences -- which processor class, and any non-default
constructor arguments -- live in the variant config instead of in 353 near-identical
factory files. Each variant is a directory named after the ``.dxnn`` stem.

A spec records the original constructor call SHAPE rather than just its arguments::

    {"class": "SimpleResizePreprocessor",
     "args": [{"token": "input_width"}, {"token": "input_height"}],
     "kwargs": {"normalize_float": true},
     "config_overrides": {"num_iterations": 4}}

``args`` is replayed in order: a ``token`` is substituted with the runner-supplied value
and a ``value`` is a literal. Replaying the shape rather than introspecting the target
signature matters -- the SuperPoint and DOPE visualisers do not take a width as their
first positional, so signature guessing mis-builds them.

A spec whose argument could not be reduced to a literal (``imagenet_mean``,
``[m * 255.0 for m in mean]``) carries a ``<expr:...>`` marker. Those variants need real
code, which lives in the family's ``custom_ops.py`` -- the same escape hatch dx-modelzoo
uses. :func:`build_processor` refuses such a spec loudly rather than passing a string
where a float is expected.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from common import processors as _processors
from common import visualizers as _visualizers

EXPR_MARKER = "<expr:"


class VariantConfigError(RuntimeError):
    """A variant config cannot be turned into a processor."""


def iter_variant_configs(family_dir: str | Path) -> list[tuple[str, Path]]:
    """``(variant name, config.json)`` for one family, sorted by name.

    The in-tree layout is ``<family>/<variant>/config.json``. A single-model
    extract keeps one ``config.json`` beside the entry script; the directory
    name is then the variant.
    """
    root = Path(family_dir)
    nested: list[tuple[str, Path]] = []
    if root.is_dir():
        for child in root.iterdir():
            config_path = child / "config.json"
            if child.is_dir() and config_path.is_file():
                nested.append((child.name, config_path))
    if nested:
        return sorted(nested, key=lambda item: item[0])
    flat = root / "config.json"
    if flat.is_file():
        return [(root.name, flat)]
    return []


@lru_cache(maxsize=None)
def load_variant_config(family_dir: str, variant: str) -> dict:
    """Read ``<family_dir>/<variant>/config.json`` (or a flat single-model config)."""
    by_name = dict(iter_variant_configs(family_dir))
    path = by_name.get(variant)
    if path is None:
        available = sorted(by_name)
        raise VariantConfigError(
            f"no variant config {variant!r} in {family_dir}. "
            f"Available: {available}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def default_variant(family_dir: str) -> str:
    """The family's default variant: the alphabetically first PUBLISHED model folder.

    Families are selected by ``--variant``; this exists so a bare invocation still
    runs something sensible rather than erroring on a missing flag.

    "Published" matters because 143 of the 499 declared variants have no .dxnn yet.
    Ordering alone would have moved three existing families onto an unpublished
    default the moment the 2_5_0 additions landed -- clip from
    clip-img_resnet50x16_384x384_openai-wit to clip-img_resnet50_224x224_openai,
    yolo11_pose from -m-pose to -l-pose, stdc_seg from stdc2-seg50 to stdc1-seg50 --
    so a bare run would have started failing on a missing model file.

    A family with no published variant at all (the 23 entirely new ones) falls back to
    the first config: there is nothing better to pick, and the runner's own
    missing-model error names the file.
    """
    configs = iter_variant_configs(family_dir)
    if not configs:
        raise VariantConfigError(f"no variant configs in {family_dir}")
    for name, path in configs:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("published", True):
                return name
        except (OSError, json.JSONDecodeError):
            continue
    return configs[0][0]


def _resolve_class(name: str):
    cls = getattr(_processors, name, None) or getattr(_visualizers, name, None)
    if cls is None:
        raise VariantConfigError(
            f"processor class {name!r} is exported by neither common.processors nor "
            "common.visualizers"
        )
    return cls


def _check_literal(where: str, value: Any) -> None:
    if isinstance(value, str) and value.startswith(EXPR_MARKER):
        raise VariantConfigError(
            f"{where} is a computed expression ({value}), not a literal. This variant "
            "needs its family's custom_ops.py rather than a config entry."
        )


def build_processor(spec: dict, *, input_width: int, input_height: int,
                    config: dict | None = None):
    """Instantiate one processor by replaying *spec*'s recorded call."""
    if not spec or not spec.get("class"):
        raise VariantConfigError(f"spec has no class: {spec!r}")

    merged = {**(config or {}), **(spec.get("config_overrides") or {})}
    tokens = {"input_width": input_width, "input_height": input_height,
              "width": input_width, "height": input_height,
              "w": input_width, "h": input_height,
              "config": merged, "cfg": merged, "self.config": merged}

    args = []
    for i, a in enumerate(spec.get("args") or []):
        if "token" in a:
            if a["token"] not in tokens:
                raise VariantConfigError(
                    f"{spec['class']} arg {i}: unknown token {a['token']!r}")
            args.append(tokens[a["token"]])
        else:
            _check_literal(f"{spec['class']} arg {i}", a.get("value"))
            args.append(a.get("value"))

    kwargs = {}
    for k, v in (spec.get("kwargs") or {}).items():
        if k == "**":
            raise VariantConfigError(
                f"{spec['class']} uses **kwargs expansion ({v}); needs custom_ops.py")
        _check_literal(f"{spec['class']}(.., {k}=)", v)
        kwargs[k] = v
    for k, v in (spec.get("config_overrides") or {}).items():
        _check_literal(f"{spec['class']} config override {k}", v)

    return _resolve_class(spec["class"])(*args, **kwargs)


def spec_is_buildable(spec: dict) -> bool:
    """True when every argument in *spec* is a literal or a known token."""
    if not spec or not spec.get("class"):
        return False
    if "**" in (spec.get("kwargs") or {}):
        return False
    values = [a.get("value") for a in (spec.get("args") or []) if "value" in a]
    values += list((spec.get("kwargs") or {}).values())
    values += list((spec.get("config_overrides") or {}).values())
    return not any(isinstance(v, str) and v.startswith(EXPR_MARKER) for v in values)
