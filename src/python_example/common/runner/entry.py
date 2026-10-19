# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Shared launcher for family-level and per-model example scripts.

Both layouts call :func:`run_entry` instead of copying the sys.path walker,
variant peek and ``parse_common_args`` boilerplate:

* ``<task>/<family>/<family>_<kind>.py`` — ``--variant`` / ``-m`` select the model.
* ``<task>/<family>/<variant>/<variant>_<kind>.py`` — the variant is fixed to the
  directory name.

Path setup prefers a vendored ``./common`` and ``./factory`` next to the script
(the single-model extract layout), then parent directories, so the same file
runs in-tree and after ``extract_model_package.sh`` with no ``PYTHONPATH``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Stop the upward search before a symlink loop can walk the whole filesystem.
_MAX_WALK = 8


def install_import_paths(script_file: str | Path) -> Path:
    """Put this script, its family directory and the ``common`` root on ``sys.path``.

    Earlier entries win. A ``common`` or ``factory`` package sitting next to the
    script is preferred over one found further up the tree.
    """
    script_dir = Path(script_file).resolve().parent
    # script_dir first, then the family that holds factory/, then the common root.
    ordered: list[Path] = [script_dir]

    parent = script_dir.parent
    if (parent / "factory").is_dir() or (parent / "common").is_dir():
        ordered.append(parent)

    cursor = script_dir
    for _ in range(_MAX_WALK):
        common_dir = cursor / "common"
        if (common_dir / "runner").is_dir():
            if cursor not in ordered:
                ordered.append(cursor)
            break
        if cursor.parent == cursor:
            break
        cursor = cursor.parent

    for path in reversed(ordered):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    return script_dir


def peek_variant(argv: list[str]) -> str | None:
    """Read ``--variant`` or the ``.dxnn`` stem of ``--model`` / ``-m``.

    The parser shape depends on the variant: an image-only model must not
    register stream flags, and some models add ``--output`` or KITTI paths.
    ``--variant`` wins over ``--model``.
    """
    from common.variants import variant_from_model_path

    model = None
    for index, arg in enumerate(argv):
        if arg == "--variant" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--variant="):
            return arg.split("=", 1)[1]
        if arg in ("--model", "-m") and index + 1 < len(argv):
            model = argv[index + 1]
        elif arg.startswith("--model="):
            model = arg.split("=", 1)[1]
    return variant_from_model_path(model) if model else None


def config_root_for(script_dir: Path) -> Path:
    """Directory :func:`load_variant_config` should search.

    In-tree per-model scripts live in ``<family>/<variant>/`` and load sibling
    configs from the family. A single-model extract keeps ``config.json`` and
    ``factory/`` beside the script, so that directory is the root. A family
    script already sits on the family directory.
    """
    has_local_config = (script_dir / "config.json").is_file()
    has_local_factory = (script_dir / "factory").is_dir()
    if has_local_config and not has_local_factory:
        parent = script_dir.parent
        if (parent / "factory").is_dir():
            return parent
    return script_dir


def ort_off_variant_dir(script_file: str | Path) -> Path:
    """Variant directory a family-level ``<model>_<kind>_ort_off.py`` script runs.

    These ORT-off debug scripts sit on the family directory, which has no
    ``factory/`` of its own, and are named after a registry ``model_name``
    (``yolov8n_sync_ort_off.py``). They run that row's variant
    (``yolov8-n_640x640``), so the factory and the default model without
    ``-m`` (:func:`common.runner.sync_runner._resolve_default_model_path`, same
    name) agree. A name the registry does not know falls back to the family's
    default variant (:func:`common.variant_config.default_variant`, the first
    published one).
    """
    from common.runner.sync_runner import example_key
    from common.variant_config import default_variant
    from common.variants import VariantNotFound, resolve_variant

    script = Path(script_file).resolve()
    family_dir = script.parent
    try:
        variant = resolve_variant(example_key(script.stem)).variant
    except VariantNotFound:
        variant = ""
    if not variant or not (family_dir / variant / "factory").is_dir():
        variant = default_variant(str(family_dir))
    return family_dir / variant


def run_entry(
    script_file: str | Path,
    *,
    factory_cls,
    runner_cls,
    description: str,
    fixed_variant: str | None = None,
) -> None:
    """Parse the variant-shaped CLI and run one inference entry point.

    *fixed_variant* is set by per-model scripts. Family scripts pass ``None``
    and honour ``--variant`` / ``-m``, falling back to the family's default.
    """
    script_dir = install_import_paths(script_file)
    from common.runner.args import parse_common_args
    from common.variant_config import default_variant, load_variant_config

    root = str(config_root_for(script_dir))
    if fixed_variant:
        variant = fixed_variant
    else:
        variant = peek_variant(sys.argv[1:]) or default_variant(root)
    cli = load_variant_config(root, variant).get("cli") or {}
    args = parse_common_args(
        description,
        include_stream_inputs=cli.get("include_stream_inputs", True),
        include_output=cli.get("include_output", False),
        include_kitti_paths=cli.get("include_kitti_paths", False),
    )
    runner_cls(factory_cls(variant=variant)).run(args)
