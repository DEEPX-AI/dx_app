#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Create one example variant under ``src/{cpp,python}_example/<task>/<family>/<variant>/``.

This is the generation core of ``scripts/add_model.sh``. It copies a reference
variant of the same postprocessor family and rewrites what names the variant:

* C++: the factory header and ``<variant>_{sync,async}.cpp`` are carried over, the
  ``dxapp::v_<reference>`` namespace becomes ``dxapp::v_<variant>`` (every character
  other than a letter or digit becomes ``_``), the include guard follows the same rule,
  ``getModelName()`` returns the new variant, and the file names change. The result
  passes ``generate_cpp_family_layout.py --variant-scope src/cpp_example --check``.
* Python: the family factory, its ``custom_ops.py`` (when the reference has one), the
  ``factory/__init__.py`` and the four entry scripts are carried over with
  ``_FIXED_VARIANT`` set to the new variant.
* Both: ``config.json`` is the reference's with ``variant``, ``dxnn_file``, ``family``,
  ``task``, ``input_width``/``input_height`` and the registry ``config`` rewritten.
* ``config/model_registry.json`` gains a row for the variant unless one exists or
  ``--no-register`` is given, and ``config/test_models.conf`` gains its line.

The reference is, in order, ``--base-model <variant>``, the first registered variant
whose ``postprocessor`` is ``--postprocessor``, the first variant of ``--family``, the
variant's own registry row's family, or the first published variant of the task.

Usage (normally through add_model.sh):
    python3 scripts/add_model_variant.py <variant> <task> [--family F]
        [--base-model V | --postprocessor PP] [--lang cpp|py|both] [--sync-only]
        [--input-size WxH] [--dxnn FILE] [--no-register] [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Task names that older add_model.sh invocations and dx_tool.sh still pass.
LEGACY_TASKS = {
    "detection": "object_detection",
    "classification": "image_classification",
    "pose": "pose_estimation",
    "obb": "oriented_object_detection",
    "obb_detection": "oriented_object_detection",
    "image_enhancement": "low_light_enhancement",
    "attribute_recognition": "person_attribute",
    "reid": "person_reid",
    "face_alignment": "face_landmark",
    "embedding": "face_recognition",
    "ppu": "object_detection",
}
# A legacy task that also implies a family (the PPU models live in one family).
LEGACY_TASK_FAMILY = {"ppu": "yolo_ppu"}

# Postprocessor keys older docs use that the registry spells differently.
POSTPROCESSOR_ALIASES = {
    "yolov11": "yolov8",
    "yolov12": "YOLOv8Postprocessor",
    "yolov13": "YOLOv8Postprocessor",
    "fast_segmentation": "deeplabv3",
    "yolov8n_ppu": "yolov8_ppu",
    "yolov8s_ppu": "yolov8_ppu",
    "clip_image": "CLIPImagePostprocessor",
    "clip_text": "CLIPTextPostprocessor",
    "embedding": "arcface",
    "classification": "efficientnet",
    "mediapipe_hand": "mediapipe_palm",
}
# ``--postprocessor fast_segmentation`` swaps the carried-over class for the fast path.
CLASS_SWAPS = {
    "fast_segmentation": {
        "cpp": ("DeepLabv3Postprocessor", "FastSegmentationPostprocessor"),
        "py": ("SemanticSegmentationPostprocessor", "FastSegmentationPostprocessor"),
    },
}
ENTRY_KINDS = ("sync", "async", "sync_cpp_postprocess", "async_cpp_postprocess")
EXPR_MARKER = "<expr:"


class AddModelError(RuntimeError):
    pass


class AlreadyExists(AddModelError):
    """The variant directory is already there (exit 3: the caller may still --verify)."""


def info(msg: str) -> None:
    print(f"[DXAPP] [INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[DXAPP] [WARN] {msg}")


def variant_namespace(variant: str) -> str:
    """Same rule as generate_cpp_family_layout.variant_namespace."""
    return "v_" + re.sub(r"[^0-9A-Za-z]", "_", variant)


def variant_guard(variant: str) -> str:
    """Same rule as generate_cpp_family_layout.variant_guard."""
    g = re.sub(r"[^0-9A-Za-z]", "_", variant).upper()
    return g if g[:1].isalpha() else "DXAPP_" + g


def python_module_stem(variant: str) -> str:
    """The factory module name is the variant itself (``factory/<variant>_factory.py``);
    ``factory/__init__.py`` loads it by path, so dashes and a leading digit are fine."""
    return variant


# ---------------------------------------------------------------------------------
# Registry and tree lookups
# ---------------------------------------------------------------------------------

class Tree:
    def __init__(self, root: Path):
        self.root = root
        self.cpp = root / "src" / "cpp_example"
        self.py = root / "src" / "python_example"
        self.registry_path = root / "config" / "model_registry.json"
        self.conf_path = root / "config" / "test_models.conf"
        self.rows: list[dict] = []
        if self.registry_path.is_file():
            self.rows = json.loads(self.registry_path.read_text(encoding="utf-8"))

    def row_for(self, variant: str) -> dict | None:
        for r in self.rows:
            if r.get("alias_of"):
                continue
            if r.get("variant") == variant or r.get("model_name") == variant:
                return r
        return None

    def src(self, lang: str) -> Path:
        return self.cpp if lang == "cpp" else self.py

    def variant_dir(self, lang: str, task: str, family: str, variant: str) -> Path:
        return self.src(lang) / task / family / variant

    def has_factory(self, lang: str, task: str, family: str, variant: str) -> bool:
        d = self.variant_dir(lang, task, family, variant)
        suffix = "_factory.hpp" if lang == "cpp" else "_factory.py"
        stem = variant if lang == "cpp" else python_module_stem(variant)
        return (d / "factory" / (stem + suffix)).is_file()

    def find_variant(self, variant: str, langs: list[str]) -> tuple[str, str] | None:
        """``(task, family)`` of an existing variant dir, searched in every language."""
        for lang in langs:
            for fac in sorted(self.src(lang).glob(f"*/*/{variant}/factory/")):
                vdir = fac.parent
                if vdir.parts[-3] == "common":
                    continue
                return vdir.parts[-3], vdir.parts[-2]
        return None

    def family_variants(self, lang: str, task: str, family: str) -> list[str]:
        fam = self.src(lang) / task / family
        if not fam.is_dir():
            return []
        return sorted(p.name for p in fam.iterdir()
                      if p.is_dir() and (p / "factory").is_dir())

    def known_tasks(self) -> set[str]:
        tasks = {r.get("task") for r in self.rows if r.get("task")}
        for src in (self.cpp, self.py):
            if src.is_dir():
                tasks |= {p.name for p in src.iterdir()
                          if p.is_dir() and p.name not in ("common", "multi_model_graph")
                          and not p.name.startswith((".", "_"))}
        return tasks


def normalize_task(task: str, tree: Tree) -> tuple[str, str | None]:
    """``(task directory, implied family or None)``; legacy names are mapped."""
    implied = LEGACY_TASK_FAMILY.get(task)
    task = LEGACY_TASKS.get(task, task)
    known = tree.known_tasks()
    if known and task not in known:
        raise AddModelError(
            f"unknown task {task!r}. Known tasks: {', '.join(sorted(known))}")
    return task, implied


def resolve_reference(tree: Tree, task: str, family: str | None, variant_row: dict | None,
                      base_model: str | None, postprocessor: str | None,
                      langs: list[str]) -> tuple[str, str, str]:
    """``(ref_task, ref_family, ref_variant)`` of the variant to copy from."""
    if base_model:
        found = tree.find_variant(base_model, langs)
        if found is None:
            raise AddModelError(f"base model not found in the example tree: {base_model}")
        return found[0], found[1], base_model

    def complete(r: dict) -> bool:
        return all(tree.has_factory(lang, r["task"], r["family"], r["variant"])
                   for lang in langs)

    if postprocessor:
        wanted = {postprocessor.lower(),
                  POSTPROCESSOR_ALIASES.get(postprocessor, postprocessor).lower()}
        rows = [r for r in tree.rows if not r.get("alias_of")
                and str(r.get("postprocessor", "")).lower() in wanted and complete(r)]
        if not rows:
            available = sorted({str(r.get("postprocessor")) for r in tree.rows
                                if r.get("postprocessor")})
            raise AddModelError(
                f"no registered example uses postprocessor {postprocessor!r}. "
                f"Available: {', '.join(available)}")
        rows.sort(key=lambda r: (r["task"] != task, family is not None and r["family"] != family,
                                 not r.get("published", True), r["variant"]))
        r = rows[0]
        return r["task"], r["family"], r["variant"]

    for fam in (family, variant_row["family"] if variant_row else None):
        if fam:
            for lang in langs:
                for v in tree.family_variants(lang, task, fam):
                    if all(tree.has_factory(lang2, task, fam, v) for lang2 in langs):
                        return task, fam, v

    rows = [r for r in tree.rows if not r.get("alias_of") and r.get("task") == task
            and complete(r)]
    rows.sort(key=lambda r: (not r.get("published", True), r["variant"]))
    if rows:
        r = rows[0]
        return r["task"], r["family"], r["variant"]
    raise AddModelError(
        f"no reference variant for task {task!r}; pass --base-model <variant> or "
        "--postprocessor <key>")


# ---------------------------------------------------------------------------------
# Text rewriting
# ---------------------------------------------------------------------------------

def sanitized(variant: str) -> str:
    """The identifier form of a variant: every non-alphanumeric character is ``_``.

    It is what the C++ namespace (``v_<sanitized>``), the include guard (upper-cased)
    and the Python factory module name (``dxapp_factory_<sanitized>_factory``) carry.
    """
    return re.sub(r"[^0-9A-Za-z]", "_", variant)


def rewrite_names(text: str, ref: str, new: str) -> str:
    """Rename every spelling of the reference variant: guard, identifier form, raw name."""
    ref_guard, new_guard = variant_guard(ref), variant_guard(new)
    if ref_guard != new_guard:
        text = re.sub(rf"\b{re.escape(ref_guard)}", new_guard, text)
    ref_id, new_id = sanitized(ref), sanitized(new)
    # Identifier contexts first: the raw name may be its own identifier form
    # (yolov7_640x640), and the new one may not be (yolov7-custom_640x640).
    text = re.sub(rf"\bv_{re.escape(ref_id)}\b", f"v_{new_id}", text)
    text = text.replace(f"dxapp_factory_{ref_id}_factory", f"dxapp_factory_{new_id}_factory")
    if ref_id != ref:
        text = text.replace(ref_id, new_id)
    return text.replace(ref, new)


def rename_file(name: str, ref: str, new: str) -> str:
    ref_mod, new_mod = python_module_stem(ref), python_module_stem(new)
    if ref_mod != ref and name.startswith(ref_mod):
        return new_mod + name[len(ref_mod):]
    return name.replace(ref, new)


def patched_config(ref_cfg: dict, *, variant: str, dxnn: str, family: str, task: str,
                   width: int, height: int, row: dict | None) -> dict:
    cfg = dict(ref_cfg)
    cfg["variant"] = variant
    cfg["dxnn_file"] = dxnn
    cfg["task"] = task
    cfg["family"] = family
    cfg["input_width"] = width
    cfg["input_height"] = height
    if row is not None:
        if row.get("config"):
            cfg["config"] = dict(row["config"])
            cfg["registry_config"] = dict(row["config"])
        if "image_only" in row:
            cfg["registry_image_only"] = bool(row["image_only"])
        cfg["published"] = bool(row.get("published", True))
        cfg["zoo_canonical"] = bool(row.get("zoo_canonical", False))
    else:
        cfg["published"] = True
        cfg["zoo_canonical"] = False
    return cfg


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------

def generate_cpp(tree: Tree, ref: tuple[str, str, str], task: str, family: str,
                 variant: str, *, sync_only: bool, cfg: dict, swap: tuple | None) -> Path:
    ref_task, ref_family, ref_variant = ref
    src = tree.variant_dir("cpp", ref_task, ref_family, ref_variant)
    dst = tree.variant_dir("cpp", task, family, variant)
    if dst.exists():
        raise AlreadyExists(f"C++ variant already exists: {dst}")
    (dst / "factory").mkdir(parents=True)

    def carry(rel_src: Path, rel_dst: Path) -> None:
        text = rewrite_names(rel_src.read_text(encoding="utf-8"), ref_variant, variant)
        if swap and swap[0] in text:
            text = text.replace(swap[0], swap[1])
        rel_dst.write_text(text, encoding="utf-8")
        print(f"  -> {rel_dst.relative_to(tree.root)}  (<- {rel_src.name})")

    for h in sorted((src / "factory").glob("*.hpp")):
        carry(h, dst / "factory" / rename_file(h.name, ref_variant, variant))
    carry(src / f"{ref_variant}_sync.cpp", dst / f"{variant}_sync.cpp")
    if not sync_only and (src / f"{ref_variant}_async.cpp").is_file():
        carry(src / f"{ref_variant}_async.cpp", dst / f"{variant}_async.cpp")
    write_json(dst / "config.json", cfg)
    print(f"  -> {(dst / 'config.json').relative_to(tree.root)}")
    return dst


def generate_py(tree: Tree, ref: tuple[str, str, str], task: str, family: str,
                variant: str, *, sync_only: bool, cfg: dict, swap: tuple | None) -> Path:
    ref_task, ref_family, ref_variant = ref
    src = tree.variant_dir("py", ref_task, ref_family, ref_variant)
    dst = tree.variant_dir("py", task, family, variant)
    if dst.exists():
        raise AlreadyExists(f"Python variant already exists: {dst}")
    (dst / "factory").mkdir(parents=True)
    family_init = dst.parent / "__init__.py"
    if not family_init.exists():
        family_init.write_text("")
        print(f"  -> {family_init.relative_to(tree.root)}  (new family package)")

    def carry(rel_src: Path, rel_dst: Path) -> None:
        text = rewrite_names(rel_src.read_text(encoding="utf-8"), ref_variant, variant)
        if ref_family != family:
            text = text.replace(f'"""{ref_family} family factory',
                                f'"""{family} family factory')
            text = text.replace(f"family of the ``{ref_family}`` family",
                                f"family of the ``{family}`` family")
        if swap and swap[0] in text:
            text = text.replace(swap[0], swap[1])
        rel_dst.write_text(text, encoding="utf-8")
        if rel_src.suffix == ".py" and rel_src.stat().st_mode & 0o111:
            rel_dst.chmod(rel_dst.stat().st_mode | 0o111)
        print(f"  -> {rel_dst.relative_to(tree.root)}  (<- {rel_src.name})")

    for p in sorted((src / "factory").glob("*.py")):
        carry(p, dst / "factory" / rename_file(p.name, ref_variant, variant))
    if (src / "custom_ops.py").is_file():
        carry(src / "custom_ops.py", dst / "custom_ops.py")
    for kind in ENTRY_KINDS:
        if sync_only and kind.startswith("async"):
            continue
        entry = src / f"{ref_variant}_{kind}.py"
        if entry.is_file():
            carry(entry, dst / f"{variant}_{kind}.py")
    write_json(dst / "config.json", cfg)
    print(f"  -> {(dst / 'config.json').relative_to(tree.root)}")

    text = json.dumps(cfg)
    if EXPR_MARKER in text:
        warn(f"{variant}: the reference config carries computed arguments ({EXPR_MARKER}...). "
             f"Add the variant to FACTORIES in {dst.relative_to(tree.root)}/custom_ops.py "
             "or replace them with literals in config.json.")
    return dst


def register(tree: Tree, ref_row: dict | None, *, variant: str, dxnn: str, family: str,
             task: str, width: int, height: int, postprocessor: str | None) -> dict:
    row = {
        "model_name": variant,
        "dxnn_file": dxnn,
        "original_name": variant,
        "csv_task": (ref_row or {}).get("csv_task", ""),
        "add_model_task": (ref_row or {}).get("add_model_task", task),
        "postprocessor": postprocessor or (ref_row or {}).get("postprocessor", ""),
        "input_width": width,
        "input_height": height,
        "config": dict((ref_row or {}).get("config") or {}),
        "source": "manual",
        "supported": True,
        "variant": variant,
        "family": family,
        "task": task,
        "task_legacy": (ref_row or {}).get("task_legacy", task),
        "image_only": bool((ref_row or {}).get("image_only", False)),
        "zoo_canonical": False,
        "alias_of": None,
        "published": True,
    }
    tree.rows.append(row)
    write_json(tree.registry_path, tree.rows)
    info(f"registered {variant} in {tree.registry_path.relative_to(tree.root)} "
         f"(family {family}, task {task}, {width}x{height})")
    return row


def add_conf_line(tree: Tree, *, family: str, task: str, dxnn: str, variant: str) -> None:
    line = f"{family}\t{task}\tassets/models/{dxnn}\t{variant}\n"
    existing = tree.conf_path.read_text(encoding="utf-8") if tree.conf_path.is_file() else ""
    if any(l.split("\t")[-1].strip() == variant for l in existing.splitlines()
           if l and not l.startswith("#")):
        return
    with tree.conf_path.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(line)
    info(f"added {variant} to {tree.conf_path.relative_to(tree.root)}")


# ---------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------

def parse_size(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", text)
    if not m:
        raise AddModelError(f"--input-size must be WIDTHxHEIGHT, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", help="the new variant: the .dxnn stem, e.g. yolov8-n_640x640")
    ap.add_argument("task", help="task directory, e.g. object_detection (legacy names accepted)")
    ap.add_argument("--family", help="family directory (default: the registry row's, "
                                     "else the reference's)")
    ap.add_argument("--base-model", help="existing variant to copy from")
    ap.add_argument("--postprocessor", help="registry postprocessor key to pick the reference by")
    ap.add_argument("--lang", choices=("cpp", "py", "both"), default="both")
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--input-size", help="WIDTHxHEIGHT (default: the registry row's, "
                                         "else the reference's)")
    ap.add_argument("--dxnn", help="model file name (default: <variant>.dxnn)")
    ap.add_argument("--no-register", action="store_true",
                    help="do not add a registry row / test_models.conf line")
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    tree = Tree(Path(a.root).resolve())
    langs = ["cpp", "py"] if a.lang == "both" else [a.lang]
    variant = a.variant
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", variant):
        raise AddModelError(f"variant name {variant!r}: use letters, digits, '.', '_' and '-'")

    task, implied_family = normalize_task(a.task, tree)
    row = tree.row_for(variant)
    if row is not None and row["variant"] != variant:
        info(f"{variant} is the registry model_name of variant {row['variant']}")
        variant = row["variant"]
    if row is not None and row.get("task") and row["task"] != task:
        warn(f"registry row for {variant} is task {row['task']}, not {task}; using {row['task']}")
        task = row["task"]
    family = a.family or (row or {}).get("family") or implied_family

    # An existing variant is refused before anything else is resolved or written, so
    # a --verify of an existing example (validate_models.sh) never depends on the
    # reference lookup.
    if family:
        for lang in langs:
            if tree.variant_dir(lang, task, family, variant).exists():
                raise AlreadyExists(
                    f"variant already exists: {tree.variant_dir(lang, task, family, variant)}")
    else:
        found = tree.find_variant(variant, langs)
        if found is not None:
            raise AlreadyExists(
                f"variant already exists: {tree.variant_dir(langs[0], found[0], found[1], variant)}")

    ref = resolve_reference(tree, task, family, row, a.base_model, a.postprocessor, langs)
    ref_task, ref_family, ref_variant = ref
    family = family or ref_family
    ref_row = tree.row_for(ref_variant)
    info(f"reference: {ref_task}/{ref_family}/{ref_variant}")

    for lang in langs:
        if not tree.has_factory(lang, ref_task, ref_family, ref_variant):
            raise AddModelError(
                f"reference {ref_variant} has no {lang} factory under "
                f"{tree.variant_dir(lang, ref_task, ref_family, ref_variant)}")
        if tree.variant_dir(lang, task, family, variant).exists():
            raise AlreadyExists(
                f"variant already exists: {tree.variant_dir(lang, task, family, variant)}")

    dxnn = a.dxnn or (row or {}).get("dxnn_file") or f"{variant}.dxnn"
    if a.input_size:
        width, height = parse_size(a.input_size)
    elif row and row.get("input_width") and row.get("input_height"):
        width, height = int(row["input_width"]), int(row["input_height"])
    else:
        ref_cfg_path = tree.variant_dir(langs[0], ref_task, ref_family, ref_variant) / "config.json"
        ref_cfg = json.loads(ref_cfg_path.read_text(encoding="utf-8")) if ref_cfg_path.is_file() else {}
        width = int(ref_cfg.get("input_width") or (ref_row or {}).get("input_width") or 0)
        height = int(ref_cfg.get("input_height") or (ref_row or {}).get("input_height") or 0)
        if not (width and height):
            raise AddModelError("input size unknown: pass --input-size WIDTHxHEIGHT")
        warn(f"input size taken from the reference ({width}x{height}); pass --input-size "
             "if the new model differs")

    if row is None and not a.no_register:
        row = register(tree, ref_row, variant=variant, dxnn=dxnn, family=family, task=task,
                       width=width, height=height, postprocessor=a.postprocessor)
    if not a.no_register:
        add_conf_line(tree, family=family, task=task, dxnn=dxnn, variant=variant)

    swaps = CLASS_SWAPS.get(a.postprocessor or "", {})
    made: dict[str, Path] = {}
    for lang in langs:
        ref_cfg_path = tree.variant_dir(lang, ref_task, ref_family, ref_variant) / "config.json"
        ref_cfg = json.loads(ref_cfg_path.read_text(encoding="utf-8")) if ref_cfg_path.is_file() else {}
        cfg = patched_config(ref_cfg, variant=variant, dxnn=dxnn, family=family, task=task,
                             width=width, height=height, row=row)
        print()
        info(f"creating {lang} package: {task}/{family}/{variant}")
        gen = generate_cpp if lang == "cpp" else generate_py
        made[lang] = gen(tree, ref, task, family, variant, sync_only=a.sync_only, cfg=cfg,
                         swap=swaps.get(lang))

    print()
    for lang, path in made.items():
        print(f"{lang.upper()}_DIR={path}")
    print(f"VARIANT={variant}")
    print(f"TASK={task}")
    print(f"FAMILY={family}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AlreadyExists as exc:
        print(f"[DXAPP] [ERROR] {exc}", file=sys.stderr)
        sys.exit(3)
    except AddModelError as exc:
        print(f"[DXAPP] [ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
