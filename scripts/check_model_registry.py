#!/usr/bin/env python3
"""Verify config/model_registry.json against the factory tree.

The model key is the variant: each row names its <task>/<family>/<variant>/
directory, which holds factory/<variant>_factory.hpp and _factory.py.

Invariants:
  1. every row's <task>/<family>/<variant>/ holds a C++ and a Python factory,
     and every factory directory is registered;
  2. one row per variant carries it; any other row on that variant is an
     alias_of row, naming a row of the same variant;
  3. every dxnn_file is claimed by at most one row, alias_of rows aside
     (an alias is not a second model);
  4. no name resolves two ways: an old model_name or an alias_of name that is
     also another variant, or one name for two variants, is ambiguous.

resolve_registry() is THE registry rule: scripts/gen_model_registry.py
imports it, so the generated graph registry and this guard cannot disagree
about which row is a model and which name is an alias.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "model_registry.json"
CPP = ROOT / "src" / "cpp_example"
PY = ROOT / "src" / "python_example"

LEGACY_NAME = "legacy"   # a row's model_name, where it differs from its variant
ALIAS_OF = "alias_of"    # an alias_of row's model_name


def resolve_registry(entries):
    """(models, aliases, errors).

    models:  variant -> the row that carries it (the one without alias_of);
    aliases: [(name, variant, kind)], kind LEGACY_NAME or ALIAS_OF, sorted;
    errors:  human-readable problems; empty means every name has one meaning.
    """
    errors = []
    models = {}
    carriers = defaultdict(list)
    by_name = {}
    for entry in entries:
        variant = entry.get("variant")
        if not isinstance(variant, str) or not variant:
            errors.append("NO VARIANT {}: the row has no variant".format(entry.get("model_name")))
            continue
        by_name.setdefault(entry.get("model_name"), entry)
        if not entry.get("alias_of"):
            carriers[variant].append(entry)
    for variant, rows in sorted(carriers.items()):
        if len(rows) > 1:
            errors.append("DUPLICATE VARIANT {} <- {} (mark the extra rows alias_of)".format(
                variant, ", ".join(r.get("model_name", "?") for r in rows)))
        models[variant] = rows[0]

    names = defaultdict(set)
    for entry in entries:
        variant, name, target = entry.get("variant"), entry.get("model_name"), entry.get("alias_of")
        if not variant or not isinstance(name, str) or not name:
            continue
        if target:
            carrier = by_name.get(target)
            if carrier is None or carrier.get("alias_of"):
                errors.append('BAD ALIAS "{}": alias_of "{}" names no registry row'.format(
                    name, target) if carrier is None else
                    'BAD ALIAS "{}": alias_of "{}" is itself an alias'.format(name, target))
                continue
            if carrier.get("variant") != variant:
                errors.append('BAD ALIAS "{}": alias_of "{}" is variant {}, but this row is {}'
                              .format(name, target, carrier.get("variant"), variant))
                continue
            kind = ALIAS_OF
        else:
            kind = LEGACY_NAME
        if name != variant:
            names[name].add((variant, kind))

    aliases = []
    for name, targets in sorted(names.items()):
        variants = sorted({v for v, _ in targets})
        if name in models:
            errors.append('AMBIGUOUS ALIAS "{}": it is variant {} and also names {}'.format(
                name, name, ", ".join(variants)))
            continue
        if len(variants) > 1:
            errors.append('AMBIGUOUS ALIAS "{}": it names {}'.format(name, ", ".join(variants)))
            continue
        kind = ALIAS_OF if any(k == ALIAS_OF for _, k in targets) else LEGACY_NAME
        aliases.append((name, variants[0], kind))
    return models, aliases, errors


# Example sources kept on disk. They are not registry rows and are not built.
KEPT_UNREGISTERED = {
    ("face_detection", "scrfd", "SCRFD500M_PPU"),
    ("pose_estimation", "yolov5_pose", "YOLOV5Pose_PPU"),
}


def factory_dirs(base, suffix):
    """(task, family, variant) of every <task>/<family>/<variant>/factory/<variant><suffix>."""
    return {tuple(p.parts[-5:-2]) for p in Path(base).glob("*/*/*/factory/*" + suffix)
            if p.name == p.parts[-3] + suffix}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registry", default=str(REGISTRY))
    parser.add_argument("--cpp-root", default=str(CPP))
    parser.add_argument("--py-root", default=str(PY))
    args = parser.parse_args(argv)

    entries = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    cpp = factory_dirs(args.cpp_root, "_factory.hpp")
    py = factory_dirs(args.py_root, "_factory.py")
    models, _aliases, problems = resolve_registry(entries)

    failures = 0
    for message in problems:
        print(message)
        failures += 1

    registered = set()
    for entry in entries:
        key = (entry.get("task"), entry.get("family"), entry.get("variant"))
        registered.add(key)
        where = "/".join(str(k) for k in key)
        if key not in cpp:
            print("NO CPP FACTORY  {} ({})".format(entry.get("variant"), where))
            failures += 1
        if key not in py:
            print("NO PY FACTORY   {} ({})".format(entry.get("variant"), where))
            failures += 1

    for orphan in sorted((cpp | py) - registered - KEPT_UNREGISTERED):
        print("UNREGISTERED    {} (factory exists, no registry entry)".format("/".join(orphan)))
        failures += 1

    by_artifact = defaultdict(list)
    for entry in entries:
        if not entry.get("alias_of"):
            by_artifact[entry["dxnn_file"]].append(entry["model_name"])
    for artifact, names in sorted(by_artifact.items()):
        if len(names) > 1:
            print("DUPLICATE DXNN  {} <- {}".format(artifact, ", ".join(names)))
            failures += 1

    if failures:
        print("{} registry inconsistency(ies)".format(failures))
        return 1
    print("model_registry.json consistent with the factory tree ({} entries, {} variants)"
          .format(len(entries), len(models)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
