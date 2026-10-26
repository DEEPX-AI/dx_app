#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
prune_common.py - Remove unreachable common/ files from an extracted standalone package.

Runs as a post-copy pass over a package directory already produced by
extract_model_package.sh. Computes the transitive dependency closure from the
package's own sources and deletes everything under <pkg>/common/ that is not in it.

C++   : follows #include "..." edges. Pure deletion.
Python: follows import edges, then rewrites the surviving __init__.py barrels so
        they no longer re-export deleted modules.

Usage:
    prune_common.py --lang cpp <package_dir>
    prune_common.py --lang py  <package_dir> [--dry-run] [--quiet]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)

CPP_SUFFIXES = (".cpp", ".hpp", ".h", ".cc", ".cxx", ".inl")


# =========================================================================
# Shared helpers
# =========================================================================
def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _iter_files(root: Path, suffixes: Tuple[str, ...]) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
            yield p


def _remove_empty_dirs(root: Path) -> None:
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            next(d.iterdir())
        except StopIteration:
            d.rmdir()
        except OSError:
            pass


# =========================================================================
# C++
# =========================================================================
def _resolve_include(inc: str, pkg: Path, including: Path) -> Path | None:
    """Resolve an #include "..." the way the generated CMakeLists.txt does."""
    for base in (pkg, including.parent, pkg / "utility", pkg / "extern"):
        cand = base / inc
        if cand.is_file():
            return cand.resolve()
    return None


class NoRootsError(RuntimeError):
    """The package has no sources to compute a dependency closure from."""


def cpp_closure(pkg: Path) -> Set[Path]:
    """Transitive #include closure rooted at the package's non-common sources."""
    common = (pkg / "common").resolve()
    roots = [
        p for p in _iter_files(pkg, CPP_SUFFIXES)
        if common not in p.resolve().parents
    ]
    if not roots:
        raise NoRootsError(
            f"no C++ sources outside common/ in {pkg} — refusing to prune, "
            "since an empty closure would delete the entire framework"
        )

    seen: Set[Path] = set()
    stack: List[Path] = list(roots)
    while stack:
        cur = stack.pop()
        for inc in INCLUDE_RE.findall(_read(cur)):
            target = _resolve_include(inc, pkg, cur)
            if target is not None and target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def prune_cpp(pkg: Path, dry_run: bool) -> Tuple[int, int]:
    common = pkg / "common"
    if not common.is_dir():
        return (0, 0)

    keep = cpp_closure(pkg)
    all_files = [p for p in common.rglob("*") if p.is_file()]
    removed = 0
    for f in all_files:
        if f.resolve() not in keep:
            if not dry_run:
                f.unlink()
            removed += 1

    if not dry_run:
        _remove_empty_dirs(common)
    return (len(all_files) - removed, len(all_files))


# =========================================================================
# Python
# =========================================================================
class PyIndex:
    """Module/package index for the copied `common` tree."""

    def __init__(self, pkg: Path):
        self.pkg = pkg
        self.root = pkg  # `common` is imported as a top-level package from here

    def module_path(self, mod: str) -> Path | None:
        """'common.a.b' -> file path, for a module (not a package)."""
        p = self.root / (mod.replace(".", "/") + ".py")
        return p if p.is_file() else None

    def package_init(self, mod: str) -> Path | None:
        """'common.a' -> its __init__.py, if `mod` is a package."""
        p = self.root / mod.replace(".", "/") / "__init__.py"
        return p if p.is_file() else None

    def exists(self, mod: str) -> bool:
        return self.module_path(mod) is not None or self.package_init(mod) is not None


def build_reexport_table(index: PyIndex) -> Dict[Tuple[str, str], str]:
    """(package, exported_name) -> defining module, from every __init__.py."""
    table: Dict[Tuple[str, str], str] = {}
    common_dir = index.root / "common"
    if not common_dir.is_dir():
        return table

    for init in sorted(common_dir.rglob("__init__.py")):
        pkg_name = ".".join(init.relative_to(index.root).parent.parts)
        try:
            tree = ast.parse(_read(init))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve_relative(pkg_name, node)
            if target is None:
                continue
            for alias in node.names:
                if alias.name != "*":
                    table[(pkg_name, alias.asname or alias.name)] = target
    return table


def _resolve_relative(cur_pkg: str, node: ast.ImportFrom) -> str | None:
    """Absolute dotted module name for an ImportFrom node whose package is `cur_pkg`.

    `cur_pkg` is the *containing package* in both cases (a module's parent, or the
    package itself for an __init__.py), which is exactly Python's `__package__`.
    So level 1 always means `cur_pkg`, level 2 its parent, and so on.
    """
    if node.level == 0:
        return node.module if node.module and node.module.split(".")[0] == "common" else None

    parts = cur_pkg.split(".") if cur_pkg else []
    up = node.level - 1
    if up > len(parts):
        return None
    base = parts[: len(parts) - up] if up else parts
    if not base or base[0] != "common":
        return None
    return ".".join(base + ([node.module] if node.module else []))


def _imports_of(path: Path, index: PyIndex, table: Dict[Tuple[str, str], str]) -> Set[str]:
    """common.* modules referenced by `path`, including function-local imports."""
    try:
        tree = ast.parse(_read(path))
    except SyntaxError:
        return set()

    rel = path.relative_to(index.root)
    cur_pkg = ".".join(rel.parent.parts)  # == Python's __package__ for this file

    found: Set[str] = set()
    # ast.walk reaches nested function/method bodies, so deferred imports count.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "common":
                    found |= _expand(alias.name, ["*"], index, table)
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative(cur_pkg, node)
            if target is None:
                continue
            found |= _expand(target, [a.name for a in node.names], index, table)
    return found


def _expand(mod: str, names: List[str], index: PyIndex,
            table: Dict[Tuple[str, str], str]) -> Set[str]:
    """Map an import statement onto the concrete common.* modules it needs."""
    if index.module_path(mod) is not None:
        return {mod}
    if index.package_init(mod) is None:
        return set()

    out: Set[str] = {mod}  # the package's own __init__.py is always needed
    for name in names:
        if name == "*":
            # `from pkg import *` — keep the whole package, we cannot narrow it.
            out |= _whole_package(mod, index)
            continue
        resolved = table.get((mod, name))
        if resolved is not None:
            out.add(resolved)
        elif index.exists(f"{mod}.{name}"):
            out.add(f"{mod}.{name}")  # submodule imported by name
        else:
            # Unknown export: fail-safe, keep everything under the package.
            out |= _whole_package(mod, index)
    return out


def _whole_package(mod: str, index: PyIndex) -> Set[str]:
    d = index.root / mod.replace(".", "/")
    out = {mod}
    for p in d.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        parts = p.relative_to(index.root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out.add(".".join(parts))
    return out


def py_closure(pkg: Path) -> Tuple[Set[str], PyIndex]:
    index = PyIndex(pkg)
    table = build_reexport_table(index)

    common_dir = (pkg / "common").resolve()
    roots = [
        p for p in _iter_files(pkg, (".py",))
        if common_dir not in p.resolve().parents
    ]
    if not roots:
        raise NoRootsError(
            f"no Python sources outside common/ in {pkg} — refusing to prune, "
            "since an empty closure would delete the entire framework"
        )

    seen: Set[str] = set()
    stack: List[str] = []
    for r in roots:
        stack.extend(_imports_of(r, index, table))

    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        # Only follow real modules. A package's __init__.py is a re-export barrel:
        # traversing it would drag in every sibling and defeat the whole pass. It
        # gets rewritten to match the surviving set instead.
        path = index.module_path(mod)
        if path is None:
            continue
        stack.extend(m for m in _imports_of(path, index, table) if m not in seen)

    # A surviving module's ancestor packages must survive too.
    for mod in list(seen):
        parts = mod.split(".")
        for i in range(1, len(parts)):
            seen.add(".".join(parts[:i]))
    return seen, index


def rewrite_init(init: Path, index: PyIndex, kept: Set[str]) -> None:
    """Drop re-exports of deleted modules and prune __all__ to match."""
    src = _read(init)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return

    pkg_name = ".".join(init.relative_to(index.root).parent.parts)
    lines = src.splitlines(keepends=True)

    drop_ranges: List[Tuple[int, int]] = []
    dropped_names: Set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _resolve_relative(pkg_name, node)
        if target is None or target in kept:
            continue
        drop_ranges.append((node.lineno - 1, node.end_lineno))
        for alias in node.names:
            if alias.name != "*":
                dropped_names.add(alias.asname or alias.name)

    new_all: List[str] | None = None
    all_range: Tuple[int, int] | None = None
    if dropped_names:
        for node in tree.body:
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "__all__" for t in node.targets)
                    and isinstance(node.value, (ast.List, ast.Tuple))):
                names = [
                    e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                new_all = [n for n in names if n not in dropped_names]
                all_range = (node.lineno - 1, node.end_lineno)
                break

    if not drop_ranges and all_range is None:
        return

    blank: Set[int] = set()
    for start, end in drop_ranges:
        blank.update(range(start, end))

    out: List[str] = []
    for i, line in enumerate(lines):
        if all_range and i == all_range[0]:
            out.append("__all__ = [\n")
            out.extend(f"    {n!r},\n" for n in (new_all or []))
            out.append("]\n")
            continue
        if all_range and all_range[0] < i < all_range[1]:
            continue
        if i in blank:
            continue
        out.append(line)

    init.write_text("".join(out), encoding="utf-8")


def prune_py(pkg: Path, dry_run: bool) -> Tuple[int, int]:
    common = pkg / "common"
    if not common.is_dir():
        return (0, 0)

    kept, index = py_closure(pkg)
    all_files = [p for p in common.rglob("*.py") if "__pycache__" not in p.parts]

    removed = 0
    for f in all_files:
        parts = f.relative_to(index.root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mod = ".".join(parts)
        if mod in kept:
            continue
        if not dry_run:
            f.unlink()
        removed += 1

    if not dry_run:
        for init in sorted(common.rglob("__init__.py")):
            rewrite_init(init, index, kept)
        for cache in common.rglob("__pycache__"):
            for p in cache.rglob("*"):
                p.unlink()
            cache.rmdir()
        _remove_empty_dirs(common)

    return (len(all_files) - removed, len(all_files))


# =========================================================================
# Main
# =========================================================================
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("package_dir")
    ap.add_argument("--lang", required=True, choices=("cpp", "py"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-no-roots", action="store_true",
                    help="a package with no sources is a warning, not an error "
                         "(used when pruning is the default rather than requested)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    pkg = Path(args.package_dir).resolve()
    if not pkg.is_dir():
        print(f"[PRUNE] [ERROR] package dir not found: {pkg}", file=sys.stderr)
        return 1

    try:
        kept, total = prune_cpp(pkg, args.dry_run) if args.lang == "cpp" \
            else prune_py(pkg, args.dry_run)
    except NoRootsError as exc:
        if args.allow_no_roots:
            print(f"  \u2192 prune skipped: {exc}")
            return 0
        print(f"[PRUNE] [ERROR] {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        if total == 0:
            print("  → prune skipped (no common/ in package)")
        else:
            tag = "would keep" if args.dry_run else "pruned common/"
            print(f"  → {tag}: {kept}/{total} files ({total - kept} removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
