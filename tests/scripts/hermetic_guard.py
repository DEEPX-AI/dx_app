"""Snapshot the parts of the repository no tests/scripts test may touch (U-74).

A test that ran the real build.sh re-configured build_x86_64 and deleted
build_x86_64/release (build.sh's `rm -rf $build_dir/release`). conftest.py
next to this file snapshots the repository before and after every test and
fails the test that changed:

  * the build and install trees - build/, build_*/, bin/, lib/, include/ -
    every entry, recursively: (mtime_ns, ctime_ns, size, inode, is_dir),
    symlinks not followed;
  * every tracked file (git ls-files), the same fields: a file rewritten
    with its own bytes still counts;
  * the names directly under src/cpp_example/ (a category made for a test).

Pure functions over a root directory: test_hermetic_guard.py runs them on a
fake tree. Not caught: an untracked file created elsewhere in the source tree
(it shows in `git status`), and a rewrite within one filesystem timestamp
tick that keeps both size and inode. Do not build while tests/scripts runs:
the guard would, rightly, report the build.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROTECTED_DIRS: Tuple[str, ...] = ("build", "bin", "lib", "include")
PROTECTED_PREFIX = "build_"
LISTED_DIRS: Tuple[str, ...] = ("src/cpp_example",)


def _entry(st: os.stat_result, is_dir: bool) -> tuple:
    return (st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino, is_dir)


def protected_roots(root: Path) -> List[Path]:
    """The build and install trees that exist under `root`."""
    root = Path(root)
    candidates = [root / name for name in PROTECTED_DIRS]
    candidates += sorted(root.glob(PROTECTED_PREFIX + "*"))
    return [p for p in candidates if p.is_dir() or p.is_symlink()]


def _walk(path: str, rel: str, out: Dict[str, tuple]) -> None:
    # Plain os calls on string paths: pathlib made each snapshot about ten
    # times slower, and the guard takes two per test.
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return  # listed by the parent's scandir, gone since: diff() says "removed"
    is_dir = stat.S_ISDIR(st.st_mode)
    out[rel] = _entry(st, is_dir)
    if not is_dir:
        return
    try:
        with os.scandir(path) as it:
            names = sorted(entry.name for entry in it)
    except OSError:
        return
    for name in names:
        _walk(os.path.join(path, name), rel + "/" + name, out)


def tracked_files(root: Path) -> List[str]:
    """`git ls-files` of `root`; [] when git or the work tree is missing."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def tracked_list_problem(root: Path, tracked: Sequence[str]) -> Optional[str]:
    """Why the tracked-file half of the guard would be silently off: `root`
    is a git checkout (a worktree's .git is a file) but git listed nothing -
    CI's "dubious ownership", git missing, a broken worktree link. None if
    fine."""
    if (Path(root) / ".git").exists() and not tracked:
        return ("hermetic guard: git ls-files listed no tracked file in {} "
                "(safe.directory? git missing?)".format(root))
    return None


def snapshot(root: Path, tracked: Sequence[str] = ()) -> Dict[str, tuple]:
    """What the guard compares: see the module docstring."""
    root = Path(root)
    base = str(root)
    out: Dict[str, tuple] = {}
    for top in protected_roots(root):
        _walk(str(top), top.relative_to(root).as_posix(), out)
    for rel in tracked:
        try:
            out[rel] = _entry(os.lstat(os.path.join(base, rel)), False)
        except FileNotFoundError:
            continue
    for rel in LISTED_DIRS:
        listed = root / rel
        if listed.is_dir():
            out["ls:" + rel] = tuple(sorted(os.listdir(listed)))
    return out


def diff(before: Dict[str, tuple], after: Dict[str, tuple]) -> List[str]:
    """Every path added, removed or changed, sorted by path."""
    changes = []
    for key in sorted(set(before) | set(after)):
        if key not in after:
            changes.append("removed " + key)
        elif key not in before:
            changes.append("added " + key)
        elif before[key] != after[key]:
            changes.append("changed " + key)
    return changes
