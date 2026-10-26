"""Where the C++ binaries are, on Linux and on Windows (SP6 U-37).

build.sh installs into bin/ with bare names (yolov7_sync). On Windows every
name ends in .exe, and the directory is bin/Release (or bin/RelWithDebInfo,
bin/Debug) when a per-configuration build left one, else bin/ - build.bat
copies an install tree's bin/ there. Pure functions over (root, os name) so
the Windows rules are unit-tested on Linux (tests/scripts/test_platform_paths.py).
Not run on Windows here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

WINDOWS_CONFIGS: Tuple[str, ...] = ("Release", "RelWithDebInfo", "Debug")


def current_os_name() -> str:
    """os.name, behind a function so tests can pretend to be Windows."""
    return os.name


def _os(os_name: Optional[str]) -> str:
    return current_os_name() if os_name is None else os_name


def resolve_bin_dir(root: Path, os_name: Optional[str] = None) -> Path:
    """bin/ on Linux; on Windows the first existing config dir, else bin/."""
    base = Path(root) / "bin"
    if _os(os_name) == "nt":
        for config in WINDOWS_CONFIGS:
            if (base / config).is_dir():
                return base / config
    return base


def exe_filename(stem: str, os_name: Optional[str] = None) -> str:
    """`stem` as a file name: `stem.exe` on Windows, unless it already is."""
    if _os(os_name) == "nt" and not stem.lower().endswith(".exe"):
        return stem + ".exe"
    return stem


def binary_path(bin_dir: Path, stem: str, os_name: Optional[str] = None) -> Path:
    return Path(bin_dir) / exe_filename(stem, os_name)
