"""Python side of build_sh_sandbox.sh: run a copy of build.sh in a throwaway
tree with cmake, ninja, column, python3 and sudo stubbed, then read back
what it did. Every build.sh run in tests/scripts goes through here (U-74):
the real build.sh re-configures build_<arch>/ and deletes
build_<arch>/release."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import List

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARNESS = HERE / "build_sh_sandbox.sh"
# DXAPP_BUILD_SH_UNDER_TEST lets a reviewer point the same cases at another
# build.sh (e.g. the pre-fix one) to see them fail.
BUILD_SH = Path(os.environ.get("DXAPP_BUILD_SH_UNDER_TEST", ROOT / "build.sh"))

NATIVE = {"arm64": "aarch64"}.get(platform.machine(), platform.machine())
CROSS = "aarch64" if NATIVE == "x86_64" else "x86_64"

needs_toolchain_file = pytest.mark.skipif(
    NATIVE not in ("x86_64", "aarch64"), reason=f"no toolchain file for host arch {NATIVE}")


class Sandbox:
    def __init__(self, path: Path):
        self.path = path
        self.repo = path / "repo"
        self.log = path / "log"

    def run(self, *args: str, **env: str) -> "Sandbox":
        subprocess.run(["bash", str(HARNESS), str(BUILD_SH), str(self.path), *args],
                       env={**os.environ, **env}, check=True, timeout=120)
        return self

    @property
    def rc(self) -> int:
        return int((self.log / "rc").read_text())

    @property
    def output(self) -> str:
        return (self.log / "stdout").read_text(errors="replace")

    @property
    def cmake_argv(self) -> List[str]:
        """argv of the FIRST cmake call (the configure)."""
        raw = (self.log / "cmake_argv").read_bytes().decode()
        return raw.split("\0")[:-1]

    @property
    def cmake_calls(self) -> List[str]:
        """One line per cmake call, in order; [] when cmake never ran."""
        return self.lines("cmake_calls")

    def lines(self, name: str) -> List[str]:
        f = self.log / name
        return f.read_text().splitlines() if f.exists() else []

    def marker(self, d: str) -> bool:
        return (self.repo / d / "marker").exists()

    def exists(self, d: str) -> bool:
        return (self.repo / d).exists()
