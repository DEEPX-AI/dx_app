"""--internal, --internal-path and --internal-video-path are accepted and ignored.

The local-copy mode they selected is gone: models and sample videos always come
from S3. External CI jobs may still pass the options, so every entry point that
used to take them still parses them, warns once and carries on, and none of them
reaches a helper script.

setup.sh and setup_sample_videos.sh run from copies in a temporary tree with
recording stubs in place of the scripts they call.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER = ROOT / "scripts" / "download_models.py"

RECORDER = """#!/bin/bash
printf '%s\\0' "$@" > "$(dirname "$0")/{name}.argv"
exit 0
"""

GUI_ENV_STUB = "dxapp_link_cv2_qt_fonts() { :; }\ndxapp_prepare_gui_env() { :; }\n"


def _copy_helpers(app: Path) -> None:
    (app / "scripts").mkdir(parents=True)
    for f in ("color_env.sh", "common_util.sh"):
        shutil.copy2(ROOT / "scripts" / f, app / "scripts" / f)


def _argv(path: Path) -> list[str]:
    return path.read_text().split("\0")[:-1]


def _setup_tree(tmp_path: Path) -> Path:
    """<tmp>/suite/runtime/dx_app with setup.sh and recording sub-scripts."""
    app = tmp_path / "suite" / "runtime" / "dx_app"
    _copy_helpers(app)
    shutil.copy2(ROOT / "setup.sh", app / "setup.sh")
    (app / "scripts" / "gui_env.sh").write_text(GUI_ENV_STUB)
    (app / "setup_sample_models.sh").write_text(RECORDER.format(name="models"))
    (app / "setup_sample_videos.sh").write_text(RECORDER.format(name="videos"))
    for f in app.glob("*.sh"):
        f.chmod(0o755)
    return app


@pytest.mark.parametrize("args", [
    ["--internal", "--internal-path=/mnt/models", "--internal-video-path=/mnt/videos.tar.gz"],
    ["--internal", "--internal-path", "/mnt/models", "--internal-video-path", "/mnt/videos.tar.gz"],
], ids=["equals", "separate"])
def test_setup_sh_warns_and_forwards_nothing(tmp_path, args):
    app = _setup_tree(tmp_path)
    result = subprocess.run(
        ["bash", str(app / "setup.sh"), f"--docker_volume_path={tmp_path / 'volume'}",
         "--all", *args],
        cwd=app, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Ignoring --internal" in combined
    for name in ("models", "videos"):
        forwarded = " ".join(_argv(app / f"{name}.argv"))
        assert "internal" not in forwarded and "/mnt/" not in forwarded, forwarded


def test_downloader_warns_and_carries_on(tmp_path):
    result = subprocess.run(
        [sys.executable, str(DOWNLOADER), "--dry-run", "--no-json",
         "--output", str(tmp_path / "models"), "--models", "yolov8n",
         "--internal", "--internal-path", "/mnt/models"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Ignoring --internal" in combined


def test_setup_sample_videos_warns_and_forwards_nothing(tmp_path):
    app = tmp_path / "dx_app"
    _copy_helpers(app)
    shutil.copy2(ROOT / "setup_sample_videos.sh", app / "setup_sample_videos.sh")
    (app / "scripts" / "get_resource.sh").write_text(RECORDER.format(name="get_resource"))
    for f in (app / "setup_sample_videos.sh", app / "scripts" / "get_resource.sh"):
        f.chmod(0o755)
    result = subprocess.run(
        ["bash", str(app / "setup_sample_videos.sh"), "--internal",
         "--internal-path=/mnt/videos.tar.gz", f"--output={tmp_path / 'videos'}"],
        cwd=app, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env={**os.environ})
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Ignoring --internal" in combined
    forwarded = " ".join(_argv(app / "scripts" / "get_resource.argv"))
    assert "internal" not in forwarded and "/mnt/" not in forwarded, forwarded
