"""Regression tests for `setup.sh` force / --no-force argument forwarding.

`setup.sh` used to encode `--no-force` as an empty `FORCE_ARGS`, which is
indistinguishable from "no flag given". Both child layers default to force-on
(`download_models.py --force default=True`, `setup_sample_videos.sh
FORCE_ARGS="--force"`), so `--no-force` was silently dropped.

These tests run `setup.sh` inside a throwaway sandbox where the two child
scripts are replaced by stubs that only record their argv. Nothing is
downloaded and no model or video assets are touched.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

STUB = """#!/bin/bash
printf '%s\\n' "$@" > "$(dirname "$0")/{record}"
exit 0
"""


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A minimal dx_app tree with stubbed child download scripts.

    Nested two levels deep so setup.sh's DX_AS_PATH (`../..`) — and the
    `workspace/res` directories it creates — stay inside tmp_path.
    """
    app = tmp_path / "suite" / "runtime" / "dx_app"
    (app / "scripts").mkdir(parents=True)

    shutil.copy(ROOT / "setup.sh", app / "setup.sh")
    for helper in ("color_env.sh", "common_util.sh"):
        shutil.copy(ROOT / "scripts" / helper, app / "scripts" / helper)

    for child, record in (
        ("setup_sample_models.sh", "models_args.txt"),
        ("setup_sample_videos.sh", "videos_args.txt"),
    ):
        stub = app / child
        stub.write_text(STUB.format(record=record))
        stub.chmod(0o755)

    # Pre-create the asset dirs so the "directory already exists" branches of
    # setup.sh are the ones under test.
    (app / "assets" / "models").mkdir(parents=True)
    (app / "assets" / "videos").mkdir(parents=True)
    return app


def run_setup(sandbox: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "./setup.sh", f"--docker_volume_path={sandbox / 'dockervol'}", *args],
        cwd=sandbox,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def recorded(sandbox: Path, name: str) -> list[str] | None:
    """argv the stub received, or None if the stub was never invoked."""
    path = sandbox / name
    if not path.exists():
        return None
    return [line for line in path.read_text().splitlines() if line]


def test_no_force_is_forwarded_to_model_downloader(sandbox: Path) -> None:
    result = run_setup(sandbox, "--no-force", "--models", "YoloV5S")
    assert result.returncode == 0, result.stdout

    args = recorded(sandbox, "models_args.txt")
    assert args is not None, f"model downloader was never invoked\n{result.stdout}"
    assert "--no-force" in args, args
    assert "--force" not in args, args


def test_no_force_is_forwarded_to_video_downloader(sandbox: Path) -> None:
    result = run_setup(sandbox, "--no-force", "--models", "YoloV5S")
    assert result.returncode == 0, result.stdout

    args = recorded(sandbox, "videos_args.txt")
    assert args is not None, f"video downloader was never invoked\n{result.stdout}"
    assert "--no-force" in args, args
    assert "--force" not in args, args


def test_default_forwards_force(sandbox: Path) -> None:
    result = run_setup(sandbox, "--models", "YoloV5S")
    assert result.returncode == 0, result.stdout

    for name in ("models_args.txt", "videos_args.txt"):
        args = recorded(sandbox, name)
        assert args is not None, f"{name}: never invoked\n{result.stdout}"
        assert "--force" in args, (name, args)
        assert "--no-force" not in args, (name, args)


def test_explicit_force_is_forwarded(sandbox: Path) -> None:
    result = run_setup(sandbox, "--force", "--models", "YoloV5S")
    assert result.returncode == 0, result.stdout

    args = recorded(sandbox, "models_args.txt")
    assert args is not None, result.stdout
    assert "--force" in args, args
    assert "--no-force" not in args, args


def test_no_force_with_all_still_runs_downloader(sandbox: Path) -> None:
    """`--no-force --all` must still fill in missing files, not skip wholesale.

    The old `[ "$FORCE_ARGS" != "" ]` guard doubled as the "force requested"
    test, so `--no-force --all` made the whole condition false and the
    downloader never ran — the opposite failure of the dropped flag.
    """
    result = run_setup(sandbox, "--no-force", "--all")
    assert result.returncode == 0, result.stdout

    args = recorded(sandbox, "models_args.txt")
    assert args is not None, (
        f"model downloader skipped entirely under --no-force --all\n{result.stdout}"
    )
    assert "--no-force" in args, args


def test_force_remove_models_overrides_no_force(sandbox: Path) -> None:
    result = run_setup(sandbox, "--no-force", "--force-remove-models", "--models", "YoloV5S")
    assert result.returncode == 0, result.stdout

    args = recorded(sandbox, "models_args.txt")
    assert args is not None, result.stdout
    assert "--force" in args, args
    assert "--no-force" not in args, args


def test_force_remove_models_does_not_leak_into_videos(sandbox: Path) -> None:
    """`--force-remove-models` must not turn force back on for videos."""
    result = run_setup(sandbox, "--no-force", "--force-remove-models", "--models", "YoloV5S")
    assert result.returncode == 0, result.stdout

    args = recorded(sandbox, "videos_args.txt")
    assert args is not None, result.stdout
    assert "--no-force" in args, args
    assert "--force" not in args, args


def test_force_remove_videos_does_not_leak_into_models(sandbox: Path) -> None:
    """`--force-remove-videos` must not turn force back on for models."""
    result = run_setup(sandbox, "--no-force", "--force-remove-videos", "--models", "YoloV5S")
    assert result.returncode == 0, result.stdout

    models = recorded(sandbox, "models_args.txt")
    assert models is not None, result.stdout
    assert "--no-force" in models, models
    assert "--force" not in models, models

    videos = recorded(sandbox, "videos_args.txt")
    assert videos is not None, result.stdout
    assert "--force" in videos, videos
