from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_download_models_demo_models_filters_to_run_demo_models():
    result = run_command(
        sys.executable,
        "scripts/download_models.py",
        "--demo-models",
        "--dry-run",
        "--no-json",
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Run demo model filter: 18 model(s) selected" in combined
    assert "YoloV7" in combined
    assert "ResNet50" in combined
    assert "AlexNet" not in combined


def test_setup_assets_forwards_demo_models_to_downloader():
    result = run_command(
        sys.executable,
        "scripts/setup_assets.py",
        "--models-only",
        "--demo-models",
        "--dry-run",
        "--no-json",
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Run demo model filter: 18 model(s) selected" in combined


def test_setup_sh_accepts_and_forwards_demo_models_option():
    text = (ROOT / "setup.sh").read_text(encoding="utf-8")

    assert "[--demo-models]" in text
    assert "DEMO_MODELS_ARG=\"--demo-models\"" in text
    assert "$DEMO_MODELS_ARG" in text


def test_setup_sh_skips_video_setup_for_list_or_dry_run():
    text = (ROOT / "setup.sh").read_text(encoding="utf-8")

    assert 'if [ -n "$LIST_ARG" ] || [ -n "$DRY_RUN_ARG" ]; then' in text
    assert "Skipping video setup for list/dry-run mode." in text


def test_setup_assets_uses_v310_sample_video_archive():
    result = run_command(
        sys.executable,
        "scripts/setup_assets.py",
        "--videos-only",
        "--dry-run",
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "sample_videos_v3.2.2.tar.gz" in combined
    assert "sample_videos_v3.1.1.tar.gz" not in combined


def test_demo_models_rejects_other_model_filters():
    result = run_command(
        sys.executable,
        "scripts/download_models.py",
        "--demo-models",
        "--models",
        "ResNet50",
        "--dry-run",
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Use only one of --demo-models, --models, --category, or --all." in combined


def test_demo_models_rejects_all_filter():
    result = run_command(
        sys.executable,
        "scripts/download_models.py",
        "--demo-models",
        "--all",
        "--dry-run",
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Use only one of --demo-models, --models, --category, or --all." in combined
