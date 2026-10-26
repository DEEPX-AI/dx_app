"""setup.sh --models= (E2) and the model-path cleanup on a failed download.

setup.sh runs from a copy in a temporary tree (it removes ./assets/models
when the downloader fails, so it must never run against the repository's
own assets). Host-mode setup.sh creates <tree>/../../workspace/res, which
is inside the temporary directory too.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
# DXAPP_SETUP_SH_UNDER_TEST lets a reviewer run these cases against another
# setup.sh (e.g. the pre-fix one) to see them fail.
SETUP_SH = Path(os.environ.get("DXAPP_SETUP_SH_UNDER_TEST", ROOT / "setup.sh"))

RECORDER = """#!/bin/bash
printf '%s\\0' "$@" > "$(dirname "$0")/{name}.argv"
{extra}
exit ${{STUB_RC:-0}}
"""

GUI_ENV_STUB = "dxapp_link_cv2_qt_fonts() { :; }\ndxapp_prepare_gui_env() { :; }\n"


def _tree(tmp_path: Path, real_downloader: bool = False) -> Path:
    """<tmp>/suite/runtime/dx_app with setup.sh and what it needs."""
    app = tmp_path / "suite" / "runtime" / "dx_app"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(SETUP_SH, app / "setup.sh")
    for f in ("color_env.sh", "common_util.sh"):
        shutil.copy2(ROOT / "scripts" / f, app / "scripts" / f)
    # The real helper would link fonts into the caller's venv.
    (app / "scripts" / "gui_env.sh").write_text(GUI_ENV_STUB)
    if real_downloader:
        shutil.copy2(ROOT / "setup_sample_models.sh", app / "setup_sample_models.sh")
        for f in ("download_models.py", "modelzoo_manifest.json"):
            shutil.copy2(ROOT / "scripts" / f, app / "scripts" / f)
        (app / "config").mkdir()
        shutil.copy2(ROOT / "config" / "model_registry.json", app / "config" / "model_registry.json")
    else:
        # A failing download may already have created ./assets/models.
        extra = 'mkdir -p ./assets/models && touch ./assets/models/partial.dxnn'
        (app / "setup_sample_models.sh").write_text(RECORDER.format(name="models", extra=extra))
    (app / "setup_sample_videos.sh").write_text(RECORDER.format(name="videos", extra=""))
    for f in app.glob("*.sh"):
        f.chmod(0o755)
    return app


def _run(app: Path, tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(app / "setup.sh"), f"--docker_volume_path={tmp_path / 'volume'}", *args],
        cwd=app, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, **env}, check=False)


def _models_forwarded(app: Path) -> list[str]:
    argv = (app / "models.argv").read_text().split("\0")[:-1]
    joined = " ".join(argv).split()
    i = joined.index("--models")
    names = []
    for token in joined[i + 1:]:
        if token.startswith("--"):
            break
        names.append(token)
    return names


@pytest.mark.parametrize("args, expected", [
    (["--models=yolov8n"], ["yolov8n"]),
    (["--models=yolov8n,resnet50"], ["yolov8n", "resnet50"]),
    (["--models=yolov8n, resnet50"], ["yolov8n", "resnet50"]),
    (["--models=yolov8n", "resnet50"], ["yolov8n", "resnet50"]),
    (["--models", "yolov8n", "resnet50"], ["yolov8n", "resnet50"]),
    (["--models", "yolov8n,resnet50", "--models=alexnet"], ["yolov8n", "resnet50", "alexnet"]),
])
def test_models_option_forms_are_forwarded(tmp_path, args, expected):
    app = _tree(tmp_path)
    result = _run(app, tmp_path, *args, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _models_forwarded(app) == expected


def test_models_option_without_names_is_rejected(tmp_path):
    app = _tree(tmp_path)
    for args in (["--models="], ["--models", "--dry-run"]):
        result = _run(app, tmp_path, *args)
        assert result.returncode != 0
        assert "--models requires at least one model name" in result.stdout + result.stderr


def test_help_documents_the_equals_form():
    text = SETUP_SH.read_text(encoding="utf-8")
    assert "[--models=<m1>[,m2...]]" in text


def test_models_equals_form_end_to_end_dry_run(tmp_path):
    app = _tree(tmp_path, real_downloader=True)
    result = _run(app, tmp_path, "--models=yolov8n,yolo26n_obb", "--dry-run", "--no-json")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Model whitelist: 2 model(s) selected" in combined
    assert "not found" not in combined


def test_failed_download_keeps_an_existing_model_dir(tmp_path):
    app = _tree(tmp_path)
    models = app / "assets" / "models"
    models.mkdir(parents=True)
    (models / "earlier.dxnn").write_bytes(b"x")
    result = _run(app, tmp_path, "--models", "a", "--no-force", STUB_RC="1")
    assert result.returncode != 0
    assert (models / "earlier.dxnn").exists()


def test_failed_download_keeps_an_existing_model_symlink(tmp_path):
    app = _tree(tmp_path)
    target = tmp_path / "shared_models"
    target.mkdir()
    (target / "earlier.dxnn").write_bytes(b"x")
    (app / "assets").mkdir()
    (app / "assets" / "models").symlink_to(target)
    result = _run(app, tmp_path, "--models", "a", STUB_RC="1")
    assert result.returncode != 0
    assert (app / "assets" / "models").is_symlink()
    assert (target / "earlier.dxnn").exists()


def test_failed_download_removes_a_model_dir_it_created(tmp_path):
    app = _tree(tmp_path)
    result = _run(app, tmp_path, "--models", "a", STUB_RC="1")
    assert result.returncode != 0
    assert "Setup models script failed." in result.stdout + result.stderr
    assert not (app / "assets" / "models").exists()


# setup_assets.py (the cross-platform twin) follows the same rule.

def _load_setup_assets():
    spec = importlib.util.spec_from_file_location("setup_assets_under_test",
                                                  ROOT / "scripts" / "setup_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pre_existing", [True, False])
def test_setup_assets_failed_download_cleanup(tmp_path, monkeypatch, pre_existing):
    sa = _load_setup_assets()
    models = tmp_path / "assets" / "models"
    if pre_existing:
        models.mkdir(parents=True)
        (models / "earlier.dxnn").write_bytes(b"x")

    def failing_setup(args):
        models.mkdir(parents=True, exist_ok=True)
        (models / "partial.dxnn").write_bytes(b"")
        return 1

    monkeypatch.setattr(sa, "MODEL_OUTPUT", models)
    monkeypatch.setattr(sa, "setup_models", failing_setup)
    monkeypatch.setattr(sys, "argv", ["setup_assets.py", "--models-only", "--models", "a"])
    with pytest.raises(SystemExit) as exc:
        sa.main()
    assert exc.value.code == 1
    assert (models / "earlier.dxnn").exists() is pre_existing
    assert models.exists() is pre_existing


def test_setup_assets_models_option_repeats_extend(monkeypatch):
    sa = _load_setup_assets()
    monkeypatch.setattr(sys, "argv", ["setup_assets.py", "--models", "a", "b", "--models", "c"])
    assert sa.parse_args().models == ["a", "b", "c"]
