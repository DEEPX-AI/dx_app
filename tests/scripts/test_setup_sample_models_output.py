"""setup_sample_models.sh never deletes an existing model directory or a
symlink it did not create.

Runs a copy of the wrapper in a temporary tree with a stub downloader that
records its --output and writes one file there. Temp dirs only.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
# DXAPP_SETUP_SAMPLE_MODELS_UNDER_TEST points the cases at another wrapper
# (e.g. the pre-fix one) to see them fail.
WRAPPER = Path(os.environ.get("DXAPP_SETUP_SAMPLE_MODELS_UNDER_TEST",
                              ROOT / "setup_sample_models.sh"))

STUB_DOWNLOADER = '''import sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--output") + 1])
Path(__file__).with_name("downloader.argv").write_text("\\n".join(args))
if "--dry-run" not in args:
    out.mkdir(parents=True, exist_ok=True)
    (out / "new.dxnn").write_bytes(b"new")
'''


def _tree(tmp_path: Path) -> Path:
    app = tmp_path / "dx_app"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(WRAPPER, app / "setup_sample_models.sh")
    for f in ("color_env.sh", "common_util.sh"):
        shutil.copy2(ROOT / "scripts" / f, app / "scripts" / f)
    (app / "scripts" / "download_models.py").write_text(STUB_DOWNLOADER)
    (app / "assets").mkdir()
    return app


def _run(app: Path, target: Path, *extra: str,
         output: str = "./assets/models") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(app / "setup_sample_models.sh"), f"--output={output}",
         f"--symlink_target_path={target}", "--models", "a", *extra],
        cwd=app, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=120)


def _downloaded_to(app: Path) -> str | None:
    argv_file = app / "scripts" / "downloader.argv"
    if not argv_file.exists():
        return None
    args = argv_file.read_text().split("\n")
    return args[args.index("--output") + 1]


def test_missing_output_becomes_a_symlink_to_the_target(tmp_path):
    app = _tree(tmp_path)
    target = tmp_path / "workspace" / "models"
    result = _run(app, target)
    assert result.returncode == 0, result.stdout + result.stderr
    models = app / "assets" / "models"
    assert models.is_symlink() and Path(os.readlink(models)) == target.resolve()
    assert (target / "new.dxnn").exists()


def test_existing_real_directory_is_used_in_place(tmp_path):
    app = _tree(tmp_path)
    models = app / "assets" / "models"
    models.mkdir()
    (models / "earlier.dxnn").write_bytes(b"old")
    target = tmp_path / "workspace" / "models"
    result = _run(app, target)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert models.is_dir() and not models.is_symlink()
    assert (models / "earlier.dxnn").read_bytes() == b"old"
    assert (models / "new.dxnn").exists()          # downloaded into it
    assert not target.exists()                     # the target was not used
    assert "used in place" in combined


def test_existing_empty_directory_is_replaced_by_the_symlink(tmp_path):
    app = _tree(tmp_path)
    models = app / "assets" / "models"
    models.mkdir()
    target = tmp_path / "workspace" / "models"
    result = _run(app, target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert models.is_symlink() and Path(os.readlink(models)) == target.resolve()


def test_existing_valid_symlink_is_kept(tmp_path):
    app = _tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "earlier.dxnn").write_bytes(b"old")
    models = app / "assets" / "models"
    models.symlink_to(elsewhere)
    target = tmp_path / "workspace" / "models"
    result = _run(app, target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert models.is_symlink() and os.readlink(models) == str(elsewhere)
    assert (elsewhere / "earlier.dxnn").read_bytes() == b"old"
    assert (elsewhere / "new.dxnn").exists()
    assert not target.exists()


def test_existing_dangling_symlink_is_an_error_and_kept(tmp_path):
    app = _tree(tmp_path)
    models = app / "assets" / "models"
    gone = tmp_path / "gone"
    models.symlink_to(gone)
    target = tmp_path / "workspace" / "models"
    result = _run(app, target)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "dangling symlink" in combined and "Fix the link" in combined
    assert models.is_symlink() and os.readlink(models) == str(gone)
    assert _downloaded_to(app) is None             # nothing was downloaded
    assert not target.exists()


def test_dry_run_changes_nothing(tmp_path):
    app = _tree(tmp_path)
    target = tmp_path / "workspace" / "models"
    result = _run(app, target, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (app / "assets" / "models").exists()
    assert not (app / "assets" / "models").is_symlink()


# A trailing slash on --output must not defeat the symlink checks: with it,
# "[ -L assets/models/ ]" follows the link and is false for a dangling link
# and a valid one alike.

@pytest.mark.parametrize("output", ["./assets/models/", "./assets/models//"])
def test_trailing_slash_over_a_dangling_symlink_is_still_an_error(tmp_path, output):
    app = _tree(tmp_path)
    models = app / "assets" / "models"
    gone = tmp_path / "gone"
    models.symlink_to(gone)
    target = tmp_path / "workspace" / "models"
    result = _run(app, target, output=output)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined
    assert "dangling symlink" in combined, combined
    assert models.is_symlink() and os.readlink(models) == str(gone)
    assert _downloaded_to(app) is None
    assert not target.exists()
    assert not gone.exists()


@pytest.mark.parametrize("populated", [True, False])
def test_trailing_slash_over_a_valid_symlink_keeps_it(tmp_path, populated):
    app = _tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if populated:
        (elsewhere / "earlier.dxnn").write_bytes(b"old")
    models = app / "assets" / "models"
    models.symlink_to(elsewhere)
    target = tmp_path / "workspace" / "models"
    result = _run(app, target, output="./assets/models/")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert models.is_symlink() and os.readlink(models) == str(elsewhere)
    assert elsewhere.is_dir() and not elsewhere.is_symlink()
    assert (elsewhere / "new.dxnn").exists()       # downloaded through the link
    if populated:
        assert (elsewhere / "earlier.dxnn").read_bytes() == b"old"
    assert not target.exists()
    assert not (elsewhere / "models").exists()     # no link made inside it


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="root ignores the read-only directory")
def test_a_failed_link_is_an_error(tmp_path):
    app = _tree(tmp_path)
    assets = app / "assets"
    target = tmp_path / "workspace" / "models"
    assets.chmod(0o555)                            # the link cannot be created
    try:
        result = _run(app, target)
    finally:
        assets.chmod(0o755)
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "Could not create the symbolic link" in combined, combined
    assert (target / "new.dxnn").exists()          # the download itself happened


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="root reads every directory")
def test_an_unreadable_output_directory_is_an_error_before_any_download(tmp_path):
    """U-55: `ls -A` on an unreadable directory prints nothing, so it was
    taken for empty: the models went to the symlink target, then rmdir
    failed with "no longer empty". Refuse it up front instead."""
    app = _tree(tmp_path)
    models = app / "assets" / "models"
    models.mkdir()
    (models / "earlier.dxnn").write_bytes(b"old")
    models.chmod(0o300)  # writable and searchable, not readable
    try:
        result = _run(app, tmp_path / "workspace" / "models")
    finally:
        models.chmod(0o755)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot read" in result.stderr, result.stderr
    assert _downloaded_to(app) is None
    assert not models.is_symlink()
    assert (models / "earlier.dxnn").read_bytes() == b"old"
