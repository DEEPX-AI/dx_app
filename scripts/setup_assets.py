#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
DEEPX Setup - Download sample models and videos (Windows-compatible).

This script is the cross-platform equivalent of setup.sh + setup_sample_models.sh
+ setup_sample_videos.sh.  It downloads models via download_models.py (subprocess)
and videos via Python requests + tarfile -- no curl, tar, or symlinks required.

Usage (directly):
    python scripts/setup_assets.py --all
    python scripts/setup_assets.py --list
    python scripts/setup_assets.py --models-only --all
    python scripts/setup_assets.py --videos-only

Usage (via batch wrapper on Windows):
    setup.bat --all
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent          # scripts/
PROJECT_DIR = SCRIPT_DIR.parent                        # dx_app/

MODEL_OUTPUT = PROJECT_DIR / "assets" / "models"
VIDEO_OUTPUT = PROJECT_DIR / "assets" / "videos"

VIDEO_BASE_URL = "https://sdk.deepx.ai/"
MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# ── ANSI colors (matching download_models.py style) ──────────────────────────

_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _C = "\033[96m"; _RST = "\033[0m"

def info(msg):  print(f"{_G}[INFO]{_RST}  {msg}", flush=True)
def warn(msg):  print(f"{_Y}[WARN]{_RST}  {msg}", flush=True)
def error(msg): print(f"{_R}[ERR ]{_RST}  {msg}", file=sys.stderr, flush=True)
def head(msg):  print(f"{_C}{msg}{_RST}", flush=True)


def _enable_ansi():
    """Enable ANSI escape processing on Windows 10+."""
    if sys.platform == "win32":
        os.system("")
        # Ensure subprocess output uses utf-8 instead of the console code page
        # (avoids UnicodeEncodeError from download_models.py's Unicode chars)
        if not os.environ.get("PYTHONIOENCODING"):
            os.environ["PYTHONIOENCODING"] = "utf-8"


def _read_release_version() -> str:
    """Read the release version from release.ver (e.g. 'v3.1.0')."""
    ver_file = PROJECT_DIR / "release.ver"
    if not ver_file.is_file():
        error(f"release.ver not found: {ver_file}")
        sys.exit(1)
    return ver_file.read_text(encoding="utf-8").strip()


# ── requests bootstrap ────────────────────────────────────────────────────────

def _ensure_requests():
    """Import requests; auto-install if missing."""
    try:
        import requests  # noqa: F401
        return requests
    except ImportError:
        info("Installing required dependency: requests")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "requests"],
            stdout=subprocess.DEVNULL,
        )
        import requests  # noqa: F401
        return requests


# ── Model setup ───────────────────────────────────────────────────────────────

def setup_models(args: argparse.Namespace) -> int:
    """Invoke download_models.py as a subprocess, forwarding relevant flags."""
    downloader = SCRIPT_DIR / "download_models.py"
    if not downloader.is_file():
        error(f"ModelZoo downloader not found: {downloader}")
        return 1

    cmd: list[str] = [sys.executable, str(downloader)]
    cmd.extend(["--output", str(MODEL_OUTPUT)])

    if args.all:            cmd.append("--all")
    if args.force:          cmd.append("--force")
    if args.dry_run:        cmd.append("--dry-run")
    if args.list:           cmd.append("--list")
    if args.no_json:        cmd.append("--no-json")
    if args.workers:        cmd.extend(["--workers", str(args.workers)])
    if args.category:       cmd.extend(["--category", args.category])
    if args.manifest:       cmd.extend(["--manifest", args.manifest])
    if args.models:         cmd.extend(["--models"] + args.models)

    if args.verbose:
        info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode


# ── Video setup ───────────────────────────────────────────────────────────────

def _safe_tar_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Filter out members with absolute paths or '..' to prevent path traversal."""
    safe = []
    for m in tar.getmembers():
        if m.name.startswith("/") or ".." in m.name.split("/"):
            warn(f"Skipping suspicious tar member: {m.name}")
            continue
        safe.append(m)
    return safe


def _flatten_single_subdir(dest_dir: Path):
    """If dest_dir has a single subdirectory with media files, flatten it.

    Ports the logic from setup_sample_videos.sh lines 122-140.
    """
    subdirs = [d for d in dest_dir.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        return

    subdir = subdirs[0]
    media_count = sum(1 for f in subdir.iterdir()
                      if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS)
    if media_count == 0:
        return

    info(f"Flattening {subdir.name}/ -> {dest_dir.name}/")
    for item in list(subdir.iterdir()):
        target = dest_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    # Remove empty subdir
    try:
        subdir.rmdir()
    except OSError:
        pass


def download_and_extract_videos(url: str, dest_dir: Path, force: bool = False):
    """Download a .tar.gz video archive and extract to dest_dir."""
    requests = _ensure_requests()

    if dest_dir.exists() and not force:
        info(f"Video directory already exists: {dest_dir} -- skipping")
        return

    if dest_dir.exists() and force:
        info(f"Force removing existing video directory: {dest_dir}")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download to a temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz", dir=dest_dir.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    try:
        info(f"Downloading videos from {url}")
        session = requests.Session()
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        filled = pct // 5
                        bar = "#" * filled + "-" * (20 - filled)
                        print(f"\r  [{bar}] {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)", end="", flush=True)
            if total > 0:
                print()  # newline after progress bar

        info("Extracting video archive...")
        with tarfile.open(str(tmp_path), "r:gz") as tar:
            members = _safe_tar_members(tar)

            # Detect single top-level directory (for --strip-components=1 equivalent)
            top_dirs = set()
            for m in members:
                parts = m.name.split("/")
                if len(parts) > 1:
                    top_dirs.add(parts[0])

            if len(top_dirs) == 1:
                prefix = top_dirs.pop() + "/"
                stripped = []
                for m in members:
                    if m.name == prefix.rstrip("/"):
                        continue  # skip the top-level dir entry itself
                    if m.name.startswith(prefix):
                        m.name = m.name[len(prefix):]
                    if m.name:
                        stripped.append(m)
                tar.extractall(dest_dir, members=stripped)
            else:
                tar.extractall(dest_dir, members=members)

        # Flatten if single media subdirectory
        _flatten_single_subdir(dest_dir)

        info(f"[OK] Videos extracted to: {dest_dir}")

    except Exception as exc:
        error(f"Video download/extraction failed: {exc}")
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        raise SystemExit(1)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def setup_videos(args: argparse.Namespace) -> int:
    """Download and extract sample videos."""
    if args.dry_run or args.list:
        version = _read_release_version()
        url = f"{VIDEO_BASE_URL}res/video/sample_videos_{version}.tar.gz"
        info(f"[dry-run] Would download videos from: {url}")
        info(f"[dry-run] Would extract to: {VIDEO_OUTPUT}")
        return 0

    force = args.force or args.force_remove_videos
    version = _read_release_version()
    url = f"{VIDEO_BASE_URL}res/video/sample_videos_{version}.tar.gz"

    download_and_extract_videos(url, VIDEO_OUTPUT, force=force)
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEEPX Setup - Download sample models and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # interactive model selection + video download
  python scripts/setup_assets.py

  # download everything non-interactively
  python scripts/setup_assets.py --all

  # list available models
  python scripts/setup_assets.py --list

  # download only models
  python scripts/setup_assets.py --models-only --all

  # download only videos
  python scripts/setup_assets.py --videos-only

  # force re-download
  python scripts/setup_assets.py --force --all
        """,
    )

    scope = parser.add_argument_group("scope")
    scope.add_argument("--models-only", action="store_true",
                       help="download models only (skip videos)")
    scope.add_argument("--videos-only", action="store_true",
                       help="download videos only (skip models)")

    dl = parser.add_argument_group("download options (forwarded to download_models.py)")
    dl.add_argument("--all",      action="store_true",
                    help="download all models non-interactively")
    dl.add_argument("--dry-run",  action="store_true",
                    help="show what would be downloaded without downloading")
    dl.add_argument("--list",     action="store_true",
                    help="list available models without downloading")
    dl.add_argument("--force",    action="store_true",
                    help="force overwrite if files already exist")
    dl.add_argument("--force-remove-models", action="store_true",
                    help="force remove and re-download models")
    dl.add_argument("--force-remove-videos", action="store_true",
                    help="force remove and re-download videos")
    dl.add_argument("--manifest", type=str, default=None, metavar="FILE",
                    help="path to manifest JSON file")
    dl.add_argument("--workers",  type=int, default=None,
                    help="parallel download threads (default: 4)")
    dl.add_argument("--no-json",  action="store_true",
                    help="skip JSON file downloads")
    dl.add_argument("--category", type=str, default=None,
                    help="download models of a specific category only")
    dl.add_argument("--models",   type=str, default=None, nargs="+",
                    metavar="MODEL",
                    help="download specific models by name")
    dl.add_argument("--verbose",  action="store_true",
                    help="enable verbose logging")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _enable_ansi()
    args = parse_args()

    head("=" * 60)
    head("  DEEPX Setup - Download sample models and videos")
    head("=" * 60)

    if args.force_remove_models:
        args.force = True

    rc = 0

    # Models
    if not args.videos_only:
        info("--- Setting up models ---")
        rc = setup_models(args)
        if rc != 0:
            error("Model setup failed.")
            if MODEL_OUTPUT.exists() and not args.list and not args.dry_run:
                shutil.rmtree(MODEL_OUTPUT, ignore_errors=True)
            sys.exit(rc)

    # Videos
    if not args.models_only:
        info("--- Setting up videos ---")
        rc = setup_videos(args)
        if rc != 0:
            error("Video setup failed.")
            sys.exit(rc)

    head("\n" + "=" * 60)
    head("  [OK] Sample models and videos setup complete")
    head("=" * 60)


if __name__ == "__main__":
    main()
