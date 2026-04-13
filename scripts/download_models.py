#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
from __future__ import annotations

"""
DEEPX ModelZoo Auto Downloader

Downloads Q-Lite DXNN models from the DEEPX S3 storage using a pre-generated
manifest file (modelzoo_manifest.json).  The manifest is bundled with each
runtime release and contains the exact model URLs for the compatible version.

Usage:
    python3 scripts/download_models.py
    python3 scripts/download_models.py --all
    python3 scripts/download_models.py --output assets/models --all
    python3 scripts/download_models.py --dry-run
    python3 scripts/download_models.py --list
    python3 scripts/download_models.py --category "Object Detection" --all
    python3 scripts/download_models.py --models YoloV8N ResNet50

Requirements:
    pip install requests
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_DIR            = Path(__file__).parent
DEFAULT_OUTPUT        = SCRIPT_DIR.parent / "assets" / "models"
DEFAULT_MANIFEST      = SCRIPT_DIR / "modelzoo_manifest.json"
DEFAULT_INTERNAL_PATH = Path("/mnt/regression_storage/atd/models_v3.1.0")

# ANSI colors
_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _C = "\033[96m"; _RST = "\033[0m"

def info(msg):  print(f"{_G}[INFO]{_RST}  {msg}", flush=True)
def warn(msg):  print(f"{_Y}[WARN]{_RST}  {msg}", flush=True)
def error(msg): print(f"{_R}[ERR ]{_RST}  {msg}", file=sys.stderr, flush=True)
def head(msg):  print(f"{_C}{msg}{_RST}", flush=True)


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load the bundled manifest file as the primary model source."""
    if not manifest_path.is_file():
        error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        error(f"Failed to load manifest: {manifest_path}")
        error(str(exc))
        sys.exit(1)

    if not isinstance(data, list):
        error(f"Invalid manifest format (expected JSON array): {manifest_path}")
        sys.exit(1)

    info(f"Loaded manifest: {manifest_path.name} ({len(data)} models)")
    return data



# ── Interactive Selection ─────────────────────────────────────────────────────────

def _parse_selection(raw: str, max_idx: int) -> set[int]:
    """Parse user input like '1 3 5', '1-3', '2,4-6' into a set of 1-based indices."""
    selected: set[int] = set()
    for token in raw.replace(",", " ").split():
        if "-" in token:
            parts = token.split("-", 1)
            try:
                lo, hi = int(parts[0]), int(parts[1])
                selected.update(range(lo, hi + 1))
            except ValueError:
                pass
        else:
            try:
                selected.add(int(token))
            except ValueError:
                pass
    return {i for i in selected if 1 <= i <= max_idx}


def _print_category_table(cat_names: list, cats: dict, output_dir: Path):
    """Print the category selection table with new/existing counts."""
    head(f"\n{'═'*64}")
    head("  Step 1 / 2  —  Select Categories")
    head(f"{'─'*64}")
    print(f"  {'#':>3}  {'Category':<30}  {'Total':>5}  {'New':>5}  {'Exists':>6}")
    print(f"  {'─'*3}  {'─'*30}  {'─'*5}  {'─'*5}  {'─'*6}")
    total_new = total_exists = 0
    for i, cat in enumerate(cat_names, 1):
        mlist = cats[cat]
        new = sum(1 for m in mlist if not (output_dir / Path(urlparse(m["dxnn_url"]).path).name).exists())
        exists = len(mlist) - new
        total_new += new
        total_exists += exists
        new_s = f"{_G}{new:5}{_RST}" if new else f"{'0':>5}"
        exists_s = f"{_Y}{exists:6}{_RST}" if exists else f"{'0':>6}"
        print(f"  {i:>3}  {cat:<30}  {len(mlist):>5}  {new_s}  {exists_s}")
    total = sum(len(v) for v in cats.values())
    print(f"  {'─'*3}  {'─'*30}  {'─'*5}  {'─'*5}  {'─'*6}")
    print(f"  {'':>3}  {'Total':<30}  {total:>5}  {_G}{total_new:>5}{_RST}  {_Y}{total_exists:>6}{_RST}")
    head(f"{'═'*64}")


def _prompt_category_selection(cat_names: list) -> set:
    """Prompt user to select categories, return selected set."""
    print(f"  Enter numbers (e.g. {_C}1 3 5{_RST} or {_C}1-3{_RST}), or press Enter for {_C}all{_RST}")
    while True:
        try:
            raw = input("  Categories > ").strip()
        except EOFError:
            warn("Non-interactive environment detected. Selecting all categories.")
            return set(cat_names)
        if not raw or raw.lower() == "all":
            return set(cat_names)
        indices = _parse_selection(raw, len(cat_names))
        if indices:
            return {cat_names[i - 1] for i in indices}
        print(f"  {_Y}Invalid input. Try again.{_RST}")


def _print_model_table(models: list[dict], output_dir: Path):
    """Print the model selection table with status."""
    head(f"\n{'═'*64}")
    head("  Step 2 / 2  —  Select Models")
    head(f"{'─'*64}")
    print(f"  {'#':>4}  {'Model':<35}  {'Category':<25}  Status")
    print(f"  {'─'*4}  {'─'*35}  {'─'*25}  {'─'*10}")
    for i, m in enumerate(models, 1):
        fname = Path(urlparse(m["dxnn_url"]).path).name
        exists = (output_dir / fname).exists()
        status = f"{_Y}exists{_RST}" if exists else f"{_G}new{_RST}"
        print(f"  {i:>4}  {m['name']:<35}  {m['category']:<25}  {status}")
    print(f"  {'─'*4}  {'─'*35}  {'─'*25}  {'─'*10}")
    head(f"{'═'*64}")


def _prompt_model_selection(models: list[dict]) -> list[dict]:
    """Prompt user to select models, return filtered list."""
    print(f"  Enter numbers (e.g. {_C}1 3 5{_RST} or {_C}1-3{_RST}), or press Enter for {_C}all{_RST}")
    while True:
        try:
            raw = input("  Models   > ").strip()
        except EOFError:
            warn("Non-interactive environment detected. Selecting all models.")
            return models
        if not raw or raw.lower() == "all":
            return models
        indices = _parse_selection(raw, len(models))
        if indices:
            return [models[i - 1] for i in sorted(indices)]
        print(f"  {_Y}Invalid input. Try again.{_RST}")


def interactive_select(models: list[dict], output_dir: Path) -> list[dict]:
    """
    Interactively prompt the user to select categories and then individual models.
    Returns the filtered model list.
    """
    cats: dict[str, list] = {}
    for m in models:
        cats.setdefault(m["category"], []).append(m)
    cat_names = list(cats.keys())

    _print_category_table(cat_names, cats, output_dir)
    selected_cats = _prompt_category_selection(cat_names)
    models = [m for m in models if m["category"] in selected_cats]
    info(f"Selected {len(selected_cats)} categor{'y' if len(selected_cats)==1 else 'ies'}: {len(models)} model(s)")

    _print_model_table(models, output_dir)
    models = _prompt_model_selection(models)

    info(f"Final selection: {len(models)} model(s)")
    print()
    return models


def _sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def download_file(url: str, dest: Path, session, force: bool = False) -> dict:
    """Download a single file. Skips if already exists, unless force=True."""
    filename = dest.name
    if dest.exists() and not force:
        return {"status": "skip", "file": filename, "size": dest.stat().st_size}

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return {"status": "error", "file": filename, "code": r.status_code, "url": url}
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    f.write(chunk)
                    downloaded += len(chunk)
        return {"status": "ok", "file": filename, "size": downloaded}
    except Exception as e:
        return {"status": "error", "file": filename, "error": str(e), "url": url}


def _handle_download_result(res: dict, name: str, kind: str, bar: str, pct: int,
                            counters: dict):
    """Process a single download result: update counters and print status."""
    status = res["status"]
    if status == "ok":
        counters["ok"] += 1
        print(f"  [{bar}] {pct:3d}%  {_G}↓{_RST} {name} ({kind}) — {_sizeof_fmt(res['size'])}")
        return
    if status == "skip":
        counters["skip"] += 1
        print(f"  [{bar}] {pct:3d}%  {_Y}–{_RST} {name} ({kind}) — already exists (skip)")
        return
    # error
    counters["err"] += 1
    detail = res.get("error") or f"HTTP {res.get('code', '?')}"
    print(f"  [{bar}] {pct:3d}%  {_R}✗{_RST} {name} ({kind}) — {detail}")
    if res.get("code") == 403:
        counters["err_403"] += 1
        if counters["first_err_url"] is None:
            counters["first_err_url"] = res.get("url", "")


def download_all(models: list[dict], output_dir: Path, session,
                 workers: int = 4, force: bool = False, with_json: bool = True):
    """Download all models in parallel."""
    if not output_dir.is_dir():
        if output_dir.is_symlink():
            output_dir.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build download task list
    tasks = []
    for m in models:
        fname = Path(urlparse(m["dxnn_url"]).path).name
        tasks.append(("dxnn", m["name"], m["dxnn_url"], output_dir / fname))
        if with_json and m.get("json_url"):
            jname = Path(urlparse(m["json_url"]).path).name
            # Save json alongside dxnn in the same output directory
            tasks.append(("json", m["name"], m["json_url"], output_dir / jname))

    total = len(tasks)
    head(f"\n{'─'*60}")
    head(f"  Starting download: {total} files → {output_dir}")
    head(f"{'─'*60}")

    counters = {"ok": 0, "skip": 0, "err": 0, "err_403": 0, "first_err_url": None}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, url, dest, session, force): (kind, name)
            for kind, name, url, dest in tasks
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            kind, name = futures[fut]
            res = fut.result()
            pct = done * 100 // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            _handle_download_result(res, name, kind, bar, pct, counters)

    elapsed = time.time() - t0
    head(f"\n{'─'*60}")
    head(f"  Done: {counters['ok']} downloaded, {counters['skip']} skipped, {counters['err']} errors  ({elapsed:.1f}s)")
    head(f"  Saved to: {output_dir.resolve()}")
    if counters["err_403"]:
        warn(f"{counters['err_403']} file(s) returned HTTP 403 (Forbidden).")
        warn("The models are listed on the page but may not be published yet.")
        if counters["first_err_url"]:
            warn(f"Example URL: {counters['first_err_url']}")
    head(f"{'─'*60}\n")


# ── Internal (local copy) ─────────────────────────────────────────────────────

def copy_file(src: Path, dest: Path, force: bool = False) -> dict:
    """Copy a single file from local path. Skips if already exists, unless force=True."""
    filename = dest.name
    if dest.exists() and not force:
        return {"status": "skip", "file": filename, "size": dest.stat().st_size}
    if not src.exists():
        return {"status": "error", "file": filename, "error": f"source not found: {src}"}

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dest)
        return {"status": "ok", "file": filename, "size": dest.stat().st_size}
    except Exception as e:
        return {"status": "error", "file": filename, "error": str(e)}


def copy_all(models: list[dict], output_dir: Path, internal_path: Path,
             workers: int = 4, force: bool = False, with_json: bool = True):
    """Copy all models in parallel from a local directory."""
    if not internal_path.is_dir():
        error(f"Internal path not found or not a directory: {internal_path}")
        sys.exit(1)

    if not output_dir.is_dir():
        if output_dir.is_symlink():
            output_dir.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for m in models:
        fname = Path(urlparse(m["dxnn_url"]).path).name
        tasks.append(("dxnn", m["name"], internal_path / fname, output_dir / fname))
        if with_json and m.get("json_url"):
            jname = Path(urlparse(m["json_url"]).path).name
            tasks.append(("json", m["name"], internal_path / jname, output_dir / jname))

    total = len(tasks)
    head(f"\n{'─'*60}")
    head(f"  Starting copy: {total} files")
    head(f"  From: {internal_path}")
    head(f"  To  : {output_dir}")
    head(f"{'─'*60}")

    counters = {"ok": 0, "skip": 0, "err": 0, "err_403": 0, "first_err_url": None}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(copy_file, src, dest, force): (kind, name)
            for kind, name, src, dest in tasks
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            kind, name = futures[fut]
            res = fut.result()
            pct = done * 100 // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            _handle_download_result(res, name, kind, bar, pct, counters)

    elapsed = time.time() - t0
    head(f"\n{'─'*60}")
    head(f"  Done: {counters['ok']} copied, {counters['skip']} skipped, {counters['err']} errors  ({elapsed:.1f}s)")
    head(f"  Saved to: {output_dir.resolve()}")
    head(f"{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DEEPX ModelZoo Auto Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # interactive (select categories then models)
  python3 scripts/download_models.py

  # download all models without interaction
  python3 scripts/download_models.py --all

  # specify output directory
  python3 scripts/download_models.py --output /data/models --all

  # list available models without downloading
  python3 scripts/download_models.py --list

  # dry-run: show what would be downloaded
  python3 scripts/download_models.py --dry-run

  # download only a specific category
  python3 scripts/download_models.py --category "Object Detection" --all

  # download only specific models by name
  python3 scripts/download_models.py --models YoloV8N ResNet50

  # use a custom manifest
  python3 scripts/download_models.py --manifest /path/to/custom_manifest.json --all
        """,
    )
    dl = parser.add_argument_group("download")
    dl.add_argument("--output",   type=str, default=str(DEFAULT_OUTPUT),
                    help=f"output directory (default: {DEFAULT_OUTPUT})")
    dl.add_argument("--workers",  type=int, default=4,  help="parallel download threads (default: 4)")
    dl.add_argument("--force",    action="store_true",  help="overwrite existing files")
    dl.add_argument("--no-json",  action="store_true",  help="skip JSON file downloads")
    dl.add_argument("--all",      action="store_true",
                    help="download all parsed models non-interactively")
    dl.add_argument("--category", type=str, default=None,
                    help="download only a specific category (e.g. 'Object Detection')")
    dl.add_argument("--models",   type=str, default=None, nargs="+", metavar="MODEL",
                    help="whitelist: download only the specified model name(s) "
                         "(e.g. --models YoloV8N ResNet50). Case-insensitive.")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--list",    action="store_true", help="list available models without downloading")
    misc.add_argument("--dry-run", action="store_true", help="list models without downloading")
    misc.add_argument("--save-manifest", type=str, default=None,
                      metavar="FILE",    help="save parsed model list to a JSON file")
    misc.add_argument("--manifest", type=str, default=None, metavar="FILE",
                      help=f"path to manifest JSON file (default: {DEFAULT_MANIFEST})")

    src = parser.add_argument_group("source")
    src.add_argument("--internal", action="store_true",
                     help="use local mount instead of S3 (internal/air-gapped network)")
    src.add_argument("--internal-path", type=str, default=str(DEFAULT_INTERNAL_PATH),
                     metavar="DIR",
                     help=f"local model directory for --internal mode (default: {DEFAULT_INTERNAL_PATH})")
    return parser.parse_args()


def _setup_session():
    """Create a requests Session for downloading."""
    try:
        import requests as _requests
    except ImportError as exc:
        print("[ERR ] Missing dependency: requests", file=sys.stderr, flush=True)
        print("[ERR ] Install it with: python3 -m pip install requests", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    session = _requests.Session()
    session.headers.update({"User-Agent": "DEEPX-ModelZoo-Downloader/1.0"})
    return session


def _apply_filters(models: list[dict], args) -> list[dict]:
    """Apply category and model whitelist filters from CLI args."""
    if args.category:
        models = [m for m in models if args.category.lower() in m["category"].lower()]
        info(f"Category filter '{args.category}': {len(models)} model(s)")
    if args.models:
        whitelist = {name.lower() for name in args.models}
        models = [m for m in models if m["name"].lower() in whitelist]
        matched = {m["name"].lower() for m in models}
        missing = [name for name in args.models if name.lower() not in matched]
        if missing:
            warn(f"Model(s) not found in page: {', '.join(missing)}")
        info(f"Model whitelist: {len(models)} model(s) selected")
    return models


def _print_filtered_model_list(models: list[dict], output_dir: Path):
    """Print a grouped model list with new/exists status."""
    head(f"\n{'─'*60}")
    head(f"  Models found: {len(models)}")
    head(f"{'─'*60}")
    cats: dict[str, list] = {}
    for m in models:
        cats.setdefault(m["category"], []).append(m)
    new_count = skip_count = 0
    for cat, mlist in cats.items():
        print(f"  {_C}{cat}{_RST} ({len(mlist)})")
        for m in mlist:
            fname = Path(urlparse(m["dxnn_url"]).path).name
            if (output_dir / fname).exists():
                skip_count += 1
                print(f"    {_Y}–{_RST} {m['name']}  {_Y}[already exists]{_RST}")
            else:
                new_count += 1
                print(f"    {_G}+{_RST} {m['name']}")
    head(f"{'─'*60}")
    print(f"  {_G}{new_count} new{_RST}  |  {_Y}{skip_count} already exist{_RST}  (use --force to re-download)")
    head(f"{'─'*60}\n")


def main():
    args = parse_args()

    # Auto-detect internal mode: if the local model path exists and --internal
    # was not explicitly requested, switch automatically (e.g. on CI runners
    # that have the internal mount available).
    if not args.internal and Path(args.internal_path).is_dir():
        info(f"Local model path detected ({args.internal_path}) — switching to internal mode automatically")
        args.internal = True

    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    output_dir = Path(args.output)

    _DOUBLE_LINE = '\u2550' * 60
    _SINGLE_LINE = '\u2500' * 60

    head(f"\n{_DOUBLE_LINE}")
    head("  DEEPX ModelZoo Auto Downloader")
    head(f"{_SINGLE_LINE}")
    print(f"  Manifest : {manifest_path}")
    print(f"  Output   : {output_dir.resolve()}")
    print(f"  Workers  : {args.workers}  |  Force : {args.force}  |  Dry-run : {args.dry_run}")
    if args.internal:
        print(f"  Source   : {_C}internal{_RST} ({args.internal_path})")
    else:
        print(f"  Source   : {_C}S3 / public{_RST}")
    head(f"{_DOUBLE_LINE}\n")

    models = load_manifest(manifest_path)
    models = _apply_filters(models, args)

    if not args.all and not args.category and not args.models and not args.dry_run and not args.list:
        models = interactive_select(models, output_dir)

    if args.save_manifest:
        Path(args.save_manifest).write_text(json.dumps(models, indent=2, ensure_ascii=False))
        info(f"Manifest saved: {args.save_manifest} ({len(models)} models)")

    if args.category or args.models or args.dry_run or args.list:
        _print_filtered_model_list(models, output_dir)

    if args.dry_run or args.list:
        info("--dry-run/--list mode: skipping download.")
        return

    if args.internal:
        copy_all(
            models=models,
            output_dir=output_dir,
            internal_path=Path(args.internal_path),
            workers=args.workers,
            force=args.force,
            with_json=not args.no_json,
        )
    else:
        session = _setup_session()
        download_all(
            models=models,
            output_dir=output_dir,
            session=session,
            workers=args.workers,
            force=args.force,
            with_json=not args.no_json,
        )


if __name__ == "__main__":
    main()
