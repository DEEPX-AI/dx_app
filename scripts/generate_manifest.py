#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
ModelZoo Manifest Version Updater

Reads models.ver (which must contain the exact S3 directory name,
e.g. 2_3_0-rc3 or 2_4_0) and updates all dxnn_url / json_url version
path segments in scripts/modelzoo_manifest.json to match.

Usage:
    python3 scripts/generate_manifest.py            # use version from models.ver
    python3 scripts/generate_manifest.py --ver 2_4_0
    python3 scripts/generate_manifest.py --dry-run  # preview without writing

Run this whenever models.ver is updated (i.e. on each release).
The value in models.ver must be the exact S3 directory name under
https://sdk.deepx.ai/modelzoo/dxnn/<version>/
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
MANIFEST_PATH = SCRIPT_DIR / "modelzoo_manifest.json"
MODELS_VER    = SCRIPT_DIR.parent / "models.ver"

# Matches the version segment in URLs like:
#   https://sdk.deepx.ai/modelzoo/dxnn/2_3_0-rc3/YoloV8N.dxnn
_VER_RE = re.compile(r"(https://sdk\.deepx\.ai/modelzoo/dxnn/)[^/]+/")


def read_version(ver_path: Path, override: str | None) -> str:
    if override:
        return override.strip()
    if not ver_path.is_file():
        print(f"[ERR] models.ver not found: {ver_path}", file=sys.stderr)
        sys.exit(1)
    ver = ver_path.read_text(encoding="utf-8").strip()
    if not ver:
        print("[ERR] models.ver is empty", file=sys.stderr)
        sys.exit(1)
    return ver


def update_manifest(ver: str, manifest_path: Path, dry_run: bool = False) -> list[dict]:
    if not manifest_path.is_file():
        print(f"[ERR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERR] Failed to read manifest: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(entries, list):
        print("[ERR] Manifest must be a JSON array", file=sys.stderr)
        sys.exit(1)

    # Detect current version from first URL found
    old_ver = None
    for e in entries:
        for field in ("dxnn_url", "json_url"):
            url = e.get(field)
            if url:
                m = _VER_RE.search(url)
                if m:
                    old_ver = m.group(0).rstrip("/").split("/")[-1]
                    break
        if old_ver:
            break

    if old_ver == ver:
        print(f"[INFO] Already at version {ver} — no changes needed.")
        return entries

    updated = 0
    for e in entries:
        for field in ("dxnn_url", "json_url"):
            if e.get(field):
                new_url = _VER_RE.sub(lambda m: m.group(1) + ver + "/", e[field])
                if new_url != e[field]:
                    e[field] = new_url
                    updated += 1

    # Summary
    print(f"Version : {old_ver or '(unknown)'} → {ver}")
    print(f"Models  : {len(entries)} entries,  {updated} URL(s) updated")
    print()
    cats: dict[str, int] = {}
    for e in entries:
        cat = e.get("category", "Unknown")
        cats[cat] = cats.get(cat, 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {n:3d}  {cat}")
    print()

    if dry_run:
        print("[dry-run] manifest not written.")
    else:
        manifest_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Written → {manifest_path}")

    return entries


def parse_args():
    p = argparse.ArgumentParser(
        description="Update modelzoo_manifest.json URLs to the version in models.ver",
        epilog="""
examples:
  # update to version written in models.ver
  python3 scripts/generate_manifest.py

  # override version explicitly (must be the exact S3 directory name)
  python3 scripts/generate_manifest.py --ver 2_4_0
  python3 scripts/generate_manifest.py --ver 2_4_0-rc1

  # preview without writing
  python3 scripts/generate_manifest.py --dry-run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ver", type=str, default=None,
        help="Override version (default: read from models.ver). "
             "Must be the exact S3 directory name, e.g. 2_4_0 or 2_4_0-rc1",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing to disk",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ver  = read_version(MODELS_VER, args.ver)
    update_manifest(ver, MANIFEST_PATH, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
