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
import os
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_DIR            = Path(__file__).parent
DEFAULT_OUTPUT        = SCRIPT_DIR.parent / "assets" / "models"
DEFAULT_MANIFEST      = SCRIPT_DIR / "modelzoo_manifest.json"
DEFAULT_REGISTRY      = SCRIPT_DIR.parent / "config" / "model_registry.json"
# OS trust stores, tried after the environment variables (Debian/Ubuntu, then
# RHEL/Fedora). Behind a TLS-inspecting proxy the corporate root is usually
# installed here but missing from the certifi bundle requests ships with.
SYSTEM_CA_BUNDLES     = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
)
CA_BUNDLE_ENV_VARS    = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")
# U-76. Python 3.13+ - and urllib3 2.3+ running on it - verify TLS with
# VERIFY_X509_STRICT, which rejects a certificate missing an extension RFC
# 5280 requires. A TLS-inspecting proxy's CA often lacks an Authority Key
# Identifier. $DXAPP_TLS_RELAX_X509_STRICT=1 clears that one flag; the
# chain, the host name and the certificate requirement stay as they are.
RELAX_X509_STRICT_ENV = "DXAPP_TLS_RELAX_X509_STRICT"
# OpenSSL's messages (crypto/x509/x509_txt.c) for checks that only strict
# mode makes, and that a proxy CA typically fails. Not exhaustive.
X509_STRICT_ONLY_ERRORS = (
    "Missing Authority Key Identifier",
    "Missing Subject Key Identifier",
    "Basic Constraints of CA cert not marked critical",
    "CA cert does not include key usage extension",
)

# ANSI colors
_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _C = "\033[96m"; _RST = "\033[0m"

def info(msg):  print(f"{_G}[DXAPP] [INFO]{_RST}  {msg}", flush=True)
def warn(msg):  print(f"{_Y}[DXAPP] [WARN]{_RST}  {msg}", flush=True)
def error(msg): print(f"{_R}[DXAPP] [ERROR]{_RST} {msg}", file=sys.stderr, flush=True)
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


def _dxnn_filename(model: dict) -> str:
    """The .dxnn file name a manifest entry downloads to."""
    return Path(urlparse(model["dxnn_url"]).path).name


def load_registry_aliases(registry_path: Path = DEFAULT_REGISTRY) -> dict[str, str]:
    """Map each registry model_name (lower case) to its dxnn_file (lower case).

    Best effort: the registry only adds aliases, so a missing or malformed
    file (or entry) just means fewer names match.
    """
    try:
        data = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    aliases: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name, dxnn = entry.get("model_name"), entry.get("dxnn_file")
        if isinstance(name, str) and isinstance(dxnn, str) and name and dxnn:
            aliases[name.lower()] = dxnn.lower()
    return aliases


def select_models(models: list[dict], requested: list[str],
                  aliases: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Select the manifest entries named by `requested` (case-insensitive).

    A name matches, in this order: the manifest name; the .dxnn file name,
    with or without the extension; a registry model_name, through its
    dxnn_file. Tokens may be comma-separated. Returns the selection in
    manifest order, each entry once, and the names that matched nothing.
    """
    by_key: dict[str, dict] = {}
    for m in models:
        by_key.setdefault(m["name"].lower(), m)
    for m in models:
        fname = _dxnn_filename(m).lower()
        by_key.setdefault(fname, m)
        if fname.endswith(".dxnn"):
            by_key.setdefault(fname[:-len(".dxnn")], m)
    by_file = {_dxnn_filename(m).lower(): m for m in models}
    for alias, dxnn in aliases.items():
        if dxnn in by_file:
            by_key.setdefault(alias, by_file[dxnn])

    chosen: set[int] = set()
    missing: list[str] = []
    for raw in requested:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            m = by_key.get(token.lower())
            if m is None:
                missing.append(token)
            else:
                chosen.add(id(m))
    return [m for m in models if id(m) in chosen], missing


def get_run_demo_model_filenames() -> set[str]:
    """Return the .dxnn filenames required by run_demo.py."""
    try:
        from run_demo import DEMOS, D_MODEL
    except Exception as exc:
        error("Failed to load run_demo model registry.")
        error(str(exc))
        raise SystemExit(1) from exc

    return {demo[D_MODEL] for demo in DEMOS}


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
        new = sum(1 for m in mlist if not (output_dir / _dxnn_filename(m)).exists())
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
        fname = _dxnn_filename(m)
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


# A model DX Model Zoo has declared but not published yet returns 403, and that has to
# read as "expected, no action" rather than as an error -- while a 403 on a published
# model stays loud.
#
# Unpublished is DECLARED: the manifest row carries `pending: true`. Do not infer it
# from the URL version. vit-l-p16_512x512_swag was the last pending row; it is now
# published at q-lite-dxnn/2_5_0, so the pending set is empty.


def is_pending(row: dict) -> bool:
    return bool(row.get("pending")) if isinstance(row, dict) else False


def _expected_length(response) -> int | None:
    """Content-Length of an unencoded response, else None (nothing to check)."""
    headers = getattr(response, "headers", None) or {}
    encoding = (headers.get("Content-Encoding") or "identity").strip().lower()
    if encoding != "identity":
        return None  # iter_content() yields decoded bytes: the header counts encoded ones
    try:
        return int(headers.get("Content-Length"))
    except (TypeError, ValueError):
        return None


def _verify_download(size: int, expected: int | None) -> str | None:
    """Why a finished download is not usable, or None when it is."""
    if size == 0:
        return "empty file"
    if expected is not None and size != expected:
        return f"incomplete download: expected {expected} bytes, got {size}"
    return None


def _remove_quietly(path: Path):
    try:
        path.unlink()
    except OSError:
        pass


def download_file(url: str, dest: Path, session, force: bool = False,
                  pending: bool = False) -> dict:
    """Download a single file. Skips if already exists, unless force=True.

    Downloads to `<dest>.part` and moves it into place only once verified
    (non-empty, and as long as Content-Length says), so a failed download
    leaves neither a truncated file nor a clobbered earlier copy behind.
    """
    filename = dest.name
    if dest.exists() and not force:
        return {"status": "skip", "file": filename, "size": dest.stat().st_size}

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        # verify passed explicitly: requests would otherwise let
        # $REQUESTS_CA_BUNDLE / $CURL_CA_BUNDLE override session.verify at
        # request time, and the bundle logged by _setup_session() would not
        # be the one used.
        with session.get(url, stream=True, timeout=60,
                         verify=getattr(session, "verify", True)) as r:
            if r.status_code != 200:
                if pending and r.status_code in (403, 404):
                    return {"status": "pending", "file": filename,
                            "code": r.status_code, "url": url}
                return {"status": "error", "file": filename, "code": r.status_code, "url": url}
            expected = _expected_length(r)
            downloaded = 0
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        problem = _verify_download(downloaded, expected)
        if problem:
            _remove_quietly(part)
            return {"status": "error", "file": filename, "error": problem, "url": url}
        os.replace(part, dest)
        return {"status": "ok", "file": filename, "size": downloaded}
    except Exception as e:
        _remove_quietly(part)
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
    if status == "pending":
        # Counted, not printed: one summary line beats 286 identical 403s.
        counters["pending"] += 1
        return
    # error
    counters["err"] += 1
    detail = res.get("error") or f"HTTP {res.get('code', '?')}"
    counters["failures"].append({"name": name, "kind": kind, "file": res.get("file", ""),
                                 "detail": detail})
    print(f"  [{bar}] {pct:3d}%  {_R}✗{_RST} {name} ({kind}) — {detail}")
    if res.get("code") == 403:
        counters["err_403"] += 1
        if counters["first_err_url"] is None:
            counters["first_err_url"] = res.get("url", "")


def _prepare_output_dir(output_dir: Path):
    """Create the output directory if it does not exist.

    An existing one - a real directory or a symlink to one - is used in
    place and never removed. A symlink that does not resolve to a directory
    (dangling) is an error: it is left alone for the user to fix.
    """
    if output_dir.is_dir():
        return
    if output_dir.is_symlink():
        error(f"Output path {output_dir} is a dangling symlink (-> {os.readlink(output_dir)}).")
        error("Fix the link or remove it, then re-run.")
        raise SystemExit(1)
    if output_dir.exists():
        error(f"Output path {output_dir} exists and is not a directory.")
        raise SystemExit(1)
    output_dir.mkdir(parents=True, exist_ok=True)


def download_all(models: list[dict], output_dir: Path, session,
                 workers: int = 4, force: bool = False, with_json: bool = True) -> dict:
    """Download all models in parallel. Returns the counters, failures included."""
    _prepare_output_dir(output_dir)

    # Build download task list
    tasks = []
    for m in models:
        fname = _dxnn_filename(m)
        pending = is_pending(m)
        tasks.append(("dxnn", m["name"], m["dxnn_url"], output_dir / fname, pending))
        if with_json and m.get("json_url"):
            jname = Path(urlparse(m["json_url"]).path).name
            # Save json alongside dxnn in the same output directory
            tasks.append(("json", m["name"], m["json_url"], output_dir / jname, pending))

    total = len(tasks)
    head(f"\n{'─'*60}")
    head(f"  Starting download: {total} files → {output_dir}")
    head(f"{'─'*60}")

    counters = {"ok": 0, "skip": 0, "err": 0, "err_403": 0, "pending": 0,
                "first_err_url": None, "failures": [], "total": total}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, url, dest, session, force,
                        pending): (kind, name)
            for kind, name, url, dest, pending in tasks
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
    if counters["pending"]:
        head(f"  Pending: {counters['pending']} file(s) are not published yet and were "
             "skipped. Their examples are already in the tree; they will download once "
             "DX Model Zoo publishes them. No action needed.")
    head(f"  Saved to: {output_dir.resolve()}")
    if counters["err_403"]:
        warn(f"{counters['err_403']} file(s) returned HTTP 403 (Forbidden).")
        warn("The models are listed on the page but may not be published yet.")
        if counters["first_err_url"]:
            warn(f"Example URL: {counters['first_err_url']}")
    head(f"{'─'*60}\n")
    return counters


def _report_failures(counters: dict | None, missing: list[str]) -> bool:
    """Print a summary of what the run could not provide; True if anything failed."""
    failures = (counters or {}).get("failures", [])
    if not failures and not missing:
        return False
    if missing:
        error(f"{len(missing)} requested model(s) not found in manifest: {', '.join(missing)}")
    if failures:
        error(f"{len(failures)} of {counters['total']} file(s) failed:")
        for f in failures:
            error(f"  {f['name']} ({f['kind']}, {f['file']}): {f['detail']}")
        hint = x509_strict_hint((f["detail"] for f in failures),
                                relaxed=os.environ.get(RELAX_X509_STRICT_ENV) == "1")
        if hint:
            for line in hint.splitlines():
                error(line)
    return True


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

environment:
  REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE, SSL_CERT_FILE
      the CA bundle to verify TLS with (else the OS trust store, else certifi)
  DXAPP_TLS_RELAX_X509_STRICT=1
      Python 3.13+ verifies TLS in X.509 strict mode; a TLS-inspecting proxy
      whose CA lacks e.g. an Authority Key Identifier then fails. This turns
      off only strict mode: the chain and the host name are still verified.
      It covers the connection to the model server (also when tunnelled);
      TLS to an https:// proxy itself stays strict.
        """,
    )
    dl = parser.add_argument_group("download")
    dl.add_argument("--output",   type=str, default=str(DEFAULT_OUTPUT),
                    help=f"output directory (default: {DEFAULT_OUTPUT})")
    dl.add_argument("--workers",  type=int, default=4,  help="parallel download threads (default: 4)")
    dl.add_argument("--force",    action="store_true",  default=True,
                    help="overwrite existing files (default: enabled)")
    dl.add_argument("--no-force",  action="store_true",
                    help="skip download if the file already exists")
    dl.add_argument("--no-json",  action="store_true",  help="skip JSON file downloads")
    dl.add_argument("--all",      action="store_true",
                    help="download all parsed models non-interactively")
    dl.add_argument("--category", type=str, default=None,
                    help="download only a specific category (e.g. 'Object Detection')")
    dl.add_argument("--models",   type=str, default=None, nargs="+", action="extend",
                    metavar="MODEL",
                    help="whitelist: download only the specified model(s), by manifest "
                         "name, .dxnn file name or config/model_registry.json model_name "
                         "(e.g. --models YoloV8N resnet50 yolov7-w6_1280x1280.dxnn). "
                         "Case-insensitive; commas also separate names; the option may "
                         "repeat. Exits non-zero if a name matches nothing.")
    dl.add_argument("--demo-models", action="store_true",
                    help="download only models required by run_demo.py/run_demo.bat")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--list",    action="store_true", help="list available models without downloading")
    misc.add_argument("--dry-run", action="store_true", help="list models without downloading")
    misc.add_argument("--save-manifest", type=str, default=None,
                      metavar="FILE",    help="save parsed model list to a JSON file")
    misc.add_argument("--manifest", type=str, default=None, metavar="FILE",
                      help=f"path to manifest JSON file (default: {DEFAULT_MANIFEST})")
    # The --internal local-copy mode is gone. Its options still parse so existing
    # command lines (CI jobs) keep working; main() ignores them with a warning.
    misc.add_argument("--internal", action="store_true", help=argparse.SUPPRESS)
    misc.add_argument("--internal-path", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_ca_bundle(environ=None, system_bundles=None) -> str | None:
    """The CA bundle to verify TLS with: the first existing file among
    $REQUESTS_CA_BUNDLE, $CURL_CA_BUNDLE, $SSL_CERT_FILE and the OS trust
    stores. None means requests' own default (certifi). Verification itself
    is never turned off."""
    environ = os.environ if environ is None else environ
    system_bundles = SYSTEM_CA_BUNDLES if system_bundles is None else system_bundles
    for var in CA_BUNDLE_ENV_VARS:
        value = environ.get(var)
        if not value:
            continue
        if Path(value).is_file():
            return value
        warn(f"${var} is set to '{value}', which is not a file; ignoring it")
    for candidate in system_bundles:
        if Path(candidate).is_file():
            return str(candidate)
    return None


def relax_x509_strict_requested(environ=None) -> bool:
    """True when $DXAPP_TLS_RELAX_X509_STRICT is exactly "1" (U-76, opt-in)."""
    environ = os.environ if environ is None else environ
    value = environ.get(RELAX_X509_STRICT_ENV, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    warn(f"${RELAX_X509_STRICT_ENV}={value!r} is not 1; X.509 strict mode stays as Python sets it")
    return False


def build_ssl_context(ca_bundle: str | None, relax_x509_strict: bool,
                      make_context=ssl.create_default_context) -> ssl.SSLContext:
    """The TLS context for downloads with X.509 strict mode relaxed (U-76).

    make_context is ssl.create_default_context: the server's certificate is
    required, its chain is verified against `ca_bundle` (None: the OS default
    store) and the host name is checked. The protocol floor is TLS 1.2, as
    in urllib3's own context (Debian/Ubuntu's Python 3.12 leaves it at
    MINIMUM_SUPPORTED). relax_x509_strict clears VERIFY_X509_STRICT from
    verify_flags and changes nothing else.
    """
    context = make_context(cafile=ca_bundle)
    context.minimum_version = max(context.minimum_version, ssl.TLSVersion.TLSv1_2)
    if relax_x509_strict:
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def _ssl_context_adapter(http_adapter_class, context):
    """A requests HTTPAdapter whose every HTTPS connection to a model server -
    direct, or tunnelled through $HTTPS_PROXY - is made with `context`.
    requests still adds the session's CA bundle and CERT_REQUIRED on top.
    The TLS connection to an https:// proxy itself is not made with it: it
    keeps urllib3's own context (proxy_ssl_context), strict where Python is."""
    class _ContextAdapter(http_adapter_class):
        def init_poolmanager(self, *args, **kwargs):
            kwargs["ssl_context"] = context
            return super().init_poolmanager(*args, **kwargs)

        def proxy_manager_for(self, proxy, **proxy_kwargs):
            proxy_kwargs["ssl_context"] = context
            return super().proxy_manager_for(proxy, **proxy_kwargs)

    return _ContextAdapter()


def x509_strict_hint(details: Iterable[str], relaxed: bool) -> str | None:
    """How to get past a download that failed only on an X.509 strict-mode
    check; None when no failure looks like that, or when relaxed already."""
    if relaxed:
        return None
    for detail in details:
        for marker in X509_STRICT_ONLY_ERRORS:
            if marker in detail:
                return (
                    f"TLS verification failed a strict-mode check ({marker}): "
                    f"{sys.executable} (Python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}) "
                    "verifies in X.509 strict mode, and the certificate chain presented here - "
                    "typically a TLS-inspecting proxy's - does not meet it.\n"
                    "Either run the downloader with a Python 3.12 or older: activate a "
                    "Python 3.12-or-older virtualenv, or DXAPP_SETUP_PYTHON=<python> ./setup.sh "
                    "(e.g. /usr/bin/python3 on Ubuntu 24.04);\n"
                    f"or opt in with {RELAX_X509_STRICT_ENV}=1 ./setup.sh, which turns off only "
                    "X.509 strict mode: the certificate chain and the host name are still verified.")
    return None


def _setup_session():
    """Create a requests Session for downloading."""
    try:
        import requests as _requests
        from requests import adapters as _adapters, certs as _certs
    except ImportError as exc:
        print("[DXAPP] [ERROR] Missing dependency: requests", file=sys.stderr, flush=True)
        print("[DXAPP] [ERROR] Install it with: python3 -m pip install requests", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    session = _requests.Session()
    session.headers.update({"User-Agent": "DEEPX-ModelZoo-Downloader/1.0"})
    bundle = _resolve_ca_bundle()
    if bundle:
        session.verify = bundle
        info(f"TLS CA bundle: {bundle}")
    else:
        info("TLS CA bundle: requests default (certifi)")
    if relax_x509_strict_requested():
        # The bundle, or certifi (requests' own default) when none was found.
        context = build_ssl_context(bundle or _certs.where(), relax_x509_strict=True)
        session.mount("https://", _ssl_context_adapter(_adapters.HTTPAdapter, context))
        warn(f"${RELAX_X509_STRICT_ENV}=1: X.509 strict mode is off for these downloads; "
             "the certificate chain and the host name are still verified")
    return session


def _apply_filters(models: list[dict], args) -> tuple[list[dict], list[str]]:
    """Apply category and model whitelist filters from CLI args.

    Returns the selected models and the requested --models names that
    matched nothing."""
    if args.demo_models and (args.models or args.category or args.all):
        error("Use only one of --demo-models, --models, --category, or --all.")
        raise SystemExit(1)

    if args.category:
        models = [m for m in models if args.category.lower() in m["category"].lower()]
        info(f"Category filter '{args.category}': {len(models)} model(s)")
    missing: list[str] = []
    if args.models:
        models, missing = select_models(models, args.models, load_registry_aliases())
        if missing:
            warn(f"Model(s) not found in manifest: {', '.join(missing)}")
        info(f"Model whitelist: {len(models)} model(s) selected")
    if args.demo_models:
        demo_filenames = get_run_demo_model_filenames()
        models = [
            m for m in models
            if _dxnn_filename(m) in demo_filenames
        ]
        matched = {_dxnn_filename(m) for m in models}
        missing = sorted(demo_filenames - matched)
        if missing:
            warn(f"Run demo model file(s) not found in manifest: {', '.join(missing)}")
        info(f"Run demo model filter: {len(models)} model(s) selected")
    return models, missing


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
            fname = _dxnn_filename(m)
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
    ignored = [opt for opt, given in (("--internal", args.internal),
                                      ("--internal-path", args.internal_path is not None))
               if given]
    if ignored:
        warn(f"Ignoring {' '.join(ignored)}: the --internal local-copy mode was removed "
             "(the options go in a future release); models come from S3.")

    # --no-force overrides --force default
    if args.no_force:
        args.force = False

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
    print(f"  Source   : {_C}S3 / public{_RST}")
    head(f"{_DOUBLE_LINE}\n")

    # Third-party license notice
    print(f"  {_Y}┌────────────────────────────────────────────────────────┐{_RST}")
    print(f"  {_Y}│  ⚠  THIRD-PARTY LICENSE NOTICE                       │{_RST}")
    print(f"  {_Y}│                                                       │{_RST}")
    print(f"  {_Y}│  The sample models (.dxnn) and dataset images are     │{_RST}")
    print(f"  {_Y}│  compiled from third-party open-source projects and   │{_RST}")
    print(f"  {_Y}│  are provided for EVALUATION AND DEVELOPMENT          │{_RST}")
    print(f"  {_Y}│  PURPOSES ONLY. NOT licensed for commercial use.      │{_RST}")
    print(f"  {_Y}│                                                       │{_RST}")
    print(f"  {_Y}│  For commercial deployment, obtain licenses from the  │{_RST}")
    print(f"  {_Y}│  original model/dataset providers.                    │{_RST}")
    print(f"  {_Y}│                                                       │{_RST}")
    print(f"  {_Y}│  Details: docs/source/docs/Appendix_Third_Party_License.md │{_RST}")
    print(f"  {_Y}└────────────────────────────────────────────────────────┘{_RST}")
    print()

    models = load_manifest(manifest_path)
    models, missing = _apply_filters(models, args)

    if (
        not args.all
        and not args.category
        and not args.models
        and not args.demo_models
        and not args.dry_run
        and not args.list
    ):
        models = interactive_select(models, output_dir)

    if args.save_manifest:
        Path(args.save_manifest).write_text(json.dumps(models, indent=2, ensure_ascii=False))
        info(f"Manifest saved: {args.save_manifest} ({len(models)} models)")

    if args.category or args.models or args.demo_models or args.dry_run or args.list:
        _print_filtered_model_list(models, output_dir)

    if args.dry_run or args.list:
        info("--dry-run/--list mode: skipping download.")
        if _report_failures(None, missing):
            sys.exit(1)
        return

    session = _setup_session()
    counters = download_all(
        models=models,
        output_dir=output_dir,
        session=session,
        workers=args.workers,
        force=args.force,
        with_json=not args.no_json,
    )

    if _report_failures(counters, missing):
        sys.exit(1)


if __name__ == "__main__":
    main()
