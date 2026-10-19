# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""modelzoo_manifest.json <-> the examples <-> config/model_registry.json.

An example is only usable if the model behind it can be downloaded: the manifest points
a name at the file's URL, an example variant config exists for the stem, and the
registry names the same file. Each pair has drifted at least once.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from test_helpers.constants import EXCLUDED_EXAMPLE_VARIANTS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = PROJECT_ROOT / "scripts" / "modelzoo_manifest.json"
REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"

URL = re.compile(r"https://sdk\.deepx\.ai/modelzoo/([a-z0-9-]+)/([0-9_]+)/([^\"' ]+\.dxnn)")


@pytest.fixture(scope="module")
def manifest() -> list:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry() -> list:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def variant_configs() -> set:
    return {p.parent.name for p in
            (PROJECT_ROOT / "src" / "python_example").glob("*/*/*/config.json")}


def _parsed(manifest: list):
    for row in manifest:
        match = URL.fullmatch(row.get("dxnn_url") or "")
        yield row, (match.groups() if match else None)


# --------------------------------------------------------------------- manifest shape
def test_every_manifest_url_is_a_modelzoo_url(manifest):
    bad = [row["name"] for row, parts in _parsed(manifest) if parts is None]
    assert not bad, f"unparsable dxnn_url on: {bad}"


def test_the_manifest_covers_every_example_variant(manifest, variant_configs):
    """An example whose model is in no manifest row cannot be downloaded at all."""
    files = {parts[2] for _row, parts in _parsed(manifest) if parts}
    orphans = sorted(
        v for v in variant_configs
        if v not in EXCLUDED_EXAMPLE_VARIANTS and f"{v}.dxnn" not in files
    )
    assert not orphans, f"example variants with no manifest row: {orphans}"


def test_no_manifest_row_is_duplicated(manifest):
    stems = [parts[2] for _row, parts in _parsed(manifest) if parts]
    duplicated = sorted({s for s in stems if stems.count(s) > 1})
    assert not duplicated, f"duplicate manifest rows: {duplicated}"


# -------------------------------------------------------------- registry consistency
def test_the_registry_and_the_manifest_agree_on_every_file(manifest, registry):
    manifest_files = {parts[2] for _row, parts in _parsed(manifest) if parts}
    registry_files = {m["dxnn_file"] for m in registry}
    assert registry_files == manifest_files, (
        f"only in registry: {sorted(registry_files - manifest_files)}\n"
        f"only in manifest: {sorted(manifest_files - registry_files)}")


def test_unpublished_rows_are_flagged_not_inferred_from_the_version(manifest, registry):
    """Which models are unpublished is DECLARED, never read off the URL.

    The downloader used to infer it -- any URL containing /2_5_0/ was "not published
    yet" -- which held only while 2_5_0 was unreleased. The zoo published it on
    2026-09-30 and moved the whole q-lite tier there, so that rule would now mark 492
    of 499 rows pending and turn every real 403 into a silent skip. Probing all 499
    URLs that day: 498 -> 200, and exactly one -> 403.

    So a row carries `pending: true` and nothing else means unpublished.
    """
    flagged = {row["name"] for row in manifest if row.get("pending")}
    unpublished = {m["model_name"] for m in registry if not m["published"]}
    assert flagged == unpublished, (
        f"manifest flags {sorted(flagged)}, registry says {sorted(unpublished)}")

