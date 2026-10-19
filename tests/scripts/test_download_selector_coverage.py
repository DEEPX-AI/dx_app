"""Every published model resolves from the runners' auto-download selectors.

An example run without ``-m`` that finds its model missing calls
``setup_sample_models.sh --models <selector>``, which hands the selector to
``scripts/download_models.py``:

* the C++ runners pass the registry ``model_name`` (``resolveExampleModel``);
* the Python runners pass the ``.dxnn`` stem (``Path(dxnn_file).stem``).

For every registry row with ``published: true``, each selector must select
exactly one manifest entry, and that entry must download the row's
``dxnn_file``. Rows that are not published are listed with their reason (run
with ``-s``); none of them may have a downloadable manifest entry.

Hermetic: the downloader's own matcher runs on the committed manifest and
registry. Nothing is downloaded and nothing is written.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER = ROOT / "scripts" / "download_models.py"
REGISTRY = ROOT / "config" / "model_registry.json"
MANIFEST = ROOT / "scripts" / "modelzoo_manifest.json"
# 497 published rows, one per variant. A lower count means a
# row was marked unpublished again.
MIN_PUBLISHED_ROWS = 497

SELECTORS = {
    "cpp_model_name": lambda row: row["model_name"],
    "python_dxnn_stem": lambda row: Path(row["dxnn_file"]).stem,
}


@pytest.fixture(scope="module")
def dm():
    spec = importlib.util.spec_from_file_location("download_models_selector", DOWNLOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.mark.parametrize("selector", sorted(SELECTORS))
def test_every_published_row_selects_exactly_its_dxnn(dm, registry, manifest, selector):
    aliases = dm.load_registry_aliases(REGISTRY)
    published = [row for row in registry if row.get("published") is True]
    assert len(published) >= MIN_PUBLISHED_ROWS

    wrong = []
    for row in published:
        name = SELECTORS[selector](row)
        chosen, missing = dm.select_models(manifest, [name], aliases)
        files = [dm._dxnn_filename(entry) for entry in chosen]
        if missing or files != [row["dxnn_file"]]:
            wrong.append("{} -> {} (want {})".format(name, files or "nothing", row["dxnn_file"]))
    assert not wrong, "{}/{} published rows do not resolve:\n  {}".format(
        len(wrong), len(published), "\n  ".join(wrong))


def test_unpublished_rows_are_listed_with_their_reason(dm, registry, manifest):
    """An unpublished row's file must not be downloadable: absent from the
    manifest or pending there. A downloadable manifest entry for it means the
    registry and the manifest disagree."""
    by_file = {dm._dxnn_filename(entry): entry for entry in manifest}
    downloadable = []
    for row in registry:
        if row.get("published") is True:
            continue
        entry = by_file.get(row["dxnn_file"])
        if entry is None:
            reason = "absent from scripts/modelzoo_manifest.json"
        elif dm.is_pending(entry):
            reason = "manifest entry is pending (declared, not downloadable)"
        else:
            reason = "listed in scripts/modelzoo_manifest.json"
            downloadable.append("{} ({})".format(row["variant"], row["dxnn_file"]))
        print("unpublished: {} ({}): published={!r} in config/model_registry.json; "
              "{}: {}".format(row["model_name"], row["variant"], row.get("published"),
                              row["dxnn_file"], reason))
    assert not downloadable, (
        "unpublished in config/model_registry.json but downloadable from "
        "scripts/modelzoo_manifest.json:\n  " + "\n  ".join(downloadable))
