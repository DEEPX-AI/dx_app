"""A Python example without -m resolves its own model (SDKREQ-529).

A per-model script ``<variant>_sync.py`` / ``<variant>_async.py`` run without
``-m`` loads ``_resolve_default_model_path()``: ``assets/models/<dxnn_file>``
of the registry row whose ``variant`` is the script's name, the canonical row
(``alias_of`` null) when an alias row shares it. No NPU, nothing written.
"""
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
_src = str(_root / "src" / "python_example")
if _src not in sys.path:
    sys.path.insert(0, _src)

from common.runner.sync_runner import _resolve_default_model_path  # noqa: E402

REGISTRY = _root / "config" / "model_registry.json"
# 497 canonical rows, no alias_of. Two PPU examples are not registered.
MIN_CANONICAL_ROWS = 497


def _rows():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _resolved(monkeypatch, script: str):
    monkeypatch.setattr(sys, "argv", [script])
    return _resolve_default_model_path()


def test_every_per_model_script_resolves_its_variant_row(monkeypatch):
    rows = [row for row in _rows() if not row.get("alias_of")]
    assert len(rows) >= MIN_CANONICAL_ROWS
    wrong = []
    for row in rows:
        for kind in ("sync", "async"):
            script = "{}_{}.py".format(row["variant"], kind)
            want = "assets/models/" + row["dxnn_file"]
            got = _resolved(monkeypatch, script)
            if got != want:
                wrong.append("{}: {} (want {})".format(script, got, want))
    assert not wrong, "{} scripts resolve wrongly:\n  {}".format(
        len(wrong), "\n  ".join(wrong[:20]))


def test_a_registry_model_name_still_resolves(monkeypatch):
    assert _resolved(monkeypatch, "yolov8n_sync.py") == "assets/models/yolov8-n_640x640.dxnn"


def test_an_unknown_script_name_resolves_nothing(monkeypatch):
    assert _resolved(monkeypatch, "no-such-variant_sync.py") is None
