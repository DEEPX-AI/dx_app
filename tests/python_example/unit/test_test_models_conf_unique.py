# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""config/test_models.conf lists each variant once, and every registry variant.

The registry used to carry an alias entry -- deit_base384_distilled was alias_of
deit_base_distilled_2, same variant -- and regen_test_models_conf.py emitted one row per
ENTRY, so that variant got two rows: run_examples.sh / bench_models.sh ran it twice and
DX AI Studio listed it twice in its Model Zoo. The duplicate row is gone, but the
per-variant invariant is what these tests hold, not the absence of aliases.
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _variants(text: str) -> list[str]:
    return [line.split("\t")[3] for line in text.splitlines()
            if line and not line.startswith("#") and len(line.split("\t")) > 3]


def test_the_generator_emits_each_variant_once(tmp_path):
    reg = json.loads((PROJECT_ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))
    out = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "regen_test_models_conf.py"),
                          "--out", str(tmp_path / "conf")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    variants = _variants((tmp_path / "conf").read_text(encoding="utf-8"))
    dups = [v for v, n in collections.Counter(variants).items() if n > 1]
    assert not dups, dups
    assert set(variants) == {e["variant"] for e in reg}


def test_the_committed_conf_is_what_the_generator_writes(tmp_path):
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "regen_test_models_conf.py"),
                    "--out", str(tmp_path / "conf")], check=True, capture_output=True)
    assert (tmp_path / "conf").read_text(encoding="utf-8") == \
        (PROJECT_ROOT / "config" / "test_models.conf").read_text(encoding="utf-8"), \
        "config/test_models.conf is stale -- run python3 scripts/regen_test_models_conf.py"
