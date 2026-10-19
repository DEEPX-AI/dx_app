"""Every CLIP variant whose factory reads a prompt bank finds it.

The zero-shot head ranks an image embedding against ``prompt_bank.json``, the
text side frozen at build time by ``scripts/build_clip_prompt_bank.py``. That
script writes the bank once per FAMILY directory, which is where it is tracked.
The factory code lives in each variant's ``custom_ops.py``, so a bank resolved
next to that file (the variant directory) is never found and every entry point
of the variant exits in ``create_visualizer``. A bank placed in the variant
directory overrides the family one.

Hermetic: imports the source tree's ``custom_ops.py`` files; no model, no NPU.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = PROJECT_ROOT / "src" / "python_example"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _variant_factories():
    """``[(variant dir, factory class)]`` for every variant with a custom_ops.py."""
    out = []
    for ops in sorted(PY_ROOT.glob("*/*/*/custom_ops.py")):
        variant = ops.parent.name
        module = _load(ops, "custom_ops_" + variant.replace("-", "_").replace(".", "_"))
        factory = getattr(module, "FACTORIES", {}).get(variant)
        if factory is not None:
            out.append((ops.parent, factory))
    return out


VARIANT_FACTORIES = _variant_factories()
BANK_READERS = [(d, f) for d, f in VARIANT_FACTORIES if hasattr(f, "_BANK")]


def test_some_variant_reads_a_prompt_bank():
    # Guards the parametrisation below against silently matching nothing.
    names = [d.name for d, _ in BANK_READERS]
    assert "clip-img_vit-b32_256x256_datacomp-s34b-b86k" in names, names


@pytest.mark.parametrize("variant_dir,factory", BANK_READERS,
                         ids=[d.name for d, _ in BANK_READERS])
def test_every_bank_reading_variant_finds_its_bank(variant_dir, factory):
    bank = Path(factory._BANK)
    assert bank.is_file(), f"{variant_dir.relative_to(PY_ROOT)}: no prompt bank at {bank}"
    labels = json.loads(bank.read_text(encoding="utf-8"))["labels"]
    assert labels, bank
    # The labels the visualizer prints come from that same bank.
    assert factory({})._bank_labels() == labels


def _family_copy(tmp_path: Path) -> Path:
    """A throwaway family dir holding one variant's custom_ops.py."""
    src_dir, _ = BANK_READERS[0]
    variant_dir = tmp_path / "family" / src_dir.name
    variant_dir.mkdir(parents=True)
    shutil.copy(src_dir / "custom_ops.py", variant_dir / "custom_ops.py")
    return variant_dir


def _bank_of(variant_dir: Path, tag: str) -> Path:
    module = _load(variant_dir / "custom_ops.py", "custom_ops_bank_" + tag)
    return Path(module.FACTORIES[variant_dir.name]._BANK)


def test_the_family_bank_is_used_when_the_variant_has_none(tmp_path):
    variant_dir = _family_copy(tmp_path)
    family_bank = variant_dir.parent / "prompt_bank.json"
    family_bank.write_text('{"labels": ["family"]}', encoding="utf-8")
    assert _bank_of(variant_dir, "family") == family_bank


def test_a_bank_in_the_variant_dir_overrides_the_family_bank(tmp_path):
    variant_dir = _family_copy(tmp_path)
    (variant_dir.parent / "prompt_bank.json").write_text('{"labels": ["family"]}',
                                                         encoding="utf-8")
    own = variant_dir / "prompt_bank.json"
    own.write_text('{"labels": ["variant"]}', encoding="utf-8")
    assert _bank_of(variant_dir, "override") == own
