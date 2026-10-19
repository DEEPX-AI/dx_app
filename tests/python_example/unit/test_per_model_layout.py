# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Per-model example folders: one directory per registry variant."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"
PY_KINDS = (
    "sync",
    "async",
    "sync_cpp_postprocess",
    "async_cpp_postprocess",
)
EXTRACT = PROJECT_ROOT / "scripts" / "extract_model_package.sh"
SAMPLE = "depth_estimation/depthanythingv2/depthanythingv2-vitb_224x224"


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _canonical(registry: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for entry in registry:
        if entry["variant"] in seen:
            continue
        seen.add(entry["variant"])
        out.append(entry)
    return out


def test_every_registry_variant_has_a_model_folder(registry):
    missing: list[str] = []
    for entry in _canonical(registry):
        for tree in ("python_example", "cpp_example"):
            config_path = (
                PROJECT_ROOT / "src" / tree / entry["task"] / entry["family"]
                / entry["variant"] / "config.json"
            )
            if not config_path.is_file():
                missing.append(str(config_path.relative_to(PROJECT_ROOT)))
    assert not missing, "missing model config.json:\n  " + "\n  ".join(missing[:20])


def test_no_leftover_variants_directories():
    leftover = []
    for tree in ("python_example", "cpp_example"):
        root = PROJECT_ROOT / "src" / tree
        leftover.extend(
            str(path.relative_to(PROJECT_ROOT))
            for path in root.glob("*/*/variants")
            if path.is_dir()
        )
    assert leftover == []


def test_each_python_model_folder_has_thin_entry_scripts(registry):
    missing: list[str] = []
    for entry in _canonical(registry):
        folder = (
            PROJECT_ROOT / "src" / "python_example" / entry["task"]
            / entry["family"] / entry["variant"]
        )
        for kind in PY_KINDS:
            script = folder / f"{entry['variant']}_{kind}.py"
            if not script.is_file():
                missing.append(str(script.relative_to(PROJECT_ROOT)))
                continue
            text = script.read_text(encoding="utf-8")
            if "fixed_variant=_FIXED_VARIANT" not in text:
                missing.append(f"{script.relative_to(PROJECT_ROOT)} is not a thin entry")
            if "_peek_variant" in text:
                missing.append(f"{script.relative_to(PROJECT_ROOT)} still inlines peek")
    assert not missing, "\n  ".join(missing[:20])


def test_default_variant_is_first_published_model_folder(tmp_path):
    from common.variant_config import default_variant, load_variant_config

    family = tmp_path / "family"
    for name, published in (("b-model", True), ("a-unpublished", False), ("c-model", True)):
        folder = family / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(
            json.dumps({"variant": name, "published": published}),
            encoding="utf-8",
        )
    load_variant_config.cache_clear()
    # Alphabetically first *published* folder, skipping a-unpublished.
    assert default_variant(str(family)) == "b-model"
    assert load_variant_config(str(family), "c-model")["variant"] == "c-model"


def _configs_under(root: Path) -> list[Path]:
    return sorted(root.rglob("config.json"))


def test_extract_one_model_keeps_a_single_config(tmp_path):
    out = tmp_path / "extract"
    py = subprocess.run(
        [str(EXTRACT), SAMPLE, "--output-dir", str(out), "--lang", "py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert py.returncode == 0, py.stderr
    py_root = out / "py"
    configs = _configs_under(py_root)
    assert len(configs) == 1, [str(p) for p in configs]
    assert configs[0].parent.name == "depthanythingv2-vitb_224x224"
    package = configs[0].parent
    assert (package / "factory").is_dir()
    assert (package / "common" / "runner" / "entry.py").is_file()
    assert not (package.parent / "depthanythingv2-vitl_224x224").exists()

    cpp = subprocess.run(
        [str(EXTRACT), SAMPLE, "--output-dir", str(out), "--lang", "cpp"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert cpp.returncode == 0, cpp.stderr
    cpp_configs = _configs_under(out / "cpp")
    assert len(cpp_configs) == 1
    variant = (
        out / "cpp" / "depth_estimation" / "depthanythingv2"
        / "depthanythingv2-vitb_224x224"
    )
    assert (variant / "depthanythingv2-vitb_224x224_sync.cpp").is_file()
    assert (variant / "factory").is_dir()


def test_extract_family_keeps_every_model_folder(tmp_path):
    out = tmp_path / "family"
    result = subprocess.run(
        [
            str(EXTRACT),
            "depth_estimation/depthanythingv2",
            "--output-dir",
            str(out),
            "--lang",
            "py",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    names = sorted(path.parent.name for path in _configs_under(out / "py"))
    assert "depthanythingv2-vitb_224x224" in names
    assert "depthanythingv2-vitl_224x224" in names
    assert "depthanythingv2-vits_224x224" in names
    family = out / "py" / "depth_estimation" / "depthanythingv2"
    variant_dir = family / "depthanythingv2-vitb_224x224"
    assert (variant_dir / "depthanythingv2-vitb_224x224_sync.py").is_file()
    assert (variant_dir / "factory").is_dir()
    assert not (family / "depthanythingv2_sync.py").exists()
