"""scripts/add_model.sh writes the per-variant layout.

A new model lands in ``src/{cpp,python}_example/<task>/<family>/<variant>/``,
its C++ factory is scoped in ``dxapp::v_<variant>`` (so
``generate_cpp_family_layout.py --variant-scope --check`` accepts it), its
config.json names the variant, and the registry gains a row for it. The
script runs against a copy of one real variant in a throwaway tree, so the
real repository is never touched (tests/scripts is hermetic).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = "yolov7_640x640"
REF_FAMILY = "yolov7"
TASK = "object_detection"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake dx_app root holding the scripts and one real variant (yolov7_640x640)."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("add_model.sh", "add_model_variant.py",
                 "generate_cpp_family_layout.py", "check_model_registry.py"):
        src = ROOT / "scripts" / name
        if src.exists():
            shutil.copy(src, scripts / name)
    (tmp_path / "src" / "cpp_example" / "common").mkdir(parents=True)
    _copy_tree(ROOT / "src" / "cpp_example" / TASK / REF_FAMILY / REF,
               tmp_path / "src" / "cpp_example" / TASK / REF_FAMILY / REF)
    py_family = tmp_path / "src" / "python_example" / TASK / REF_FAMILY
    _copy_tree(ROOT / "src" / "python_example" / TASK / REF_FAMILY / REF, py_family / REF)
    (py_family / "__init__.py").write_text("")
    rows = json.loads((ROOT / "config" / "model_registry.json").read_text())
    row = next(r for r in rows if r["variant"] == REF)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "model_registry.json").write_text(json.dumps([row], indent=2) + "\n")
    (tmp_path / "config" / "test_models.conf").write_text(
        "# Format: family<TAB>task<TAB>model_file<TAB>variant\n"
        f"{REF_FAMILY}\t{TASK}\tassets/models/{REF}.dxnn\t{REF}\n")
    return tmp_path


def _add_model(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(["bash", str(repo / "scripts" / "add_model.sh"), *args],
                          cwd=repo, capture_output=True, text=True, timeout=300, env=env)


def _check_scope(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "generate_cpp_family_layout.py"),
         "--variant-scope", str(repo / "src" / "cpp_example"), "--check"],
        cwd=repo, capture_output=True, text=True, timeout=120)


def _check_registry(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_model_registry.py"),
         "--registry", str(repo / "config" / "model_registry.json"),
         "--cpp-root", str(repo / "src" / "cpp_example"),
         "--py-root", str(repo / "src" / "python_example")],
        cwd=repo, capture_output=True, text=True, timeout=120)


def test_base_model_lands_in_the_reference_family_with_its_own_namespace(repo: Path):
    new = "yolov7-custom_640x640"
    r = _add_model(repo, new, TASK, "--base-model", REF)
    assert r.returncode == 0, r.stdout + r.stderr

    cpp = repo / "src" / "cpp_example" / TASK / REF_FAMILY / new
    assert (cpp / "factory" / f"{new}_factory.hpp").is_file(), r.stdout
    assert (cpp / f"{new}_sync.cpp").is_file()
    assert (cpp / f"{new}_async.cpp").is_file()
    assert not (repo / "src" / "cpp_example" / TASK / new).exists(), "old flat layout"

    hpp = (cpp / "factory" / f"{new}_factory.hpp").read_text()
    assert "namespace v_yolov7_custom_640x640 {" in hpp
    assert "}  // namespace v_yolov7_custom_640x640" in hpp
    assert "v_yolov7_640x640" not in hpp
    assert "YOLOV7_CUSTOM_640X640_FACTORY_HPP" in hpp
    assert "YOLOV7_640X640_FACTORY_HPP" not in hpp
    assert f'return "{new}";' in hpp
    for kind in ("sync", "async"):
        entry = (cpp / f"{new}_{kind}.cpp").read_text()
        assert "dxapp::v_yolov7_custom_640x640::Yolov7Factory" in entry
        assert f'#include "factory/{new}_factory.hpp"' in entry
        assert REF not in entry

    cfg = json.loads((cpp / "config.json").read_text())
    assert cfg["variant"] == new
    assert cfg["dxnn_file"] == f"{new}.dxnn"
    assert cfg["family"] == REF_FAMILY
    assert cfg["task"] == TASK

    scope = _check_scope(repo)
    assert scope.returncode == 0, scope.stdout + scope.stderr


def test_python_side_is_a_fixed_variant_package(repo: Path):
    new = "yolov7-custom_640x640"
    r = _add_model(repo, new, TASK, "--base-model", REF)
    assert r.returncode == 0, r.stdout + r.stderr

    py = repo / "src" / "python_example" / TASK / REF_FAMILY / new
    assert (py / "factory" / f"{new}_factory.py").is_file(), r.stdout
    assert (py / "factory" / "__init__.py").is_file()
    assert (py / "custom_ops.py").is_file(), "the reference's custom_ops.py is carried over"
    entries = [py / f"{new}_{k}.py" for k in
               ("sync", "async", "sync_cpp_postprocess", "async_cpp_postprocess")]
    for e in entries:
        assert e.is_file(), e
        text = e.read_text()
        assert f'_FIXED_VARIANT = "{new}"' in text
        assert REF not in text
        compile(text, str(e), "exec")  # a syntax check that writes no .pyc
    assert "from .yolov7-custom_640x640_factory" not in \
        (py / "factory" / "__init__.py").read_text()
    assert json.loads((py / "config.json").read_text())["variant"] == new


def test_the_registry_and_test_models_conf_gain_the_variant(repo: Path):
    new = "yolov7-custom_640x640"
    r = _add_model(repo, new, TASK, "--base-model", REF, "--input-size", "1280x1280")
    assert r.returncode == 0, r.stdout + r.stderr

    rows = json.loads((repo / "config" / "model_registry.json").read_text())
    row = [x for x in rows if x["variant"] == new]
    assert len(row) == 1, rows
    row = row[0]
    assert row["family"] == REF_FAMILY and row["task"] == TASK
    assert row["dxnn_file"] == f"{new}.dxnn"
    assert (row["input_width"], row["input_height"]) == (1280, 1280)
    assert row["alias_of"] is None and row["published"] is True
    for lang in ("cpp", "python"):
        cfg = json.loads((repo / "src" / f"{lang}_example" / TASK / REF_FAMILY / new
                          / "config.json").read_text())
        assert (cfg["input_width"], cfg["input_height"]) == (1280, 1280)
    conf = (repo / "config" / "test_models.conf").read_text()
    assert f"{REF_FAMILY}\t{TASK}\tassets/models/{new}.dxnn\t{new}\n" in conf

    reg = _check_registry(repo)
    assert reg.returncode == 0, reg.stdout + reg.stderr


def test_an_existing_registry_row_drives_family_and_size(repo: Path):
    new = "yolov7-big_1280x1280"
    rows = json.loads((repo / "config" / "model_registry.json").read_text())
    row = dict(rows[0], model_name=new, variant=new, dxnn_file=f"{new}.dxnn",
               original_name=new, input_width=1280, input_height=1280)
    (repo / "config" / "model_registry.json").write_text(json.dumps(rows + [row], indent=2) + "\n")

    r = _add_model(repo, new, TASK)
    assert r.returncode == 0, r.stdout + r.stderr
    cpp = repo / "src" / "cpp_example" / TASK / REF_FAMILY / new
    assert (cpp / "factory" / f"{new}_factory.hpp").is_file(), r.stdout
    cfg = json.loads((cpp / "config.json").read_text())
    assert (cfg["input_width"], cfg["input_height"]) == (1280, 1280)
    rows = json.loads((repo / "config" / "model_registry.json").read_text())
    assert sum(x["variant"] == new for x in rows) == 1, "no duplicate row"
    assert _check_registry(repo).returncode == 0
    assert _check_scope(repo).returncode == 0


def test_postprocessor_and_family_open_a_new_family(repo: Path):
    new = "custom-det_640x640"
    r = _add_model(repo, new, "detection", "--postprocessor", "yolov5",
                   "--family", "custom_det", "--lang", "both")
    assert r.returncode == 0, r.stdout + r.stderr
    cpp = repo / "src" / "cpp_example" / TASK / "custom_det" / new
    py = repo / "src" / "python_example" / TASK / "custom_det" / new
    assert (cpp / "factory" / f"{new}_factory.hpp").is_file(), r.stdout
    assert (py / "factory" / f"{new}_factory.py").is_file()
    assert (py.parent / "__init__.py").is_file(), "a new family is a package"
    hpp = (cpp / "factory" / f"{new}_factory.hpp").read_text()
    assert "namespace v_custom_det_640x640 {" in hpp
    assert json.loads((py / "config.json").read_text())["family"] == "custom_det"
    assert _check_scope(repo).returncode == 0
    assert _check_registry(repo).returncode == 0


def test_an_existing_variant_is_refused_not_overwritten(repo: Path):
    r = _add_model(repo, REF, TASK, "--base-model", REF)
    assert r.returncode != 0
    assert "exists" in (r.stdout + r.stderr).lower()


def test_sync_only_skips_the_async_entries(repo: Path):
    new = "yolov7-sync_640x640"
    r = _add_model(repo, new, TASK, "--base-model", REF, "--sync-only")
    assert r.returncode == 0, r.stdout + r.stderr
    cpp = repo / "src" / "cpp_example" / TASK / REF_FAMILY / new
    py = repo / "src" / "python_example" / TASK / REF_FAMILY / new
    assert (cpp / f"{new}_sync.cpp").is_file() and not (cpp / f"{new}_async.cpp").exists()
    assert (py / f"{new}_sync.py").is_file() and not (py / f"{new}_async.py").exists()
    assert not (py / f"{new}_async_cpp_postprocess.py").exists()
