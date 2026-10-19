# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The tests read an example's task from the task/family/variant layout.

Examples live at ``src/{python,cpp}_example/<task>/<family>/<variant>/``. The
task is the first directory under the example root; the grandparent of a
script is the FAMILY. Reading the grandparent as the task (the pre-family
layout rule) silently disables every image-only skip, which is how dope,
sfa3d and espcn stream / multi-loop cases ran video and failed.

Hermetic: reads the source tree only, never a build tree or a model store.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TESTS_DIR.parent
PY_ROOT = PROJECT_ROOT / "src" / "python_example"
CPP_ROOT = PROJECT_ROOT / "src" / "cpp_example"

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_helpers import utils  # noqa: E402
from test_helpers.constants import EXCLUDED_EXAMPLE_VARIANTS  # noqa: E402


def _variant_configs(root: Path) -> list:
    """``[(variant_dir, config_task)]`` for every ``<task>/<family>/<variant>/``."""
    out = []
    for cfg in sorted(root.glob("*/*/*/config.json")):
        out.append((cfg.parent, json.loads(cfg.read_text(encoding="utf-8"))["task"]))
    return out


PY_VARIANTS = _variant_configs(PY_ROOT)
CPP_VARIANTS = _variant_configs(CPP_ROOT)


def test_every_variant_tree_is_discovered():
    # 499 variants per language at the time of writing; the floor only guards
    # against a glob that silently matches nothing.
    assert len(PY_VARIANTS) >= 400, len(PY_VARIANTS)
    assert len(CPP_VARIANTS) >= 400, len(CPP_VARIANTS)


def test_py_script_task_matches_every_variant_config():
    wrong = []
    checked = 0
    for variant_dir, task in PY_VARIANTS:
        scripts = sorted(variant_dir.glob("*_sync.py"))
        assert scripts, f"no *_sync.py in {variant_dir}"
        for script in scripts:
            checked += 1
            got = utils.py_script_task(script)
            if got != task:
                wrong.append(f"{script.relative_to(PY_ROOT)}: {got!r} != {task!r}")
    assert checked >= len(PY_VARIANTS)
    assert not wrong, f"{len(wrong)} scripts:\n" + "\n".join(wrong[:20])


def test_py_script_task_accepts_a_relative_path():
    variant_dir, task = PY_VARIANTS[0]
    script = sorted(variant_dir.glob("*_sync.py"))[0]
    assert utils.py_script_task(script.relative_to(PROJECT_ROOT)) == task


def test_py_script_task_rejects_a_path_outside_python_example():
    with pytest.raises(ValueError):
        utils.py_script_task(CPP_ROOT / "super_resolution" / "x" / "y" / "y_sync.cpp")


def test_cpp_exe_task_map_matches_every_variant_config():
    mapping = utils.cpp_exe_task_map()
    wrong = []
    for variant_dir, task in CPP_VARIANTS:
        # Kept on disk as source only, never built: cpp_exe_task_map skips them.
        if variant_dir.name in EXCLUDED_EXAMPLE_VARIANTS:
            continue
        for suffix in ("_sync", "_async"):
            exe = variant_dir.name + suffix
            if mapping.get(exe) != task:
                wrong.append(f"{exe}: {mapping.get(exe)!r} != {task!r}")
    assert not wrong, f"{len(wrong)} executables:\n" + "\n".join(wrong[:20])


def test_cpp_exe_task_map_leaves_out_the_source_only_examples():
    listed = sorted(exe for exe in utils.cpp_exe_task_map()
                    if exe.rsplit("_", 1)[0] in EXCLUDED_EXAMPLE_VARIANTS)
    assert not listed, listed


# ---------------------------------------------------------------------------
# Regression guard: the call sites that used the grandparent as the task.
# ---------------------------------------------------------------------------

_PY_CALL_SITES = (
    TESTS_DIR / "python_example" / "test_dump_tensors.py",
    TESTS_DIR / "python_example" / "test_e2e.py",
    TESTS_DIR / "python_example" / "test_multi_loop.py",
)
_CPP_E2E = TESTS_DIR / "cpp_example" / "test_e2e.py"


def _is_grandparent_name(node: ast.AST) -> bool:
    """``<x>.parent.parent.name``."""
    return (isinstance(node, ast.Attribute) and node.attr == "name"
            and isinstance(node.value, ast.Attribute) and node.value.attr == "parent"
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "parent")


@pytest.mark.parametrize("path", _PY_CALL_SITES, ids=lambda p: p.name)
def test_no_call_site_reads_the_grandparent_as_the_task(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = [n.lineno for n in ast.walk(tree) if _is_grandparent_name(n)]
    assert not hits, (
        f"{path.name}:{hits} reads <script>.parent.parent.name -- that is the "
        "family on the task/family/variant layout; use py_script_task()")


def test_cpp_e2e_uses_the_shared_task_map():
    tree = ast.parse(_CPP_E2E.read_text(encoding="utf-8"), filename=str(_CPP_E2E))
    local_builders = [n.name for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "_build_exe_task_map"]
    assert not local_builders, "test_e2e.py keeps its own family-keyed task map"
    sources = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_EXE_TASK_MAP" for t in n.targets)]
    assert len(sources) == 1
    call = sources[0]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
        and call.func.id == "cpp_exe_task_map", ast.dump(call)


# ---------------------------------------------------------------------------
# Image-only variants of video-capable tasks (casvit, clip-img).
# ---------------------------------------------------------------------------

def test_py_variant_image_only_matches_every_variant_config():
    wrong = []
    for variant_dir, _task in PY_VARIANTS:
        expected = bool(json.loads((variant_dir / "config.json").read_text(
            encoding="utf-8")).get("image_only"))
        for script in sorted(variant_dir.glob("*_sync.py")):
            if utils.py_variant_image_only(script) != expected:
                wrong.append(f"{script.relative_to(PY_ROOT)}: expected {expected}")
    assert not wrong, f"{len(wrong)} scripts:\n" + "\n".join(wrong[:20])


def test_task_rule_alone_misses_image_only_variants():
    """The variant flag is needed: some image-only variants sit under a
    video-capable task, so IMAGE_ONLY_TASKS cannot skip them."""
    from test_helpers.constants import IMAGE_ONLY_TASKS
    missed = sorted(d.name for d, task in PY_VARIANTS
                    if task not in IMAGE_ONLY_TASKS
                    and utils.py_variant_image_only(next(d.glob("*_sync.py"))))
    assert "casvit-t_224x224" in missed, missed


def test_py_variant_image_only_without_config(tmp_path):
    script = tmp_path / "x_sync.py"
    script.write_text("", encoding="utf-8")
    assert utils.py_variant_image_only(script) is False


def test_cpp_variant_image_only_matches_every_variant_config():
    """A config.json image_only variant is image-only; any other variant is
    image-only only because its runner takes no -v (see the next test)."""
    wrong = []
    checked = 0
    for variant_dir, _task in CPP_VARIANTS:
        expected = bool(json.loads((variant_dir / "config.json").read_text(
            encoding="utf-8")).get("image_only"))
        for source in sorted(variant_dir.glob("*_sync.cpp")) + sorted(variant_dir.glob("*_async.cpp")):
            checked += 1
            reason = utils.cpp_variant_image_only_reason(source.stem)
            if expected and (reason is None or "config.json" not in reason):
                wrong.append(f"{source.stem}: config image_only, helper says {reason!r}")
            if not expected and reason is not None and not reason.startswith("image-only runner"):
                wrong.append(f"{source.stem}: config not image_only, helper says {reason!r}")
    assert checked >= 400, checked
    assert not wrong, f"{len(wrong)} executables:\n" + "\n".join(wrong[:20])


def test_cpp_clip_zero_shot_is_image_only_by_its_runner():
    """The variant config is shared with Python, where clip-img zero-shot runs on
    video; the C++ example runs it through the embedding runner, which has no
    -v option ("Option 'v' does not exist")."""
    variant = "clip-img_vit-b32_256x256_datacomp-s34b-b86k"
    config = next(CPP_ROOT.glob(f"*/*/{variant}/config.json"))
    assert json.loads(config.read_text(encoding="utf-8"))["image_only"] is False
    for exe, runner in ((variant + "_sync", "SyncEmbeddingRunner"),
                        (variant + "_async", "AsyncEmbeddingRunner")):
        assert utils.cpp_variant_image_only(exe) is True, exe
        assert runner in utils.cpp_variant_image_only_reason(exe), exe


def test_cpp_runner_stream_option_follows_aliases():
    """SyncKeypointRunner is SyncPoseRunner, which registers -v; the embedding
    runners include the detection runner's header but not its options."""
    runner_dir = str(CPP_ROOT / "common" / "runner")
    assert utils._cpp_runner_takes_stream("SyncKeypointRunner", runner_dir) is True
    assert utils._cpp_runner_takes_stream("SyncDetectionRunner", runner_dir) is True
    assert utils._cpp_runner_takes_stream("SyncEmbeddingRunner", runner_dir) is False
    assert utils._cpp_runner_takes_stream("NoSuchRunner", runner_dir) is None


def test_cpp_variant_image_only_tells_casvit_classifier_from_casvit_segmenter():
    """The keyword "casvit" matched both; only the classifier is image-only."""
    assert utils.cpp_variant_image_only("casvit-t_224x224_sync") is True
    assert utils.cpp_variant_image_only("casvit-t-fpn-resnet50_512x512_sync") is False


def test_cpp_variant_image_only_of_an_unknown_executable_is_false():
    assert utils.cpp_variant_image_only("no-such-variant_sync") is False


def test_cpp_e2e_stream_skip_reads_the_variant_config_not_name_keywords():
    tree = ast.parse(_CPP_E2E.read_text(encoding="utf-8"), filename=str(_CPP_E2E))
    keyword_lists = [t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     for t in n.targets
                     if isinstance(t, ast.Name) and t.id == "_IMAGE_ONLY_KEYWORDS"]
    assert not keyword_lists, "test_e2e.py still skips streams by a name keyword list"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id in ("cpp_variant_image_only", "cpp_variant_image_only_reason")]
    assert calls, "test_e2e.py does not read the variant config's image_only"
