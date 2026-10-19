"""scripts/check_factory_uniqueness.py and scripts/check_model_registry.py on
the <task>/<family>/<variant> tree: class names are compared fully qualified
(each variant's factory lives in dxapp::v_<variant>), and an alias_of row is an
alias of another row, not a second claim on its .dxnn file.

Hermetic: every case builds a throwaway tree under tmp_path and points the
scripts at it; the real-tree cases only read.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UNIQUENESS = ROOT / "scripts" / "check_factory_uniqueness.py"
REGISTRY_CHECK = ROOT / "scripts" / "check_model_registry.py"

HEADER = """\
#ifndef {guard}
#define {guard}
#include "common/base/i_factory.hpp"
namespace dxapp {{
{open}class {cls} : public IDetectionFactory {{
 public:
    PreprocessorPtr createPreprocessor(int w, int h) override;
}};
{close}}}  // namespace dxapp
#endif
"""


def ns_of(variant):
    return "v_" + re.sub(r"[^0-9A-Za-z]", "_", variant)


def header(variant, cls="WidgetFactory", scoped=True):
    ns = ns_of(variant)
    return HEADER.format(guard=ns.upper() + "_FACTORY_HPP", cls=cls,
                         open="namespace {} {{\n".format(ns) if scoped else "",
                         close="}}  // namespace {}\n".format(ns) if scoped else "")


def write_cpp(root, task, family, variant, text):
    d = root / "cpp_example" / task / family / variant / "factory"
    d.mkdir(parents=True, exist_ok=True)
    (d / "{}_factory.hpp".format(variant)).write_text(text, encoding="utf-8")


def write_py(root, task, family, variant):
    d = root / "python_example" / task / family / variant / "factory"
    d.mkdir(parents=True, exist_ok=True)
    (d / "{}_factory.py".format(variant)).write_text("class F: pass\n", encoding="utf-8")


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=str(ROOT),
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False, timeout=300)


def uniqueness(tmp_path):
    return run(UNIQUENESS, "--cpp-root", tmp_path / "cpp_example")


# ── check_factory_uniqueness.py ─────────────────────────────────────────────


def test_the_same_class_in_two_variant_namespaces_is_unique(tmp_path):
    write_cpp(tmp_path, "object_detection", "yolo11", "yolo11-n_640x640", header("yolo11-n_640x640"))
    write_cpp(tmp_path, "object_detection", "yolo11", "yolo11-s_640x640", header("yolo11-s_640x640"))
    result = uniqueness(tmp_path)
    assert result.returncode == 0, result.stdout
    assert "factory class names unique (2 headers)" in result.stdout


def test_unscoped_copies_of_one_class_are_duplicates(tmp_path):
    """Before --variant-scope, every copy is dxapp::WidgetFactory: the ODR clash."""
    for v in ("yolo11-n_640x640", "yolo11-s_640x640"):
        write_cpp(tmp_path, "object_detection", "yolo11", v, header(v, scoped=False))
    result = uniqueness(tmp_path)
    assert result.returncode == 1, result.stdout
    assert ("DUPLICATE class ::dxapp::WidgetFactory: "
            "object_detection/yolo11/yolo11-n_640x640/factory/yolo11-n_640x640_factory.hpp, "
            "object_detection/yolo11/yolo11-s_640x640/factory/yolo11-s_640x640_factory.hpp"
            ) in result.stdout


def test_one_qualified_name_from_two_headers_is_a_duplicate(tmp_path):
    """"a-b" and "a_b" both scope into v_a_b."""
    write_cpp(tmp_path, "object_detection", "fam", "a-b", header("a-b"))
    write_cpp(tmp_path, "image_classification", "fam", "a_b", header("a_b"))
    result = uniqueness(tmp_path)
    assert result.returncode == 1, result.stdout
    assert "DUPLICATE class ::dxapp::v_a_b::WidgetFactory:" in result.stdout


def test_a_header_at_another_depth_is_still_compared(tmp_path):
    """A family-level factory/ left behind is scanned too, never skipped."""
    write_cpp(tmp_path, "object_detection", "yolo11", "yolo11-n_640x640",
              header("yolo11-n_640x640", scoped=False))
    stray = tmp_path / "cpp_example" / "object_detection" / "yolo11" / "factory"
    stray.mkdir(parents=True)
    (stray / "yolo11_factory.hpp").write_text(header("x", scoped=False), encoding="utf-8")
    result = uniqueness(tmp_path)
    assert result.returncode == 1, result.stdout
    assert "object_detection/yolo11/factory/yolo11_factory.hpp" in result.stdout


def test_the_real_tree_has_unique_qualified_factory_classes():
    result = run(UNIQUENESS)
    assert result.returncode == 0, result.stdout
    count = sum(1 for _ in (ROOT / "src" / "cpp_example").glob("*/*/*/factory/*_factory.hpp"))
    assert "factory class names unique ({} headers)".format(count) in result.stdout


# ── check_model_registry.py ─────────────────────────────────────────────────


def row(variant, task="object_detection", family="widget", **kwargs):
    r = {"model_name": variant, "variant": variant, "family": family, "task": task,
         "dxnn_file": variant + ".dxnn", "alias_of": None}
    r.update(kwargs)
    return r


def registry_check(tmp_path, rows, variants):
    for task, family, variant in variants:
        write_cpp(tmp_path, task, family, variant, header(variant))
        write_py(tmp_path, task, family, variant)
    registry = tmp_path / "model_registry.json"
    registry.write_text(json.dumps(rows), encoding="utf-8")
    return run(REGISTRY_CHECK, "--registry", registry, "--cpp-root", tmp_path / "cpp_example",
               "--py-root", tmp_path / "python_example")


TREE = [("object_detection", "widget", "widget-n"), ("object_detection", "widget", "widget-s")]


def test_the_registry_joins_the_tree_on_task_family_variant(tmp_path):
    rows = [row("widget-n", model_name="widgetn"), row("widget-s", model_name="widgets")]
    result = registry_check(tmp_path, rows, TREE)
    assert result.returncode == 0, result.stdout
    assert "consistent with the factory tree (2 entries, 2 variants)" in result.stdout


def test_an_alias_of_row_is_an_alias_not_a_duplicate_dxnn(tmp_path):
    """8d0b748's deit rows: two model_names on one variant and one .dxnn, the
    second marked alias_of the first."""
    rows = [row("widget-n", model_name="widgetn"), row("widget-s"),
            row("widget-n", model_name="widget_n_distilled", alias_of="widgetn")]
    result = registry_check(tmp_path, rows, TREE)
    assert result.returncode == 0, result.stdout
    assert "DUPLICATE DXNN" not in result.stdout


def test_two_rows_on_one_variant_without_alias_of_fail(tmp_path):
    rows = [row("widget-n", model_name="widgetn"), row("widget-s"),
            row("widget-n", model_name="widget_n_distilled")]
    result = registry_check(tmp_path, rows, TREE)
    assert result.returncode == 1, result.stdout
    assert "DUPLICATE VARIANT widget-n <- widgetn, widget_n_distilled" in result.stdout
    assert "DUPLICATE DXNN  widget-n.dxnn <- widgetn, widget_n_distilled" in result.stdout


@pytest.mark.parametrize("target,message", [
    ("nobody", 'BAD ALIAS "old": alias_of "nobody" names no registry row'),
    ("widget-s", 'BAD ALIAS "old": alias_of "widget-s" is variant widget-s, but this row is widget-n'),
])
def test_an_alias_of_must_name_a_row_of_the_same_variant(tmp_path, target, message):
    rows = [row("widget-n"), row("widget-s"), row("widget-n", model_name="old", alias_of=target)]
    result = registry_check(tmp_path, rows, TREE)
    assert result.returncode == 1, result.stdout
    assert message in result.stdout


def test_a_legacy_name_equal_to_another_variant_is_ambiguous(tmp_path):
    rows = [row("widget-n", model_name="widget-s"), row("widget-s")]
    result = registry_check(tmp_path, rows, TREE)
    assert result.returncode == 1, result.stdout
    assert 'AMBIGUOUS ALIAS "widget-s"' in result.stdout


def test_a_row_in_the_wrong_family_directory_has_no_factory(tmp_path):
    rows = [row("widget-n", family="gizmo"), row("widget-s")]
    result = registry_check(tmp_path, rows, TREE)
    assert result.returncode == 1, result.stdout
    assert "NO CPP FACTORY  widget-n (object_detection/gizmo/widget-n)" in result.stdout
    assert "NO PY FACTORY   widget-n (object_detection/gizmo/widget-n)" in result.stdout
    assert "UNREGISTERED    object_detection/widget/widget-n" in result.stdout


def test_the_real_registry_is_consistent_with_the_tree():
    entries = json.loads((ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))
    result = run(REGISTRY_CHECK)
    assert result.returncode == 0, result.stdout
    assert "({} entries, {} variants)".format(
        len(entries), len({e["variant"] for e in entries})) in result.stdout
