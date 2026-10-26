"""Tests for scripts/gen_model_registry.py.

The happy path is proved by the build itself: if the generator emitted the
wrong thing, its translation units would not compile. What a build cannot
prove is that the generator FAILS when it should, and those cases are the
whole reason this table is generated instead of hand-maintained:

  * a header declaring two I...Factory-deriving classes (ambiguous pick),
  * a header the shared factory-scanning rule cannot parse (silent skip),
  * a registry entry whose task disagrees with the factory's task directory,
  * a variant header left outside its own dxapp::v_<variant> namespace, and
    two headers declaring the same fully qualified class (ODR).

The factory tree is <task>/<family>/<variant>/factory/<variant>_factory.hpp,
each header scoped in dxapp::v_<variant> (generate_cpp_family_layout.py
--variant-scope). Each test builds a throwaway tree with --cpp-root/--registry
so it never touches the real tree; the few real-tree cases write only to
tmp_path.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "gen_model_registry.py"
REGISTRY = ROOT / "config" / "model_registry.json"

GOOD_HEADER = """\
#ifndef X_FACTORY_HPP
#define X_FACTORY_HPP
#include "common/base/i_factory.hpp"
namespace dxapp {
class WidgetFactory : public IDetectionFactory {
 public:
    PreprocessorPtr createPreprocessor(int w, int h) override;
};
}  // namespace dxapp
#endif
"""


def ns_of(variant):
    """The variant's namespace, as --variant-scope names it."""
    return "v_" + re.sub(r"[^0-9A-Za-z]", "_", variant)


def qualified(variant, cls):
    return "::dxapp::{}::{}".format(ns_of(variant), cls)


def scope(text, variant):
    """What --variant-scope writes: dxapp::v_<variant> inside namespace dxapp."""
    ns = ns_of(variant)
    return (text.replace("namespace dxapp {\n", "namespace dxapp {\nnamespace %s {\n" % ns, 1)
                .replace("}  // namespace dxapp", "}  // namespace %s\n}  // namespace dxapp" % ns, 1))


def _key(key):
    """(task, variant) is shorthand for a family named like its variant."""
    return key if len(key) == 3 else (key[0], key[1], key[1])


def write_tree(tmp_path, headers, entries, scoped=True, configs=None):
    """headers: {(task, [family,] variant): text}; entries: registry dicts;
    configs: {(task, [family,] variant): config.json dict}."""
    cpp_root = tmp_path / "cpp_example"
    for key, text in headers.items():
        task, family, variant = _key(key)
        factory_dir = cpp_root / task / family / variant / "factory"
        factory_dir.mkdir(parents=True, exist_ok=True)
        (factory_dir / "{}_factory.hpp".format(variant)).write_text(
            scope(text, variant) if scoped else text, encoding="utf-8")
    for key, config in (configs or {}).items():
        task, family, variant = _key(key)
        (cpp_root / task / family / variant).mkdir(parents=True, exist_ok=True)
        (cpp_root / task / family / variant / "config.json").write_text(
            json.dumps(config), encoding="utf-8")
    registry = tmp_path / "model_registry.json"
    registry.write_text(json.dumps(entries), encoding="utf-8")
    return cpp_root, registry


def run_generator(tmp_path, cpp_root, registry, *extra):
    return subprocess.run(
        [sys.executable, str(GENERATOR),
         "--cpp-root", str(cpp_root),
         "--registry", str(registry),
         "--out-dir", str(tmp_path / "generated"),
         "--docs", str(tmp_path / "graph_models.md")] + list(extra),
        cwd=str(ROOT), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def entry(variant, task, family=None, **kwargs):
    """A registry row as 8d0b748 writes them; the family defaults to the variant."""
    row = {"model_name": variant, "variant": variant, "family": family or variant,
           "task": task, "add_model_task": task, "dxnn_file": variant + ".dxnn",
           "input_width": 640, "input_height": 640, "alias_of": None, "published": True}
    row.update(kwargs)
    return row


def unit(tmp_path, stem):
    return (tmp_path / "generated" / "graph_registry_{}.cpp".format(stem)).read_text()


def load(module):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return __import__(module)
    finally:
        sys.path.pop(0)


def test_happy_path_emits_sources_and_docs(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget", "widget-n_640x640"): GOOD_HEADER},
        [entry("widget-n_640x640", "object_detection", "widget", model_name="widgetn")],
    )
    result = run_generator(tmp_path, cpp_root, registry, "--strict")

    assert result.returncode == 0, result.stdout
    emitted = unit(tmp_path, "object_detection")
    assert ("MakeTypedStage<::dxapp::v_widget_n_640x640::WidgetFactory, DetectionResult>"
            in emitted)
    assert ('#include "object_detection/widget/widget-n_640x640/factory/'
            'widget-n_640x640_factory.hpp"') in emitted
    for line in ('info.model_name = "widget-n_640x640";', 'info.variant = "widget-n_640x640";',
                 'info.family = "widget";', 'info.task = "object_detection";',
                 'info.published = true;'):
        assert line in emitted
    assert (tmp_path / "graph_models.md").is_file()


def test_second_factory_class_in_one_header_is_a_hard_error(tmp_path):
    """Ambiguity must be an error, not a silent first-match pick."""
    ambiguous = GOOD_HEADER.replace(
        "}  // namespace dxapp",
        "class OtherFactory : public IClassificationFactory {\n"
        " public:\n"
        "    PreprocessorPtr createPreprocessor(int w, int h) override;\n"
        "};\n"
        "}  // namespace dxapp",
    )
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget"): ambiguous},
        [entry("widget", "object_detection")],
    )
    result = run_generator(tmp_path, cpp_root, registry)

    assert result.returncode != 0, result.stdout
    assert "AMBIGUOUS" in result.stdout
    assert "WidgetFactory" in result.stdout and "OtherFactory" in result.stdout


def test_private_wrapper_class_before_the_factory_still_resolves(tmp_path):
    """The shared rule scans every class, not just the first one.

    Three real headers in this tree declare a private wrapper ahead of the
    factory; a naive 'first class in the file' match picks the wrapper.
    """
    wrapped = GOOD_HEADER.replace(
        "class WidgetFactory",
        "class WidgetPostprocessorAdapter : public IPostprocessor<DetectionResult> {\n"
        " public:\n"
        "    std::vector<DetectionResult> process(const dxrt::TensorPtrs&,\n"
        "                                         const PreprocessContext&) override;\n"
        "};\n"
        "class WidgetFactory",
    )
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget"): wrapped},
        [entry("widget", "object_detection")],
    )
    result = run_generator(tmp_path, cpp_root, registry, "--strict")

    assert result.returncode == 0, result.stdout
    emitted = unit(tmp_path, "object_detection")
    assert "MakeTypedStage<{}, DetectionResult>".format(qualified("widget", "WidgetFactory")) in emitted
    assert "WidgetPostprocessorAdapter" not in emitted


def test_unparseable_header_is_a_hard_error(tmp_path):
    """A header the rule cannot parse fails the build, never a silent skip."""
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget"): GOOD_HEADER,
         ("object_detection", "broken"): "// no class declaration at all\n"},
        [entry("widget", "object_detection"), entry("broken", "object_detection")],
    )
    result = run_generator(tmp_path, cpp_root, registry)

    assert result.returncode != 0, result.stdout
    assert "UNPARSEABLE" in result.stdout
    assert "broken_factory.hpp" in result.stdout


def test_scan_reads_task_family_variant_and_qualified_class(tmp_path):
    """The tree is <task>/<family>/<variant>/factory/<variant>_factory.hpp and the
    class is read inside the variant's namespace, then emitted fully qualified."""
    header = GOOD_HEADER.replace("WidgetFactory", "FFactory")
    cpp_root, _ = write_tree(tmp_path, {("t", "f", "v-1_8x8"): header}, [])
    found, errors = load("gen_model_registry").scan_factories(cpp_root)

    assert errors == []
    assert sorted(found) == ["v-1_8x8"]
    info = found["v-1_8x8"]
    assert (info.task, info.family, info.variant) == ("t", "f", "v-1_8x8")
    assert info.cls == "::dxapp::v_v_1_8x8::FFactory"
    assert info.header == "t/f/v-1_8x8/factory/v-1_8x8_factory.hpp"


def test_registry_joins_on_variant_and_checks_task(tmp_path):
    """The model key is the variant; model_name is only a legacy alias. TASK
    MISMATCH compares the registry's task (and family) with the directory,
    because TypedStage reads <task>/<family>/<variant>/config.json."""
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget", "widget-s_640x640"): GOOD_HEADER},
        [entry("widget-s_640x640", "object_detection", "widget", model_name="widgets",
               add_model_task="detection")])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout
    emitted = unit(tmp_path, "object_detection")
    assert 'info.model_name = "widget-s_640x640";' in emitted
    assert 'info.model_name = "widgets"' not in emitted

    for row, wrong in ((entry("widget-s_640x640", "image_classification", "widget"),
                        "image_classification"),
                       (entry("widget-s_640x640", "object_detection", "gizmo"), "gizmo")):
        bad = tmp_path / wrong
        bad_root, bad_registry = write_tree(
            bad, {("object_detection", "widget", "widget-s_640x640"): GOOD_HEADER}, [row])
        result = run_generator(bad, bad_root, bad_registry)
        assert result.returncode != 0, result.stdout
        assert "TASK MISMATCH widget-s_640x640" in result.stdout
        assert wrong in result.stdout and "object_detection/widget/" in result.stdout


def test_duplicate_is_on_qualified_names(tmp_path):
    """Two variants of one family share a class name: each is scoped, so that is
    no duplicate. Two headers whose QUALIFIED names coincide are."""
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget", "widget-n_640x640"): GOOD_HEADER,
         ("object_detection", "widget", "widget-s_640x640"): GOOD_HEADER},
        [entry("widget-n_640x640", "object_detection", "widget"),
         entry("widget-s_640x640", "object_detection", "widget")])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout
    emitted = unit(tmp_path, "object_detection")
    assert "MakeTypedStage<::dxapp::v_widget_n_640x640::WidgetFactory," in emitted
    assert "MakeTypedStage<::dxapp::v_widget_s_640x640::WidgetFactory," in emitted

    # "widget-n" and "widget_n" both scope into v_widget_n: one qualified name.
    clash = tmp_path / "clash"
    clash_root, clash_registry = write_tree(
        clash,
        {("object_detection", "widget", "widget-n"): GOOD_HEADER,
         ("image_classification", "widget", "widget_n"): GOOD_HEADER},
        [entry("widget-n", "object_detection", "widget"),
         entry("widget_n", "image_classification", "widget")])
    result = run_generator(clash, clash_root, clash_registry)
    assert result.returncode != 0, result.stdout
    assert "DUPLICATE class ::dxapp::v_widget_n::WidgetFactory" in result.stdout
    assert ("image_classification/widget/widget_n/factory/widget_n_factory.hpp, "
            "object_detection/widget/widget-n/factory/widget-n_factory.hpp") in result.stdout


UNUSED_PARAMETER_HEADER = GOOD_HEADER.replace(
    "PreprocessorPtr createPreprocessor(int w, int h) override;",
    "PreprocessorPtr createPreprocessor(int w, int h) override { return nullptr; }")


def test_missing_namespace_is_refused_at_configure_with_the_fix_command(tmp_path):
    """An unscoped variant header would link as dxapp::WidgetFactory next to every
    other copy: configure stops and names the fix. An unused create*()
    parameter is not a configure error (--variant-scope --check reports it)."""
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget", "widget-n_640x640"): GOOD_HEADER},
        [entry("widget-n_640x640", "object_detection", "widget")], scoped=False)
    result = run_generator(tmp_path, cpp_root, registry)
    assert result.returncode != 0, result.stdout
    assert ("object_detection/widget/widget-n_640x640/factory/widget-n_640x640_factory.hpp: "
            "not scoped in namespace dxapp::v_widget_n_640x640") in result.stdout
    assert ("python3 scripts/generate_cpp_family_layout.py --variant-scope {}".format(cpp_root)
            in result.stdout)
    assert not (tmp_path / "generated").exists()

    loose = tmp_path / "loose"
    loose_root, loose_registry = write_tree(
        loose, {("object_detection", "widget", "widget-n_640x640"): UNUSED_PARAMETER_HEADER},
        [entry("widget-n_640x640", "object_detection", "widget")])
    result = run_generator(loose, loose_root, loose_registry, "--strict")
    assert result.returncode == 0, result.stdout
    assert "never used" not in result.stdout


def test_registry_entry_without_a_factory_is_not_ready_but_not_fatal(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget"): GOOD_HEADER},
        [entry("widget", "object_detection"), entry("ghost", "object_detection")],
    )
    result = run_generator(tmp_path, cpp_root, registry)

    assert result.returncode == 0, result.stdout
    emitted = (tmp_path / "generated" / "graph_registry_object_detection.cpp").read_text()
    assert 'info.model_name = "ghost"' in emitted
    assert "info.ready = false" in emitted
    assert "no factory" in emitted


def test_name_drift_fails_only_under_strict(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget"): GOOD_HEADER},
        [entry("widget", "object_detection"), entry("ghost", "object_detection")],
    )
    result = run_generator(tmp_path, cpp_root, registry, "--strict")

    assert result.returncode != 0, result.stdout
    assert "ghost" in result.stdout


def test_generator_agrees_with_check_factory_uniqueness_on_the_real_tree(tmp_path):
    """One scanning rule, not two.

    scripts/check_factory_uniqueness.py owns parse_factory_class(es); the
    generator imports it rather than re-implementing it. This pins that the
    two agree on every one of the real factory headers, so a drift between
    them cannot go unnoticed.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_factory_uniqueness as guard
        import gen_model_registry as gen
    finally:
        sys.path.pop(0)

    assert gen.parse_factory_classes is guard.parse_factory_classes

    found, _ = gen.scan_factories(ROOT / "src" / "cpp_example")
    by_class, errors = guard.collect()
    assert errors == []
    assert sorted(info.cls for info in found.values()) == sorted(by_class)


def test_download_name_comes_from_the_manifest(tmp_path):
    """The build-time join PrepareGraph used at run time: basename(dxnn_url) -> name."""
    gadget = GOOD_HEADER.replace("WidgetFactory", "GadgetFactory")
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget"): GOOD_HEADER,
         ("object_detection", "gadget"): gadget},
        [entry("widget", "object_detection"), entry("gadget", "object_detection")],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"name": "Widget-640", "dxnn_url": "https://example.invalid/m/widget.dxnn"},
        {"name": 7, "dxnn_url": "https://example.invalid/m/gadget.dxnn"},  # malformed: skipped
        "not an entry",
    ]), encoding="utf-8")
    result = run_generator(tmp_path, cpp_root, registry, "--manifest", str(manifest))

    assert result.returncode == 0, result.stdout
    emitted = (tmp_path / "generated" / "graph_registry_object_detection.cpp").read_text()
    assert 'info.download_name = "Widget-640";' in emitted
    assert emitted.count("info.download_name") == 1


def test_an_unsafe_manifest_name_is_dropped_not_fatal(tmp_path):
    """Final review M2: the manifest is not ours, so a name that cannot go into
    a C++ string literal costs that model its zoo name (the registry name is
    printed instead), not the whole configure step."""
    gadget = GOOD_HEADER.replace("WidgetFactory", "GadgetFactory")
    cpp_root, registry = write_tree(
        tmp_path,
        {("object_detection", "widget"): GOOD_HEADER,
         ("object_detection", "gadget"): gadget},
        [entry("widget", "object_detection"), entry("gadget", "object_detection")],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"name": 'Widget"640', "dxnn_url": "https://example.invalid/m/widget.dxnn"},
        {"name": "Gadget-1", "dxnn_url": "https://example.invalid/m/gadget.dxnn"},
    ]), encoding="utf-8")
    result = run_generator(tmp_path, cpp_root, registry, "--manifest", str(manifest))

    assert result.returncode == 0, result.stdout
    assert "WARNING manifest name 'Widget\"640' for widget.dxnn" in result.stdout
    emitted = (tmp_path / "generated" / "graph_registry_object_detection.cpp").read_text()
    assert 'info.download_name = "Gadget-1";' in emitted
    assert emitted.count("info.download_name") == 1  # widget: the registry name


def test_configure_shows_generator_warnings_on_success(tmp_path):
    """Final review M1: configure captures the generator's stderr in
    GRAPH_CODEGEN_STDERR, which was printed only on failure, so a warning from
    a successful run (an unsafe manifest name, an unreadable manifest) was
    never seen. The real execute_process block, run by cmake -P against a
    stand-in generator that warns and succeeds."""
    import re
    import shutil
    cmake = shutil.which("cmake")
    assert cmake is not None, "cmake is needed to build this project"
    text = (ROOT / "src" / "cpp_example" / "CMakeLists.txt").read_text()
    block = re.search(r"execute_process\(\s*COMMAND \$\{Python3_EXECUTABLE\} "
                      r"\$\{PROJECT_ROOT\}/scripts/gen_model_registry\.py.*?\nendif\(\)\n",
                      text, re.S)
    assert block is not None, "the codegen execute_process block moved"
    fake_root = tmp_path / "root"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "scripts" / "gen_model_registry.py").write_text(
        "import sys\nprint('WARNING stand-in generator note', file=sys.stderr)\n"
        "print('a.cpp')\n", encoding="utf-8")
    script = tmp_path / "codegen.cmake"
    script.write_text(
        'set(Python3_EXECUTABLE "{}")\nset(PROJECT_ROOT "{}")\n'
        'set(GRAPH_GENERATED_DIR "{}")\n'.format(sys.executable, fake_root, tmp_path / "gen")
        + block.group(0), encoding="utf-8")
    completed = subprocess.run([cmake, "-P", str(script)], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=False, timeout=120)
    assert completed.returncode == 0, completed.stdout
    assert "WARNING stand-in generator note" in completed.stdout, completed.stdout
    assert "CMake Warning" in completed.stdout, completed.stdout


def test_check_docs_passes_on_a_fresh_doc_and_fails_on_a_stale_one(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): GOOD_HEADER},
        [entry("widget", "object_detection")])
    assert run_generator(tmp_path, cpp_root, registry).returncode == 0

    fresh = run_generator(tmp_path, cpp_root, registry, "--check-docs")
    assert fresh.returncode == 0, fresh.stdout

    doc = tmp_path / "graph_models.md"
    doc.write_text(doc.read_text() + "| `ghost` | object_detection | boxes | frame | 1x1 | yes |\n")
    stale = run_generator(tmp_path, cpp_root, registry, "--check-docs")
    assert stale.returncode == 1, stale.stdout
    assert "STALE" in stale.stdout and "ghost" in stale.stdout
    assert "--docs-only" in stale.stdout

    doc.unlink()
    assert run_generator(tmp_path, cpp_root, registry, "--check-docs").returncode == 1


def test_check_docs_writes_nothing(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): GOOD_HEADER},
        [entry("widget", "object_detection")])
    result = run_generator(tmp_path, cpp_root, registry, "--check-docs")
    assert result.returncode == 1, result.stdout  # no doc yet: stale
    assert not (tmp_path / "graph_models.md").exists()
    assert not (tmp_path / "generated").exists()


def test_docs_only_writes_the_doc_and_no_sources(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): GOOD_HEADER},
        [entry("widget", "object_detection")])
    result = run_generator(tmp_path, cpp_root, registry, "--docs-only")
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "graph_models.md").is_file()
    assert not (tmp_path / "generated").exists()


def _main_in_fake_root(monkeypatch, tmp_path, *argv):
    """main() with ROOT patched to an empty fake repository (the defaults
    that name the tracked doc point there), on a one-model tree."""
    gen = load("gen_model_registry")
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): GOOD_HEADER},
        [entry("widget", "object_detection")])
    fake_root = tmp_path / "root"
    (fake_root / "docs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gen, "ROOT", fake_root)
    monkeypatch.setattr(sys, "argv", ["gen_model_registry.py", "--cpp-root", str(cpp_root),
                                      "--registry", str(registry)] + list(argv))
    return gen.main(), fake_root / "docs" / "graph_models.md"


def test_a_plain_run_writes_the_doc_under_out_dir_not_the_tracked_one(tmp_path, monkeypatch):
    """Only --docs-only writes the tracked docs/graph_models.md: a run that
    emits sources and names no --docs writes its table next to them."""
    out_dir = tmp_path / "generated"
    code, tracked = _main_in_fake_root(monkeypatch, tmp_path, "--out-dir", str(out_dir))
    assert code == 0
    assert (out_dir / "graph_models.md").is_file()
    assert not tracked.exists()


def test_docs_only_and_check_docs_default_to_the_tracked_doc(tmp_path, monkeypatch):
    code, tracked = _main_in_fake_root(monkeypatch, tmp_path, "--docs-only")
    assert code == 0 and tracked.is_file()
    code, _ = _main_in_fake_root(monkeypatch, tmp_path, "--check-docs")
    assert code == 0


def test_docs_count_only_what_a_camera_graph_can_run(tmp_path):
    """U-19: the doc's count is what a graph can run. A boxes3d model is ready
    but refused in every graph (its input is a LiDAR point cloud; a graph's
    sources are camera frames), and an unpublished model has no .dxnn to
    download; the table says so and the summary counts them apart."""
    gen = load("gen_model_registry")
    base = {"task": "t", "family": "f", "shape": "kBoxes", "contract": "kFullFrame",
            "input_width": 8, "input_height": 8, "ports": [], "ready": True,
            "reason": "", "published": True}
    rows = [dict(base, model_name="cam_8x8"),
            dict(base, model_name="lidar_8x8", shape="kBoxes3d"),
            dict(base, model_name="pending_8x8", published=False),
            dict(base, model_name="anomaly_8x8", ready=False, reason="not graph-ready")]
    doc = gen.render_docs(rows, [])

    assert "| model | task | produces | consumes | input | published | ready |" in doc
    assert "| `cam_8x8` | t | boxes | frame | 8x8 | yes | yes |" in doc
    assert "| `pending_8x8` | t | boxes | frame | 8x8 | NO | yes |" in doc
    assert ("| `lidar_8x8` | t | boxes3d | frame | 8x8 | yes | yes, but refused in "
            "every graph: boxes3d needs a LiDAR point-cloud source |") in doc
    assert "| `anomaly_8x8` | t | boxes | frame | 8x8 | yes | NO - not graph-ready |" in doc
    assert ("4 models registered, 3 ready. 2 run in a graph: `lidar_8x8` is refused "
            "(boxes3d). 1 ready model has no published `.dxnn`: `pending_8x8`.") in doc
    assert "usable in a graph." not in doc


def test_the_tracked_graph_models_doc_is_fresh():
    """The guard, on the real tree: configure no longer rewrites the doc, so
    only this notices when a registry change did not regenerate it."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_graph_models_doc.py")],
        cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, timeout=300)
    assert completed.returncode == 0, completed.stdout


def test_configure_writes_the_doc_into_the_build_tree():
    text = (ROOT / "src" / "cpp_example" / "CMakeLists.txt").read_text()
    assert "--docs ${GRAPH_GENERATED_DIR}/graph_models.md" in text
    assert "${PROJECT_ROOT}/docs/graph_models.md" not in text
    assert "${PROJECT_ROOT}/scripts/modelzoo_manifest.json" in text


TRAIT_HEADER = """\
#ifndef P_FACTORY_HPP
#define P_FACTORY_HPP
#include "common/base/i_factory.hpp"
namespace dxapp {{
class PointsFactory : public IPoseFactory {{
 public:
    PreprocessorPtr createPreprocessor(int w, int h) override;
{traits}
}};
}}  // namespace dxapp
#endif
"""




def _points(tmp_path, traits):
    cpp_root, registry = write_tree(
        tmp_path, {("pose_estimation", "points"): TRAIT_HEADER.format(traits=traits)},
        [entry("points", "pose_estimation")])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    unit = tmp_path / "generated" / "graph_registry_pose_estimation.cpp"
    return result, (unit.read_text() if unit.is_file() else "")


def test_a_factory_overrides_its_input_contract_in_one_line(tmp_path):
    result, emitted = _points(
        tmp_path, "    static constexpr GraphInput graphInput() { return GraphInput::kEither; }")
    assert result.returncode == 0, result.stdout
    assert "info.input_contract = InputContract::kEither;" in emitted
    assert "static_assert(detail::DeclaredGraphInput<::dxapp::v_points::PointsFactory>(0) == GraphInput::kEither," in emitted
    doc = (tmp_path / "graph_models.md").read_text()
    assert "| `points` | pose_estimation | keypoints | either | 640x640 | yes |" in doc


def test_without_a_declaration_the_interface_default_holds_and_is_asserted(tmp_path):
    result, emitted = _points(tmp_path, "")
    assert result.returncode == 0, result.stdout
    assert "info.input_contract = InputContract::kFullFrame;" in emitted
    assert "static_assert(detail::DeclaredGraphInput<::dxapp::v_points::PointsFactory>(0) == GraphInput::kDefault," in emitted
    assert 'static_assert(detail::SameText(detail::DeclaredGraphPorts<::dxapp::v_points::PointsFactory>(0), ""),' in emitted
    assert "info.ports.push_back" not in emitted


def test_a_trait_the_parser_cannot_read_is_a_hard_error(tmp_path):
    result, _ = _points(
        tmp_path, "    static constexpr GraphInput graphInput() {\n        return kInput;\n    }")
    assert result.returncode == 1
    assert "UNPARSEABLE TRAIT pose_estimation/points/points/factory/points_factory.hpp" in result.stdout
    # The declaration may span lines; what the parser needs is one of its form.
    assert "declare it once, in the form `static constexpr GraphInput graphInput()" in result.stdout
    assert "one line" not in result.stdout


def test_a_trait_declared_over_several_lines_parses(tmp_path):
    result, emitted = _points(
        tmp_path, "    static constexpr GraphInput graphInput() {\n        return GraphInput::kEither;\n    }")
    assert result.returncode == 0, result.stdout
    assert "info.input_contract = InputContract::kEither;" in emitted


def test_a_factory_declares_extra_output_ports(tmp_path):
    result, emitted = _points(
        tmp_path, '    static constexpr const char* graphPorts() { return "descriptors"; }')
    assert result.returncode == 0, result.stdout
    assert 'info.ports.push_back(PortInfo("descriptors", Shape::kDenseMap));' in emitted
    assert 'static_assert(detail::SameText(detail::DeclaredGraphPorts<::dxapp::v_points::PointsFactory>(0), "descriptors"),' in emitted


def test_an_unknown_port_is_a_hard_error(tmp_path):
    result, _ = _points(tmp_path, '    static constexpr const char* graphPorts() { return "wings"; }')
    assert result.returncode == 1
    assert 'UNKNOWN PORT pose_estimation/points/points/factory/points_factory.hpp: PoseResult has no port "wings"' in result.stdout


CONFIG_HEADER = """\
#ifndef X_FACTORY_HPP
#define X_FACTORY_HPP
#include "common/base/i_factory.hpp"
namespace dxapp {{
class WidgetFactory : public IDetectionFactory {{
 public:
    PreprocessorPtr createPreprocessor(int w, int h) override;
    void loadConfig(const ModelConfig& config) override {{
        score_ = config.get<float>("score_threshold", score_);
        names_ = config.get_string_list("class_names");
        label_ = config.get<std::string>("label_file", label_);
        {extra}
    }}
}};
}}  // namespace dxapp
#endif
"""


def test_text_params_come_from_the_factory_header(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): CONFIG_HEADER.format(extra="")},
        [entry("widget", "object_detection")])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout
    emitted = (tmp_path / "generated" / "graph_registry_object_detection.cpp").read_text()
    assert 'info.text_params.push_back(ParamInfo("class_names", ParamInfo::kTextList));' in emitted
    assert 'info.text_params.push_back(ParamInfo("label_file", ParamInfo::kText));' in emitted
    assert emitted.index('"class_names"') < emitted.index('"label_file"')
    assert "score_threshold" not in emitted


def test_a_key_read_as_a_number_and_as_a_list_is_a_hard_error(tmp_path):
    header = CONFIG_HEADER.format(extra='top_ = config.get<int>("class_names", top_);')
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): header}, [entry("widget", "object_detection")])
    result = run_generator(tmp_path, cpp_root, registry)
    assert result.returncode == 1
    assert 'CONFLICTING PARAM object_detection/widget/widget/factory/widget_factory.hpp: "class_names"' in result.stdout


@pytest.mark.parametrize("number", ["unsigned", "size_t", "std::size_t", "long"])
def test_a_key_read_as_a_number_and_as_text_is_a_hard_error(tmp_path, number):
    header = CONFIG_HEADER.format(extra='n_ = config.get<{}>("label_file", n_);'.format(number))
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): header}, [entry("widget", "object_detection")])
    result = run_generator(tmp_path, cpp_root, registry)
    assert result.returncode == 1
    assert ('CONFLICTING PARAM object_detection/widget/widget/factory/widget_factory.hpp: '
            '"label_file" is read as a number and as text') in result.stdout


def test_a_key_read_as_text_and_as_a_list_is_a_hard_error(tmp_path):
    header = CONFIG_HEADER.format(extra='more_ = config.get_string_list("label_file");')
    cpp_root, registry = write_tree(
        tmp_path, {("object_detection", "widget"): header}, [entry("widget", "object_detection")])
    result = run_generator(tmp_path, cpp_root, registry)
    assert result.returncode == 1
    assert ('CONFLICTING PARAM object_detection/widget/widget/factory/widget_factory.hpp: '
            '"label_file" is read as text and as a list') in result.stdout


RESTORE_HEADER = GOOD_HEADER.replace("WidgetFactory : public IDetectionFactory",
                                     "SharpFactory : public IRestorationFactory")


def test_restoration_models_get_the_restoration_stage_maker(tmp_path):
    cpp_root, registry = write_tree(
        tmp_path, {("super_resolution", "sharp"): RESTORE_HEADER}, [entry("sharp", "super_resolution")])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout
    emitted = (tmp_path / "generated" / "graph_registry_super_resolution.cpp").read_text()
    assert "registry->Add(info, &MakeRestorationStage<::dxapp::v_sharp::SharpFactory>);" in emitted
    assert '#include "common/registry/tiled_sr_stage.hpp"' in emitted
    det = tmp_path / "det"
    det_root, det_registry = write_tree(
        det, {("object_detection", "widget"): GOOD_HEADER}, [entry("widget", "object_detection")])
    assert run_generator(det, det_root, det_registry).returncode == 0
    detection = (det / "generated" / "graph_registry_object_detection.cpp").read_text()
    assert "MakeTypedStage<::dxapp::v_widget::WidgetFactory, DetectionResult>" in detection
    assert "tiled_sr_stage.hpp" not in detection


def test_the_model_table_names_extra_ports(tmp_path):
    header = TRAIT_HEADER.format(traits='    static constexpr const char* graphPorts() { return "descriptors"; }')
    cpp_root, registry = write_tree(tmp_path, {("pose_estimation", "points"): header},
                                    [entry("points", "pose_estimation")])
    assert run_generator(tmp_path, cpp_root, registry).returncode == 0
    doc = (tmp_path / "graph_models.md").read_text()
    assert "| `points` | pose_estimation | keypoints + descriptors (densemap) | frame |" in doc


def test_the_produces_cell_groups_ports_only_when_they_share_a_shape():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import gen_model_registry as gen
    finally:
        sys.path.pop(0)
    shared = {"shape": "kBoxes", "ports": [("drivable", "kLabelMap"), ("lane", "kLabelMap")]}
    mixed = {"shape": "kKeypoints", "ports": [("descriptors", "kDenseMap"), ("pose", "kVector")]}
    assert gen.produces_cell(shared) == "boxes + drivable, lane (labelmap)"
    assert gen.produces_cell(mixed) == "keypoints + descriptors (densemap), pose (vector)"
    assert gen.produces_cell({"shape": "kScores", "ports": []}) == "scores"


@pytest.mark.parametrize("manifest_text", [None, json.dumps({"name": "Widget-640"})],
                         ids=["missing", "not-an-array"])
def test_an_unusable_manifest_warns_and_falls_back_to_registry_names(tmp_path, manifest_text):
    """SP1 leave item: a manifest that cannot be read, or is not a JSON
    array, costs every model its zoo name (the registry name is printed),
    with a WARNING - never the configure step."""
    cpp_root, registry = write_tree(tmp_path, {("object_detection", "widget"): GOOD_HEADER},
                                    [entry("widget", "object_detection")])
    manifest = tmp_path / "manifest.json"
    if manifest_text is not None:
        manifest.write_text(manifest_text, encoding="utf-8")
    result = run_generator(tmp_path, cpp_root, registry, "--manifest", str(manifest))
    assert result.returncode == 0, result.stdout
    assert "WARNING" in result.stdout and "manifest.json" in result.stdout, result.stdout
    emitted = (tmp_path / "generated" / "graph_registry_object_detection.cpp").read_text()
    assert "info.download_name" not in emitted


# ── Model key = variant; old names are aliases (R6) ─────────────────────────


def test_legacy_name_and_alias_of_resolve_to_the_variant(tmp_path):
    """A row's model_name, when it differs from its variant, is the old name of
    that variant; an alias_of row names another row and resolves to ITS variant.
    Neither is a second model: one ModelInfo per variant."""
    cpp_root, registry = write_tree(
        tmp_path, {("image_classification", "deit", "deit-b_384x384"): GOOD_HEADER.replace(
            "IDetectionFactory", "IClassificationFactory")},
        [entry("deit-b_384x384", "image_classification", "deit", model_name="deitbase384"),
         entry("deit-b_384x384", "image_classification", "deit",
               model_name="deit_base384_distilled", alias_of="deitbase384")])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout

    emitted = unit(tmp_path, "image_classification")
    assert emitted.count("registry->Add(info,") == 1
    assert 'info.model_name = "deit-b_384x384";' in emitted
    everything = (tmp_path / "generated" / "graph_registry_all.cpp").read_text()
    assert ('registry->AddAlias("deitbase384", "deit-b_384x384", ModelAlias::kLegacyName);'
            in everything)
    assert ('registry->AddAlias("deit_base384_distilled", "deit-b_384x384", '
            'ModelAlias::kAliasOf);') in everything
    assert everything.index("RegisterGraphModels_image_classification(registry);") \
        < everything.index("registry->AddAlias(")

    doc = (tmp_path / "graph_models.md").read_text()
    assert "| `deit-b_384x384` | image_classification |" in doc
    assert "| `deitbase384` | `deit-b_384x384` | old name |" in doc
    assert "| `deit_base384_distilled` | `deit-b_384x384` | alias of |" in doc


@pytest.mark.parametrize("case", ["legacy", "alias_of", "twice", "unknown", "variant"])
def test_alias_equal_to_another_variant_is_ambiguous(tmp_path, case):
    """An old name that is also another row's variant, or one name for two
    variants, would resolve by guesswork: the generator refuses it. So does an
    alias_of that names no row, or a row of another variant."""
    headers = {("object_detection", "widget", "widget-n"): GOOD_HEADER,
               ("object_detection", "widget", "widget-s"): GOOD_HEADER}
    rows = {
        "legacy": [entry("widget-n", "object_detection", "widget", model_name="widget-s"),
                   entry("widget-s", "object_detection", "widget")],
        "alias_of": [entry("widget-n", "object_detection", "widget"),
                     entry("widget-s", "object_detection", "widget"),
                     entry("widget-n", "object_detection", "widget", model_name="widget-s",
                           alias_of="widget-n")],
        "twice": [entry("widget-n", "object_detection", "widget", model_name="widget"),
                  entry("widget-s", "object_detection", "widget"),
                  entry("widget-s", "object_detection", "widget", model_name="widget",
                        alias_of="widget-s")],
        "unknown": [entry("widget-n", "object_detection", "widget"),
                    entry("widget-s", "object_detection", "widget"),
                    entry("widget-n", "object_detection", "widget", model_name="old",
                          alias_of="nobody")],
        "variant": [entry("widget-n", "object_detection", "widget"),
                    entry("widget-s", "object_detection", "widget"),
                    entry("widget-n", "object_detection", "widget", model_name="old",
                          alias_of="widget-s")],
    }[case]
    cpp_root, registry = write_tree(tmp_path, headers, rows)
    result = run_generator(tmp_path, cpp_root, registry)
    assert result.returncode != 0, result.stdout
    expected = {"legacy": 'AMBIGUOUS ALIAS "widget-s"',
                "alias_of": 'AMBIGUOUS ALIAS "widget-s"',
                "twice": 'AMBIGUOUS ALIAS "widget"',
                "unknown": 'BAD ALIAS "old"',
                "variant": 'BAD ALIAS "old"'}[case]
    assert expected in result.stdout
    assert not (tmp_path / "generated").exists()


# ── Graph traits come from a table, not from teammate headers (R5) ──────────


POSE_HEADER = TRAIT_HEADER


def test_graph_traits_table(tmp_path):
    """vitpose and dark_hrnet take a crop or a frame; superpoint has a
    descriptors port. The headers declare nothing, so the static_asserts pin
    kDefault/"" while ModelInfo carries the table's value. A header that does
    declare a trait must agree with the table."""
    headers = {("pose_estimation", "vitpose", "vitpose-s_256x192"): POSE_HEADER.format(traits=""),
               ("pose_estimation", "dark_hrnet", "dark-hrnet-w32_256x192"): POSE_HEADER.format(traits=""),
               ("keypoint_detection", "superpoint", "superpoint_480x640"): POSE_HEADER.format(traits=""),
               ("pose_estimation", "yolov8_pose", "yolov8-n-pose_640x640"): POSE_HEADER.format(traits="")}
    cpp_root, registry = write_tree(
        tmp_path, headers, [entry(v, t, f) for (t, f, v) in headers])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout

    pose = unit(tmp_path, "pose_estimation")
    blocks = {m.group(1): m.group(0) for m in re.finditer(
        r'  \{\n    ModelInfo info;\n    info.model_name = "([^"]+)";.*?\n  \}', pose, re.S)}
    for variant in ("vitpose-s_256x192", "dark-hrnet-w32_256x192"):
        assert "info.input_contract = InputContract::kEither;" in blocks[variant]
        assert "DeclaredGraphInput<{}>(0) == GraphInput::kDefault,".format(
            qualified(variant, "PointsFactory")) in blocks[variant]
    assert "info.input_contract = InputContract::kFullFrame;" in blocks["yolov8-n-pose_640x640"]
    keypoints = unit(tmp_path, "keypoint_detection")
    assert 'info.ports.push_back(PortInfo("descriptors", Shape::kDenseMap));' in keypoints
    assert 'DeclaredGraphPorts<{}>(0), ""),'.format(
        qualified("superpoint_480x640", "PointsFactory")) in keypoints
    doc = (tmp_path / "graph_models.md").read_text()
    assert "| `vitpose-s_256x192` | pose_estimation | keypoints | either | 640x640 | yes |" in doc
    assert "| `superpoint_480x640` | keypoint_detection | keypoints + descriptors (densemap) |" in doc

    agree = tmp_path / "agree"
    agree_root, agree_registry = write_tree(agree, {
        ("pose_estimation", "vitpose", "vitpose-s_256x192"): POSE_HEADER.format(
            traits="    static constexpr GraphInput graphInput() { return GraphInput::kEither; }")},
        [entry("vitpose-s_256x192", "pose_estimation", "vitpose")])
    result = run_generator(agree, agree_root, agree_registry)
    assert result.returncode == 0, result.stdout
    assert "== GraphInput::kEither," in unit(agree, "pose_estimation")

    clash = tmp_path / "clash"
    clash_root, clash_registry = write_tree(clash, {
        ("pose_estimation", "vitpose", "vitpose-s_256x192"): POSE_HEADER.format(
            traits="    static constexpr GraphInput graphInput() { return GraphInput::kFullFrame; }")},
        [entry("vitpose-s_256x192", "pose_estimation", "vitpose")])
    result = run_generator(clash, clash_root, clash_registry)
    assert result.returncode != 0, result.stdout
    assert ("TRAIT MISMATCH pose_estimation/vitpose/vitpose-s_256x192/factory/"
            "vitpose-s_256x192_factory.hpp") in result.stdout


# ── Resources a model needs beside its .dxnn (R9) ───────────────────────────


EMBEDDING_HEADER = GOOD_HEADER.replace("IDetectionFactory", "IEmbeddingFactory")
ANOMALY_HEADER = GOOD_HEADER.replace("IDetectionFactory", "IAnomalyDetectionFactory")


def test_gallery_and_companion_resources(tmp_path):
    """A gallery file comes from the variant's config.json (the nested "config"
    wins); EfficientAD's companion engines from its roles; the CLIP prompt bank
    is Python-only, and the zero-shot rows say so."""
    vpr = ("visual_place_recognition", "eigenplaces", "eigenplaces-resnet18_512x512")
    ad = ("anomaly_detection", "efficientad", "efficientad-m-teacher_256x256")
    clip = ("zero_shot_image_classification", "clip", "clip-img_vit-b32_256x256_x")
    plain = ("image_retrieval", "clip_rn50", "clip-text_resnet50_77x512_openai")
    cpp_root, registry = write_tree(
        tmp_path,
        {vpr: EMBEDDING_HEADER, ad: ANOMALY_HEADER, clip: EMBEDDING_HEADER, plain: EMBEDDING_HEADER},
        [entry(v, t, f) for (t, f, v) in (vpr, ad, clip, plain)],
        configs={vpr: {"gallery": "sample/gallery/old.bin",
                       "config": {"gallery": "sample/gallery/vpr_eigenplaces-resnet18_512x512.bin"}},
                 ad: {"config": {"roles": ["teacher", "student", "autoencoder"]}},
                 plain: {"config": {}}})
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout

    vpr_unit = unit(tmp_path, "visual_place_recognition")
    assert ('info.resources.push_back(ResourceInfo(ResourceInfo::kGallery, '
            '"sample/gallery/vpr_eigenplaces-resnet18_512x512.bin"));') in vpr_unit
    assert "old.bin" not in vpr_unit
    assert "info.ready = true;" in vpr_unit

    ad_unit = unit(tmp_path, "anomaly_detection")
    for companion in ("efficientad-m-student_256x256.dxnn", "efficientad-m-autoencoder_256x256.dxnn"):
        assert ('info.resources.push_back(ResourceInfo(ResourceInfo::kCompanion, "{}"));'
                .format(companion)) in ad_unit
        assert companion in re.search(r'info.not_ready_reason = "([^"]*)";', ad_unit).group(1)
    assert "efficientad-m-teacher_256x256.dxnn\"))" not in ad_unit

    clip_unit = unit(tmp_path, "zero_shot_image_classification")
    note = re.search(r'ResourceInfo\(ResourceInfo::kNote, "([^"]*)"\)', clip_unit)
    assert note and "Python-only" in note.group(1) and "embedding" in note.group(1)
    assert "info.resources" not in unit(tmp_path, "image_retrieval")


SEGMENTATION_HEADER = GOOD_HEADER.replace("IDetectionFactory", "ISegmentationFactory")


def test_matting_alpha_and_gallery_matches_are_ports(tmp_path):
    """PP-Matting's soft alpha matte and a gallery row's ranking reach a graph
    as named ports: "alpha" (densemap) for the ppmatting family, from
    GRAPH_TRAITS, and "matches" (scores) on every embedding row that has a
    gallery. An embedding row without a gallery declares no port."""
    matting = ("image_matting", "ppmatting", "ppmatting-hrnet-w48-composition_512x512")
    vpr = ("visual_place_recognition", "eigenplaces", "eigenplaces-resnet18_512x512")
    plain = ("image_retrieval", "clip_rn50", "clip-text_resnet50_77x512_openai")
    cpp_root, registry = write_tree(
        tmp_path, {matting: SEGMENTATION_HEADER, vpr: EMBEDDING_HEADER, plain: EMBEDDING_HEADER},
        [entry(v, t, f) for (t, f, v) in (matting, vpr, plain)],
        configs={vpr: {"config": {"gallery": "sample/gallery/vpr_eigenplaces-resnet18_512x512.bin"}},
                 plain: {"config": {}}})
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout

    assert ('info.ports.push_back(PortInfo("alpha", Shape::kDenseMap));'
            in unit(tmp_path, "image_matting"))
    assert ('info.ports.push_back(PortInfo("matches", Shape::kScores));'
            in unit(tmp_path, "visual_place_recognition"))
    assert "info.ports" not in unit(tmp_path, "image_retrieval")
    doc = (tmp_path / "graph_models.md").read_text()
    assert ("| `ppmatting-hrnet-w48-composition_512x512` | image_matting | "
            "labelmap + alpha (densemap) |") in doc
    assert ("| `eigenplaces-resnet18_512x512` | visual_place_recognition | "
            "vector + matches (scores) |") in doc


def test_anomaly_interface_is_not_ready_not_an_error(tmp_path):
    """IAnomalyDetectionFactory has no graph stage this release: the row is
    registered, not ready, with a reason - and the header is not compiled into
    the registry."""
    cpp_root, registry = write_tree(
        tmp_path, {("anomaly_detection", "patchcore", "patchcore_224x224"): ANOMALY_HEADER},
        [entry("patchcore_224x224", "anomaly_detection", "patchcore")])
    result = run_generator(tmp_path, cpp_root, registry, "--strict")
    assert result.returncode == 0, result.stdout
    emitted = unit(tmp_path, "anomaly_detection")
    assert "info.ready = false;" in emitted
    assert "registry->Add(info, NULL);" in emitted
    assert "patchcore_224x224_factory.hpp" not in emitted
    reason = re.search(r'info.not_ready_reason = "([^"]*)";', emitted).group(1)
    assert reason.startswith("anomaly detection is not graph-ready in this release")
    assert "unmapped interface" not in emitted
    doc = (tmp_path / "graph_models.md").read_text()
    assert "| `patchcore_224x224` | anomaly_detection |" in doc and "| NO - anomaly detection" in doc


def test_list_models_rows_are_every_variant_plus_alias_of_rows(tmp_path):
    """--list-models prints one row per registered ModelInfo and one per
    ModelAlias::kAliasOf alias (`alias of <variant>`); old model_names stay
    out of that list. Counted here on the real tree against the registry.

    Since the duplicate deit_base_distilled_2 row was dropped no entry carries
    alias_of, so the kAliasOf list is empty and deit_base384_distilled -- the
    one name that used to be there -- is a kLegacyName alias instead."""
    entries = json.loads(REGISTRY.read_text(encoding="utf-8"))
    variants = {e["variant"] for e in entries}
    alias_of = {e["model_name"]: e for e in entries if e.get("alias_of")}
    by_name = {e["model_name"]: e for e in entries}

    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--strict", "--out-dir", str(tmp_path / "generated"),
         "--docs", str(tmp_path / "graph_models.md")],
        cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, timeout=300)
    assert result.returncode == 0, result.stdout

    units = [p.read_text() for p in sorted((tmp_path / "generated").glob("graph_registry_*.cpp"))]
    added = [m for text in units for m in re.findall(r'info.model_name = "([^"]+)";', text)]
    assert sorted(added) == sorted(variants)
    assert sum(text.count("registry->Add(info,") for text in units) == len(variants)
    everything = (tmp_path / "generated" / "graph_registry_all.cpp").read_text()
    listed = re.findall(r'registry->AddAlias\("([^"]+)", "([^"]+)", ModelAlias::kAliasOf\);',
                        everything)
    assert sorted(name for name, _ in listed) == sorted(alias_of)
    for name, variant in listed:
        assert variant == by_name[alias_of[name]["alias_of"]]["variant"]
    assert listed == [], "no entry carries alias_of any more"
    legacy = re.findall(
        r'registry->AddAlias\("([^"]+)", "([^"]+)", ModelAlias::kLegacyName\);', everything)
    assert ("deit_base384_distilled", "deit-b_384x384_distilled") in legacy


def test_only_emits_a_registry_of_the_named_variants(tmp_path):
    """An exported graph project's registry (scripts/export_graph_project.py): the used models, and
    every old name or alias of them, so the graph's own model names still resolve."""
    out = tmp_path / "gen"
    r = subprocess.run([sys.executable, str(GENERATOR), "--out-dir", str(out), "--only", "yolov8-n_640x640",
                        "--print-sources"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    sources = r.stdout.split(";")
    assert sorted(Path(s).name for s in sources) == ["graph_registry_all.cpp",
                                                     "graph_registry_object_detection.cpp"]
    text = "".join(Path(s).read_text() for s in sources)
    assert re.findall(r'info\.variant = "([^"]+)"', text) == ["yolov8-n_640x640"]
    assert set(re.findall(r'AddAlias\("[^"]+", "([^"]+)"', text)) <= {"yolov8-n_640x640"}


def test_only_refuses_a_variant_the_registry_lacks(tmp_path):
    r = subprocess.run([sys.executable, str(GENERATOR), "--out-dir", str(tmp_path), "--only",
                        "yolov8-n_640x640,no-such-variant"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "not in the registry: no-such-variant" in r.stderr


def test_subset_entries_keeps_registry_order_and_the_aliases_of_kept_rows():
    sys.path.insert(0, str(ROOT / "scripts"))
    import gen_model_registry as gen
    rows = [{"variant": "a"}, {"variant": "b"}, {"variant": "c"}]
    aliases = [("old_a", "a", "legacy"), ("old_b", "b", "legacy")]
    kept, kept_aliases = gen.subset_entries(rows, aliases, ["c", "a"])
    assert [r["variant"] for r in kept] == ["a", "c"] and kept_aliases == [("old_a", "a", "legacy")]
    with pytest.raises(gen.GenError, match="not in the registry: z"):
        gen.subset_entries(rows, aliases, ["z"])


@pytest.mark.parametrize("extra", [["--docs-only"], ["--check-docs"]])
def test_only_is_refused_with_the_tracked_doc(tmp_path, extra):
    """A registry of a few variants is an exported project's; it must never become the tracked table."""
    r = subprocess.run([sys.executable, str(GENERATOR), "--out-dir", str(tmp_path), "--only", "yolov8-n_640x640"]
                       + extra, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "--only" in r.stderr


def test_only_refuses_an_empty_list(tmp_path):
    r = subprocess.run([sys.executable, str(GENERATOR), "--out-dir", str(tmp_path), "--only", ","],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "--only" in r.stderr
