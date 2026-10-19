"""scripts/export_graph_project.py: a graph becomes a standalone C++ project (Graph Composer Phase 3).

These tests write projects and read them; none builds one (tests/cpp_example/test_graph_export.py
builds and runs an export outside the repository). Model files are tiny stand-ins in tmp_path, so a
test copies kilobytes, not the 8 MB of yolov8-n.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_graph_project.py"
EXPORT = [sys.executable, str(SCRIPT)]
USER_ZONE = ROOT / "tests" / "cpp_example" / "graph_export" / "user_zone.json"
SAMPLES = ROOT / "src" / "cpp_example" / "multi_model_graph"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
PROJECT_INCLUDES = ("", "extern", "src/utility", "src/cpp_example", "src/cpp_example/common/processors")


def run(*args, timeout=120):
    return subprocess.run(EXPORT + [str(a) for a in args], capture_output=True, text=True, cwd=str(ROOT),
                          timeout=timeout)


def fake_models(tmp_path, *files):
    d = tmp_path / "models"
    d.mkdir(exist_ok=True)
    for f in files:
        (d / f).write_bytes(b"not a real model " + f.encode())
    return d


def write_graph(tmp_path, graph, name="g.json"):
    p = tmp_path / name
    p.write_text(json.dumps(graph), encoding="utf-8")
    return p


def user_zone():
    return json.loads(USER_ZONE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """user_zone.json exported once, with a stand-in yolov8-n file."""
    tmp = tmp_path_factory.mktemp("export")
    models = fake_models(tmp, "yolov8-n_640x640.dxnn")
    out = tmp / "user-zone"
    r = run("--graph", USER_ZONE, "--out", out, "--model-dir", models, "--json")
    assert r.returncode == 0, r.stderr
    return out, json.loads(r.stdout)


def test_version_names_the_schema():
    r = run("--version")
    assert r.returncode == 0 and r.stdout.strip() == "export_graph_project 1"


def test_usage_errors_exit_2():
    assert run().returncode == 2
    assert run("--graph", USER_ZONE).returncode == 2          # no --out and no --dry-run


def test_the_project_holds_what_the_spec_lists(project):
    out, summary = project
    for rel in ("CMakeLists.txt", "main.cpp", "graph.json", ".dxapp-root", "setup.sh", "run.sh", "README.md",
                "export.json", "user_logic/tag_people.hpp", "user_logic/tag_people.cpp", "user_logic/register.hpp",
                "user_logic/register.cpp", "models/yolov8-n_640x640.dxnn", "models/models.tsv",
                "inputs/sample_people.jpg", "src/cpp_example/multi_model_graph/graph_cli.cpp",
                "src/cpp_example/common/graph/user_logic.hpp",
                "src/cpp_example/object_detection/yolov8/yolov8-n_640x640/config.json"):
        assert (out / rel).is_file(), rel
    for script in ("setup.sh", "run.sh"):
        assert (out / script).stat().st_mode & stat.S_IXUSR, script
    assert summary["ok"] is True and summary["name"] == "user-zone" and summary["binary"] == "user_zone"
    assert [m["model"] for m in summary["models"]] == ["yolov8-n_640x640"]
    assert summary["user_logic"] == [{"name": "tag_people", "input": "boxes", "nodes": ["tag"],
                                      "files": ["user_logic/tag_people.hpp", "user_logic/tag_people.cpp"]}]
    assert summary["files"] == len([p for p in out.rglob("*") if p.is_file()])


def test_only_the_used_models_factories_and_registry_rows_go_in(project):
    out, _ = project
    factories = sorted(p.parent.relative_to(out / "src" / "cpp_example").as_posix()
                       for p in (out / "src" / "cpp_example").rglob("factory/*.hpp"))
    assert factories and all(f.startswith("object_detection/yolov8/yolov8-n_640x640/") for f in factories)
    generated = "".join(p.read_text() for p in (out / "src" / "generated").glob("*.cpp"))
    assert re.findall(r'info\.variant = "([^"]+)"', generated) == ["yolov8-n_640x640"]
    assert set(re.findall(r'AddAlias\("[^"]+", "([^"]+)"', generated)) <= {"yolov8-n_640x640"}


def test_the_graph_points_at_its_copied_input(project):
    out, summary = project
    graph = json.loads((out / "graph.json").read_text())
    assert [n["uri"] for n in graph["nodes"] if n.get("type") == "source"] == ["inputs/sample_people.jpg"]
    assert (out / "inputs/sample_people.jpg").read_bytes() == (ROOT / "sample/img/sample_people.jpg").read_bytes()
    assert summary["inputs"] == [{"source": "cam", "uri": "inputs/sample_people.jpg", "copied": True}]
    assert [n for n in graph["nodes"] if n["id"] == "tag"][0]["params"]["name"] == "tag_people"


def test_cmake_compiles_files_that_exist_and_defines_no_repository_root(project):
    out, _ = project
    text = (out / "CMakeLists.txt").read_text()
    sources = re.search(r"add_executable\(user_zone\n(.*?)\n\)", text, re.S).group(1).split()
    files = [s for s in sources if not s.startswith("${")]
    assert "main.cpp" in files and "src/cpp_example/common/graph/user_logic.cpp" in files
    assert all((out / f).is_file() for f in files), [f for f in files if not (out / f).is_file()]
    assert "PROJECT_ROOT_DIR" not in text
    assert "find_package(dxrt REQUIRED" in text and "user_logic/*.cpp" in text


def test_every_include_found_in_dx_app_is_found_in_the_project(project):
    """The compile-time half of "self-contained": what an include reaches in dx_app, it reaches here."""
    out, _ = project
    lost = []
    for path in out.rglob("*"):
        if path.suffix not in (".cpp", ".hpp", ".h"):
            continue
        for name in INCLUDE_RE.findall(path.read_text(encoding="utf-8", errors="replace")):
            here = [path.parent] + [out / d for d in PROJECT_INCLUDES]
            there = [ROOT / d for d in ("extern", "src/utility", "src/cpp_example",
                                        "src/cpp_example/common/processors")]
            if any((b / name).is_file() for b in there) and not any((b / name).is_file() for b in here):
                lost.append((path.relative_to(out).as_posix(), name))
    assert not lost, lost


def test_no_file_names_a_path_of_this_repository(project):
    out, _ = project
    needles = [str(ROOT)]
    suite = ROOT.parent.parent
    if (suite / "dx-runtime").is_dir():
        needles.append(str(suite))
    for path in out.rglob("*"):
        if path.is_file() and path.suffix not in (".dxnn", ".jpg", ".png", ".bin", ".mp4"):
            text = path.read_text(encoding="utf-8", errors="replace")
            assert not [n for n in needles if n in text], path


def test_the_stub_says_what_comes_in_and_what_to_return_and_passes_through(project):
    out, _ = project
    hpp = (out / "user_logic/tag_people.hpp").read_text()
    cpp = (out / "user_logic/tag_people.cpp").read_text()
    assert "class tag_people {" in hpp and "Input: boxes - the boxes, in source-frame pixels" in hpp
    assert "Return: boxes - what the nodes after this one receive" in hpp
    assert "BoxesData Run(const dxapp::graph::BoxesData& input, dxapp::graph::UserContext& ctx);" in hpp
    assert "// Write your logic here." in cpp and "return input;  // pass-through, as in Studio" in cpp
    reg = (out / "user_logic/register.cpp").read_text()
    assert 'RegisterUserLogic<dxapp::graph::BoxesData, user_logic::tag_people>("tag_people");' in reg
    main = (out / "main.cpp").read_text()
    assert "RegisterProjectLogic();" in main and 'defaults.graph = "graph.json";' in main


def test_run_sh_writes_results_and_readme_says_where_events_go(project):
    out, _ = project
    run_sh = (out / "run.sh").read_text()
    assert 'export DXAPP_ROOT="$SCRIPT_DIR"' in run_sh
    assert '--output "$SCRIPT_DIR/results/result.jpg" --events "$SCRIPT_DIR/results/events.jsonl"' in run_sh
    readme = (out / "README.md").read_text()
    for text in ("./setup.sh", "./run.sh", "results/events.jsonl", "`user_logic/tag_people.cpp` | tag | boxes",
                 "[EVENT]", "only for the program it starts"):
        assert text in readme, text
    assert subprocess.run(["bash", "-n", str(out / "setup.sh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(out / "run.sh")]).returncode == 0


def test_models_tsv_names_each_download_and_no_models_leaves_the_files_out(tmp_path):
    out = tmp_path / "p"
    r = run("--graph", USER_ZONE, "--out", out, "--no-models", "--model-dir", tmp_path / "none", "--json")
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (out / "models").iterdir()) == ["models.tsv"]
    file, url = (out / "models/models.tsv").read_text().strip().split("\t")
    assert file == "yolov8-n_640x640.dxnn" and url.startswith("https://") and url.endswith("/" + file)
    assert "downloads the model files" in (out / "README.md").read_text()


def test_dry_run_writes_nothing_and_sizes_the_export(tmp_path):
    models = fake_models(tmp_path, "yolov8-n_640x640.dxnn")
    r = run("--graph", USER_ZONE, "--dry-run", "--model-dir", models, "--json")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["dry_run"] is True and plan["out"] is None
    assert plan["model_bytes"] == (models / "yolov8-n_640x640.dxnn").stat().st_size
    assert plan["source_bytes"] > 500_000 and plan["bytes"] == plan["source_bytes"] + plan["input_bytes"] + \
        plan["model_bytes"]
    assert plan["models"][0]["present"] is True
    assert sorted(p.name for p in tmp_path.iterdir()) == ["models"]


def test_dry_run_marks_a_model_file_that_is_not_installed_as_of_unknown_size(tmp_path):
    """The model zoo manifest gives each file's address, not its size: a file not on disk is of unknown size
    (null), never 0 bytes, and the plan counts such files apart from the bytes it knows."""
    r = run("--graph", USER_ZONE, "--dry-run", "--model-dir", tmp_path / "none", "--json")
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["models"][0]["present"] is False and plan["models"][0]["bytes"] is None
    assert plan["models"][0]["url"].startswith("https://")
    assert plan["model_bytes"] == 0 and plan["model_bytes_unknown"] == 1
    assert plan["bytes"] == plan["source_bytes"] + plan["input_bytes"]
    models = fake_models(tmp_path, "yolov8-n_640x640.dxnn")
    known = json.loads(run("--graph", USER_ZONE, "--dry-run", "--model-dir", models, "--json").stdout)
    assert known["model_bytes_unknown"] == 0 and known["models"][0]["bytes"] > 0


@pytest.mark.parametrize("uri", ["rtsps://cam.example/live", "RTSPS://cam.example/live"])
def test_an_rtsps_source_is_refused_as_studio_refuses_it(tmp_path, uri):
    """dx_app opens rtsp:// streams only (graph_consumer.cpp's OpenInput); an rtsps:// address is not one."""
    g = user_zone()
    g["nodes"][0]["uri"] = uri
    r = run("--graph", write_graph(tmp_path, g), "--out", tmp_path / "p", "--no-models", "--json")
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["error_key"] == "input_unsupported" and "rtsp://" in out["error"]
    assert not (tmp_path / "p").exists()


def test_an_http_source_is_a_remote_video_file_not_a_live_stream(tmp_path):
    """dx_app opens an http(s) address as a file, by its extension (InputFactory::createFromFile): a video
    address plays as a video (run.sh writes result.mp4); any other address is refused, since it cannot open."""
    g = user_zone()
    g["nodes"][0]["uri"] = "https://example.com/clips/hall.mp4"
    out = tmp_path / "p"
    r = run("--graph", write_graph(tmp_path, g), "--out", out, "--no-models", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["inputs"] == [{"source": g["nodes"][0]["id"],
                                               "uri": "https://example.com/clips/hall.mp4", "copied": False}]
    run_sh = (out / "run.sh").read_text()
    assert "result.mp4" in run_sh and "--display" not in run_sh
    for uri in ("https://example.com/stream", "http://example.com/a.jpg"):
        g["nodes"][0]["uri"] = uri
        r = run("--graph", write_graph(tmp_path, g), "--out", tmp_path / "q", "--no-models", "--json")
        assert r.returncode == 1 and json.loads(r.stdout)["error_key"] == "input_unsupported", uri


def test_a_camera_or_rtsp_source_is_live(tmp_path):
    g = user_zone()
    for uri in ("camera:0", "rtsp://cam.example/live"):
        g["nodes"][0]["uri"] = uri
        out = tmp_path / uri.split(":")[0]
        r = run("--graph", write_graph(tmp_path, g), "--out", out, "--no-models", "--json")
        assert r.returncode == 0, r.stderr
        assert "--display" in (out / "run.sh").read_text(), uri


def test_a_gallery_param_is_copied_into_data(tmp_path):
    out = tmp_path / "p"
    r = run("--graph", SAMPLES / "gallery_known_person.json", "--out", out, "--no-models", "--json")
    assert r.returncode == 0, r.stderr
    graph = json.loads((out / "graph.json").read_text())
    known = [n for n in graph["nodes"] if n["id"] == "known"][0]
    assert known["params"]["gallery"] == "data/reid_repvgg-a0-reid_256x128.bin"
    assert (out / "data/reid_repvgg-a0-reid_256x128.bin").is_file()
    assert sorted(m["model"] for m in json.loads(r.stdout)["models"]) == ["repvgg-a0-reid_256x128",
                                                                         "yolov8-n_640x640"]


@pytest.mark.parametrize("change,key", [
    (lambda g: g["nodes"][1].update(model="no-such-model"), "model_unknown"),
    (lambda g: g["nodes"][3]["params"].update(name="class"), "user_name_invalid"),
    (lambda g: g["nodes"][3]["params"].update(output="records"), "user_kind_invalid"),
    (lambda g: g["nodes"][0].update(uri="sample/img/no_such.jpg"), "input_missing"),
])
def test_a_graph_that_cannot_be_exported_says_why(tmp_path, change, key):
    g = user_zone()
    change(g)
    r = run("--graph", write_graph(tmp_path, g), "--out", tmp_path / "p", "--no-models", "--json")
    assert r.returncode == 1
    assert json.loads(r.stdout)["error_key"] == key and "ERROR [{}]".format(key) in r.stderr
    assert not (tmp_path / "p").exists(), "a failed export leaves nothing behind"


def test_two_user_nodes_with_one_name_must_take_one_kind(tmp_path):
    g = user_zone()
    g["nodes"].append({"id": "count", "type": "logic", "op": "count"})
    g["nodes"].append({"id": "tag2", "type": "logic", "op": "user",
                       "params": {"name": "tag_people", "input": "records", "output": "records"}})
    g["edges"] += [{"from": "od", "to": "count", "carry": "result"},
                   {"from": "count", "to": "tag2", "carry": "result"}]
    r = run("--graph", write_graph(tmp_path, g), "--dry-run", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["error_key"] == "user_kind_conflict"


def test_refusals_of_the_command_line(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    r = run("--graph", USER_ZONE, "--out", tmp_path / "full", "--no-models", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["error_key"] == "out_exists"
    r = run("--graph", USER_ZONE, "--out", tmp_path / "p", "--name", "9lives", "--json")
    assert json.loads(r.stdout)["error_key"] == "name_invalid"
    r = run("--graph", USER_ZONE, "--out", tmp_path / "q", "--model-dir", tmp_path / "empty", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["error_key"] == "model_file_missing"


def test_the_keyword_list_is_the_engines():
    """A name the exporter takes, --check takes: CPP_KEYWORDS here, CppKeyword() in graph_config.cpp."""
    src = (ROOT / "src/cpp_example/common/graph/graph_config.cpp").read_text(encoding="utf-8")
    block = src[src.index("bool CppKeyword("):src.index("};", src.index("bool CppKeyword("))]
    engine = set(re.findall(r'"([a-z_0-9]+)"', block))
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_graph_project
    assert engine == set(export_graph_project.CPP_KEYWORDS)


# ---------------------------------------------------------------------------
# Review fixes: every kind, cleanup of a given empty folder, sources, setup.sh's model loop
# ---------------------------------------------------------------------------

def test_every_kind_a_user_node_takes_gets_a_stub(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_graph_project
    g = user_zone()
    for kind in export_graph_project.KINDS:
        g["nodes"].append({"id": "n_" + kind, "type": "logic", "op": "user",
                           "params": {"name": "rule_" + kind, "input": kind, "output": kind}})
    out = tmp_path / "p"
    r = run("--graph", write_graph(tmp_path, g), "--out", out, "--no-models", "--json")
    assert r.returncode == 0, r.stderr
    reg = (out / "user_logic/register.cpp").read_text()
    for kind, (cpp, _doc) in export_graph_project.KINDS.items():
        hpp = (out / "user_logic/rule_{}.hpp".format(kind)).read_text()
        assert "dxapp::graph::{} Run(const dxapp::graph::{}& input".format(cpp, cpp) in hpp
        assert "RegisterUserLogic<dxapp::graph::{}, user_logic::rule_{}>".format(cpp, kind) in reg
    assert "next_s_" in (out / "user_logic/rule_records.cpp").read_text()
    assert "Boxes3dData" in (out / "user_logic/rule_boxes3d.hpp").read_text()


def test_a_failed_export_into_a_given_empty_folder_leaves_it_empty(tmp_path):
    g = user_zone()
    g["nodes"][0]["uri"] = "sample/img/no_such.jpg"
    out = tmp_path / "p"
    out.mkdir()
    r = run("--graph", write_graph(tmp_path, g), "--out", out, "--no-models", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["error_key"] == "input_missing"
    assert out.is_dir() and not any(out.iterdir())


def test_a_source_without_a_uri_gets_an_empty_one(tmp_path):
    g = user_zone()
    del g["nodes"][0]["uri"]
    out = tmp_path / "p"
    r = run("--graph", write_graph(tmp_path, g), "--out", out, "--no-models", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads((out / "graph.json").read_text())["nodes"][0]["uri"] == ""
    assert json.loads(r.stdout)["warnings"]


@pytest.fixture(scope="module")
def no_models(tmp_path_factory):
    """user_zone.json exported without its model files."""
    tmp = tmp_path_factory.mktemp("nomodels")
    r = run("--graph", USER_ZONE, "--out", tmp / "p", "--no-models", "--json")
    assert r.returncode == 0, r.stderr
    return tmp / "p"


def _setup(project, *args):
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
    r = subprocess.run(["bash", str(project / "setup.sh"), "--no-build"] + [str(a) for a in args],
                       capture_output=True, text=True, env=env, timeout=120)
    if "the DEEPX runtime (dx_rt) is not installed" in r.stderr:
        pytest.skip("dx_rt is not installed here")
    return r


def test_setup_copies_the_left_out_models_from_a_folder(no_models, tmp_path):
    fake_models(tmp_path, "yolov8-n_640x640.dxnn")
    r = _setup(no_models, "--models-from", tmp_path / "models")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (no_models / "models/yolov8-n_640x640.dxnn").read_bytes() == b"not a real model yolov8-n_640x640.dxnn"
    (no_models / "models/yolov8-n_640x640.dxnn").unlink()


def test_setup_says_what_it_cannot_fetch(no_models):
    tsv = no_models / "models/models.tsv"
    kept = tsv.read_text()
    try:
        tsv.write_text("yolov8-n_640x640.dxnn\t\n")
        r = _setup(no_models)
        assert r.returncode == 1 and "has no download address" in r.stderr
        tsv.unlink()
        r = _setup(no_models)
        assert r.returncode == 1 and "models/models.tsv is missing" in r.stderr
    finally:
        tsv.write_text(kept)


def test_setup_finds_dx_rt_without_a_pipe_that_can_break(no_models):
    """`ldconfig -p | grep -q` fails under pipefail when grep stops early (SIGPIPE)."""
    text = (no_models / "setup.sh").read_text()
    assert "| grep -q" not in text and "/sbin/ldconfig" in text
