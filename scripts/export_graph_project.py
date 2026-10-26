#!/usr/bin/env python3
"""Export a node-graph JSON as a standalone C++ (CMake) project: Graph Composer's "take away".

    python3 scripts/export_graph_project.py --graph <file.json> --out <dir> [--name <name>]
            [--no-models] [--model-dir <dir>] [--json]
    python3 scripts/export_graph_project.py --graph <file.json> --dry-run [--name <name>] [--json]
    python3 scripts/export_graph_project.py --version

The project builds the engine and the CLI of multi_model_graph_{sync,async} with only the
factories of the models the graph names. It needs the DEEPX runtime (dx_rt), CMake 3.14+, a
C++17 compiler and OpenCV 4, and nothing of this repository:

    CMakeLists.txt         one program, <name> (no path of this repository in it)
    main.cpp               registers the user logic, then cli::ProjectMain: graph.json, models/
    graph.json             the graph; its inputs and galleries point into the project
    .dxapp-root            marks the project root (RuntimeRoot(), common/utility/repo_path.hpp)
    src/, extern/          the engine, the CLI and the used factories, at their dx_app paths
    src/generated/         the model registry of those models (gen_model_registry.py)
    user_logic/            one stub class per "user" node name, and register.{hpp,cpp}
    models/                the .dxnn files (unless --no-models) and models.tsv (file, URL)
    inputs/, data/         the graph's input files and gallery files, copied
    setup.sh, run.sh, README.md, export.json

Exit codes: 0 exported (or planned, --dry-run); 1 the graph cannot be exported: the reason on
stderr as "ERROR [<key>] <text>", and with --json {"ok": false, "error", "error_key"} on stdout;
2 a usage error.
A dry run's model row has "bytes": null for a file that is not installed (the manifest has no sizes);
"model_bytes" sums the known ones and "model_bytes_unknown" counts the others.
Contract: tests/scripts/test_export_graph_project.py, tests/cpp_example/test_graph_export.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

SCHEMA = 1
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CPP = ROOT / "src" / "cpp_example"
sys.path.insert(0, str(SCRIPT_DIR))
import gen_model_registry as registry_gen  # noqa: E402

NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
# The -I order of the graph targets in src/cpp_example/CMakeLists.txt. A quoted include is looked
# up next to the including file first, then here, as the compiler does.
INCLUDE_DIRS = ("extern", "src/utility", "src/cpp_example", "src/cpp_example/common/processors",
                "src/postprocess/sfa3d", "src/postprocess/superpoint", "src/postprocess/dope",
                "src/postprocess/yolopv2", "src/postprocess/vitpose")
POSTPROCESS_LIBS = ("sfa3d", "superpoint", "dope", "yolopv2", "vitpose")
CLI_SOURCES = ("src/cpp_example/common/registry/static_model_registry.cpp",
               "src/cpp_example/multi_model_graph/graph_cli.cpp",
               "src/cpp_example/multi_model_graph/graph_consumer.cpp")
# What dx_app's graph CLI opens as a live stream (graph_consumer.cpp's OpenInput): a camera and rtsp://. It opens
# any other uri as a file, by its extension (InputFactory::createFromFile): an http(s) address ending in a video
# extension plays as a video file, any other one does not open, and rtsps:// never does.
LIVE_PREFIXES = ("camera:", "rtsp://")
REMOTE_PREFIXES = ("http://", "https://")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
# Files never searched for this repository's path: binary data, whatever it embeds.
BINARY_EXT = (".dxnn", ".bin", ".onnx") + IMAGE_EXT + VIDEO_EXT
# The C++ keywords and alternative tokens: a "user" node's name becomes a class name. The same
# list is CppKeyword() in src/cpp_example/common/graph/graph_config.cpp (a test compares them).
CPP_KEYWORDS = frozenset("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t
char32_t class compl concept const consteval constexpr constinit const_cast continue co_await
co_return co_yield decltype default delete do double dynamic_cast else enum explicit export
extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq
nullptr operator or or_eq private protected public register reinterpret_cast requires return
short signed sizeof static static_assert static_cast struct switch template this thread_local
throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while
xor xor_eq""".split())

# kind -> (C++ type, what the stub's Run receives)
KINDS = {
    "boxes": ("BoxesData",
              "the boxes, in source-frame pixels. input.items[i]: .box (cv::Rect2f: x, y, width, "
              "height), .score (0..1), .class_id, .class_name, .track_id (-1 when untracked)"),
    "obboxes": ("BoxesData",
                "the rotated boxes, in source-frame pixels. input.items[i]: .box, .angle (radians), "
                ".score, .class_id, .class_name, .track_id (-1 when untracked)"),
    "instances": ("BoxesData",
                  "the instances, in source-frame pixels. input.items[i]: .box, .mask (cv::Mat), "
                  ".score, .class_id, .class_name, .track_id (-1 when untracked)"),
    "keypoints": ("KeypointsData",
                  "one PoseResult per person in input.items: .box (x1, y1, x2, y2), .confidence, "
                  ".keypoints[k] (.x, .y, .confidence), as the node before it reports them"),
    "labelmap": ("LabelMapData", "input.labels: one class index per pixel (cv::Mat, CV_8U or CV_32S)"),
    "densemap": ("DenseMapData", "input.values: one value per pixel (cv::Mat, CV_32F), e.g. depth"),
    "image": ("ImageData", "input.image: the image the node before it made (cv::Mat, BGR)"),
    "scores": ("ScoresData",
               "input.items[i]: one ClassificationResult each: .class_id, .class_name, .confidence, "
               ".top_k"),
    "vector": ("VectorData", "input.values: the vector (std::vector<float>), e.g. an embedding"),
    "boxes3d": ("Boxes3dData",
                "input.items[i]: one Detection3DResult per object: .class_id, .class_name, .confidence, "
                ".x3d, .y3d, .z3d, .dim_h, .dim_w, .dim_l, .yaw (and the bird's-eye box .bev_x/y/w/h)"),
    "records": ("RecordsData",
                "input.items[i]: one RecordItem each: .numbers and .text, lists of (name, value)"),
}
BOX_KINDS = ("boxes", "obboxes", "instances")


class ExportError(Exception):
    """A graph that cannot be exported: `key` names the reason for a front end."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def load_graph(path) -> dict:
    try:
        graph = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExportError("graph_unreadable", "cannot read the graph {}: {}".format(path, exc))
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ExportError("graph_unreadable", "{} is not a graph: it has no \"nodes\" list".format(path))
    return graph


def default_name(graph: dict) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(graph.get("name") or "")).strip("_-")
    if not text or not text[0].isalpha():
        text = "graph_" + text if text else "graph_project"
    return text[:48]


def check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ExportError("name_invalid", "the project name {!r} must be 1 to 48 letters, digits, _ or -, "
                          "starting with a letter".format(name))
    return name


def binary_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def nodes_of(graph: dict) -> list:
    return [n for n in graph.get("nodes") or [] if isinstance(n, dict)]


def is_model_node(node: dict) -> bool:
    return "model" in node and node.get("type") not in ("source", "cpu", "logic")


def is_user_node(node: dict) -> bool:
    return node.get("type") == "logic" and node.get("op") == "user"


def user_logic_name_ok(name) -> bool:
    if not isinstance(name, str) or not re.match(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z", name):
        return False
    if name in CPP_KEYWORDS or "__" in name:
        return False
    return not (len(name) > 1 and name[0] == "_" and name[1].isupper())


def user_nodes(graph: dict) -> dict:
    """{name: {"input": kind, "nodes": [ids]}} of the "user" nodes, in graph order."""
    out = {}
    for node in nodes_of(graph):
        if not is_user_node(node):
            continue
        params = node.get("params") if isinstance(node.get("params"), dict) else {}
        nid, name, kind = node.get("id"), params.get("name"), params.get("input")
        if not user_logic_name_ok(name):
            raise ExportError("user_name_invalid",
                              "node {!r}: \"name\" {!r} must be a C++ name: letters, digits and _, not "
                              "starting with a digit, not a C++ keyword".format(nid, name))
        if kind not in KINDS or params.get("output") != kind:
            raise ExportError("user_kind_invalid",
                              "node {!r}: \"input\" {!r} and \"output\" {!r} must name one kind of "
                              "data: {}".format(nid, kind, params.get("output"), ", ".join(KINDS)))
        if name in out and out[name]["input"] != kind:
            raise ExportError("user_kind_conflict",
                              "node {!r} and node {!r} share the name {!r} but take {} and {}: one "
                              "name is one class".format(out[name]["nodes"][0], nid, name,
                                                         out[name]["input"], kind))
        out.setdefault(name, {"input": kind, "nodes": []})["nodes"].append(nid)
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_registry():
    try:
        return registry_gen.build_entries(registry_gen.DEFAULT_REGISTRY, registry_gen.DEFAULT_CPP, False,
                                          registry_gen.load_download_names(registry_gen.DEFAULT_MANIFEST))
    except registry_gen.GenError as exc:
        raise ExportError("registry_broken", "the model registry cannot be read: {}".format(exc))


def resolve_models(graph: dict, rows: list, aliases: list) -> list:
    """The registry rows of the models the graph names, in first-use order."""
    by_variant = {row["variant"]: row for row in rows}
    alias_to = {name: variant for name, variant, _kind in aliases}
    used = []
    for node in nodes_of(graph):
        if not is_model_node(node):
            continue
        name = node.get("model")
        variant = name if name in by_variant else alias_to.get(name)
        if variant is None:
            raise ExportError("model_unknown", "node {!r}: model {!r} is not in config/model_registry.json"
                              .format(node.get("id"), name))
        row = by_variant[variant]
        if not row["ready"]:
            raise ExportError("model_not_ready", "node {!r}: model {!r} cannot run in a graph ({})"
                              .format(node.get("id"), name, row["reason"]))
        if row not in used:
            used.append(row)
    return used


def manifest_urls() -> dict:
    """.dxnn file name -> its download URL (scripts/modelzoo_manifest.json)."""
    try:
        entries = json.loads(Path(registry_gen.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    urls = {}
    for item in entries if isinstance(entries, list) else []:
        url = item.get("dxnn_url") if isinstance(item, dict) else None
        if isinstance(url, str) and url.startswith("https://"):
            urls[url.rsplit("/", 1)[-1]] = url
    return urls


def model_files(used: list, model_dir: Path) -> list:
    """[{"model", "file", "bytes", "present", "url"}]: one .dxnn per used model. "bytes" is None for a file that is
    not on disk: the manifest gives its address, not its size."""
    urls = manifest_urls()
    out = []
    for row in used:
        path = Path(model_dir) / row["dxnn_file"]
        size = path.stat().st_size if path.is_file() else 0
        out.append({"model": row["variant"], "file": row["dxnn_file"], "bytes": size if size > 0 else None,
                    "present": size > 0, "url": urls.get(row["dxnn_file"], "")})
    return out


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def resolve_include(name: str, including: Path):
    for base in (including.parent,) + tuple(ROOT / d for d in INCLUDE_DIRS):
        candidate = base / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def include_closure(roots) -> set:
    """Every file the roots reach through #include "...", the roots too."""
    seen, stack = set(), [Path(r).resolve() for r in roots]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for name in INCLUDE_RE.findall(cur.read_text(encoding="utf-8", errors="replace")):
            hit = resolve_include(name, cur)
            if hit is not None:
                stack.append(hit)
    return seen


def source_set(generated: list):
    """(files under ROOT to copy, files under ROOT to compile). The compiled set grows by the
    out-of-line .cpp of a header the closure reaches (src/utility/common_util.cpp, the
    src/postprocess/<lib>/*.cpp), until it stops growing."""
    compiled = sorted((CPP / "common" / "graph").glob("*.cpp")) + [ROOT / p for p in CLI_SOURCES]
    compiled = [p.resolve() for p in compiled]
    while True:
        files = include_closure(compiled + [Path(g) for g in generated])
        extra = []
        if (ROOT / "src" / "utility" / "common_util.hpp").resolve() in files:
            extra.append((ROOT / "src" / "utility" / "common_util.cpp").resolve())
        for lib in POSTPROCESS_LIBS:
            folder = (ROOT / "src" / "postprocess" / lib).resolve()
            if any(folder in f.parents for f in files):
                extra += [p.resolve() for p in sorted(folder.glob("*.cpp"))]
        new = [p for p in extra if p not in compiled]
        if not new:
            break
        compiled += new
    copied = sorted(f for f in files if ROOT in f.parents)
    return copied, sorted(compiled)


def config_files(used: list) -> list:
    out = []
    for row in used:
        path = CPP / row["task"] / row["family"] / row["variant"] / "config.json"
        if path.is_file():
            out.append(path.resolve())
    return out


# ---------------------------------------------------------------------------
# Files the graph names
# ---------------------------------------------------------------------------

def local_file(text: str, what: str, node_id) -> Path:
    path = Path(text) if Path(text).is_absolute() else ROOT / text
    if not path.is_file():
        raise ExportError("{}_missing".format(what), "node {!r}: the {} {} does not exist".format(
            node_id, "input" if what == "input" else "gallery", text))
    return path.resolve()


def localize_files(graph: dict, out, write: bool):
    """Points the graph's local files at copies in inputs/ (sources) and data/ (galleries).
    Returns ([{"source", "uri", "copied"}], [{"file", "bytes"}] copied, warnings)."""
    taken, inputs, copies, warnings = set(), [], [], []

    def place(src: Path, folder: str) -> str:
        name, k = src.name, 2
        while "{}/{}".format(folder, name) in taken:
            name = "{}-{}{}".format(src.stem, k, src.suffix)
            k += 1
        rel = "{}/{}".format(folder, name)
        taken.add(rel)
        copies.append({"file": rel, "bytes": src.stat().st_size})
        if write:
            (out / folder).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out / rel)
        return rel

    for node in nodes_of(graph):
        if node.get("type") == "source":
            uri = node.get("uri") or ""
            if not uri:
                node["uri"] = ""  # the engine requires the key; --input fills it at run time
                warnings.append("source {!r} has no input: run it with --input {}=<file>".format(
                    node.get("id"), node.get("id")))
                inputs.append({"source": node.get("id"), "uri": "", "copied": False})
            elif str(uri).lower().startswith("rtsps://"):
                raise ExportError("input_unsupported", "node {!r}: rtsps:// is not supported: dx_app opens "
                                  "rtsp:// streams".format(node.get("id")))
            elif str(uri).startswith(LIVE_PREFIXES):
                inputs.append({"source": node.get("id"), "uri": uri, "copied": False})
            elif str(uri).lower().startswith(REMOTE_PREFIXES):
                if not str(uri).lower().endswith(VIDEO_EXT):
                    raise ExportError("input_unsupported", "node {!r}: dx_app opens an http(s) address only as a "
                                      "video file ({}); {} is not one".format(node.get("id"), ", ".join(VIDEO_EXT),
                                                                             uri))
                inputs.append({"source": node.get("id"), "uri": uri, "copied": False})
            else:
                node["uri"] = place(local_file(uri, "input", node.get("id")), "inputs")
                inputs.append({"source": node.get("id"), "uri": node["uri"], "copied": True})
        params = node.get("params")
        if isinstance(params, dict) and isinstance(params.get("gallery"), str) and params["gallery"].strip():
            params["gallery"] = place(local_file(params["gallery"], "gallery", node.get("id")), "data")
    return inputs, copies, warnings


def resource_files(used: list) -> list:
    """The galleries a used model's config.json names (relative to the repository): copied to
    the same relative path, where its config.json looks."""
    out = []
    for row in used:
        for kind, text in row["resources"]:
            if kind == "kGallery" and (ROOT / text).is_file():
                out.append((ROOT / text).resolve())
    return out


# ---------------------------------------------------------------------------
# Generated files
# ---------------------------------------------------------------------------

CMAKE = """\
# GENERATED by dx_app scripts/export_graph_project.py from the graph "{graph}".
# One program, {binary}: dx_app's graph engine and CLI with the factories of
# {models}. ./setup.sh builds it; by hand:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cmake_minimum_required(VERSION 3.14)
project({binary} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
# The DEEPX runtime installs its CMake package (dx_rt: /usr/local/lib/cmake/dxrt).
# Installed elsewhere: cmake -DDXRT_INSTALLED_DIR=<its prefix> ...
find_package(dxrt REQUIRED HINTS ${{DXRT_INSTALLED_DIR}})

# Your logic: every .cpp in user_logic/ (after adding a file, run cmake again).
file(GLOB USER_LOGIC_SOURCES CONFIGURE_DEPENDS ${{CMAKE_CURRENT_SOURCE_DIR}}/user_logic/*.cpp)

add_executable({binary}
    main.cpp
    ${{USER_LOGIC_SOURCES}}
{sources}
)
target_include_directories({binary} PRIVATE
    ${{CMAKE_CURRENT_SOURCE_DIR}}
{includes}
    ${{OpenCV_INCLUDE_DIRS}}
)
target_compile_options({binary} PRIVATE -Wall -Wextra -O2)
target_link_libraries({binary} PRIVATE ${{OpenCV_LIBS}} dxrt Threads::Threads)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries({binary} PRIVATE stdc++fs)
endif()
"""

MAIN = """\
// main.cpp - GENERATED by dx_app scripts/export_graph_project.py for the graph "{graph}".
// The program is dx_app's graph CLI: ./{binary} --help lists its options. With no --graph it
// runs graph.json, with no --model-dir it reads models/, both next to .dxapp-root; --sync picks
// the synchronous executor.
#include "multi_model_graph/graph_cli.hpp"
#include "user_logic/register.hpp"

int main(int argc, char** argv) {{
    RegisterProjectLogic();  // user_logic/register.cpp: the class behind each "user" node
    dxapp::graph::cli::ProjectDefaults defaults;
    defaults.graph = "graph.json";
    defaults.model_dir = "models";
    return dxapp::graph::cli::ProjectMain(argc, argv, defaults);
}}
"""

REGISTER_HPP = """\
// user_logic/register.hpp - GENERATED by dx_app scripts/export_graph_project.py.
#ifndef USER_LOGIC_REGISTER_HPP
#define USER_LOGIC_REGISTER_HPP

/// Registers the class behind each "user" node name of graph.json with the engine. main()
/// calls it before the graph is built; a name with no class passes its input through.
void RegisterProjectLogic();

#endif  // USER_LOGIC_REGISTER_HPP
"""

STUB_HPP = """\
// user_logic/{name}.hpp - the code behind the "user" node(s) {nodes} of graph.json.
// Written once by dx_app scripts/export_graph_project.py; it is yours to edit.
#ifndef USER_LOGIC_{guard}_HPP
#define USER_LOGIC_{guard}_HPP

#include "common/graph/user_logic.hpp"

namespace user_logic {{

/**
 * Node(s) {nodes} of graph.json ("name": "{name}").
 *
{input_doc}
 * Return: {kind} - what the nodes after this one receive. Returning `input`
 *   unchanged passes it through, which is what Studio's runs do.
 *
 * One object per node and per stream: members keep state from one frame to the
 * next (counts, last positions, timers). Run is called once per frame, in frame
 * order, never from two threads at once.
 *
 *   ctx.time_s()               seconds since the stream started (a file: frame / fps)
 *   ctx.frame_size()           the source frame's size in pixels
 *   ctx.Raise("name", id, d)   an event: an [EVENT] line on stdout and a line in
 *                              --events (./run.sh: results/events.jsonl); id is a
 *                              track id or -1, d a RecordItem of numbers and text
 *   ctx.Label("text")          a line of text in the result's top-left panel
 */
class {name} {{
 public:
    dxapp::graph::{cpp} Run(const dxapp::graph::{cpp}& input, dxapp::graph::UserContext& ctx);
}};

}}  // namespace user_logic

#endif  // USER_LOGIC_{guard}_HPP
"""

STUB_CPP = """\
// user_logic/{name}.cpp - written once by dx_app scripts/export_graph_project.py; yours to edit.
#include "user_logic/{name}.hpp"

#include <string>

namespace user_logic {{

dxapp::graph::{cpp} {name}::Run(const dxapp::graph::{cpp}& input, dxapp::graph::UserContext& ctx) {{
    // Write your logic here.
    //   input:  {kind} ({doc_short})
    //   return: {kind}, the same kind: what the nodes after this one receive.
{example}
    (void)ctx;
    return input;  // pass-through, as in Studio
}}

}}  // namespace user_logic
"""

BOX_EXAMPLE = """\
    // For example, keep the confident boxes and say how many there are:
    //   dxapp::graph::BoxesData out = input;
    //   out.items.clear();
    //   for (const dxapp::graph::BoxItem& item : input.items) {
    //       if (item.score >= 0.5f) out.items.push_back(item);
    //   }
    //   ctx.Label("kept " + std::to_string(out.items.size()));
    //   if (out.items.size() > 3) ctx.Raise("crowded");
    //   return out;"""

OTHER_EXAMPLE = """\
    // For example, raise an event once per second of stream time:
    //   if (ctx.time_s() >= next_s_) { ctx.Raise("tick"); next_s_ = ctx.time_s() + 1.0; }
    //   (declare `double next_s_ = 0.0;` in the class in user_logic/{name}.hpp)"""


def stub_files(users: dict) -> dict:
    """{relative path: text} for user_logic/: one .hpp/.cpp per name, and register.{hpp,cpp}."""
    files = {"user_logic/register.hpp": REGISTER_HPP}
    includes, calls = [], []
    for name, info in users.items():
        cpp, doc = KINDS[info["input"]]
        nodes = ", ".join(info["nodes"])
        fields = {"name": name, "guard": name.upper(), "nodes": nodes, "kind": info["input"], "cpp": cpp,
                  "doc": doc, "doc_short": doc.split(".")[0]}
        fields["input_doc"] = textwrap.fill("Input: {} - {}".format(info["input"], doc), width=96,
                                            initial_indent=" * ", subsequent_indent=" *   ")
        fields["example"] = BOX_EXAMPLE if info["input"] in BOX_KINDS else OTHER_EXAMPLE.replace("{name}", name)
        files["user_logic/{}.hpp".format(name)] = STUB_HPP.format(**fields)
        files["user_logic/{}.cpp".format(name)] = STUB_CPP.format(**fields)
        includes.append('#include "user_logic/{}.hpp"'.format(name))
        calls.append("    // node(s) {}: {} in, {} out\n"
                     "    dxapp::graph::RegisterUserLogic<dxapp::graph::{}, user_logic::{}>(\"{}\");"
                     .format(nodes, info["input"], info["input"], cpp, name, name))
    if not calls:
        calls.append("    // graph.json has no \"user\" node. For one named my_rule taking boxes, add\n"
                     "    // user_logic/my_rule.{hpp,cpp} (a class with BoxesData Run(const BoxesData&,\n"
                     "    // UserContext&)) and register it here:\n"
                     "    //   dxapp::graph::RegisterUserLogic<dxapp::graph::BoxesData, user_logic::my_rule>(\"my_rule\");")
    files["user_logic/register.cpp"] = (
        "// user_logic/register.cpp - GENERATED by dx_app scripts/export_graph_project.py. Edit it when\n"
        "// you add a \"user\" node to graph.json.\n"
        '#include "user_logic/register.hpp"\n\n#include "common/graph/user_logic.hpp"\n'
        + "".join(line + "\n" for line in includes) +
        "\nvoid RegisterProjectLogic() {\n" + "\n".join(calls) + "\n}\n")
    return files


SETUP = r"""#!/usr/bin/env bash
# setup.sh - GENERATED by dx_app scripts/export_graph_project.py.
# Checks what the build needs, fetches the models that models/ lacks, builds {binary}.
#   ./setup.sh [--models-from <dir>] [--no-build]
#   BUILD_DIR=<dir> ./setup.sh     builds there instead of ./build
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd -P)"
BUILD_DIR="${{BUILD_DIR:-$SCRIPT_DIR/build}}"
MODELS_FROM=""
BUILD=1

usage() {{
    echo "usage: ./setup.sh [--models-from <dir>] [--no-build]"
    echo "  --models-from <dir>  copy missing .dxnn files from <dir> instead of downloading them"
    echo "  --no-build           check and fetch only"
}}
while [ $# -gt 0 ]; do
    case "$1" in
        --models-from) [ $# -ge 2 ] || {{ usage >&2; exit 2; }}; MODELS_FROM="$2"; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# 1. The DEEPX runtime (dx_rt): the one prerequisite outside the usual build tools.
# (ldconfig's list is read into a variable: piped into an early-exiting grep, it fails under
# pipefail.)
DXRT_LIBS="$( (ldconfig -p || /sbin/ldconfig -p) 2>/dev/null || true)"
if [ -z "${{DXRT_INSTALLED_DIR:-}}" ] && [ ! -f /usr/local/lib/cmake/dxrt/dxrtConfig.cmake ] \
   && [[ "$DXRT_LIBS" != *libdxrt.so* ]]; then
    echo "ERROR: the DEEPX runtime (dx_rt) is not installed. Install it first, or set" >&2
    echo "       DXRT_INSTALLED_DIR to where it is installed." >&2
    exit 1
fi
# 2. Build tools and OpenCV.
for tool in cmake c++; do
    command -v "$tool" >/dev/null 2>&1 || {{ echo "ERROR: $tool is missing (Ubuntu: sudo apt install cmake g++)" >&2; exit 1; }}
done
if ! pkg-config --exists opencv4 2>/dev/null && ! ls /usr/lib/*/cmake/opencv4/OpenCVConfig.cmake >/dev/null 2>&1 \
   && ! ls /usr/local/lib/cmake/opencv4/OpenCVConfig.cmake >/dev/null 2>&1; then
    echo "ERROR: OpenCV 4 is missing (Ubuntu: sudo apt install libopencv-dev)" >&2
    exit 1
fi

# 3. Models: models/models.tsv lists each .dxnn and where to download it.
mkdir -p "$SCRIPT_DIR/models"
if [ ! -f "$SCRIPT_DIR/models/models.tsv" ]; then
    echo "ERROR: models/models.tsv is missing; export the project again" >&2
    exit 1
fi
while IFS=$'\t' read -r file url; do
    [ -n "$file" ] || continue
    target="$SCRIPT_DIR/models/$file"
    [ -s "$target" ] && continue
    if [ -n "$MODELS_FROM" ]; then
        [ -s "$MODELS_FROM/$file" ] || {{ echo "ERROR: $MODELS_FROM/$file not found" >&2; exit 1; }}
        cp "$MODELS_FROM/$file" "$target"
    elif [ -n "$url" ]; then
        echo "downloading $file"
        if command -v curl >/dev/null 2>&1; then
            curl -fL --retry 3 -o "$target.part" "$url"
        else
            wget -q -O "$target.part" "$url"
        fi
        mv "$target.part" "$target"
    else
        echo "ERROR: models/$file is missing and has no download address; copy it there" >&2
        exit 1
    fi
done < "$SCRIPT_DIR/models/models.tsv"

# 4. Build.
[ "$BUILD" = 1 ] || exit 0
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      ${{DXRT_INSTALLED_DIR:+-DDXRT_INSTALLED_DIR="$DXRT_INSTALLED_DIR"}}
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || echo 2)"
echo "built: $BUILD_DIR/{binary}  (run it: ./run.sh)"
"""

RUN = r"""#!/usr/bin/env bash
# run.sh - GENERATED by dx_app scripts/export_graph_project.py.
#   ./run.sh                runs graph.json on its own input; results in results/
#   ./run.sh <options>      passes the options to {binary} instead (./run.sh --help)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd -P)"
BUILD_DIR="${{BUILD_DIR:-$SCRIPT_DIR/build}}"
BIN="$BUILD_DIR/{binary}"
if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN is not built; run ./setup.sh first" >&2
    exit 1
fi
export DXAPP_ROOT="$SCRIPT_DIR"
if [ $# -eq 0 ]; then
    mkdir -p "$SCRIPT_DIR/results"
    set -- {default_args}
fi
exec "$BIN" "$@"
"""

README = """\
# {name}

A standalone C++ program made by DX-AI Studio's Graph Composer from the graph
"{graph}": {summary}. It is dx_app's graph engine and CLI with only the
factories of the models this graph uses.

## What you need

- The DEEPX runtime (dx_rt) and a DEEPX NPU. Nothing else of the DEEPX SDK.
- CMake 3.14 or newer, a C++17 compiler, OpenCV 4 (Ubuntu:
  `sudo apt install cmake g++ libopencv-dev`).

## Build and run

```bash
./setup.sh      # checks the above, fetches missing models, builds build/{binary}
./run.sh        # runs graph.json on its input; results go to results/
```

`./run.sh` with options passes them to the program instead, for example:

```bash
./run.sh --input {first_source}=other.mp4 --output out.mp4 --events events.jsonl
./run.sh --display              # show each frame in a window
./run.sh --sync                 # the synchronous executor (the reference)
./run.sh --check                # check graph.json and the models without running
./run.sh --help                 # every option
```

The project can be moved or copied anywhere: the program finds graph.json,
models/ and src/ from `.dxapp-root` (or `DXAPP_ROOT`, which run.sh sets
only for the program it starts; do not export it in your shell, because
dx_app's own programs read it too).
`BUILD_DIR=<dir> ./setup.sh` builds elsewhere; run.sh reads the same variable.

## Where events go

Each event is one JSON object
`{{"frame", "time", "stream", "node", "event", "track_id", "data"}}`:

- on stdout as `[EVENT] {{...}}`, right after the frame's `frame N:` line;
- one per line in the `--events <file>` file (`./run.sh`: `results/events.jsonl`);
- under the frame's `"events"` in `--report <file>` (`./run.sh`: `results/report.json`).

## Your logic

{user_section}

After editing, rebuild with `cmake --build build -j` (or `./setup.sh`).

## The graph

`graph.json` is read when the program starts: change a threshold, a zone or a
model's params there and run again, no rebuild. Its nodes:

{node_table}

## Models

{model_section}

## Files

| Path | What |
|---|---|
| `graph.json` | the graph |
| `main.cpp` | registers your logic, then runs dx_app's graph CLI |
| `user_logic/` | your logic, and `register.cpp` that hands it to the engine |
| `models/` | the .dxnn files; `models.tsv` lists where each comes from |
| `inputs/`, `data/` | the graph's input files and galleries |
| `src/`, `extern/` | dx_app's graph engine, CLI and the used factories (do not edit) |
| `src/generated/` | the model registry of these models |
| `setup.sh`, `run.sh` | build, run |
| `export.json` | what was exported, for tools |

Exported from dx_app {commit} on {date}.
"""


def readme_user_section(users: dict) -> str:
    if not users:
        return ("graph.json has no \"User logic\" node. Add one in the Graph Composer and export again, "
                "or follow the comment in `user_logic/register.cpp`.")
    lines = ["Each \"User logic\" node is a class in `user_logic/`. Studio runs it as a pass-through; "
             "here it runs your `Run`:", "", "| Class | Node(s) | Takes and returns |", "|---|---|---|"]
    for name, info in users.items():
        lines.append("| `user_logic/{0}.cpp` | {1} | {2} |".format(name, ", ".join(info["nodes"]), info["input"]))
    lines += ["", "`Run(input, ctx)` receives what the node before it produced and returns what the nodes "
              "after it receive (the same kind). The comment above the class in its `.hpp` lists the "
              "fields of the input, and `ctx.Raise(...)` / `ctx.Label(...)` raise events and draw text. "
              "One object exists per node and per stream, so members keep state across frames."]
    return "\n".join(lines)


def readme_node_table(graph: dict) -> str:
    lines = ["| Node | Kind | What |", "|---|---|---|"]
    for node in nodes_of(graph):
        if node.get("type") == "source":
            kind, what = "source", node.get("uri") or "(no input: pass --input)"
        elif node.get("type") in ("logic", "cpu"):
            kind, what = node["type"], node.get("op", "")
        else:
            kind, what = "model", node.get("model", "")
        lines.append("| `{}` | {} | {} |".format(node.get("id"), kind, what))
    return "\n".join(lines)


def readme_model_section(models: list, included: bool) -> str:
    rows = ["| File | Size |", "|---|---|"]
    for m in models:
        rows.append("| `models/{}` | {} |".format(m["file"], "{:.1f} MB".format(m["bytes"] / 1048576)
                                                    if m["bytes"] else "-"))
    head = ("The model files are in `models/`." if included else
            "`./setup.sh` downloads the model files into `models/` from the DEEPX Model Zoo "
            "(`models/models.tsv` lists each address), or copies them with "
            "`./setup.sh --models-from <dir>`.")
    return head + "\n\n" + "\n".join(rows)


def default_run_args(graph: dict) -> str:
    sources = [n for n in nodes_of(graph) if n.get("type") == "source"]
    uris = [str(n.get("uri") or "") for n in sources]
    args = ['--events "$SCRIPT_DIR/results/events.jsonl"']
    if any(u.startswith(LIVE_PREFIXES) for u in uris):
        return "--display " + args[0]
    video = any(u.lower().endswith(VIDEO_EXT) for u in uris)
    args = ['--output "$SCRIPT_DIR/results/{}"'.format("result.mp4" if video else "result.jpg")] + args
    args.append('--report "$SCRIPT_DIR/results/report.json"')
    return " ".join(args)


def dx_app_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

def forbidden_paths() -> list:
    """This repository's path, and the suite's when dx_app sits in one (dx-runtime/dx_app)."""
    out = [str(ROOT)]
    suite = ROOT.parent.parent
    if (suite / "dx-runtime").is_dir() and (suite / "dx-compiler").is_dir():
        out.append(str(suite))
    return out


def leaks(out: Path) -> list:
    """Project files that name a path of this repository or its suite."""
    needles = [p.encode() for p in forbidden_paths()]
    found = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix.lower() not in BINARY_EXT:
            data = path.read_bytes()
            if any(n in data for n in needles):
                found.append(str(path.relative_to(out)))
    return found


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def export(graph_path, out=None, name=None, include_models=True, model_dir=None, dry_run=False) -> dict:
    graph = load_graph(graph_path)
    name = check_name(name or default_name(graph))
    binary = binary_name(name)
    model_dir = Path(model_dir) if model_dir else ROOT / "assets" / "models"
    if not dry_run:
        if out is None:
            raise ExportError("out_missing", "--out names the project folder to create")
        out = Path(out).absolute()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ExportError("out_exists", "{} exists and is not an empty folder".format(out))
    rows, aliases = load_registry()
    used = resolve_models(graph, rows, aliases)
    users = user_nodes(graph)
    models = model_files(used, model_dir)
    if include_models and not dry_run:
        absent = [m["file"] for m in models if not m["present"]]
        if absent:
            raise ExportError("model_file_missing", "not in {}: {}; download it, or export without the "
                              "model files".format(model_dir, ", ".join(absent)))
    sub_rows, sub_aliases = registry_gen.subset_entries(rows, aliases, [r["variant"] for r in used])
    stubs = stub_files(users)
    summary = {"ok": True, "schema": SCHEMA, "name": name, "binary": binary, "graph": graph.get("name") or "",
               "dry_run": dry_run, "out": str(out) if out else None, "include_models": include_models,
               "models": models, "user_logic": [{"name": n, "input": i["input"], "nodes": i["nodes"],
                                                 "files": ["user_logic/{}.hpp".format(n),
                                                           "user_logic/{}.cpp".format(n)]}
                                                for n, i in users.items()]}
    created = False
    with tempfile.TemporaryDirectory(prefix="graph-export-") as scratch:
        target = Path(scratch) if dry_run else out
        if not dry_run:
            created = not out.exists()
            out.mkdir(parents=True, exist_ok=True)
        try:
            generated = registry_gen.emit_sources(sub_rows, sub_aliases, target / "src" / "generated")
            copied, compiled = source_set(generated)
            data_files = copied + config_files(used) + resource_files(used)
            inputs, copies, warnings = localize_files(graph, target, write=not dry_run)
            source_bytes = sum(p.stat().st_size for p in data_files) + \
                sum(Path(g).stat().st_size for g in generated)
            model_bytes = sum(m["bytes"] or 0 for m in models)
            summary.update(inputs=inputs, warnings=warnings, source_bytes=source_bytes, model_bytes=model_bytes,
                           model_bytes_unknown=sum(1 for m in models if m["bytes"] is None),
                           input_bytes=sum(c["bytes"] for c in copies),
                           bytes=source_bytes + sum(c["bytes"] for c in copies) +
                           (model_bytes if include_models else 0))
            if dry_run:
                return summary
            for path in data_files:
                dest = out / path.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
            sources = ["    " + p.relative_to(ROOT).as_posix() for p in compiled] + \
                      ["    src/generated/" + Path(g).name for g in generated]
            includes = ["    ${CMAKE_CURRENT_SOURCE_DIR}/" + d for d in INCLUDE_DIRS if (out / d).is_dir()]
            fields = {"graph": graph.get("name") or name, "binary": binary,
                      "models": ", ".join(r["variant"] for r in used) or "no model"}
            write_text(out / "CMakeLists.txt", CMAKE.format(sources="\n".join(sources),
                                                            includes="\n".join(includes), **fields))
            write_text(out / "main.cpp", MAIN.format(**fields))
            for rel, text in stubs.items():
                write_text(out / rel, text)
            write_text(out / "graph.json", json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
            write_text(out / ".dxapp-root", "dx_app graph project: RuntimeRoot() is this folder\n")
            (out / "models").mkdir(exist_ok=True)
            write_text(out / "models" / "models.tsv", "".join("{}\t{}\n".format(m["file"], m["url"])
                                                              for m in models))
            if include_models:
                for m in models:
                    shutil.copyfile(model_dir / m["file"], out / "models" / m["file"])
            write_text(out / "setup.sh", SETUP.format(binary=binary), executable=True)
            write_text(out / "run.sh", RUN.format(binary=binary, default_args=default_run_args(graph)),
                       executable=True)
            sources_of = [n.get("id") for n in nodes_of(graph) if n.get("type") == "source"]
            summary_text = "{} node(s), model(s) {}".format(len(nodes_of(graph)), fields["models"])
            write_text(out / "README.md", README.format(
                name=name, graph=fields["graph"], summary=summary_text, binary=binary,
                first_source=(sources_of or ["cam"])[0], user_section=readme_user_section(users),
                node_table=readme_node_table(graph), model_section=readme_model_section(models, include_models),
                commit=dx_app_commit() or "(unknown commit)",
                date=__import__("datetime").date.today().isoformat()))
            leaked = leaks(out)
            if leaked:
                raise ExportError("path_leak", "these files name a path of this repository: " + ", ".join(leaked))
            files = [p for p in out.rglob("*") if p.is_file()]
            summary.update(files=len(files) + 1, bytes=sum(p.stat().st_size for p in files))
            record = {k: v for k, v in summary.items() if k != "out"}   # where it was written is not portable
            write_text(out / "export.json", json.dumps(dict(record, dx_app=dx_app_commit()), indent=2) + "\n")
            return summary
        except BaseException:
            if created:
                shutil.rmtree(out, ignore_errors=True)
            elif not dry_run:
                # The folder was given empty: it is left empty, so a retry is not refused (out_exists).
                for child in list(out.iterdir()):
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink()
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export a node-graph JSON as a standalone C++ (CMake) project.")
    parser.add_argument("--graph", help="the graph JSON to export")
    parser.add_argument("--out", help="the project folder to create (absent or empty)")
    parser.add_argument("--name", help="project and program name (default: from the graph's name)")
    parser.add_argument("--no-models", action="store_true",
                        help="leave the .dxnn files out; setup.sh downloads them")
    parser.add_argument("--model-dir", help="where the .dxnn files are (default: assets/models)")
    parser.add_argument("--dry-run", action="store_true", help="plan only: write nothing")
    parser.add_argument("--json", action="store_true", help="print one JSON summary on stdout")
    parser.add_argument("--version", action="store_true", help="print the export schema and exit")
    args = parser.parse_args(argv)
    if args.version:
        print("export_graph_project {}".format(SCHEMA))
        return 0
    if not args.graph or (not args.out and not args.dry_run):
        parser.error("--graph and --out (or --dry-run) are required")
    try:
        summary = export(args.graph, args.out, args.name, not args.no_models, args.model_dir, args.dry_run)
    except ExportError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "error_key": exc.key}))
        print("ERROR [{}] {}".format(exc.key, exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary))
    elif args.dry_run:
        unknown = summary.get("model_bytes_unknown", 0)
        print("plan: {} ({}{} bytes with the models{}, {} without)".format(
            summary["name"], "at least " if unknown else "",
            summary["source_bytes"] + summary["input_bytes"] + summary["model_bytes"],
            " ({} model file{} of unknown size: not installed)".format(unknown, "" if unknown == 1 else "s")
            if unknown else "", summary["source_bytes"] + summary["input_bytes"]))
    else:
        print("exported {} to {} ({} files, {} bytes)".format(summary["name"], summary["out"], summary["files"],
                                                             summary["bytes"]))
        print("build and run: cd {} && ./setup.sh && ./run.sh".format(summary["out"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
