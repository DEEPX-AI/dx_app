#!/usr/bin/env python3
"""Generate the graph engine's model registry from the factory tree.

Inputs: config/model_registry.json, the factory headers under
src/cpp_example/<task>/<family>/<variant>/factory/<variant>_factory.hpp, each
variant's config.json beside them, and the interface-to-shape and graph-trait
tables below. Nobody maintains a list by hand.
scripts/modelzoo_manifest.json (--manifest) adds each model's download name.

Outputs: the registry sources (--out-dir) and the model table (--docs).
The configure step writes both into the build tree. Without --docs, a run
that emits sources writes the table as <out-dir>/graph_models.md; only
--docs-only and --check-docs default to the tracked docs/graph_models.md,
so the tracked doc is written only by --docs-only, and --check-docs
(scripts/check_graph_models_doc.py) fails when it is stale.

The model key is the VARIANT (R6)
---------------------------------
One ModelInfo per variant: model_name == variant. A registry row's own
model_name, where it differs, is the variant's old name, and an alias_of row
is an alias of the row it names; both become ModelAlias entries
(registry->AddAlias(name, variant, kind)), never a second model.
scripts/check_model_registry.py's resolve_registry() decides which is which,
imported, so the guard and this generator cannot disagree. The graph CLI's
--list-models (multi_model_graph_{sync,async}) lists every ModelInfo plus
the ModelAlias::kAliasOf aliases.

What the generated sources expect of the engine:
  ModelInfo::variant, ::family (TypedStage reads
  <task>/<family>/<variant>/config.json), ::published (bool), ::resources
  (std::vector<ResourceInfo>); ResourceInfo(ResourceInfo::Kind, path_or_text)
  with Kind kGallery (a gallery .bin, relative to the repository), kCompanion
  (a .dxnn looked up in --model-dir) and kNote (text for --check);
  StaticModelRegistry::AddAlias(name, variant, ModelAlias::Kind) with Kind
  kLegacyName and kAliasOf, called after every Add().

ONE scanning rule, not two
--------------------------
The factory headers are scanned through scripts/check_factory_uniqueness.py's
parse_factory_classes(), imported, never re-implemented. It returns each class
FULLY QUALIFIED (`::dxapp::v_<variant>::Cls`), and that is the name emitted:
every variant's copy of a family's class is its own type. A second copy of
the rule living here would be free to drift, and the failure mode of drift is
the worst one available: a header this file cannot parse silently vanishes
from the generated registry and the model goes missing at RUNTIME instead of
at build time.

Failure policy
--------------
Hard errors, always, regardless of --strict - each one is a case where
guessing would put a wrong or missing model in front of a user:

  * NAMESPACE    a variant header not scoped in dxapp::v_<variant>
                 (generate_cpp_family_layout.namespace_problems()). Linked
                 unscoped, two variants' factories are one weak symbol and
                 the linker silently keeps one. The message names the fix.
                 An unused create*() parameter is NOT an error here:
                 --variant-scope --check reports it (guards, CI).
  * UNPARSEABLE  a factory header parse_factory_classes() cannot read.
  * AMBIGUOUS    a header declaring a SECOND I...Factory-deriving class.
  * DUPLICATE    one qualified factory class declared by two headers.
  * TASK MISMATCH  a registry row whose task or family disagrees with the
                 directory its variant's factory lives in. TypedStage
                 derives <task>/<family>/<variant>/config.json from
                 ModelInfo, so a disagreement silently loads the wrong config.
  * AMBIGUOUS ALIAS / BAD ALIAS / DUPLICATE VARIANT  a name that would
                 resolve two ways (check_model_registry.resolve_registry()).
  * TRAIT MISMATCH  a header declaring a graph trait the table disagrees with.

Soft, and deliberately so:

  * name drift between the registry and the factory tree fails only under
    --strict (scripts/ci_checks.sh: codegen-strict), because a normal build
    must not break while a rename is in flight;
  * a registry entry with no factory is normal - a model is often registered
    before its factory lands - so it is emitted with ready=false and a reason
    rather than dropped; so is an interface with no graph stage yet
    (anomaly detection, R9).

Shape comes from the FACTORY INTERFACE, never from the registry's task
string. That is why "ppu" - one task, three interfaces - needs no special
case anywhere in this file.
"""
import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# THE factory-scanning rule. Imported, not copied - see the module docstring.
from check_factory_uniqueness import (  # noqa: E402
    ParseError,
    parse_factory_classes,
)
# THE registry rule (which row is a model, which name an alias), and the
# namespace check configure refuses to build without.
from check_model_registry import ALIAS_OF, resolve_registry  # noqa: E402
from generate_cpp_family_layout import (  # noqa: E402
    VARIANT_SCOPE_FIX,
    namespace_problems,
)

DEFAULT_CPP = ROOT / "src" / "cpp_example"
DEFAULT_REGISTRY = ROOT / "config" / "model_registry.json"
DEFAULT_MANIFEST = ROOT / "scripts" / "modelzoo_manifest.json"
DOCS_UPDATE_COMMAND = "python3 scripts/gen_model_registry.py --docs-only"

# Interface -> (Shape, result type, default input contract, produces landmarks,
#               default extra ports)
INTERFACE_TABLE = {
    "IDetectionFactory":            ("kBoxes",     "DetectionResult",              "kFullFrame", False, ()),
    "IFaceDetectionFactory":        ("kBoxes",     "FaceDetectionResult",          "kFullFrame", True, ()),
    "IOBBFactory":                  ("kObBoxes",   "OBBResult",                    "kFullFrame", False, ()),
    "IInstanceSegmentationFactory": ("kInstances", "InstanceSegmentationResult",   "kFullFrame", False, ()),
    "IPanopticDrivingFactory":      ("kBoxes",     "PanopticResult",               "kFullFrame", False, ("drivable", "lane")),
    "ISegmentationFactory":         ("kLabelMap",  "SegmentationResult",           "kFullFrame", False, ()),
    "IDepthEstimationFactory":      ("kDenseMap",  "DepthResult",                  "kFullFrame", False, ()),
    "IRestorationFactory":          ("kImage",     "RestorationResult",            "kEither",    False, ()),
    "IClassificationFactory":       ("kScores",    "ClassificationResult",         "kEither",    False, ()),
    "IEmbeddingFactory":            ("kVector",    "EmbeddingResult",              "kEither",    False, ()),
    "IPoseFactory":                 ("kKeypoints", "PoseResult",                   "kFullFrame", False, ()),
    "IKeypointDetectionFactory":    ("kKeypoints", "PoseResult",                   "kFullFrame", False, ()),
    "IObjectPoseFactory":           ("kKeypoints", "PoseResult",                   "kFullFrame", False, ()),
    "IFaceAlignmentFactory":        ("kKeypoints", "FaceAlignmentResult",          "kEither",    False, ("pose",)),
    "IHandLandmarkFactory":         ("kKeypoints", "HandLandmarkResult",           "kEither",    False, ("handedness",)),
    "I3DDetectionFactory":          ("kBoxes3d",   "Detection3DResult",            "kFullFrame", False, ()),
}

# (result type, port name) -> Shape of that extra graph output port. A port
# a factory or an interface names must appear here, or generation fails.
PORT_SHAPES = {
    ("PanopticResult", "drivable"): "kLabelMap",
    ("PanopticResult", "lane"): "kLabelMap",
    ("PoseResult", "descriptors"): "kDenseMap",
    ("HandLandmarkResult", "handedness"): "kScores",
    ("FaceAlignmentResult", "pose"): "kVector",
    ("SegmentationResult", "alpha"): "kDenseMap",
    ("EmbeddingResult", "matches"): "kScores",
}

# Graph traits per family (R5), in place of lines in the teammate's generated
# headers. A header may still declare graphInput()/graphPorts() itself; it
# must then agree with this table.
GRAPH_TRAITS = {
    "vitpose":    {"input": "kEither"},        # top-down pose: an ROI consumer
    "dark_hrnet": {"input": "kEither"},
    "superpoint": {"ports": ("descriptors",)},  # per-keypoint descriptors
    "ppmatting":  {"ports": ("alpha",)},        # the soft alpha matte
}

# An embedding row with a gallery ranks against it (GalleryRetrievalPostprocessor):
# its ranking is this extra port (R9).
GALLERY_PORT = ("EmbeddingResult", "matches")

# Interfaces registered but not graph-ready in this release, with the reason.
NOT_READY_INTERFACES = {
    "IAnomalyDetectionFactory": "anomaly detection is not graph-ready in this release",
}

# A note --check prints for every model of a task (R9).
TASK_NOTES = {
    "zero_shot_image_classification":
        "the CLIP prompt bank is Python-only; in a graph this model outputs its "
        "image embedding, not zero-shot scores",
}

MODELS_PER_TU = 64

# Only ASCII identifiers and simple relative paths ever reach the generated
# source, so nothing here needs escaping - but a stray quote or backslash in
# the registry JSON would produce a source file that does not compile with a
# baffling error, so it is rejected at the source instead.
_SAFE_STRING = re.compile(r'^[A-Za-z0-9._/+\- ]*$')


class GenError(Exception):
    """A condition the generator refuses to guess its way past."""


# Factory graph traits (B7 / U-08). A factory overrides its interface's
# defaults with one line each; the generated unit static_asserts the parse.
_TRAIT_INPUT = re.compile(
    r'static\s+constexpr\s+(?:dxapp::)?GraphInput\s+graphInput\s*\(\s*\)\s*'
    r'\{\s*return\s+(?:dxapp::)?GraphInput::(kFullFrame|kRoi|kEither)\s*;\s*\}')
_TRAIT_PORTS = re.compile(
    r'static\s+constexpr\s+const\s+char\s*\*\s*graphPorts\s*\(\s*\)\s*'
    r'\{\s*return\s+"([a-z][a-z0-9_]*(?:,[a-z][a-z0-9_]*)*)"\s*;\s*\}')


def parse_graph_traits(text, relative):
    """(contract or None, ports tuple or None). A header that mentions a trait
    in a form this parser cannot read is an error, never the default.

    Any mention of graphInput/graphPorts in the header, comments included,
    must be exactly one declaration of the form _TRAIT_INPUT/_TRAIT_PORTS
    match (it may span lines). A comment that names a trait in a header that
    declares none, or quotes the declaration a second time, fails generation."""
    contract = ports = None
    if "graphInput" in text:
        found = _TRAIT_INPUT.findall(text)
        if len(found) != 1:
            raise GenError(
                "UNPARSEABLE TRAIT {}: declare it once, in the form `static constexpr "
                "GraphInput graphInput() {{ return GraphInput::kEither; }}`".format(relative))
        contract = found[0]
    if "graphPorts" in text:
        found = _TRAIT_PORTS.findall(text)
        if len(found) != 1:
            raise GenError(
                'UNPARSEABLE TRAIT {}: declare it once, in the form `static constexpr '
                'const char* graphPorts() {{ return "a,b"; }}`'.format(relative))
        ports = tuple(found[0].split(","))
    return contract, ports


# Config keys a factory header reads (U-62). A key read as text or a list is
# declared in ModelInfo::text_params; every other key is numeric, so a string
# for a key read outside the header is refused, never silently dropped.
_CONFIG_TEXT = re.compile(r'get\s*<\s*std::string\s*>\s*\(\s*"([A-Za-z0-9_]+)"')
_CONFIG_LIST = re.compile(r'get_string_list\s*\(\s*"([A-Za-z0-9_]+)"')
_CONFIG_NUMBER = re.compile(
    r'get\s*<\s*(?:float|double|bool|int|unsigned(?:\s+int)?|(?:std::)?size_t|long(?:\s+long)?)'
    r'\s*>\s*\(\s*"([A-Za-z0-9_]+)"')


def scan_config_kinds(text, relative):
    """Keys this factory reads as text: [(key, "kText"|"kTextList")], sorted.
    A key read both as text and as a number is a hard error."""
    kinds = {}
    for key in _CONFIG_TEXT.findall(text):
        kinds[key] = "kText"
    for key in _CONFIG_LIST.findall(text):
        if kinds.get(key, "kTextList") != "kTextList":
            raise GenError('CONFLICTING PARAM {}: "{}" is read as text and as a list'.format(relative, key))
        kinds[key] = "kTextList"
    for key in _CONFIG_NUMBER.findall(text):
        if key in kinds:
            raise GenError('CONFLICTING PARAM {}: "{}" is read as a number and as {}'.format(
                relative, key, "a list" if kinds[key] == "kTextList" else "text"))
    return sorted(kinds.items())


class FactoryInfo(object):
    """One variant's factory header, resolved."""

    def __init__(self, task, family, variant, header, cls, interface,
                 contract_override=None, ports_override=None, text_params=()):
        self.task = task          # <task> directory
        self.family = family      # <family> directory
        self.variant = variant    # <variant> directory: the model key
        self.header = header      # include path, relative to src/cpp_example
        self.cls = cls            # fully qualified C++ class name
        self.interface = interface
        self.contract_override = contract_override  # graphInput(), or None
        self.ports_override = ports_override        # graphPorts() tuple, or None
        self.text_params = list(text_params)       # [(key, "kText"|"kTextList")]

    def __repr__(self):
        return "FactoryInfo({}, {}, {})".format(self.variant, self.cls, self.interface)


def scan_factories(cpp_root):
    """variant -> FactoryInfo, for every <task>/<family>/<variant>/factory/<variant>_factory.hpp.

    Returns (found, errors) where errors is the list of human-readable
    problems; the caller decides when to stop, so every problem in the tree
    is reported in one run rather than one per invocation.
    """
    cpp_root = Path(cpp_root)
    found = {}
    errors = []
    by_class = defaultdict(list)
    by_variant = defaultdict(list)

    for header in sorted(cpp_root.glob("*/*/*/factory/*_factory.hpp")):
        relative = header.relative_to(cpp_root).as_posix()
        task, family, variant = header.parts[-5:-2]
        if task == "common":
            continue
        if header.name != variant + "_factory.hpp":
            errors.append("UNPLACED {}: a variant directory's factory is "
                          "factory/{}_factory.hpp".format(relative, variant))
            continue
        try:
            text = header.read_text(encoding="utf-8", errors="replace")
            classes = parse_factory_classes(text)
        except ParseError as exc:
            # Never a silent skip: see the module docstring.
            errors.append("UNPARSEABLE {}: {}".format(relative, exc))
            continue
        except OSError as exc:
            errors.append("UNREADABLE {}: {}".format(relative, exc))
            continue

        if len(classes) > 1:
            errors.append(
                "AMBIGUOUS {}: {} classes derive from an I...Factory ({}) - "
                "which one is the model's factory is a guess, and a wrong "
                "guess compiles".format(
                    relative, len(classes),
                    ", ".join("{}:{}".format(c, i) for c, i in classes)))
            continue

        try:
            contract, ports = parse_graph_traits(text, relative)
            text_params = scan_config_kinds(text, relative)
        except GenError as exc:
            errors.append(str(exc))
            continue

        cls, interface = classes[0]
        found[variant] = FactoryInfo(task, family, variant, relative, cls, interface,
                                     contract, ports, text_params)
        by_class[cls].append(relative)
        by_variant[variant].append(relative)

    for cls in sorted(by_class):
        if len(by_class[cls]) > 1:
            errors.append("DUPLICATE class {}: {}".format(
                cls, ", ".join(by_class[cls])))
    for variant in sorted(by_variant):
        if len(by_variant[variant]) > 1:
            errors.append("DUPLICATE VARIANT {}: {}".format(
                variant, ", ".join(by_variant[variant])))

    return found, errors


def _checked(value, field, model):
    text = "" if value is None else str(value)
    if not _SAFE_STRING.match(text):
        raise GenError("UNSAFE {} for model {}: {!r} would not survive being "
                       "written into a C++ string literal".format(
                           field, model, text))
    return text


def load_download_names(path):
    """dxnn filename -> the model-zoo manifest name, printed after
    `./setup.sh --models`. download_models.py accepts the manifest name, the
    .dxnn file name, and a registry model_name through the aliases it reads
    from config/model_registry.json - but those aliases are best effort (no
    registry file next to the script, no alias), while the manifest name is
    the one spelling that always matches. The join is the basename of the
    manifest's own dxnn_url.

    Nothing here fails the configure step. An unreadable manifest, or one
    that is not a JSON array, gives {} with a warning, and a name that would
    not survive being written into a C++ string literal is dropped with a
    warning; either way that model falls back to its registry name."""
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print("WARNING cannot read {} ({}); MODEL_MISSING will print registry "
              "names".format(path, exc), file=sys.stderr)
        return {}
    names = {}
    if not isinstance(entries, list):
        print("WARNING {} is not a JSON array of models; MODEL_MISSING will print "
              "registry names".format(path), file=sys.stderr)
        return names
    for item in entries:
        if not isinstance(item, dict):
            continue
        url, name = item.get("dxnn_url"), item.get("name")
        if isinstance(url, str) and isinstance(name, str):
            if not _SAFE_STRING.match(name):
                print("WARNING manifest name {!r} for {} is not safe in a C++ "
                      "string literal; MODEL_MISSING will print the registry "
                      "name for it".format(name, url.rsplit("/", 1)[-1]),
                      file=sys.stderr)
                continue
            names[url.rsplit("/", 1)[-1]] = name
    return names


def variant_config(cpp_root, info):
    """The values TypedStage's ModelConfig reads from the variant's config.json:
    the top-level non-object values with a top-level "config" object's on top
    (nested wins). {} when the variant has no config.json."""
    path = Path(cpp_root) / info.task / info.family / info.variant / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise GenError("UNREADABLE CONFIG {}: {}".format(
            path.relative_to(cpp_root).as_posix(), exc))
    if not isinstance(data, dict):
        raise GenError("UNREADABLE CONFIG {}: not a JSON object".format(
            path.relative_to(cpp_root).as_posix()))
    values = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested = data.get("config")
    if isinstance(nested, dict):
        values.update({k: v for k, v in nested.items() if not isinstance(v, dict)})
    return values


def companion_files(dxnn_file, roles):
    """EfficientAD's other networks, named as its factory names them: the
    primary's role in the .dxnn name (`-teacher_`) replaced by each other role."""
    if not isinstance(roles, list):
        return []
    primary = next((r for r in roles if isinstance(r, str)
                    and "-{}_".format(r) in dxnn_file), None)
    if primary is None:
        return []
    return [dxnn_file.replace("-{}_".format(primary), "-{}_".format(r), 1)
            for r in roles if isinstance(r, str) and r != primary]


def model_resources(row, config):
    """[(kind, text)]: what the model needs beside its .dxnn (R9)."""
    name = row["model_name"]
    resources = []
    gallery = config.get("gallery")
    if isinstance(gallery, str) and gallery:
        resources.append(("kGallery", _checked(gallery, "gallery", name)))
    if row["interface"] in NOT_READY_INTERFACES:
        for companion in companion_files(row["dxnn_file"], config.get("roles")):
            resources.append(("kCompanion", _checked(companion, "companion", name)))
    note = TASK_NOTES.get(row["task"])
    if note:
        resources.append(("kNote", note))
    return resources


def _graph_traits(info, errors):
    """(contract override or None, ports override or None) after GRAPH_TRAITS."""
    table = GRAPH_TRAITS.get(info.family, {})
    contract, ports = info.contract_override, info.ports_override
    if contract is not None and "input" in table and contract != table["input"]:
        errors.append("TRAIT MISMATCH {}: it declares graphInput() {}, but GRAPH_TRAITS "
                      "in scripts/gen_model_registry.py says {} for family {}".format(
                          info.header, contract, table["input"], info.family))
    if ports is not None and "ports" in table and tuple(ports) != tuple(table["ports"]):
        errors.append('TRAIT MISMATCH {}: it declares graphPorts() "{}", but GRAPH_TRAITS '
                      'in scripts/gen_model_registry.py says "{}" for family {}'.format(
                          info.header, ",".join(ports), ",".join(table["ports"]),
                          info.family))
    return (contract if contract is not None else table.get("input"),
            ports if ports is not None else table.get("ports"))


def build_entries(registry_path, cpp_root, strict, download_names):
    """(rows, aliases): one row per registered variant, in registry order, and
    [(name, variant, kind)]. Raises GenError after printing every problem."""
    entries = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    scope_problems = namespace_problems(Path(cpp_root))
    factories, errors = scan_factories(cpp_root)
    models, aliases, registry_errors = resolve_registry(entries)
    errors = scope_problems + errors + registry_errors

    # TypedStage reads <task>/<family>/<variant>/config.json from ModelInfo,
    # so the registry's task and family and the factory's directory must
    # agree. They are two independent statements of the same fact; a
    # disagreement is a build error, not something to paper over.
    for entry in entries:
        info = factories.get(entry.get("variant"))
        if info is not None and (entry.get("task"), entry.get("family")) != (info.task, info.family):
            errors.append(
                'TASK MISMATCH {}: model_registry.json row "{}" says task="{}", '
                'family="{}", but the factory lives in src/cpp_example/{}/{}/{}/ - '
                "TypedStage resolves <task>/<family>/<variant>/config.json from the "
                "registry, so these must agree".format(
                    info.variant, entry.get("model_name"), entry.get("task"),
                    entry.get("family"), info.task, info.family, info.variant))

    rows = []
    drift = []
    carriers = [e for e in entries if not e.get("alias_of") and models.get(e.get("variant")) is e]
    for entry in carriers:
        variant = entry["variant"]
        row = {
            "model_name": _checked(variant, "variant", variant),
            "variant": variant,
            "family": _checked(entry.get("family", ""), "family", variant),
            "task": _checked(entry.get("task", ""), "task", variant),
            "dxnn_file": _checked(entry.get("dxnn_file", ""), "dxnn_file", variant),
            "download_name": _checked(
                download_names.get(entry.get("dxnn_file", ""), ""),
                "download_name", variant),
            "published": entry.get("published") is True,
            "input_width": int(entry.get("input_width", 0)),
            "input_height": int(entry.get("input_height", 0)),
            "ready": False,
            "reason": "",
            "shape": "kBoxes",
            "contract": "kFullFrame",
            "landmarks": False,
            "header": None,
            "cls": None,
            "result": None,
            "ports": [],
            "declared_input": "kDefault",
            "declared_ports": "",
            "text_params": [],
            "interface": None,
            "resources": [],
        }

        info = factories.get(variant)
        if info is None:
            drift.append(variant)
            row["reason"] = "no factory"
            rows.append(row)
            continue
        row["interface"] = info.interface
        try:
            row["resources"] = model_resources(row, variant_config(cpp_root, info))
        except GenError as exc:
            errors.append(str(exc))
            continue

        if info.interface in NOT_READY_INTERFACES:
            companions = [text for kind, text in row["resources"] if kind == "kCompanion"]
            row["reason"] = NOT_READY_INTERFACES[info.interface] + (
                " (companion engines: {})".format(", ".join(companions)) if companions else "")
            rows.append(row)
            continue

        mapped = INTERFACE_TABLE.get(info.interface)
        if mapped is None:
            row["reason"] = "unmapped interface " + info.interface
            rows.append(row)
            continue

        shape, result, contract, landmarks, default_ports = mapped
        contract_override, declared = _graph_traits(info, errors)
        if contract_override:
            contract = contract_override
        port_rows = []
        ports = list(declared if declared is not None else default_ports)
        has_gallery = any(kind == "kGallery" for kind, _ in row["resources"])
        if has_gallery and result == GALLERY_PORT[0] and GALLERY_PORT[1] not in ports:
            ports.append(GALLERY_PORT[1])
        for port in ports:
            port_shape = PORT_SHAPES.get((result, port))
            if port_shape is None:
                known = sorted(n for r, n in PORT_SHAPES if r == result)
                errors.append('UNKNOWN PORT {}: {} has no port "{}" (known: {})'.format(
                    info.header, result, port, ", ".join(known) or "none"))
                continue
            if port == SHAPE_NAMES[shape]:
                errors.append('PORT NAME {}: "{}" is the primary output\'s name'.format(
                    info.header, port))
                continue
            port_rows.append((port, port_shape))
        # The static_asserts compare the compiler's view of the HEADER with
        # this script's parse of it; the table's values reach ModelInfo only.
        row.update({"ports": port_rows,
                    "declared_input": info.contract_override or "kDefault",
                    "declared_ports": ",".join(info.ports_override)
                    if info.ports_override is not None else ""})
        row.update({
            "ready": True,
            "shape": shape,
            "contract": contract,
            "landmarks": landmarks,
            "header": info.header,
            "cls": info.cls,
            "result": result,
            "text_params": info.text_params,
        })
        rows.append(row)

    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        if scope_problems:
            print("fix: " + VARIANT_SCOPE_FIX.format(root=cpp_root)
                  + "  (code outside a variant namespace must be moved by hand)",
                  file=sys.stderr)
        raise GenError("{} factory tree error(s); refusing to generate a "
                       "registry that would be wrong".format(len(errors)))

    if drift and strict:
        for name in sorted(drift):
            print("DRIFT registry variant has no factory directory: {}".format(name),
                  file=sys.stderr)
        raise GenError("{} name drift(s); run scripts/check_model_registry.py"
                       .format(len(drift)))

    return rows, aliases


def subset_entries(rows, aliases, variants):
    """(rows, aliases) of only `variants`, in registry order, with the aliases
    (old names, alias_of) of the rows kept: the registry of an exported graph
    project (scripts/export_graph_project.py). Raises GenError naming every
    variant the registry lacks."""
    wanted = set(variants)
    unknown = sorted(wanted - {row["variant"] for row in rows})
    if unknown:
        raise GenError("not in the registry: " + ", ".join(unknown))
    return ([row for row in rows if row["variant"] in wanted],
            [alias for alias in aliases if alias[1] in wanted])


def _split_units(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"] or "unassigned"].append(row)

    units = []
    for task, task_rows in sorted(by_task.items()):
        chunks = [task_rows[i:i + MODELS_PER_TU]
                  for i in range(0, len(task_rows), MODELS_PER_TU)]
        for n, chunk in enumerate(chunks):
            stem = task if len(chunks) == 1 else "{}_{}".format(task, n)
            units.append((stem, chunk))
    return units


def _write_if_changed(path, text):
    """Rewrite only on a real change, so a reconfigure does not force ninja
    to recompile 24 translation units that did not change."""
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except (OSError, UnicodeDecodeError):
        pass
    path.write_text(text, encoding="utf-8")


def _is_restoration(row):
    """A ready restoration row gets MakeRestorationStage, which tiles
    super-resolution models as their runner does (U-66)."""
    return row["ready"] and row["interface"] == "IRestorationFactory"


def emit_sources(rows, aliases, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units = _split_units(rows)

    for stem, chunk in units:
        lines = [
            "// GENERATED by scripts/gen_model_registry.py - do not edit.",
            '#include "common/registry/static_model_registry.hpp"',
            '#include "common/registry/typed_stage.hpp"',
        ]
        if any(_is_restoration(row) for row in chunk):
            lines.append('#include "common/registry/tiled_sr_stage.hpp"')
        for row in chunk:
            if row["header"]:
                lines.append('#include "{}"'.format(row["header"]))
        lines += ["", "namespace dxapp {", "namespace graph {", ""]
        lines.append("void RegisterGraphModels_{}(StaticModelRegistry* registry) {{".format(stem))
        for row in chunk:
            lines.append("  {")
            lines.append("    ModelInfo info;")
            lines.append('    info.model_name = "{}";'.format(row["model_name"]))
            lines.append('    info.variant = "{}";'.format(row["variant"]))
            lines.append('    info.family = "{}";'.format(row["family"]))
            lines.append('    info.task = "{}";'.format(row["task"]))
            lines.append('    info.dxnn_file = "{}";'.format(row["dxnn_file"]))
            if row["download_name"]:
                lines.append('    info.download_name = "{}";'.format(row["download_name"]))
            lines.append("    info.published = {};".format(
                "true" if row["published"] else "false"))
            lines.append("    info.input_width = {};".format(row["input_width"]))
            lines.append("    info.input_height = {};".format(row["input_height"]))
            lines.append("    info.output_shape = Shape::{};".format(row["shape"]))
            lines.append("    info.input_contract = InputContract::{};".format(row["contract"]))
            lines.append("    info.produces_landmarks = {};".format(
                "true" if row["landmarks"] else "false"))
            lines.append("    info.ready = {};".format("true" if row["ready"] else "false"))
            lines.append('    info.not_ready_reason = "{}";'.format(row["reason"]))
            for port, port_shape in row["ports"]:
                lines.append('    info.ports.push_back(PortInfo("{}", Shape::{}));'.format(
                    port, port_shape))
            for key, kind in row["text_params"]:
                lines.append('    info.text_params.push_back(ParamInfo("{}", ParamInfo::{}));'.format(
                    key, kind))
            for kind, text in row["resources"]:
                lines.append('    info.resources.push_back(ResourceInfo(ResourceInfo::{}, "{}"));'
                             .format(kind, text))
            if row["ready"]:
                lines.append("    static_assert(detail::DeclaredGraphInput<{}>(0) == "
                             "GraphInput::{},".format(row["cls"], row["declared_input"]))
                lines.append('                  "{}: graphInput() and gen_model_registry.py '
                             'disagree");'.format(row["header"]))
                lines.append('    static_assert(detail::SameText(detail::DeclaredGraphPorts<{}>(0), '
                             '"{}"),'.format(row["cls"], row["declared_ports"]))
                lines.append('                  "{}: graphPorts() and gen_model_registry.py '
                             'disagree");'.format(row["header"]))
                if _is_restoration(row):
                    lines.append("    registry->Add(info, &MakeRestorationStage<{}>);".format(
                        row["cls"]))
                else:
                    lines.append("    registry->Add(info, &MakeTypedStage<{}, {}>);".format(
                        row["cls"], row["result"]))
            else:
                lines.append("    registry->Add(info, NULL);")
            lines.append("  }")
        lines += ["}", "", "}  // namespace graph", "}  // namespace dxapp", ""]
        _write_if_changed(out_dir / "graph_registry_{}.cpp".format(stem),
                          "\n".join(lines))

    entry = ["// GENERATED by scripts/gen_model_registry.py - do not edit.",
             '#include "common/registry/static_model_registry.hpp"', "",
             "namespace dxapp {", "namespace graph {", ""]
    for stem, _ in units:
        entry.append("void RegisterGraphModels_{}(StaticModelRegistry*);".format(stem))
    entry += ["", "void RegisterAllGraphModels(StaticModelRegistry* registry) {"]
    for stem, _ in units:
        entry.append("  RegisterGraphModels_{}(registry);".format(stem))
    # After every Add(): an alias names a variant that is already registered.
    for name, variant, kind in aliases:
        entry.append('  registry->AddAlias("{}", "{}", ModelAlias::{});'.format(
            _checked(name, "alias", variant), variant,
            "kAliasOf" if kind == ALIAS_OF else "kLegacyName"))
    entry += ["}", "", "}  // namespace graph", "}  // namespace dxapp", ""]
    _write_if_changed(out_dir / "graph_registry_all.cpp", "\n".join(entry))

    return [str(out_dir / "graph_registry_{}.cpp".format(stem)) for stem, _ in units] + \
           [str(out_dir / "graph_registry_all.cpp")]


SHAPE_NAMES = {
    "kBoxes": "boxes", "kObBoxes": "obboxes", "kInstances": "instances",
    "kKeypoints": "keypoints", "kLabelMap": "labelmap",
    "kDenseMap": "densemap", "kImage": "image", "kScores": "scores",
    "kVector": "vector", "kBoxes3d": "boxes3d",
}
CONTRACT_NAMES = {"kFullFrame": "frame", "kRoi": "roi", "kEither": "either"}


def produces_cell(row):
    """The doc's `produces` cell: the primary shape, then the extra ports.
    `boxes + drivable, lane (labelmap)` when every port has one shape, else
    `keypoints + descriptors (densemap), pose (vector)`."""
    primary = SHAPE_NAMES.get(row["shape"], "?")
    ports = row.get("ports") or []
    if not ports:
        return primary
    shapes = [SHAPE_NAMES.get(shape, "?") for _, shape in ports]
    if len(set(shapes)) == 1:
        return "{} + {} ({})".format(
            primary, ", ".join(name for name, _ in ports), shapes[0])
    return "{} + {}".format(primary, ", ".join(
        "{} ({})".format(name, shape) for (name, _), shape in zip(ports, shapes)))


def _ready_cell(row):
    if not row["ready"]:
        return "NO - " + row["reason"]
    if row["shape"] == "kBoxes3d":
        return "yes, but refused in every graph: boxes3d needs a LiDAR point-cloud source"
    return "yes"


def _names(names):
    return ", ".join("`{}`".format(n) for n in names)


def render_docs(rows, aliases):
    ready = [r for r in rows if r["ready"]]
    lines = [
        "# Models usable in a graph",
        "",
        "GENERATED by `scripts/gen_model_registry.py --docs-only` - do not edit. "
        "`scripts/check_graph_models_doc.py` fails when it is stale.",
        "",
        "`produces` is the shape this model emits; `consumes` says whether it "
        "accepts a cropped region, a full frame, or either. A cascade edge "
        "needs a producer whose shape is `boxes`, `obboxes` or `instances` and "
        "a consumer that accepts `roi`. Extra outputs after `+` are named "
        "ports: an edge selects one with `\"port\"`, and `--report` lists "
        "them under `\"ports\"`. `published` is NO for a model whose `.dxnn` "
        "the model zoo does not publish yet, so `setup.sh` cannot download it.",
        "",
        "| model | task | produces | consumes | input | published | ready |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["task"], r["model_name"])):
        lines.append("| `{}` | {} | {} | {} | {}x{} | {} | {} |".format(
            row["model_name"], row["task"],
            produces_cell(row),
            CONTRACT_NAMES.get(row["contract"], "?"),
            row["input_width"], row["input_height"],
            "yes" if row["published"] else "NO",
            _ready_cell(row)))
    # U-19: count what a graph can run. The engine refuses a boxes3d node in
    # every graph (graph_config.cpp: "consumes LiDAR point clouds, not camera
    # frames"), and an unpublished model cannot be downloaded.
    lidar = [r["model_name"] for r in ready if r["shape"] == "kBoxes3d"]
    unpublished = [r["model_name"] for r in ready if not r["published"]]
    summary = "{} models registered, {} ready. {} run in a graph".format(
        len(rows), len(ready), len(ready) - len(lidar))
    summary += (": {} {} refused (boxes3d).".format(
        _names(lidar), "is" if len(lidar) == 1 else "are") if lidar else ".")
    if unpublished:
        summary += " {} ready model{} no published `.dxnn`: {}.".format(
            len(unpublished), " has" if len(unpublished) == 1 else "s have",
            _names(unpublished))
    lines += ["", summary, ""]
    lines += [
        "## Old names",
        "",
        "A model is named by its variant. A graph may still use a model's old "
        "registry name, or an `alias_of` name; `--check` resolves it to the "
        "variant with a note, and `--list-models` lists the `alias of` rows.",
        "",
        "| name | resolves to | kind |",
        "|---|---|---|",
    ]
    for name, variant, kind in sorted(aliases):
        lines.append("| `{}` | `{}` | {} |".format(
            name, variant, "alias of" if kind == ALIAS_OF else "old name"))
    lines += ["", "{} old names.".format(len(aliases)), ""]
    return "\n".join(lines)


def emit_docs(rows, aliases, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(path, render_docs(rows, aliases))


def check_docs(rows, aliases, path):
    """0 when the doc at path is what render_docs() would write, 1 when it is
    stale or missing. Writes nothing."""
    expected = render_docs(rows, aliases)
    try:
        actual = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        actual = None
    if actual == expected:
        print("{} is up to date".format(path))
        return 0
    print("STALE {}: it no longer matches config/model_registry.json and the "
          "factory tree".format(path), file=sys.stderr)
    if actual is None:
        print("  (the file is missing or unreadable)", file=sys.stderr)
    else:
        diff = list(difflib.unified_diff(
            actual.splitlines(), expected.splitlines(),
            "tracked", "generated", lineterm="", n=0))
        for line in diff[:40]:
            print(line, file=sys.stderr)
        if len(diff) > 40:
            print("  ... {} more diff line(s)".format(len(diff) - 40),
                  file=sys.stderr)
    print("Regenerate it with: " + DOCS_UPDATE_COMMAND, file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-root", default=str(DEFAULT_CPP),
                        help="factory tree to scan (tests point this elsewhere)")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--out-dir", default=str(ROOT / "build" / "generated"))
    parser.add_argument("--docs", default=None,
                        help="the model table; default <out-dir>/graph_models.md, or the "
                             "tracked docs/graph_models.md with --docs-only / --check-docs")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="model-zoo manifest: the download name of each .dxnn")
    parser.add_argument("--docs-only", action="store_true",
                        help="write only --docs (no sources): the tracked doc's update path")
    parser.add_argument("--check-docs", action="store_true",
                        help="write nothing; exit 1 when --docs is stale")
    parser.add_argument("--strict", action="store_true",
                        help="fail on registry/factory name drift (scripts/ci_checks.sh: codegen-strict)")
    parser.add_argument("--print-sources", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma-separated variants: emit a registry of only these "
                             "(an exported graph project, scripts/export_graph_project.py)")
    args = parser.parse_args()
    if args.only is not None:
        if args.docs_only or args.check_docs:
            parser.error("--only makes an exported project's registry; it never writes or checks the "
                         "tracked docs (--docs-only, --check-docs)")
        if not [v for v in args.only.split(",") if v]:
            parser.error("--only names no variant")
    if args.docs is None:
        tracked = args.docs_only or args.check_docs
        args.docs = str((ROOT / "docs" if tracked else Path(args.out_dir)) / "graph_models.md")

    try:
        rows, aliases = build_entries(args.registry, args.cpp_root, args.strict,
                                      load_download_names(args.manifest))
        if args.only:
            rows, aliases = subset_entries(rows, aliases,
                                           [v for v in args.only.split(",") if v])
    except GenError as exc:
        print("ERROR {}".format(exc), file=sys.stderr)
        return 1

    if args.check_docs:
        return check_docs(rows, aliases, args.docs)
    if args.docs_only:
        emit_docs(rows, aliases, args.docs)
        print("wrote {}".format(args.docs))
        return 0

    sources = emit_sources(rows, aliases, args.out_dir)
    emit_docs(rows, aliases, args.docs)

    if args.print_sources:
        sys.stdout.write(";".join(sources))
    else:
        ready = sum(1 for r in rows if r["ready"])
        print("{} models, {} graph-ready, {} old names, {} translation units".format(
            len(rows), ready, len(aliases), len(sources)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
