#!/usr/bin/env python3
"""Enforce i_registry.hpp rule 1: the engine never names a concrete registry
and never switches on a task string.

Reviewers forget. This does not. But read the "What this guard cannot catch"
section below before trusting it as a complete substitute for review — it is
a tripwire, not a parser.

Design note (Task 1's lesson): an earlier guard in this project used a regex
strategy that silently skipped files it could not parse, so a violation
sitting in one of those files passed unnoticed. To avoid repeating that:

  * This guard does not use a fixed list of "engine filenames" to check. It
    walks every .hpp/.cpp/.h/.cc file under common/graph/ (recursively) and
    checks all of them, except the test/ fixtures, which legitimately define
    and reference a concrete FakeModelRegistry. A new engine file added later
    (a new executor, a new router variant) is covered automatically — it does
    not need to be added to an allowlist for the guard to see it.
  * A file this guard cannot read (decode error, permission error, etc.) is
    reported as a GUARD FAILURE, never silently skipped. "Passed because it
    saw nothing" is treated as worse than "failed loudly".
  * An empty or missing common/graph/ directory is also a guard failure, not
    a silent pass — a guard that finds zero files to check and reports
    success is indistinguishable from a guard that is no longer wired up.
  * The patterns below deliberately over-match (several spellings of "the
    engine compares a task string" and "the engine names a concrete
    registry") rather than pinning to the exact identifiers used in this
    project's current code, because the next violation will not necessarily
    be spelled StaticModelRegistry or `task == "..."`.
  * This guard does not strip comments before scanning, on purpose: a
    comment-stripping regex is itself exactly the kind of "parser" that can
    silently misparse a file and hide a violation inside a string literal or
    an unusual comment style. Scanning raw text means a violation quoted
    inside a comment produces a false positive a reviewer has to wave
    through — worse than that would be a violation hidden by a stripping
    step that got a file wrong.

What this guard CATCHES:
  * A concrete registry type/symbol named or included in an engine file,
    however it is spelled (ModelRegistry/OperatorRegistry suffix, snake_case
    *_registry, or a bare #include of it) — two independent mechanisms:
      1. ENGINE_INCLUDE_ALLOWLIST / _resolve_include() / _include_violation():
         an engine file's #include is resolved the way this project's own
         compiler resolves it (against every -I root src/cpp_example/
         CMakeLists.txt wires up — see INCLUDE_ROOTS), quote or
         angle-bracket, identically; if it resolves to a real file inside
         this repo, that file's path must be on the allowed surface. A
         concrete registry renamed to something with no "registry" in its
         name or filename at all (e.g. a class `Zoo` in `zoo.hpp`) is still
         rejected because `zoo.hpp` was never on ENGINE_INCLUDE_ALLOWLIST —
         name-blindness does not help it, and spelling the include with
         `<>` instead of `""` does not help it either, because resolution
         does not branch on the quote character. An include that does NOT
         resolve to a real file is a VIOLATION unless it is recognizably
         external (KNOWN_EXTERNAL_HEADERS/KNOWN_EXTERNAL_PREFIXES) — this
         is fail-closed, not fail-open: an earlier draft guessed "internal"
         only for text that started with a known-internal prefix and let
         anything else through, which let a bare-relative phantom include
         (`#include "zoo_config.hpp"`, no internal-looking prefix, file not
         on disk) evade it; now the only way an unresolvable include is
         allowed is by matching the closed, short list of what this project
         genuinely depends on externally (the C++ standard library, OpenCV,
         dxrt), so a phantom internal header a codegen step hasn't produced
         yet is correctly flagged the same as any other bare relative
         include this guard has never heard of.
      2. REGISTRY_IN_INCLUDE_PATH: independently, any include path
         containing the substring "registry" (quoted or angle-bracket) is
         flagged by name on sight, whether or not it resolves to a real
         local file — this is not a superset/subset of check 1 in either
         direction; each catches cases the other does not.
  * A literal task-string comparison in several spellings (`task ==`,
    `.task_name() ==`, `switch (getTask())`, `task.compare(...)`, etc).
  * An engine file reaching outside its allowed dependency surface (see
    ENGINE_INCLUDE_ALLOWLIST / ENGINE_INCLUDE_PREFIXES / INCLUDE_ROOTS).

What this guard CANNOT CATCH (an honest limit, not an oversight):
  * A task decision routed through an indirect helper, e.g.
    `SomeTaskHelper(info) == kYoloFamily` or a macro/constexpr table that
    encodes the same branch without the literal word "task" or a string
    literal anywhere nearby. Recognizing that requires understanding what a
    helper function *does*, not just what it is *named* — that needs a real
    C++ semantic parse (a compiler front end), which is out of scope for a
    dependency-free regex script. **This class of violation is the task
    review's responsibility, not this guard's** — a human (or a reviewing
    agent) reading a new helper function's body is what catches "this is a
    task switch wearing a trenchcoat".
  * Anything hidden behind conditional compilation (#ifdef) that changes
    meaning per build, since this guard reads one text form of the file.
This guard is a tripwire that raises the cost of adding a violation. It is
not a proof that none exists.

Self-tested (see task-5-report.md) across four review-and-fix rounds after
this guard's first draft: a concrete registry named outside the two literal
names in that first draft; a concrete registry pulled in only via
#include; three different spellings of a task-string switch; an unreadable
(non-UTF-8) file; an empty/missing common/graph/ directory (first draft). A
registry class named `Zoo` with no "registry" anywhere in its name or
filename, included from a file outside ENGINE_INCLUDE_ALLOWLIST, both as a
bare `"zoo.hpp"` and as a full `"common/graph/zoo.hpp"` sibling path (fix
round 1). The SAME `Zoo` evasion again spelled with `<>` instead of `""` —
the hole round 1 actually left open, since round 1 still branched on the
quote character — plus `registry_utils.hpp` (round 1's own regression,
restored), a genuine system header, a genuine OpenCV header, a genuine
dxrt header, and a legitimate allowlisted engine header, each tried BOTH
quoted and angle-bracket (fix round 2). A phantom include — an
internal-shaped path (`common/graph/zoo_config.hpp`) that does not exist
on disk at scan time, e.g. produced by a codegen build step — in both
quote forms; a phantom that IS genuinely external (`some_vendor/thing.hpp`)
in both quote forms, which must stay clean; a real allowlisted header,
which must also stay clean; and the INCLUDE_ROOTS drift tripwire itself,
fired by removing one of its expected substrings from a copy of
CMakeLists.txt, then cleared by restoring it, then fired again by removing
CMakeLists.txt entirely (fix round 3). A BARE-relative phantom include
(`#include "zoo_config.hpp"` / `#include <zoo_config2.hpp>`, no
"common/"-shaped prefix at all, file not on disk — the round-3 fix's own
gap) in both quote forms; a phantom rooted for a non-"common/" name; a
genuine stdlib header (both `<vector>` and `"vector"` forms, plus the
`<c*>`/`*.h` C-compatibility forms), a genuine `opencv2/` header and a
genuine `dxrt/` header, all confirmed clean; an unresolvable header NOT in
KNOWN_EXTERNAL (`<boost/optional.hpp>`), confirmed flagged; the drift
tripwire's scoping itself, reproduced exactly as a reviewer did against a
copy of the real production CMakeLists.txt — removing `${PROJECT_ROOT}/
src/utility/` from only the two graph-relevant target blocks while leaving
it in the top-level `include_directories()`, `dxapp_common_obj`'s own
block, and the source list — now correctly fires FATAL instead of passing
clean; and a target block deleted from CMakeLists.txt entirely, which also
correctly fires FATAL rather than assuming the transcription is still fine
(fix round 4). All violations fail loudly; none pass silently, and nothing
legitimate was ever rejected.
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CPP_EXAMPLE_DIR = PROJECT_ROOT / "src" / "cpp_example"
GRAPH_DIR = CPP_EXAMPLE_DIR / "common" / "graph"
SOURCE_SUFFIXES = (".hpp", ".cpp", ".h", ".cc")

# Fixture code: FakeModelRegistry is a legitimate concrete IModelRegistry
# that tests inject explicitly. It is never included by the engine itself
# (that is exactly what this guard checks for everywhere else).
EXEMPT_DIRS = {GRAPH_DIR / "test"}

# A concrete registry type: anything ending in ModelRegistry or
# OperatorRegistry other than the interfaces themselves. Matches
# StaticModelRegistry, FakeModelRegistry, DxModelRegistry, a disguised
# IModelRegistryImpl, PluginOperatorRegistry, etc. — not just the two
# literal names a first draft of this guard happened to spell out.
_INTERFACE_NAMES = ("IModelRegistry", "IOperatorRegistry")
CONCRETE_REGISTRY_TYPE = re.compile(
    r'\b(?!(?:%s)\b)[A-Za-z_][A-Za-z0-9_]*(?:ModelRegistry|OperatorRegistry)[A-Za-z0-9_]*\b'
    % "|".join(_INTERFACE_NAMES)
)
# snake_case concrete registry symbols, e.g. graph_registry_impl,
# static_model_registry. i_registry is this project's own interface header
# stem (i_registry.hpp) and is the one snake_case "*_registry" spelling
# that is allowed everywhere.
CONCRETE_REGISTRY_SNAKE = re.compile(
    r'\b(?!i_registry\b)[a-z][a-z0-9]*_registry(?:_[a-z0-9]+)*\b'
)

# --- Structural include allowlist -------------------------------------
# An engine file under common/graph/ (i.e. every file this guard checks —
# see discover_files()) may #include only:
#   (a) another header that is a direct sibling of i_registry.hpp inside
#       common/graph/ itself — ENGINE_INCLUDE_ALLOWLIST is that explicit,
#       exact set of filenames. It is a literal list on purpose: adding a
#       legitimate new engine header (graph_config.hpp, graph_runner_sync.hpp,
#       roi_router.hpp, graph_visualizer.hpp, stage_graph.hpp, ...) is meant
#       to be a deliberate one-line addition to THIS list as part of that
#       task's own diff, not something that passes for free because it
#       happens to live in the right directory. A file placed directly under
#       common/graph/ that is NOT on this list — or a file placed anywhere
#       under common/graph/test/ other than by the exempted test/ directory
#       rule below — fails, regardless of what its class is named. This is
#       the check that catches a concrete registry renamed to something with
#       no "registry" in its name or filename at all (e.g. a class `Zoo` in
#       `zoo.hpp`): `zoo.hpp` is simply not on this list.
#   (b) a header under one of ENGINE_INCLUDE_PREFIXES — the framework the
#       engine is allowed to consume (base result structs, processors,
#       trackers, inputs, visualizers).
#   (c) a genuinely EXTERNAL header (standard library, OpenCV, dxrt) — see
#       _resolve_include() below. This is decided by asking the filesystem
#       whether the include resolves to a real file inside this repository,
#       via every -I root src/cpp_example/CMakeLists.txt actually wires up
#       for a common/graph/ translation unit. It is NOT decided by the
#       quote character. Round-1 of this guard branched on `<` vs `"` and
#       treated every angle-bracket include as external; that was wrong —
#       src/cpp_example/ is on this project's plain -I path (not -isystem),
#       so `#include <common/graph/zoo.hpp>` resolves and COMPILES exactly
#       like the quoted form (confirmed: CMakeLists.txt:130-137, target_
#       include_directories(... PRIVATE ${CPP_EXAMPLE_DIR} ...) with no
#       SYSTEM keyword). A reviewer reproduced the round-1 evasion end to
#       end with that one-character change and got a clean pass. Branching
#       on the quote character answers a question the compiler does not
#       ask; resolving the path is the question the compiler actually asks,
#       so that is what this guard asks too.
# Anything else — a bare relative include, a path under common/graph/test/,
# a path outside common/ entirely, a concrete registry pulled in from
# somewhere else — is a violation, independent of the name-based checks
# above and independent of what class the included file defines.
ENGINE_INCLUDE_ALLOWLIST = frozenset([
    "i_registry.hpp",
    "shape.hpp",
    "result_to_shape.hpp",
    "graph_error.hpp",
    "graph_config.hpp",
    "roi_router.hpp",
    "stage_graph.hpp",
    "graph_runner_sync.hpp",
    "graph_runner_async.hpp",
    "graph_visualizer.hpp",
    # The engine's own CPU nodes ("type": "cpu", ops headpose and volume):
    # both executors call ReduceCpu once a cpu node's "carry" parents have
    # finished. It includes only i_registry.hpp and shape.hpp from the graph
    # surface and names no concrete registry and no task string.
    "cpu_reduce.hpp",
    # Logic nodes ("type": "logic"): the op catalog, the typed config, the
    # per-stream state and RunLogic. The header includes only graph_config.hpp
    # and shape.hpp and forward-declares RetrievalGallery; the DXGAL1 reader,
    # common/processors/gallery_retrieval_postprocessor.hpp, is included by
    # logic_ops.cpp (MakeLogicConfig, RunGallery) and graph_config.cpp
    # (--check reads the gallery file). Names no concrete registry and no
    # task string.
    "logic_ops.hpp",
    # The user-logic hook ("op": "user"): the class a program registers under
    # a user node's name (an exported project's user_logic/), run by
    # logic_ops.cpp's RunUser. It includes only logic_ops.hpp and shape.hpp
    # from the graph surface and names no concrete registry and no task
    # string.
    "user_logic.hpp",
])
ENGINE_INCLUDE_PREFIXES = (
    "common/base/",
    "common/processors/",
    "common/trackers/",
    "common/inputs/",
    "common/visualizers/",
    # Vendored framework surface, exactly like common/base/ — the framework
    # already consumes it from common/utility/verify_serialize.hpp. Task 6
    # (graph_config.cpp) includes common/third_party/nlohmann_json.hpp as
    # the graph parser's JSON library; the spec names nlohmann for this
    # purpose and it is already vendored, not a new third-party dependency.
    "common/third_party/",
)
# Single shared headers outside those prefixes, by exact path. ROI crops use
# common/utility/roi_crop.hpp (I17); the rest of common/utility/ stays outside
# the engine's surface.
ENGINE_INCLUDE_SHARED_HEADERS = frozenset([
    "common/utility/roi_crop.hpp",
])
ALL_INCLUDE = re.compile(r'#\s*include\s*[<"]([^">]+)[>"]')

# Every -I root src/cpp_example/CMakeLists.txt wires up for a common/graph/
# translation unit (dxapp_graph_obj / graph_engine_test), lines 108-137 and
# 146-153, plus the including file's own directory (checked separately,
# first, in _resolve_include — true "" semantics). OpenCV_INCLUDE_DIRS is
# deliberately excluded: those are genuinely external (not inside this
# repository), which is exactly the case this list does not need to name.
#
# Paired with the literal substring each root's own relative path segment
# must appear as somewhere in CMakeLists.txt — check_include_roots_drift()
# greps for these at guard start-up. Not a CMake parser (this project's own
# docstring above rejects a parser that can silently misparse); a plain
# substring `in` check cannot misparse, it can only be stale, and staleness
# is exactly what this catches.
_INCLUDE_ROOT_SPECS = (
    (CPP_EXAMPLE_DIR, "${CPP_EXAMPLE_DIR}"),
    (CPP_EXAMPLE_DIR / "common" / "processors", "${CPP_EXAMPLE_DIR}/common/processors"),
    (PROJECT_ROOT / "src" / "utility", "src/utility"),
    (PROJECT_ROOT / "extern", "${PROJECT_ROOT}/extern"),
    (PROJECT_ROOT / "src" / "postprocess" / "sfa3d", "src/postprocess/sfa3d"),
    (PROJECT_ROOT / "src" / "postprocess" / "superpoint", "src/postprocess/superpoint"),
    (PROJECT_ROOT / "src" / "postprocess" / "dope", "src/postprocess/dope"),
    (PROJECT_ROOT / "src" / "postprocess" / "yolopv2", "src/postprocess/yolopv2"),
    (PROJECT_ROOT / "src" / "postprocess" / "vitpose", "src/postprocess/vitpose"),
)
INCLUDE_ROOTS = tuple(root for root, _substring in _INCLUDE_ROOT_SPECS)
CMAKE_LISTS_PATH = CPP_EXAMPLE_DIR / "CMakeLists.txt"

# --- Fail-closed classification of unresolvable includes ---------------
# An include that _resolve_include() cannot find on disk is a VIOLATION
# unless it is recognizably external. This is the opposite default from an
# earlier draft of this guard, which treated "not found" as "external"
# unless the text happened to start with a known-internal prefix
# (OWNED_PATH_PREFIXES = ("common/",)) — that is backwards for a guard
# whose own docstring says it never silently passes: it let a bare-relative
# phantom include (`#include "zoo_config.hpp"`, no "common/" prefix, file
# not on disk) evade both the resolution check and the shape check, even
# though the identical text, had the file existed, would have resolved via
# including_file.parent into common/graph/ and been rejected.
#
# KNOWN_EXTERNAL is deliberately a CLOSED, SHORT list of what is outside
# this repository — the C++ standard library, OpenCV, dxrt — rather than an
# open list of what is inside. A closed list of externals is far shorter
# and far more stable than an open list of internals: this project depends
# on a small, slow-changing set of external libraries, and adding a new one
# is rare enough that requiring a deliberate one-line edit here is not a
# burden. Anything NOT on this list that does not resolve to a real file is
# flagged, naming the file, so a maintainer adding a genuinely new external
# dependency extends this list on purpose instead of the guard silently
# assuming the best.
#
# Extensionless C++ standard library and C-compatibility headers actually
# reachable from a common/graph/ translation unit in this project (C++14).
# Built from the canonical C-library basenames so the c<name>/<name>.h pairs
# cannot drift out of sync with each other by a typo.
_C_LIBRARY_BASENAMES = (
    "assert", "ctype", "errno", "fenv", "float", "inttypes", "iso646",
    "limits", "locale", "math", "setjmp", "signal", "stdalign", "stdarg",
    "stdbool", "stddef", "stdint", "stdio", "stdlib", "string", "tgmath",
    "time", "uchar", "wchar", "wctype",
)
KNOWN_EXTERNAL_HEADERS = frozenset(
    {"c" + name for name in _C_LIBRARY_BASENAMES}       # <cstdio>, <cstring>, ...
    | {name + ".h" for name in _C_LIBRARY_BASENAMES}    # <stdio.h>, <string.h>, ...
    | {
        # C++-only extensionless standard library headers.
        "algorithm", "array", "atomic", "bitset", "chrono", "codecvt",
        "complex", "condition_variable", "deque", "exception",
        "forward_list", "fstream", "functional", "future",
        "initializer_list", "iomanip", "ios", "iosfwd", "iostream",
        "istream", "iterator", "limits", "list", "locale", "map", "memory",
        "mutex", "new", "numeric", "ostream", "queue", "random", "ratio",
        "regex", "scoped_allocator", "set", "sstream", "stack",
        "stdexcept", "streambuf", "string", "strstream", "system_error",
        "thread", "tuple", "type_traits", "typeindex", "typeinfo",
        "unordered_map", "unordered_set", "utility", "valarray", "vector",
    }
    | {
        # A single exact KNOWN_EXTERNAL_HEADERS entry — not a "sys/" prefix,
        # do not widen this to one. sys/stat.h is used by stage_graph.cpp
        # (Task 9) for the --model-dir artifact existence check (stat());
        # genuinely external (libc), not a project header.
        "sys/stat.h",
    }
)
# External libraries reached by directory prefix rather than a single
# extensionless name.
KNOWN_EXTERNAL_PREFIXES = ("opencv2/", "dxrt/")


def _is_known_external(normalized_token):
    if normalized_token in KNOWN_EXTERNAL_HEADERS:
        return True
    return normalized_token.startswith(KNOWN_EXTERNAL_PREFIXES)


# Note: Path.is_file() resolution below is case-sensitive on this
# project's ext4 checkout, matching a real compile on the same filesystem;
# it would be case-insensitive (and so slightly more permissive) on a
# case-insensitive filesystem (e.g. default macOS/Windows). Not exploitable
# here — noted for anyone porting this guard to run on one.
def _resolve_include(including_file, included_token):
    """Resolve included_token the way this project's compiler resolves it
    for a common/graph/ translation unit: against the including file's own
    directory first, then against every -I root above. Returns the
    resolved Path if included_token names a real file inside this
    repository, or None if it does not — which could mean either a genuine
    external header (stdlib, OpenCV, dxrt/...) or an internal-shaped
    phantom that does not exist on disk yet. This function does not (and
    should not) try to tell those two apart: that is not a filesystem
    question, and _include_violation() answers it by checking
    KNOWN_EXTERNAL_HEADERS/KNOWN_EXTERNAL_PREFIXES — failing closed (a
    violation) for anything not on that list, per the M1 ruling."""
    for root in (including_file.parent,) + INCLUDE_ROOTS:
        candidate = root / included_token
        if candidate.is_file():
            return candidate.resolve()
    return None


def _classify_canonical_path(canonical):
    """Classify a path already known/assumed to be repo-relative-to-
    src/cpp_example/ (e.g. "common/graph/zoo.hpp") against the engine's
    allowed surface. Shared by both the resolves-to-a-real-file case and
    the phantom-but-owned-shape case in _include_violation, so the two
    paths through that function apply identical rules."""
    if canonical.startswith(ENGINE_INCLUDE_PREFIXES):
        return None  # the framework surface the engine may consume
    if canonical in ENGINE_INCLUDE_SHARED_HEADERS:
        return None  # a shared header named on its own, see above
    if canonical.startswith("common/graph/"):
        remainder = canonical[len("common/graph/"):]
        if "/" in remainder:
            # A subdirectory of common/graph/ (e.g. test/) — never a
            # "sibling", regardless of what the file is named.
            return "path under common/graph/ outside the sibling allowlist"
        if remainder in ENGINE_INCLUDE_ALLOWLIST:
            return None
        return "common/graph/ header not on ENGINE_INCLUDE_ALLOWLIST"
    return "outside the engine's allowed include surface"


def _include_violation(including_file, included_token):
    """Return a human-readable reason the include is NOT allowed, or None
    if it is allowed.

    Fails closed for anything that does not resolve to a real file, per
    the M1 ruling — "does not resolve on disk" is not the same claim as
    "is genuinely external", and treating the two as equivalent is exactly
    what let a bare-relative phantom include (`#include "zoo_config.hpp"`,
    no internal-looking prefix, file not on disk) evade an earlier draft
    of this guard:

      1. Resolves to a real file inside the repo -> classify by its
         canonical path (the common case).
      2. Does not resolve, but the text is a recognized external header
         (KNOWN_EXTERNAL_HEADERS/KNOWN_EXTERNAL_PREFIXES) -> allowed; this
         is the ONLY way an unresolvable include is allowed.
      3. Does not resolve and is not a recognized external header ->
         VIOLATION, regardless of what the text looks like. A build step
         (configure_file, add_custom_command, codegen) that produces an
         internal header this checkout does not have yet still hits this
         branch and is correctly flagged, exactly like any other bare
         relative include this guard has never heard of.
    """
    resolved = _resolve_include(including_file, included_token)
    if resolved is not None:
        try:
            canonical = resolved.relative_to(CPP_EXAMPLE_DIR).as_posix()
        except ValueError:
            # Resolves inside the repo but outside src/cpp_example/ (e.g.
            # under extern/ or src/utility/ or a postprocess/ dir) — not
            # part of the engine's authorized consumption surface.
            return "resolves inside the repo but outside the engine's allowed surface"
        return _classify_canonical_path(canonical)

    normalized = included_token.replace("\\", "/")
    if _is_known_external(normalized):
        return None
    return ("cannot be resolved to a file inside this repository and is not "
            "a recognized external header — if this is a new external "
            "dependency, extend KNOWN_EXTERNAL_HEADERS/KNOWN_EXTERNAL_PREFIXES "
            "deliberately")


# Belt-and-braces alongside the path-resolution check above, NOT subsumed
# by it and NOT a superset relationship in either direction: any include
# path containing "registry" as a substring, quoted or angle-bracket,
# other than the interface header itself. This independently catches e.g.
# `#include <common/graph/registry_utils.hpp>` — a name that neither ends
# in ModelRegistry/OperatorRegistry/_registry (so the name-based checks
# miss it) nor needs to resolve to a real file to be worth flagging on
# sight (so a reviewer sees it even before running the guard against a
# real checkout). Round 1 of this guard had this check, and the round-2
# rewrite's claim that the path-resolution check "subsumed" it was false —
# demonstrated by exactly the registry_utils.hpp example above, which the
# path check alone does not catch (it isn't on ENGINE_INCLUDE_ALLOWLIST,
# true, but that only fires if the file resolves locally; written as a
# bare include with no local file present it resolves to None and passes
# the path check silently). Restored here, kept, not replaced again.
REGISTRY_IN_INCLUDE_PATH = re.compile(
    r'#\s*include\s*[<"]([^">]*registry[^">]*)[>"]', re.IGNORECASE)

# A "task"-like identifier: task, Task, task_name, getTask(), obj->task,
# obj.task_name(), etc. Deliberately permissive about the prefix and the
# optional call parens so `model.task ==`, `info->task_name() ==`,
# `getTask() ==` and plain `task ==` are all covered by one pattern instead
# of one literal spelling.
_TASK_IDENT = r'(?:[\w:>.]*[\->.])?(?:task|Task)(?:_?[Nn]ame)?(?:\s*\(\s*\))?'
TASK_STRING_COMPARE = re.compile(
    r'%s\s*(?:==|!=)\s*"' % _TASK_IDENT
    + r'|"\s*(?:==|!=)\s*%s' % _TASK_IDENT
)
TASK_SWITCH_STATEMENT = re.compile(r'\bswitch\s*\(\s*%s\s*\)' % _TASK_IDENT)
TASK_STRING_COMPARE_CALL = re.compile(
    r'\b(?:strcmp|strcasecmp)\s*\([^,]*%s' % _TASK_IDENT
    + r'|%s\s*\.\s*compare\s*\(' % _TASK_IDENT
)

CHECKS = (
    (CONCRETE_REGISTRY_TYPE, "names a concrete registry type"),
    (CONCRETE_REGISTRY_SNAKE, "names a concrete registry symbol"),
    (TASK_STRING_COMPARE, "switches on task name (string compare)"),
    (TASK_SWITCH_STATEMENT, "switches on task name (switch statement)"),
    (TASK_STRING_COMPARE_CALL, "switches on task name (compare call)"),
)


# M2 ruling: the tripwire must search only the CMakeLists.txt blocks it
# claims to verify, not the whole file. src/cpp_example/CMakeLists.txt
# repeats several of the _INCLUDE_ROOT_SPECS substrings in unrelated
# places — a top-level include_directories() call, dxapp_common_obj's own
# target_include_directories() block, and the postprocess source list —
# so a whole-file grep stays green even when the graph-relevant target
# blocks themselves lose a root (a realistic accidental CMakeLists.txt
# refactor, not a contrived attack). The two graph-relevant call sites are
# these targets' own target_include_directories() blocks:
GRAPH_CMAKE_TARGETS = ("dxapp_graph_obj", "graph_engine_test")
# Both blocks reach their postprocess roots only through this variable, not
# by writing the literal path inline (see CMakeLists.txt lines 130-137 and
# 146-153, both of which end in "${DXAPP_POSTPROCESS_INCLUDES}") — so the
# postprocess entries in _INCLUDE_ROOT_SPECS are truly and singularly
# defined here, and this is the third block in scope alongside the two
# targets above, not a loosening of the M2 scoping: it is still not the
# top-level include_directories(), not dxapp_common_obj's own block, and
# not the source list — the three places the reviewer's reproduction left
# the substring in on purpose to prove the whole-file grep was too loose.
GRAPH_CMAKE_VARIABLE_DEFS = ("DXAPP_POSTPROCESS_INCLUDES",)
# Each graph target must list the variable itself: the substrings below are
# otherwise found in its set() block alone, and a target that dropped the
# token would pass (U-52).
GRAPH_TARGET_INCLUDE_TOKEN = "${DXAPP_POSTPROCESS_INCLUDES}"

_TARGET_INCLUDE_BLOCK = re.compile(
    r'target_include_directories\s*\(\s*(\w+)\b.*?\)', re.DOTALL)
_SET_BLOCK = re.compile(r'set\s*\(\s*(\w+)\b.*?\)', re.DOTALL)


def _extract_named_block(pattern, text, name):
    """Return the literal text of the first regex match whose captured
    identifier equals `name`, from that call's opening paren to its FIRST
    closing paren. src/cpp_example/CMakeLists.txt never nests parentheses
    inside these calls (verified by inspection — no $(...) or function
    calls appear inside them), so "first close paren after the call" is
    the matching one. A file that ever did nest parens here would need a
    real parser, which this guard deliberately does not carry (see the
    module docstring's stance on not risking a silent misparse); it would
    fail this extraction and correctly report FATAL rather than silently
    matching the wrong, truncated span."""
    for match in pattern.finditer(text):
        if match.group(1) == name:
            return match.group(0)
    return None


def check_include_roots_drift():
    """Cheap tripwire, not a CMake parser (a real parse risks the exact
    "silent misparse" failure mode this script's docstring rejects for
    C++). Fails loudly if any _INCLUDE_ROOT_SPECS substring is no longer
    present within the specific CMakeLists.txt blocks this script's
    INCLUDE_ROOTS transcription actually depends on (GRAPH_CMAKE_TARGETS'
    target_include_directories() calls and GRAPH_CMAKE_VARIABLE_DEFS'
    set() definitions) — NOT the whole file (see the M2 comment above
    GRAPH_CMAKE_TARGETS for why that matters). Also fails loudly, rather
    than silently assuming everything is fine, if any of those blocks
    cannot be located at all — that means this guard has lost its footing
    and must say so. Also fails loudly if a GRAPH_CMAKE_TARGETS block no
    longer lists GRAPH_TARGET_INCLUDE_TOKEN itself: its roots are then not
    the ones INCLUDE_ROOTS transcribes, even though the set() block still
    carries every substring (U-52). Returns True when the transcription
    still checks out.
    """
    if not CMAKE_LISTS_PATH.is_file():
        print("FATAL: {} not found — cannot verify INCLUDE_ROOTS against it".format(
            CMAKE_LISTS_PATH))
        return False
    text = CMAKE_LISTS_PATH.read_text(encoding="utf-8")

    scoped_blocks = []
    ok = True
    for target in GRAPH_CMAKE_TARGETS:
        block = _extract_named_block(_TARGET_INCLUDE_BLOCK, text, target)
        if block is None:
            print("FATAL: could not locate target_include_directories({} ...) "
                  "in {} — this guard cannot verify INCLUDE_ROOTS without it, "
                  "and will not assume it is fine".format(target, CMAKE_LISTS_PATH))
            ok = False
            continue
        # A "# ${...}" comment left in the block lists nothing; CMake has no
        # '#' inside these include lists, so stripping to end of line is safe.
        if GRAPH_TARGET_INCLUDE_TOKEN not in re.sub(r"#[^\n]*", "", block):
            print("FATAL: target_include_directories({} ...) in {} no longer lists {}; "
                  "its postprocess include roots are then not the ones this guard "
                  "resolves against".format(target, CMAKE_LISTS_PATH, GRAPH_TARGET_INCLUDE_TOKEN))
            ok = False
            continue
        scoped_blocks.append(block)
    for varname in GRAPH_CMAKE_VARIABLE_DEFS:
        block = _extract_named_block(_SET_BLOCK, text, varname)
        if block is None:
            print("FATAL: could not locate set({} ...) in {} — this guard "
                  "cannot verify INCLUDE_ROOTS without it, and will not "
                  "assume it is fine".format(varname, CMAKE_LISTS_PATH))
            ok = False
            continue
        scoped_blocks.append(block)
    if not ok:
        return False

    scoped_text = "\n".join(scoped_blocks)
    missing = [substring for _root, substring in _INCLUDE_ROOT_SPECS
               if substring not in scoped_text]
    if not missing:
        return True
    print("FATAL: INCLUDE_ROOTS drift detected — the following substring(s) "
          "are no longer present in {}'s {}/{} block(s):".format(
              CMAKE_LISTS_PATH, GRAPH_CMAKE_TARGETS, GRAPH_CMAKE_VARIABLE_DEFS))
    for substring in missing:
        print("  {!r}".format(substring))
    print("Re-check the INCLUDE_ROOTS / _INCLUDE_ROOT_SPECS transcription in "
          "this script against those blocks specifically — a substring "
          "present ELSEWHERE in the file (a different target, a top-level "
          "include_directories(), a source list) does not count.")
    return False


def discover_files():
    if not GRAPH_DIR.is_dir():
        print("FATAL: {} does not exist or is not a directory".format(GRAPH_DIR))
        sys.exit(2)
    files = []
    for path in sorted(GRAPH_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(exempt == path or exempt in path.parents for exempt in EXEMPT_DIRS):
            continue
        files.append(path)
    if not files:
        print("FATAL: no {} files found under {} — a guard that checks zero "
              "files is not a guard".format(SOURCE_SUFFIXES, GRAPH_DIR))
        sys.exit(2)
    return files


def check_file(path):
    """Return the number of violations found in path. Raises on unreadable
    files so the caller can turn that into a hard failure, not a skip."""
    text = path.read_text(encoding="utf-8")  # let decode errors propagate
    rel = path.relative_to(GRAPH_DIR)
    violations = 0

    for pattern, label in CHECKS:
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            print("{}:{}: engine {}: {!r}".format(rel, line, label, match.group(0).strip()))
            violations += 1

    for match in ALL_INCLUDE.finditer(text):
        included = match.group(1)
        reason = _include_violation(path, included)
        if reason is None:
            continue
        line = text[: match.start()].count("\n") + 1
        print("{}:{}: engine includes a header outside its allowed surface "
              "({}): {}".format(rel, line, reason, included))
        violations += 1

    for match in REGISTRY_IN_INCLUDE_PATH.finditer(text):
        included = match.group(1)
        if included.endswith("i_registry.hpp"):
            continue
        line = text[: match.start()].count("\n") + 1
        print("{}:{}: engine includes a header naming a registry: {}".format(
            rel, line, included))
        violations += 1

    return violations


def main():
    if not check_include_roots_drift():
        return 2

    failures = 0
    unreadable = 0
    files = discover_files()

    for path in files:
        try:
            failures += check_file(path)
        except (UnicodeDecodeError, OSError) as exc:
            # Never silently skip: a file the guard could not read is a
            # guard failure, not a pass. This is the exact failure mode a
            # prior guard in this project had.
            print("{}: GUARD FAILURE — could not read file ({}); this file "
                  "was NOT verified".format(path.relative_to(GRAPH_DIR), exc))
            unreadable += 1

    total_failures = failures + unreadable
    if total_failures:
        print("{} boundary violation(s), {} unreadable file(s), across "
              "{} file(s) checked".format(failures, unreadable, len(files)))
        return 1
    print("graph engine boundary clean ({} file(s) checked)".format(len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
