"""
Shared utility functions for C++ and Python test scripts.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from .constants import (
    EXCLUDED_EXAMPLE_VARIANTS,
    BIN_DIR,
    LIB_DIR,
    MODELS_DIR,
    MULTI_MODEL_EXECUTABLES,
    PROJECT_ROOT,
    REGISTRY_PATH,
    SKIP_MODELS,
    TASK_IMAGE_MAP,
    MODEL_IMAGE_OVERRIDE,
)
from .platform_paths import binary_path, exe_filename, resolve_bin_dir  # noqa: F401 (re-exported)
from .proc import apply_headless_display


# ======================================================================
# Normalisation helpers
# ======================================================================

# ======================================================================
# Module constants
# ======================================================================
_DXNN_GLOB = "*.dxnn"
_DEFAULT_IMAGE = "sample/img/sample_kitchen.jpg"
_SKIP_DIRS = frozenset({"common", "__pycache__"})

def family_dxnn_map() -> dict:
    """``{family -> [Path, ...]}`` -- every model a family's example can run.

    Under the dx-modelzoo family/variant layout an executable basename IS the family,
    and a family serves N variants, so this is a direct registry lookup. It replaces
    the normalise/prefix/strip/alias cascade below, which existed only because example
    and executable names did not match their ``.dxnn``; they do now, and a heuristic
    that silently resolves a typo to a neighbouring model is worse than no match.
    """
    out: dict = {}
    for e in load_registry():
        p = MODELS_DIR / e["dxnn_file"]
        out.setdefault(e["family"], [])
        if p not in out[e["family"]]:
            out[e["family"]].append(p)
    return out


@lru_cache(maxsize=1)
def _variant_dxnn_name() -> dict:
    """Map a variant directory / executable basename to its ``dxnn_file``."""
    out: dict = {}
    for entry in load_registry():
        variant = entry.get("variant")
        dxnn_file = entry.get("dxnn_file")
        if variant and dxnn_file:
            out.setdefault(variant, dxnn_file)
    return out


def _search_model_file(filename: str) -> Optional[Path]:
    """Find ``filename`` under ``assets/models``, then ``workspace/res/models``.

    ``filename`` may be a bare stem. The search walks parents of the project
    root so a suite checkout and a standalone dx_app tree both resolve.
    """
    name = filename if filename.endswith(".dxnn") else f"{filename}.dxnn"
    local = MODELS_DIR / name
    if local.is_file():
        return local
    directory = PROJECT_ROOT
    while directory != directory.parent:
        candidate = directory / "workspace" / "res" / "models" / name
        if candidate.is_file():
            return candidate
        directory = directory.parent
    return None


def dxnn_for_exe(base_name: str) -> Optional[Path]:
    """Existing ``.dxnn`` for an executable or script basename.

    The basename is the variant directory (``alexnet_224x224``). A family name
    still resolves to the first file of that family that is on disk.
    """
    variant_file = _variant_dxnn_name().get(base_name)
    if variant_file:
        found = _search_model_file(variant_file)
        if found is not None:
            return found
    for path in family_dxnn_map().get(base_name, []):
        found = _search_model_file(path.name)
        if found is not None:
            return found
    return _search_model_file(base_name)


def _find_dxnn_for_name(base_name: str) -> Optional[Path]:
    """Name kept for callers that still use the pre-layout helper."""
    return dxnn_for_exe(base_name)


def setup_environment(*, extra_lib_dirs: Optional[List[Path]] = None) -> dict:
    """Return an ``os.environ`` copy with ``LD_LIBRARY_PATH`` set.

    Parameters
    ----------
    extra_lib_dirs : list[Path], optional
        Additional directories to prepend (e.g. ``dx_rt/build_x86_64/lib``).

    Without a display, ``QT_QPA_PLATFORM`` defaults to ``offscreen``
    (see :func:`test_helpers.proc.apply_headless_display`).
    """
    env = os.environ.copy()
    dirs = []
    if extra_lib_dirs:
        dirs.extend(str(d) for d in extra_lib_dirs if d.exists())
    if LIB_DIR.exists():
        dirs.append(str(LIB_DIR))
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        dirs.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(dirs) if dirs else ""
    apply_headless_display(env)
    return env


# ======================================================================
# Image resolution
# ======================================================================

def resolve_image_for_model(model_name: str, task: str) -> Optional[str]:
    """Return the sample image relative path for a given model and task.

    Priority: per-model override → task default.
    """
    img_rel = MODEL_IMAGE_OVERRIDE.get(model_name) or TASK_IMAGE_MAP.get(task)
    if img_rel and (PROJECT_ROOT / img_rel).exists():
        return img_rel
    return None


def strip_variant_suffix(executable: str) -> str:
    """``sfa3d_608x608_sync`` → ``sfa3d_608x608`` (``_sync``/``_async`` removed)."""
    for suffix in ("_sync", "_async"):
        if executable.endswith(suffix):
            return executable[: -len(suffix)]
    return executable


def resolve_cpp_exe_input(
    executable: str,
    default: Optional[Path] = None,
) -> Optional[Path]:
    """Return the sample input an *executable* actually accepts (``-i`` argument).

    Most C++ examples take an image, but a few HARD-REJECT one: 3D object
    detection (``sfa3d_*``) requires a KITTI LiDAR point cloud
    (``sample/kitti/velodyne/000049.bin``) and object-pose requires a DOPE frame.
    Handing those a JPG aborts the run with rc=255 before inference, so every CLI
    test that passes ``-i`` must resolve per task instead of hardcoding one image
    (``TASK_IMAGE_MAP`` / ``MODEL_IMAGE_OVERRIDE`` are the source of truth).

    Falls back to *default* when the task is unknown or its sample is missing.
    """
    base = strip_variant_suffix(executable)
    task_map = _cpp_exe_task_map_cached()
    task = task_map.get(executable) or task_map.get(base, "")
    rel = resolve_image_for_model(base, task)
    if rel:
        return PROJECT_ROOT / rel
    return default


# ======================================================================
# Model registry helpers
# ======================================================================

def load_registry() -> list:
    """Load ``config/model_registry.json`` and return supported entries."""
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    return [
        e for e in registry
        if e.get("supported") and e.get("model_name") not in SKIP_MODELS
    ]


@lru_cache(maxsize=1)
def _build_multi_model_args(base_name: str) -> Optional[list]:
    """Return CLI args list for a multi-model executable, or None."""
    if base_name not in MULTI_MODEL_EXECUTABLES:
        return None
    flag_models = MULTI_MODEL_EXECUTABLES[base_name]
    if not all((MODELS_DIR / f).exists() for _, f in flag_models):
        return None
    args: list = []
    for flag, fname in flag_models:
        args.extend([flag, str(MODELS_DIR / fname)])
    return args


def _resolve_cpp_candidate(
    stem: str, task: str, suffixes: Tuple[str, ...],
) -> 'Optional[Tuple[str, str, list, bool, str]]':
    """Resolve a single C++ candidate to test case tuple, or None."""
    if not any(stem.endswith(s) for s in suffixes):
        return None
    if not binary_path(BIN_DIR, stem).exists():
        return None
    base_name = stem.rsplit("_", 1)[0]
    multi_args = _build_multi_model_args(base_name)
    if multi_args is not None:
        img_rel = resolve_image_for_model(base_name, task) or _DEFAULT_IMAGE
        return (task, stem, multi_args, True, img_rel)
    model = _find_dxnn_for_name(base_name)
    if model is None:
        return None
    img_rel = resolve_image_for_model(base_name, task) or TASK_IMAGE_MAP.get(task, _DEFAULT_IMAGE)
    return (task, stem, ["-m", str(model)], False, img_rel)


def discover_cpp_executables(
    suffixes: Tuple[str, ...] = ("_sync", "_async"),
) -> List[Tuple[str, str, list, bool, str]]:
    """Discover ``(task, exe_name, model_args, is_multi, image_rel)`` from source tree.

    Scans ``src/cpp_example/<task>/`` for ``*.cpp`` files whose stems end with
    one of *suffixes*, then verifies the binary exists in ``BIN_DIR``.
    """
    src_cpp = PROJECT_ROOT / "src" / "cpp_example"
    cases: list = []

    for task_dir in sorted(src_cpp.iterdir()):
        if not task_dir.is_dir() or task_dir.name in _SKIP_DIRS:
            continue
        task = task_dir.name

        for cpp_file in sorted(task_dir.rglob("*.cpp")):
            if cpp_file.parent.name in EXCLUDED_EXAMPLE_VARIANTS:
                continue
            result = _resolve_cpp_candidate(cpp_file.stem, task, suffixes)
            if result is not None:
                cases.append(result)

    return cases


def discover_cpp_model_cases(
    suffix: str,
    bin_dir: Path,
) -> List[Tuple[str, Path]]:
    """Sorted ``(exe_name, model_path)`` for every single-model C++ example
    ending in *suffix* whose binary is in *bin_dir* and whose ``.dxnn`` is present.

    Executables come from the ``src/cpp_example/<task>/`` source tree
    (:func:`cpp_exe_task_map`). Under the family/variant layout an executable is
    ``<variant><suffix>`` and its model is :func:`dxnn_for_exe` of the variant
    (``yolov5-s_640x640_sync`` -> ``yolov5-s_640x640.dxnn``). Multi-model
    executables are skipped because callers launch ``-m <model>``.
    """
    out: dict = {}
    for exe in cpp_exe_task_map((suffix,)):
        base = exe[: -len(suffix)]
        if base in MULTI_MODEL_EXECUTABLES or not binary_path(bin_dir, exe).exists():
            continue
        model = dxnn_for_exe(base)
        if model is not None:
            out[exe] = model
    return sorted(out.items(), key=lambda c: c[0])


def stream_rejecting_cpp_cases(
    tasks: frozenset,
    bin_dir: Path,
    suffixes: Tuple[str, ...] = ("_sync",),
) -> List[Tuple[str, Path]]:
    """One ``(exe_name, model_path)`` per task in *tasks*, for NEGATIVE tests that
    assert image-only single-model examples reject stream (``-v``) input.

    Uses :func:`discover_cpp_executables` (src-tree scan + fuzzy ``.dxnn`` match)
    rather than the model-filename-normalisation discovery some suites use, because
    image-only models are frequently published with a ``.dxnn`` filename that does
    NOT normalise to the executable name — e.g. ``arcface_iresnet100_112x112_ms1m
    .dxnn`` ↔ exe ``arcface_iresnet100_ms1m_sync`` — so a filename-based matcher
    silently misses them and the negative test collapses to an empty parameter set.
    Multi-model executables are skipped (the test launches ``-m <model>``), and
    binary existence is re-checked against *bin_dir* (the test launch bin dir).
    """
    seen: set = set()
    out: List[Tuple[str, Path]] = []
    for task, exe, model_args, is_multi, _img in discover_cpp_executables(suffixes):
        if is_multi or len(model_args) != 2 or model_args[0] != "-m":
            continue
        if task not in tasks or task in seen:
            continue
        if not binary_path(bin_dir, exe).exists():
            continue
        out.append((exe, Path(model_args[1])))
        seen.add(task)
    return out


def cpp_exe_task_map(
    suffixes: Tuple[str, ...] = ("_sync", "_async"),
) -> dict:
    """Map each C++ example executable name to its task category.

    e.g. ``{"sfa3d_608x608_sync": "3d_object_detection", ...}``. Derived purely
    from the ``src/cpp_example/<task>/`` source layout — the executable's task is
    the directory its ``*.cpp`` lives under, a source fact that does NOT depend on
    whether the binary or its ``.dxnn`` is present (unlike
    :func:`discover_cpp_executables`, which gates on both). Used by stream/video
    tests to skip image-only tasks (see ``IMAGE_ONLY_TASKS``) whose single-model
    examples reject ``-v``/``-c``/``-r`` input.
    """
    src_cpp = PROJECT_ROOT / "src" / "cpp_example"
    mapping: dict = {}
    if not src_cpp.is_dir():
        return mapping
    for task_dir in sorted(src_cpp.iterdir()):
        if not task_dir.is_dir() or task_dir.name in _SKIP_DIRS:
            continue
        for cpp_file in sorted(task_dir.rglob("*.cpp")):
            if cpp_file.parent.name in EXCLUDED_EXAMPLE_VARIANTS:
                continue
            stem = cpp_file.stem
            if any(stem.endswith(s) for s in suffixes):
                mapping[stem] = task_dir.name
    return mapping


def py_script_task(path) -> str:
    """Task category of a Python example script.

    e.g. ``"super_resolution"`` for
    ``src/python_example/super_resolution/espcn/espcn-x2_17x17/espcn-x2_17x17_sync.py``.
    Derived purely from the ``src/python_example/<task>/<family>/<variant>/``
    source layout, as :func:`cpp_exe_task_map` is for C++: the task is the first
    directory under ``src/python_example/``. The script's grandparent is the
    FAMILY, so ``script.parent.parent.name`` is not the task. Used by stream/video
    tests to skip image-only tasks (see ``IMAGE_ONLY_TASKS``). A relative *path*
    is taken relative to the project root; a path outside
    ``src/python_example/`` raises ``ValueError``.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    rel = path.resolve().relative_to((PROJECT_ROOT / "src" / "python_example").resolve())
    if len(rel.parts) < 2:
        raise ValueError(f"not an example script under src/python_example/: {path}")
    return rel.parts[0]


def py_variant_image_only(path) -> bool:
    """``image_only`` of a Python example script's variant ``config.json``.

    The runner decides image-only-ness from the variant config, so a variant of a
    video-capable task can still reject ``--video`` (casvit under
    ``image_classification``); ``IMAGE_ONLY_TASKS`` alone misses it. False when
    the script's directory has no ``config.json``.
    """
    config = Path(path).parent / "config.json"
    if not config.is_file():
        return False
    return bool(json.loads(config.read_text(encoding="utf-8")).get("image_only"))


@lru_cache(maxsize=1)
def _cpp_exe_source_map() -> dict:
    """``{"<variant>_sync"/"_async": <variant dir>/<exe>.cpp}`` from the source tree."""
    src_cpp = PROJECT_ROOT / "src" / "cpp_example"
    mapping: dict = {}
    if not src_cpp.is_dir():
        return mapping
    for task_dir in sorted(src_cpp.iterdir()):
        if not task_dir.is_dir() or task_dir.name in _SKIP_DIRS:
            continue
        for cpp_file in sorted(task_dir.rglob("*.cpp")):
            if cpp_file.stem.endswith(("_sync", "_async")):
                mapping[cpp_file.stem] = cpp_file
    return mapping


_CPP_RUNNER_USE = re.compile(r"dxapp::(\w+Runner)\s*<")
# A runner that takes stream input registers the cxxopts option "v, video_path".
_CPP_STREAM_OPTION = re.compile(r'\(\s*"v\s*,\s*video')


@lru_cache(maxsize=None)
def _cpp_runner_takes_stream(runner: str, runner_dir: str) -> Optional[bool]:
    """Whether the C++ runner class *runner* registers ``-v/--video_path``.

    Read from the runner headers in *runner_dir*: the header that defines the
    class, or the alias it is (``using SyncKeypointRunner = SyncPoseRunner<F>``)
    followed to the header of its target. Only that header's own text counts; a
    runner header includes another runner's header (the embedding runners
    include sync_detection_runner.hpp) without taking its options. None when no
    header defines *runner*.
    """
    alias = re.compile(r"\busing\s+" + re.escape(runner) + r"\s*=\s*(\w+)\s*<")
    definition = re.compile(r"\bclass\s+" + re.escape(runner) + r"\b\s*(?:final\s*)?[:{]")
    for header in sorted(Path(runner_dir).glob("*.hpp")):
        text = header.read_text(encoding="utf-8", errors="replace")
        target = alias.search(text)
        if target:
            return _cpp_runner_takes_stream(target.group(1), runner_dir)
        if definition.search(text):
            return bool(_CPP_STREAM_OPTION.search(text))
    return None


def cpp_variant_image_only_reason(executable: str) -> Optional[str]:
    """Why a C++ example executable takes image input only, or None.

    Two facts of the source tree make it image-only:

    * its variant ``config.json`` says ``image_only`` -- a variant of a
      video-capable task can still be image-only (casvit-t under
      ``image_classification``), and a variant whose name merely looks like one
      (casvit-t-fpn-resnet50, semantic segmentation) is not;
    * the runner its ``.cpp`` instantiates registers no ``-v/--video_path``
      option, so the binary rejects ``-v`` ("Option 'v' does not exist") whatever
      the config says. The config is the one the Python tree uses, and the Python
      clip-img zero-shot example does run on video while the C++ one runs it
      through the image-only embedding runner.

    None when the executable has no source, or neither fact holds.
    """
    source = _cpp_exe_source_map().get(executable)
    if source is None:
        return None
    config = source.parent / "config.json"
    if config.is_file() and json.loads(config.read_text(encoding="utf-8")).get("image_only"):
        return "image-only variant (config.json image_only)"
    runner = _CPP_RUNNER_USE.search(source.read_text(encoding="utf-8", errors="replace"))
    if runner is not None:
        runner_dir = PROJECT_ROOT / "src" / "cpp_example" / "common" / "runner"
        if _cpp_runner_takes_stream(runner.group(1), str(runner_dir)) is False:
            return "image-only runner ({} registers no -v/--video_path)".format(runner.group(1))
    return None


def cpp_variant_image_only(executable: str) -> bool:
    """True when :func:`cpp_variant_image_only_reason` gives a reason."""
    return cpp_variant_image_only_reason(executable) is not None


@lru_cache(maxsize=1)
def _cpp_exe_task_map_cached() -> dict:
    """Read-only, cached ``cpp_exe_task_map()`` for per-test input resolution.

    :func:`resolve_cpp_exe_input` is called once per parametrised test, and the
    uncached map rglobs the whole ``src/cpp_example`` tree each time.
    """
    return cpp_exe_task_map()


# ======================================================================
# Python script discovery
# ======================================================================

def _discover_scripts_in_dir(
    model_dir: Path, suffixes: Tuple[str, ...],
) -> Tuple[List[Path], List[Path]]:
    """Return ``(sync_scripts, async_scripts)`` from one example directory.

    Includes ``*_sync*`` / ``*_async*`` scripts, including cpp_postprocess.
    ``ort_off`` scripts are local experiments and are not part of run_tc.
    """
    sync_scripts: List[Path] = []
    async_scripts: List[Path] = []
    for py in sorted(model_dir.glob("*.py")):
        if py.name.startswith("__") or "ort_off" in py.stem:
            continue
        if "_sync" in py.stem and "_sync" in suffixes:
            sync_scripts.append(py)
        elif "_async" in py.stem and "_async" in suffixes:
            async_scripts.append(py)
    return sync_scripts, async_scripts


def _variant_dirs(family_dir: Path) -> List[Path]:
    """Child folders that are one compiled model (``config.json`` present)."""
    found: List[Path] = []
    for child in sorted(family_dir.iterdir()):
        if not child.is_dir() or child.name in ("factory", "__pycache__"):
            continue
        if (child / "config.json").is_file():
            found.append(child)
    return found


def discover_python_scripts(
    suffixes: Tuple[str, ...] = ("_sync", "_async"),
) -> List[Tuple[str, str, List[Path], List[Path], Optional[Path]]]:
    """Discover Python example scripts organised by task and variant.

    Returns ``(task, variant, sync_scripts, async_scripts, model_path)``.
    Scripts live in ``src/python_example/<task>/<family>/<variant>/``.
    A family that still has entry scripts directly under it is kept for
    layouts that have not been split yet.
    """
    src_py = PROJECT_ROOT / "src" / "python_example"
    cases: list = []

    for task_dir in sorted(src_py.iterdir()):
        if not task_dir.is_dir() or task_dir.name in _SKIP_DIRS:
            continue
        task = task_dir.name

        for family_dir in sorted(task_dir.iterdir()):
            if not family_dir.is_dir() or family_dir.name in ("__pycache__", "factory"):
                continue
            variant_dirs = _variant_dirs(family_dir)
            if variant_dirs:
                for variant_dir in variant_dirs:
                    if variant_dir.name in EXCLUDED_EXAMPLE_VARIANTS:
                        continue
                    sync_scripts, async_scripts = _discover_scripts_in_dir(
                        variant_dir, suffixes,
                    )
                    if not sync_scripts and not async_scripts:
                        continue
                    model_path = _find_model_for_name(variant_dir.name)
                    cases.append(
                        (task, variant_dir.name, sync_scripts, async_scripts, model_path)
                    )
                continue
            sync_scripts, async_scripts = _discover_scripts_in_dir(family_dir, suffixes)
            if not sync_scripts and not async_scripts:
                continue
            model_path = _find_model_for_name(family_dir.name)
            cases.append(
                (task, family_dir.name, sync_scripts, async_scripts, model_path)
            )

    return cases


def _find_model_for_name(model_name: str) -> Optional[Path]:
    """A representative ``.dxnn`` for a python example directory name.

    The directory name is the FAMILY under the dx-modelzoo layout, so the registry
    answers this directly. It replaces a four-pass cascade (registry, normalise,
    dot-swap, underscore-strip, variant-suffix-strip) that existed only because
    directory names did not match their ``.dxnn``.
    """
    return dxnn_for_exe(model_name)


