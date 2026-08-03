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
    BIN_DIR,
    LIB_DIR,
    MODELS_DIR,
    MODEL_DXNN_ALIAS,
    MULTI_MODEL_EXECUTABLES,
    PROJECT_ROOT,
    REGISTRY_PATH,
    SKIP_MODELS,
    TASK_IMAGE_MAP,
    MODEL_IMAGE_OVERRIDE,
)


# ======================================================================
# Normalisation helpers
# ======================================================================

def normalize_model_name(stem: str) -> str:
    """Normalise a model filename stem to its expected executable prefix.

    ``YoloV5M_6.1`` → ``yolov5m_6_1``
    """
    return stem.lower().replace(".", "_").replace("-", "_")


# ======================================================================
# Environment setup
# ======================================================================

def setup_environment(*, extra_lib_dirs: Optional[List[Path]] = None) -> dict:
    """Return an ``os.environ`` copy with ``LD_LIBRARY_PATH`` set.

    Parameters
    ----------
    extra_lib_dirs : list[Path], optional
        Additional directories to prepend (e.g. ``dx_rt/build_x86_64/lib``).
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
def registry_dxnn_map() -> dict:
    """Authoritative ``{example base-name → .dxnn Path}`` from ``model_registry.json``.

    Consulted BEFORE the filename-normalisation heuristics below so that models
    whose registry ``dxnn_file`` differs from the example directory / executable
    name by more than a size/variant suffix still resolve — cases the
    normalise/prefix/strip passes miss because the difference is a *mid-name*
    token, e.g.::

        arcface_r50               → arcface_resnet50_112x112.dxnn   (r50 ↔ resnet50)
        beit_large_patch16        → beit-l-p16_224x224.dxnn         (fully renamed)
        arcface_iresnet100_ms1m   → arcface_iresnet100_112x112_ms1m.dxnn (size inserted)
        casvit_t_seg_fpn_resnet50 → casvit-t-fpn-resnet50_512x512.dxnn   (seg ↔ fpn)

    This is the same ``model_name → dxnn_file`` resolution ``_rt_resolve.py``
    already uses. A version-suffix-stripped alias (``beit_large_patch16_1`` →
    ``beit_large_patch16``) is added so example dir names without the re-publish
    suffix still hit. The returned dict is cached; callers must not mutate it.
    """
    out: dict = {}
    for e in load_registry():
        name, dxnn = e.get("model_name"), e.get("dxnn_file")
        if name and dxnn:
            out[name] = MODELS_DIR / dxnn
    for name in list(out):
        base = re.sub(r"_\d+$", "", name)
        if base != name and base not in out:
            out[base] = out[name]
    return out


# ======================================================================
# Constants
# ======================================================================

_DXNN_GLOB = "*.dxnn"
_DEFAULT_IMAGE = "sample/img/sample_kitchen.jpg"
_SKIP_DIRS = frozenset({"common", "__pycache__"})

# Trailing tokens that some re-published .dxnn files carry but the model
# directory / executable names do not: a version suffix (``-1`` / ``-2`` →
# normalised to ``_1`` / ``_2``) or a quantisation suffix (``_q-lite`` →
# ``_q_lite``). e.g. ``DeiTBase-1.dxnn`` for model ``deit_base``,
# ``deeplabv3plus-drn_512x512_q-lite.dxnn`` for ``deeplabv3plus_drn_512x512``.
_VARIANT_SUFFIX_RE = re.compile(r"(?:_q_lite|_\d+)$")


def _strip_variant_suffix(norm_stem: str) -> str:
    """Remove ONE trailing version/quantisation token from a *normalised* stem.

    ``deitbase_1`` → ``deitbase``; ``..._512x512_q_lite`` → ``..._512x512``.
    Stripped exactly once (not repeatedly) so a legitimate trailing number that
    is part of the real name is preserved: ``fcn8_resnet_v1_18_1`` → only the
    ``_1`` version suffix goes, leaving ``fcn8_resnet_v1_18`` (the ``_18`` is
    kept). Only a trailing ``_<digits>`` or ``_q_lite`` is removed, so names
    whose digits are not preceded by ``_`` (``resnet50``, ``512x512``) are
    untouched.
    """
    return _VARIANT_SUFFIX_RE.sub("", norm_stem, count=1)


def find_dxnn_ignoring_variant(models_dir: Path, normalized_name: str) -> Optional[Path]:
    """Last-resort .dxnn match: compare with underscores stripped *and* trailing
    version/quantisation suffixes removed from the filename side only.

    Handles files re-published with a ``-1`` / ``-2`` version suffix or a
    ``_q-lite`` quantisation suffix, which the exact / prefix / plain-stripped
    passes miss (e.g. ``DeiTBase-1.dxnn`` ↔ model dir ``deit_base``). The
    suffix is stripped only from the filename, never from *normalized_name*,
    so version-numbered directory names (``deit_base_distilled_1``) keep
    matching their own file and do not collide with siblings.

    An explicit ``MODEL_DXNN_ALIAS`` entry (for arbitrary renames that no rule
    can derive) takes priority when the aliased file is present.
    """
    alias = MODEL_DXNN_ALIAS.get(normalized_name)
    if alias:
        aliased = models_dir / alias
        if aliased.exists():
            return aliased

    target = normalized_name.replace("_", "")
    for m in sorted(models_dir.glob(_DXNN_GLOB)):
        mn = _strip_variant_suffix(normalize_model_name(m.stem)).replace("_", "")
        if mn == target:
            return m
    return None

# ======================================================================
# C++ discovery
# ======================================================================

def _find_dxnn_for_name(base_name: str) -> Optional[Path]:
    """Find a .dxnn model matching *base_name* (exact, then prefix, then
    underscore-insensitive).

    Prefix match is skipped when a more specific binary exists for that model.
    e.g. yolov7_w6_face.dxnn won't match yolov7_w6 if yolov7_w6_face binary exists.
    """
    # Authoritative registry lookup first (model_name → dxnn_file). Bridges names
    # the normalisation heuristics below cannot (e.g. arcface_r50 → arcface_resnet50).
    reg = registry_dxnn_map().get(base_name)
    if reg is not None and reg.exists():
        return reg
    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        if normalize_model_name(m.stem) == base_name:
            return m
    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        mn = normalize_model_name(m.stem)
        if mn.startswith(base_name + "_") or mn.startswith(base_name + "-"):
            # Skip if a dedicated binary exists for this model
            if (BIN_DIR / f"{mn}_sync").exists() or (BIN_DIR / f"{mn}_async").exists():
                continue
            return m
    # Fallback: compare with underscores stripped (e.g. YoloV7W6 ↔ yolov7_w6)
    stripped = base_name.replace("_", "")
    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        mn = normalize_model_name(m.stem).replace("_", "")
        if mn == stripped:
            return m
    # Final fallback: ignore trailing -1/-2/_q-lite variant suffixes on files.
    return find_dxnn_ignoring_variant(MODELS_DIR, base_name)


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
    if not (BIN_DIR / stem).exists():
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
            result = _resolve_cpp_candidate(cpp_file.stem, task, suffixes)
            if result is not None:
                cases.append(result)

    return cases


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
        if not (bin_dir / exe).exists():
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
            stem = cpp_file.stem
            if any(stem.endswith(s) for s in suffixes):
                mapping[stem] = task_dir.name
    return mapping


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
    """Return ``(sync_scripts, async_scripts)`` from a model directory.

    Includes all ``*_sync*`` / ``*_async*`` scripts, including
    ``*_sync_cpp_postprocess*`` / ``*_async_cpp_postprocess*`` variants.
    """
    sync_scripts: List[Path] = []
    async_scripts: List[Path] = []
    for py in sorted(model_dir.glob("*.py")):
        if py.name.startswith("__"):
            continue
        if "_sync" in py.stem and "_sync" in suffixes:
            sync_scripts.append(py)
        elif "_async" in py.stem and "_async" in suffixes:
            async_scripts.append(py)
    return sync_scripts, async_scripts


def discover_python_scripts(
    suffixes: Tuple[str, ...] = ("_sync", "_async"),
) -> List[Tuple[str, str, List[Path], List[Path], Optional[Path]]]:
    """Discover Python example scripts organised by task/model.

    Returns ``(task, model_name, sync_scripts, async_scripts, model_path)``
    where ``sync_scripts`` / ``async_scripts`` are lists that include all
    ``*_sync*`` / ``*_async*`` variants (e.g. cpp_postprocess too).
    """
    src_py = PROJECT_ROOT / "src" / "python_example"
    cases: list = []

    for task_dir in sorted(src_py.iterdir()):
        if not task_dir.is_dir() or task_dir.name in _SKIP_DIRS:
            continue
        task = task_dir.name

        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name == "__pycache__":
                continue
            sync_scripts, async_scripts = _discover_scripts_in_dir(model_dir, suffixes)
            if not sync_scripts and not async_scripts:
                continue
            model_path = _find_model_for_name(model_dir.name)
            cases.append((task, model_dir.name, sync_scripts, async_scripts, model_path))

    return cases


def _find_model_for_name(model_name: str) -> Optional[Path]:
    """Find the best matching .dxnn file for a Python model directory name."""
    # Authoritative registry lookup first (model_name → dxnn_file). Bridges names
    # the normalisation heuristics below cannot (e.g. arcface_r50 → arcface_resnet50).
    reg = registry_dxnn_map().get(model_name)
    if reg is not None and reg.exists():
        return reg

    norm = normalize_model_name(model_name)

    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        if normalize_model_name(m.stem) == norm:
            return m

    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        if m.stem.lower().replace(".", "_") == norm:
            return m

    # Fallback: compare with underscores stripped (e.g. YoloV7W6 ↔ yolov7_w6)
    stripped = norm.replace("_", "")
    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        mn = normalize_model_name(m.stem).replace("_", "")
        if mn == stripped:
            return m

    # Final fallback: ignore trailing -1/-2/_q-lite variant suffixes on files.
    return find_dxnn_ignoring_variant(MODELS_DIR, norm)
