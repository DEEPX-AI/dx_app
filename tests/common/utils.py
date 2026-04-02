"""
Shared utility functions for C++ and Python test scripts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .constants import (
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


# ======================================================================
# Normalisation helpers
# ======================================================================

def normalize_model_name(stem: str) -> str:
    """Normalise a model filename stem to its expected executable prefix.

    ``YoloV5M_6.1`` → ``yolov5m_6_1``
    """
    return stem.lower().replace(".", "_")


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


# ======================================================================
# Constants
# ======================================================================

_DXNN_GLOB = "*.dxnn"
_DEFAULT_IMAGE = "sample/img/sample_kitchen.jpg"
_SKIP_DIRS = frozenset({"common", "__pycache__"})

# ======================================================================
# C++ discovery
# ======================================================================

def _find_dxnn_for_name(base_name: str) -> Optional[Path]:
    """Find a .dxnn model matching *base_name* (exact, then prefix)."""
    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        if normalize_model_name(m.stem) == base_name:
            return m
    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        mn = normalize_model_name(m.stem)
        if mn.startswith(base_name) or base_name.startswith(mn):
            return m
    return None


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


# ======================================================================
# Python script discovery
# ======================================================================

def _discover_scripts_in_dir(
    model_dir: Path, suffixes: Tuple[str, ...],
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return ``(sync_script, async_script)`` from a model directory."""
    sync_script: Optional[Path] = None
    async_script: Optional[Path] = None
    for py in model_dir.glob("*.py"):
        if py.name.startswith("__") or "cpp_postprocess" in py.stem:
            continue
        if "_sync" in py.stem and "_sync" in suffixes:
            sync_script = py
        elif "_async" in py.stem and "_async" in suffixes:
            async_script = py
    return sync_script, async_script


def discover_python_scripts(
    suffixes: Tuple[str, ...] = ("_sync", "_async"),
) -> List[Tuple[str, str, Optional[Path], Optional[Path], Optional[Path]]]:
    """Discover Python example scripts organised by task/model.

    Returns ``(task, model_name, sync_script, async_script, model_path)``.
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
            sync_script, async_script = _discover_scripts_in_dir(model_dir, suffixes)
            if sync_script is None and async_script is None:
                continue
            model_path = _find_model_for_name(model_dir.name)
            cases.append((task, model_dir.name, sync_script, async_script, model_path))

    return cases


def _find_model_for_name(model_name: str) -> Optional[Path]:
    """Find the best matching .dxnn file for a Python model directory name."""
    norm = normalize_model_name(model_name)

    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        if normalize_model_name(m.stem) == norm:
            return m

    for m in sorted(MODELS_DIR.glob(_DXNN_GLOB)):
        if m.stem.lower().replace(".", "_") == norm:
            return m

    return None
