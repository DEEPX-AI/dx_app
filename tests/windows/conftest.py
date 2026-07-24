"""
Configuration and fixtures for Windows integration tests.

These tests validate invalid argument handling for C++, Python, and pybinding
layers using the YOLOv7 demo as the reference model.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Register DXRT DLL directories so dx_postprocess can load ---
if sys.platform == "win32":
    _DEEPX_SDK_DIR = os.environ.get("DEEPX_SDK_DIR", r"C:\Program Files\DEEPX\DXNN\sdk")
    _DLL_SEARCH_DIRS = [
        os.path.join(_DEEPX_SDK_DIR, "csharp"),
        os.path.join(_DEEPX_SDK_DIR, "bin"),
        str(PROJECT_ROOT / "bin" / "Release"),
        str(PROJECT_ROOT / "lib"),
    ]
    for _dll_dir in _DLL_SEARCH_DIRS:
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)
BIN_DIR = PROJECT_ROOT / "bin" / "Release" if os.name == "nt" else PROJECT_ROOT / "bin"
MODELS_DIR = PROJECT_ROOT / "assets" / "models"
VIDEOS_DIR = PROJECT_ROOT / "assets" / "videos"
SAMPLE_DIR = PROJECT_ROOT / "sample" / "img"
PYTHON_EXAMPLE_DIR = PROJECT_ROOT / "src" / "python_example"

# Reference demo: YOLOv7
YOLOV7_MODEL = MODELS_DIR / "YoloV7.dxnn"
YOLOV7_VIDEO = VIDEOS_DIR / "snowboard.mp4"
YOLOV7_IMAGE = SAMPLE_DIR / "sample_street.jpg"
YOLOV7_CPP_SYNC = BIN_DIR / "yolov7_sync.exe"
YOLOV7_CPP_ASYNC = BIN_DIR / "yolov7_async.exe"
YOLOV7_PY_SYNC = PYTHON_EXAMPLE_DIR / "object_detection" / "yolov7" / "yolov7_sync.py"
YOLOV7_PY_ASYNC = PYTHON_EXAMPLE_DIR / "object_detection" / "yolov7" / "yolov7_async.py"


def setup_environment() -> dict:
    """Return environment dict with library paths set."""
    env = os.environ.copy()
    lib_dir = PROJECT_ROOT / "lib"
    if os.name == "nt":
        path_dirs = [str(BIN_DIR), str(lib_dir)]
        existing = env.get("PATH", "")
        env["PATH"] = ";".join(path_dirs) + ";" + existing
    else:
        dirs = []
        if lib_dir.exists():
            dirs.append(str(lib_dir))
        existing = env.get("LD_LIBRARY_PATH", "")
        if existing:
            dirs.append(existing)
        env["LD_LIBRARY_PATH"] = ":".join(dirs)
    return env


def run_command(cmd, timeout=15):
    """Run a subprocess command and return the CompletedProcess."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=setup_environment(),
        cwd=str(PROJECT_ROOT),
    )


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def bin_dir():
    return BIN_DIR


@pytest.fixture
def models_dir():
    return MODELS_DIR


@pytest.fixture
def yolov7_model():
    if not YOLOV7_MODEL.exists():
        pytest.skip(f"Model not found: {YOLOV7_MODEL} (run setup.bat)")
    return YOLOV7_MODEL


@pytest.fixture
def yolov7_video():
    if not YOLOV7_VIDEO.exists():
        pytest.skip(f"Video not found: {YOLOV7_VIDEO} (run setup.bat)")
    return YOLOV7_VIDEO


@pytest.fixture
def yolov7_image():
    if not YOLOV7_IMAGE.exists():
        pytest.skip(f"Image not found: {YOLOV7_IMAGE} (run setup.bat)")
    return YOLOV7_IMAGE


@pytest.fixture
def yolov7_cpp_sync():
    if not YOLOV7_CPP_SYNC.exists():
        pytest.skip(f"Executable not found: {YOLOV7_CPP_SYNC} (run build.bat)")
    return YOLOV7_CPP_SYNC


@pytest.fixture
def yolov7_cpp_async():
    if not YOLOV7_CPP_ASYNC.exists():
        pytest.skip(f"Executable not found: {YOLOV7_CPP_ASYNC} (run build.bat)")
    return YOLOV7_CPP_ASYNC


@pytest.fixture
def yolov7_py_sync():
    if not YOLOV7_PY_SYNC.exists():
        pytest.skip(f"Script not found: {YOLOV7_PY_SYNC}")
    return YOLOV7_PY_SYNC


@pytest.fixture
def yolov7_py_async():
    if not YOLOV7_PY_ASYNC.exists():
        pytest.skip(f"Script not found: {YOLOV7_PY_ASYNC}")
    return YOLOV7_PY_ASYNC
