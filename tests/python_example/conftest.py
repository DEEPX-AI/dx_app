"""pytest configuration for the Python example suite (``src/python_example``).

Scope: the CLI/help contract tests and E2E inference tests in this directory,
plus the fully mocked unit tests under ``unit/``. Option and fixture wiring
shared with the C++ suite lives in ``tests/test_helpers/pytest_support.py`` --
see ``tests/README.md``.
"""
import importlib.util
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_EXAMPLE_PATH = PROJECT_ROOT / "src" / "python_example"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PYTHON_EXAMPLE_PATH))
# Make tests/test_helpers importable here and in every test module of this suite.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_helpers.pytest_support import (  # noqa: E402
    add_loop_option,
    add_stream_options,
    configure_stdio_errors,
)
# Re-exported fixtures -- importing them here is what makes pytest discover them.
from test_helpers.pytest_support import (  # noqa: E402,F401
    camera_index,
    rtsp_url,
    stream_duration,
    wait_for_temperature,
)

configure_stdio_errors()

from test_helpers.proc import (  # noqa: E402
    TEST_PYTHON_ENV,
    apply_headless_display,
    example_python,
)

# No display: Qt-backed OpenCV windows go offscreen instead of aborting.
apply_headless_display(os.environ)


def pytest_report_header(config):
    return "example python ({}): {}".format(TEST_PYTHON_ENV, example_python())


_rng = np.random.default_rng(42)
logger = logging.getLogger(__name__)

#: Default inference iterations for this suite (the C++ image suite uses 20).
DEFAULT_LOOP_COUNT = 1


def pytest_addoption(parser):
    """Register the options shared with the C++ suite."""
    add_loop_option(parser, default=str(DEFAULT_LOOP_COUNT))
    add_stream_options(parser)


@pytest.fixture
def loop_count(request):
    """Number of inference loops (from --loop CLI option).

    Function-scoped here -- the C++ suite scopes it per session -- because
    Python E2E cases may vary it per test.
    """
    try:
        return int(request.config.getoption("--loop"))
    except (TypeError, ValueError):
        return DEFAULT_LOOP_COUNT


def load_module_from_file(file_path: str, module_name: str):
    try:
        script_dir = str(Path(file_path).parent)
        # src/python_example/ root — two directories above the model dir
        # (src/python_example/<task>/<model>/<script>.py)
        py_example_root = str(Path(file_path).resolve().parent.parent.parent)

        # Remove stale 'factory' modules so each script loads its
        # own versions.
        stale_keys = [k for k in sys.modules.keys()
                      if k in ('factory',)
                      or k.startswith('factory.')]
        for key in stale_keys:
            sys.modules.pop(key, None)

        # Ensure sys.path order: script_dir first (factory/ lives here),
        # then py_example_root (common/ lives here).
        for p in (py_example_root, script_dir):
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, py_example_root)
        sys.path.insert(0, script_dir)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Failed to load module {module_name}: {e}")
        return None


def get_mock_outputs(config, script_name: str):

    is_ort_off = "ort_off" in script_name.lower()
    is_ppu = "ppu" in script_name.lower()

    if is_ort_off and config.ort_off_output_shapes:
        shapes = config.ort_off_output_shapes
    elif config.ort_on_output_shapes:
        shapes = config.ort_on_output_shapes
    else:
        raise ValueError(
            f"Model config for '{config.name}' missing output_shapes!\n"
            f"Please define 'ort_on_output_shapes' (and optionally 'ort_off_output_shapes') "
            f"in config.py for model '{config.name}'.\n"
            f"Script: {script_name}"
        )

    if is_ppu:
        return [_rng.integers(0, 256, shape, dtype=np.uint8) for shape in shapes]
    else:
        return [_rng.random(shape).astype(np.float32) for shape in shapes]


def pytest_configure(config):

    if "dx_engine" not in sys.modules:
        mock_dx_engine = Mock()
        mock_dx_engine.InferenceEngine = Mock
        mock_dx_engine.InferenceOption = Mock
        mock_dx_engine.Configuration = Mock
        sys.modules["dx_engine"] = mock_dx_engine
        print("[OK] dx_engine mocked")

    if "dx_postprocess" not in sys.modules:
        mock_dx_postprocess = Mock()
        sys.modules["dx_postprocess"] = mock_dx_postprocess
        print("[OK] dx_postprocess mocked")

    # Dynamically register per-model markers. ``test_e2e.py`` builds one marker per
    # model with ``getattr(pytest.mark, model_name)`` and pytest.ini sets
    # ``--strict-markers``, so every name it can produce must be registered here or
    # collection of the WHOLE suite dies with "not found in `markers` configuration
    # option".
    #
    # Both depths are registered because the tree is ``<task>/<family>/<variant>/``:
    # registering only the second level yields the FAMILY (``sfa3d``) while test_e2e
    # asks for the VARIANT (``sfa3d_608x608``). That mismatch is latent until enough
    # .dxnn files are on disk for the missing variant to be collected -- running
    # ``./setup.sh --demo-models`` was enough to turn it into a hard collection error.
    if PYTHON_EXAMPLE_PATH.exists():
        seen: set[str] = set()
        for task_dir in sorted(PYTHON_EXAMPLE_PATH.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "__pycache__":
                continue
            for family_dir in sorted(task_dir.iterdir()):
                if not family_dir.is_dir() or family_dir.name == "__pycache__":
                    continue
                for name_dir in (family_dir, *sorted(family_dir.iterdir())):
                    if not name_dir.is_dir() or name_dir.name == "__pycache__":
                        continue
                    if name_dir.name in seen:
                        continue
                    seen.add(name_dir.name)
                    config.addinivalue_line(
                        "markers",
                        f"{name_dir.name}: {name_dir.name} model tests",
                    )

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)

    htmlcov_dir = PROJECT_ROOT / "htmlcov"
    htmlcov_dir.mkdir(exist_ok=True)


def pytest_sessionfinish(session, exitstatus):

    from test_e2e import get_collector

    collector = get_collector()

    if not collector.results:
        return

    collector.print_report()

    output_dir = Path(__file__).parent / "performance_reports"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Include 'e2e_short' in filename when running the short suite
    markers_used = session.config.option.markexpr or ""
    label = "e2e_short" if "e2e_short" in markers_used else "e2e"
    csv_path = output_dir / f"performance_report_{label}_{timestamp}.csv"
    collector.save_csv(str(csv_path))

    print(f"\n Report saved: {csv_path}")


@pytest.fixture(scope="function", autouse=True)
def mock_inference_engine(request):

    if "e2e" in request.keywords:
        yield None
        return

    if "dx_engine" not in sys.modules:
        sys.modules["dx_engine"] = Mock()

    mock_engine_class = Mock()
    sys.modules["dx_engine"].InferenceEngine = mock_engine_class

    engine_instance = Mock()

    engine_instance.get_model_version.return_value = "7"

    engine_instance.get_input_tensors_info.return_value = [
        {"shape": [1, 640, 640, 3], "dtype": "float32", "name": "images"}
    ]

    mock_engine_class.return_value = engine_instance

    yield engine_instance


@pytest.fixture(autouse=True)
def mock_cv2_display(request):

    if "e2e" in request.keywords:
        yield
    else:
        with patch("cv2.imshow"), patch("cv2.waitKey", return_value=-1), patch(
            "cv2.destroyAllWindows"
        ):
            yield


@pytest.fixture
def mock_video_capture():

    mock_image = _rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    mock_cap_instance = Mock()
    mock_cap_instance.isOpened.return_value = True

    def mock_get(prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 100
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        else:
            return 0

    mock_cap_instance.get.side_effect = mock_get

    mock_cap_instance.read.side_effect = [
        (True, mock_image),
        (True, mock_image),
        (True, mock_image),
        (False, None),
    ]

    mock_cap_class = Mock(return_value=mock_cap_instance)

    return {
        "mock_class": mock_cap_class,
        "instance": mock_cap_instance,
        "image": mock_image,
    }
