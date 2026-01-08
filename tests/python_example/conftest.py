import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_EXAMPLE_PATH = PROJECT_ROOT / "src" / "python_example"
sys.path.insert(0, str(PYTHON_EXAMPLE_PATH))


def load_module_from_file(file_path: str, module_name: str):
    try:
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
        return [np.random.rand(*shape).astype(np.uint8) for shape in shapes]
    else:
        return [np.random.rand(*shape).astype(np.float32) for shape in shapes]


def pytest_configure(config):

    if "dx_engine" not in sys.modules:
        mock_dx_engine = Mock()
        mock_dx_engine.InferenceEngine = Mock
        mock_dx_engine.InferenceOption = Mock
        mock_dx_engine.Configuration = Mock
        sys.modules["dx_engine"] = mock_dx_engine
        print("✓ dx_engine mocked")

    if "dx_postprocess" not in sys.modules:
        mock_dx_postprocess = Mock()
        sys.modules["dx_postprocess"] = mock_dx_postprocess
        print("✓ dx_postprocess mocked")

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)

    htmlcov_dir = PROJECT_ROOT / "htmlcov"
    htmlcov_dir.mkdir(exist_ok=True)


def pytest_sessionfinish(session, exitstatus):

    from framework.performance_collector import get_collector

    collector = get_collector()

    if not collector.results:
        return

    collector.print_report()

    output_dir = Path(__file__).parent / "performance_reports"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"performance_report_{timestamp}.csv"
    collector.save_csv(str(csv_path))

    print(f"\n Report saved: {csv_path}")


@pytest.fixture(scope="function", autouse=True)
def mock_inference_engine(request):

    if "e2e" in request.keywords:
        yield None
        return

    mock_inference_engine

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

    mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

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
