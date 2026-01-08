import pytest
from framework.base_test import BaseTestFramework
from framework.cli_test import CLITestFramework
from framework.config import YOLOV12_CONFIG
from framework.e2e_test import E2ETestFramework
from framework.groups_test import GroupsTestFramework
from framework.integration_test import IntegrationTestFramework

pytestmark = pytest.mark.yolov12


@pytest.mark.unit
class TestYOLOv12Base:

    @classmethod
    def setup_class(cls):
        cls.framework = BaseTestFramework(YOLOV12_CONFIG)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_letterbox(self, script_name):
        self.framework.test_letterbox(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_preprocess(self, script_name):
        self.framework.test_preprocess(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "cpp_postprocess" not in s]
    )
    def test_postprocess(self, script_name):
        self.framework.test_postprocess(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_convert_to_original_coordinates(self, script_name):
        self.framework.test_convert_to_original_coordinates(script_name)


@pytest.mark.unit
class TestYOLOv12Groups:

    @classmethod
    def setup_class(cls):
        cls.framework = GroupsTestFramework(YOLOV12_CONFIG)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "cpp_postprocess" not in s]
    )
    def test_python_postprocess_has_method(self, script_name):
        self.framework.test_python_postprocess_has_method(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "cpp_postprocess" in s]
    )
    def test_cpp_postprocess_no_method(self, script_name):
        self.framework.test_cpp_postprocess_no_method(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "_sync" in s]
    )
    def test_sync_has_image_inference(self, script_name):
        self.framework.test_sync_has_image_inference(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "_async" in s]
    )
    def test_async_no_image_inference(self, script_name):
        self.framework.test_async_no_image_inference(script_name)


@pytest.mark.integration
class TestYOLOv12Integration:

    @classmethod
    def setup_class(cls):
        cls.framework = IntegrationTestFramework(YOLOV12_CONFIG)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "_sync" in s]
    )
    def test_image_inference_file_not_found(self, script_name):
        self.framework.test_image_inference_file_not_found(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "_sync" in s]
    )
    def test_image_inference_success(self, script_name):
        self.framework.test_image_inference_success(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "_sync" in s]
    )
    def test_image_inference_save_output(self, script_name):
        self.framework.test_image_inference_save_output(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_stream_inference_video_not_found(self, script_name, mock_video_capture):
        self.framework.test_stream_inference_video_not_found(
            script_name, mock_video_capture
        )

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_stream_inference_empty_video(self, script_name, mock_video_capture):
        self.framework.test_stream_inference_empty_video(
            script_name, mock_video_capture
        )

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_stream_inference_keyboard_interrupt(self, script_name, mock_video_capture):
        self.framework.test_stream_inference_keyboard_interrupt(
            script_name, mock_video_capture
        )

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_stream_inference_success(self, script_name, mock_video_capture):
        self.framework.test_stream_inference_success(script_name, mock_video_capture)


@pytest.mark.cli
class TestYOLOv12CLI:

    @classmethod
    def setup_class(cls):
        cls.framework = CLITestFramework(YOLOV12_CONFIG)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_cli_help(self, script_name):
        self.framework.test_cli_help(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_cli_missing_required_args(self, script_name):
        self.framework.test_cli_missing_required_args(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_cli_unrecognized_argument(self, script_name):
        self.framework.test_cli_unrecognized_argument(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "_sync" in s]
    )
    def test_cli_image_mode(self, script_name):
        self.framework.test_cli_image_mode(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_cli_video_mode(self, script_name):
        self.framework.test_cli_video_mode(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_cli_camera_mode(self, script_name):
        self.framework.test_cli_camera_mode(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_cli_rtsp_mode(self, script_name):
        self.framework.test_cli_rtsp_mode(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_cli_display_options(self, script_name):
        self.framework.test_cli_display_options(script_name)


@pytest.mark.e2e
class TestYOLOv12E2E:

    @classmethod
    def setup_class(cls):
        cls.framework = E2ETestFramework(YOLOV12_CONFIG)

    @pytest.mark.parametrize(
        "script_name", [s for s in YOLOV12_CONFIG.variants if "_sync" in s]
    )
    def test_image_inference_real(self, script_name):
        self.framework.test_image_inference_real(script_name)

    @pytest.mark.parametrize("script_name", YOLOV12_CONFIG.variants)
    def test_stream_inference_real(self, script_name):
        self.framework.test_stream_inference_real(script_name)
