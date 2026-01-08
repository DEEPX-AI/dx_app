import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from conftest import get_mock_outputs, load_module_from_file

from .config import ModelConfig


class IntegrationTestFramework:

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.mock_model_path = "/fake/model.dxnn"
        tests_root = Path(__file__).parent.parent
        src_root = tests_root.parent.parent / "src" / "python_example"
        self.model_path = src_root / self.config.base_path

    def _setup_dx_postprocessor_mock(self):
        mock_postprocessor = Mock()
        output_size = self.config.postprocess_result_shape
        
        if isinstance(output_size, list):
            mock_result = [np.random.rand(*s).astype(np.float32) for s in output_size]
        else:
            mock_result = np.random.rand(*output_size).astype(np.float32)
            
        mock_postprocessor.postprocess.return_value = mock_result

        class DynamicDxPostprocess:
            def __getattr__(self, name):
                return Mock(return_value=mock_postprocessor)

        sys.modules["dx_postprocess"] = DynamicDxPostprocess()

    def _load_module(self, script_name: str):
        script_path = self.model_path / script_name

        if not script_path.exists():
            pytest.skip(f"{script_name} not found")

        return load_module_from_file(str(script_path), script_name.replace(".py", ""))

    def _create_model_instance(self, script_name: str):
        if "cpp_postprocess" in script_name:
            self._setup_dx_postprocessor_mock()

        module = self._load_module(script_name)

        with patch("os.path.exists", return_value=True):
            return getattr(module, self.config.class_name)(self.mock_model_path)

    def test_image_inference_file_not_found(self, script_name: str):
        with patch("cv2.imread", return_value=None):
            model = self._create_model_instance(script_name)

            with pytest.raises(SystemExit):
                model.image_inference("nonexistent.jpg")

    def test_image_inference_success(self, script_name: str):
        if "async" in script_name:
            pytest.skip(f"Async variants do not support image_inference")

        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch("cv2.imread", return_value=mock_image):
            outputs = get_mock_outputs(self.config, script_name)
            model = self._create_model_instance(script_name)
            model.ie.run.return_value = outputs

            model.image_inference("test.jpg")

    def test_image_inference_save_output(self, script_name: str):
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch("cv2.imread", return_value=mock_image), patch(
            "cv2.imwrite"
        ) as mock_imwrite:

            outputs = get_mock_outputs(self.config, script_name)
            model = self._create_model_instance(script_name)
            model.ie.run.return_value = outputs

            model.image_inference("test.jpg", display=False)

            mock_imwrite.assert_called_once()

    def test_stream_inference_video_not_found(
        self, script_name: str, mock_video_capture
    ):
        mock_video_capture["instance"].isOpened.return_value = False

        with patch("cv2.VideoCapture", mock_video_capture["mock_class"]):
            model = self._create_model_instance(script_name)

            with pytest.raises(SystemExit):
                model.stream_inference("nonexistent.mp4")

            mock_video_capture["instance"].release.assert_not_called()

    def test_stream_inference_empty_video(self, script_name: str, mock_video_capture):
        mock_video_capture["instance"].get.side_effect = [0, 30, 640, 480]
        mock_video_capture["instance"].read.side_effect = [(False, None)]

        with patch("cv2.VideoCapture", mock_video_capture["mock_class"]):
            outputs = get_mock_outputs(self.config, script_name)
            model = self._create_model_instance(script_name)

            if "async" in script_name:
                model.ie.wait.return_value = outputs
            else:
                model.ie.run.return_value = outputs

            model.stream_inference("empty.mp4")

            mock_video_capture["instance"].release.assert_called_once()

    def test_stream_inference_keyboard_interrupt(
        self, script_name: str, mock_video_capture
    ):
        mock_video_capture["instance"].read.side_effect = KeyboardInterrupt()

        with patch("cv2.VideoCapture", mock_video_capture["mock_class"]):
            model = self._create_model_instance(script_name)

            model.stream_inference("test.mp4")

            mock_video_capture["instance"].release.assert_called_once()

    def test_stream_inference_success(self, script_name: str, mock_video_capture):
        with patch("cv2.VideoCapture", mock_video_capture["mock_class"]):
            outputs = get_mock_outputs(self.config, script_name)
            model = self._create_model_instance(script_name)

            if "async" in script_name:
                model.ie.wait.return_value = outputs
            else:
                model.ie.run.return_value = outputs

            model.stream_inference("test.mp4")

            mock_video_capture["instance"].release.assert_called_once()
