from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from conftest import get_mock_outputs, load_module_from_file

_rng = np.random.default_rng(42)

from .config import ModelConfig, TaskType


class BaseTestFramework:

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.mock_model_path = "/fake/model.dxnn"
        tests_root = Path(__file__).parent.parent
        src_root = tests_root.parent.parent / "src" / "python_example"
        self.model_path = src_root / self.config.base_path

    def _load_module(self, script_name: str):
        script_path = self.model_path / script_name

        if not script_path.exists():
            pytest.skip(f"{script_name} not found")

        return load_module_from_file(str(script_path), script_name.replace(".py", ""))

    def _create_model_instance(self, script_name: str):
        module = self._load_module(script_name)
        if module is None:
            pytest.skip(f"Failed to load module: {script_name}")
        cls = getattr(module, self.config.class_name, None)
        if cls is None:
            pytest.skip(f"Class '{self.config.class_name}' not found in {script_name} (v3.0.0 runner pattern)")
        with patch("os.path.exists", return_value=True):
            return cls(self.mock_model_path)

    def test_letterbox(self, script_name: str):
        model = self._create_model_instance(script_name)
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.letterbox(mock_image, (640, 640))

        assert isinstance(result, tuple), "letterbox should return tuple"
        assert len(result) == 2, "letterbox should return (image, pad)"

        img, pad = result
        assert isinstance(img, np.ndarray), "First element should be image (np.ndarray)"
        assert isinstance(pad, tuple), "Second element should be pad (tuple)"
        assert len(pad) == 2, "pad should be (top, left)"

    def test_preprocess(self, script_name: str):
        model = self._create_model_instance(script_name)
        test_img = _rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        input_tensor = model.preprocess(test_img)

        assert isinstance(input_tensor, np.ndarray)
        assert input_tensor.ndim in [
            3,
            4,
        ], f"{script_name}: input should be 3D or 4D, got {input_tensor.ndim}D"

    def test_postprocess(self, script_name: str):
        model = self._create_model_instance(script_name)
        mock_outputs = get_mock_outputs(self.config, script_name)
        result = model.postprocess(mock_outputs)

        if self.config.task == TaskType.CLASSIFICATION:
            assert isinstance(
                result, tuple
            ), f"{script_name}: Classification should return tuple"
            assert len(result) == 2, f"{script_name}: Should return (class_id, score)"

            class_id, score = result
            assert isinstance(
                class_id, (int, np.integer)
            ), f"{script_name}: class_id should be int, got {type(class_id)}"
            assert isinstance(
                score, (float, np.floating)
            ), f"{script_name}: score should be float, got {type(score)}"
            assert (
                0 <= score <= 1
            ), f"{script_name}: score should be in [0, 1], got {score}"

        elif self.config.task == TaskType.OBJECT_DETECTION:
            assert isinstance(
                result, np.ndarray
            ), f"{script_name}: Detection should return np.ndarray"
            assert (
                result.ndim == 2
            ), f"{script_name}: Detection should return 2D array, got {result.ndim}D"

            if result.shape[0] > 0:
                expected_size = self.config.detection_output_size
                assert (
                    result.shape[1] == expected_size
                ), f"Expected detection output size {expected_size}, got {result.shape[1]}"

        elif self.config.task in (
            TaskType.SEMANTIC_SEGMENTATION,
            TaskType.DEPTH_ESTIMATION,
            TaskType.IMAGE_RESTORATION,
            TaskType.IMAGE_ENHANCEMENT,
            TaskType.SUPER_RESOLUTION,
        ):
            assert isinstance(
                result, np.ndarray
            ), f"{script_name}: Segmentation/Restoration should return np.ndarray"
            assert (
                result.ndim == 2
            ), f"{script_name}: Should return 2D array (H, W), got {result.ndim}D"

            expected_shape = self.config.postprocess_result_shape
            assert (
                result.shape == expected_shape
            ), f"{script_name}: Expected shape {expected_shape}, got {result.shape}"
        
        elif self.config.task == TaskType.INSTANCE_SEGMENTATION:
            assert isinstance(
                result, tuple
            ), f"{script_name}: Instance Segmentation should return tuple"
            assert len(result) == 2, f"{script_name}: Should return (detections, masks)"

            detections, masks = result
            assert isinstance(
                detections, np.ndarray
            ), f"{script_name}: Detections should be np.ndarray"
            assert (
                detections.ndim == 2
            ), f"{script_name}: Detections should be 2D array, got {detections.ndim}D"

            assert isinstance(
                masks, np.ndarray
            ), f"{script_name}: Masks should be np.ndarray"
            assert (
                masks.ndim == 3
            ), f"{script_name}: Masks should be 3D array (N, H, W), got {masks.ndim}D"

    def test_convert_to_original_coordinates(self, script_name: str):
        model = self._create_model_instance(script_name)

        img = _rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        model.preprocess(img)

        output_size = self.config.detection_output_size
        self._assert_empty_coords(model, output_size, script_name)

        det_args, _ = self._build_detection_args(model, output_size)
        result = model.convert_to_original_coordinates(*det_args)
        self._assert_detection_result(result, output_size, script_name)

    def _assert_empty_coords(self, model, output_size, script_name):
        """Verify convert_to_original_coordinates returns empty for empty input."""
        empty_dets = np.empty((0, output_size), dtype=np.float32)
        args = [empty_dets]
        if self.config.task == TaskType.INSTANCE_SEGMENTATION:
            mask_h, mask_w = model.input_height // 4, model.input_width // 4
            args.append(np.empty((0, mask_h, mask_w), dtype=np.float32))
        result = model.convert_to_original_coordinates(*args)
        if isinstance(result, tuple):
            assert all(len(r) == 0 for r in result), (
                f"{script_name}: empty input should return empty results"
            )
        else:
            assert len(result) == 0, f"{script_name}: empty input should return empty"

    def _build_detection_args(self, model, output_size):
        """Build detection args and mask dimensions for convert_to_original_coordinates."""
        mask_h, mask_w = None, None
        if self.config.task == TaskType.INSTANCE_SEGMENTATION:
            mask_h = model.input_height // 4
            mask_w = model.input_width // 4
        detections = self._make_detections(output_size)
        args = [detections]
        if self.config.task == TaskType.INSTANCE_SEGMENTATION:
            args.append(np.zeros((1, mask_h, mask_w), dtype=np.float32))
        return args, (mask_h, mask_w)

    def _make_detections(self, output_size):
        """Build a single-row detection array based on config."""
        if self.config.has_keypoints:
            data = [100, 100, 200, 200, 0.9, 0]
            for _ in range(self.config.num_keypoints):
                if self.config.keypoint_dim == 2:
                    data.extend([150, 150])
                else:
                    data.extend([150, 150, 0.9])
            return np.array([data], dtype=np.float32)
        if output_size == 7:
            return np.array([[100, 100, 200, 200, 0.9, 0, 45.0]], dtype=np.float32)
        return np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)

    def _assert_detection_result(self, result, output_size, script_name):
        """Assert convert_to_original_coordinates result has expected shape."""
        if self.config.task == TaskType.INSTANCE_SEGMENTATION:
            assert isinstance(result, tuple)
            res_dets, res_masks = result
            assert isinstance(res_dets, np.ndarray)
            assert isinstance(res_masks, np.ndarray)
            assert res_dets.shape == (1, output_size)
        else:
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, output_size), (
                f"{script_name}: output shape should be (1, {output_size}), got {result.shape}"
            )
