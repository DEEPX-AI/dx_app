from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import load_module_from_file

from .config import ModelConfig


class GroupsTestFramework:

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
        with patch("os.path.exists", return_value=True):
            return getattr(module, self.config.class_name)(self.mock_model_path)

    def test_python_postprocess_has_method(self, script_name: str):
        model = self._create_model_instance(script_name)
        assert hasattr(
            model, "postprocess"
        ), f"{script_name}: should have postprocess method"

    def test_cpp_postprocess_no_method(self, script_name: str):
        model = self._create_model_instance(script_name)

        assert not hasattr(
            model, "postprocess"
        ), f"{script_name}: should NOT have postprocess method"

        assert hasattr(
            model, "postprocessor"
        ), f"{script_name}: should have postprocessor attribute"

    def test_sync_has_image_inference(self, script_name: str):
        model = self._create_model_instance(script_name)

        assert hasattr(
            model, "image_inference"
        ), f"{script_name}: Sync variant MUST have image_inference method"

    def test_async_no_image_inference(self, script_name: str):
        model = self._create_model_instance(script_name)

        assert not hasattr(
            model, "image_inference"
        ), f"{script_name}: Async variant MUST NOT have image_inference method"
