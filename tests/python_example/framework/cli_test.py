from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import load_module_from_file

from .config import ModelConfig


class CLITestFramework:

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        tests_root = Path(__file__).parent.parent
        src_root = tests_root.parent.parent / "src" / "python_example"
        self.model_path = src_root / self.config.base_path

    def _load_module(self, script_name: str):
        script_path = self.model_path / script_name

        if not script_path.exists():
            pytest.skip(f"{script_name} not found")

        return load_module_from_file(str(script_path), script_name.replace(".py", ""))

    def test_cli_help(self, script_name: str):
        module = self._load_module(script_name)

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [script_name, "--help"]):
                module.parse_arguments()

        assert exc_info.value.code == 0

    def test_cli_missing_required_args(self, script_name: str):
        module = self._load_module(script_name)

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [script_name]):
                module.parse_arguments()

        assert exc_info.value.code != 0

    def test_cli_unrecognized_argument(self, script_name: str):
        module = self._load_module(script_name)

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [script_name, "--invalid-option", "value"]):
                module.parse_arguments()

        assert exc_info.value.code != 0

    def test_cli_image_mode(self, script_name: str):
        module = self._load_module(script_name)

        with patch(
            "sys.argv", [script_name, "--model", "test.dxnn", "--image", "test.jpg"]
        ):
            with patch("os.path.exists", return_value=True):
                args = module.parse_arguments()

        assert hasattr(args, "model") and args.model == "test.dxnn"
        assert hasattr(args, "image") and args.image == "test.jpg"

    def test_cli_video_mode(self, script_name: str):
        module = self._load_module(script_name)

        with patch(
            "sys.argv", [script_name, "--model", "test.dxnn", "--video", "test.mp4"]
        ):
            with patch("os.path.exists", return_value=True):
                args = module.parse_arguments()

        assert hasattr(args, "model") and args.model == "test.dxnn"
        assert hasattr(args, "video") and args.video == "test.mp4"

    def test_cli_camera_mode(self, script_name: str):
        module = self._load_module(script_name)

        with patch("sys.argv", [script_name, "--model", "test.dxnn", "--camera", "0"]):
            with patch("os.path.exists", return_value=True):
                args = module.parse_arguments()

        assert hasattr(args, "model") and args.model == "test.dxnn"
        assert hasattr(args, "camera") and args.camera == 0

    def test_cli_rtsp_mode(self, script_name: str):
        module = self._load_module(script_name)

        rtsp_url = "rtsp://fake.url/stream"
        with patch(
            "sys.argv", [script_name, "--model", "test.dxnn", "--rtsp", rtsp_url]
        ):
            with patch("os.path.exists", return_value=True):
                args = module.parse_arguments()

        assert hasattr(args, "model") and args.model == "test.dxnn"
        assert hasattr(args, "rtsp") and args.rtsp == rtsp_url

    def test_cli_display_options(self, script_name: str):
        module = self._load_module(script_name)

        with patch(
            "sys.argv", [script_name, "--model", "test.dxnn", "--video", "test.mp4"]
        ):
            with patch("os.path.exists", return_value=True):
                args = module.parse_arguments()

        assert hasattr(args, "display")
        assert args.display is True

        with patch(
            "sys.argv",
            [
                script_name,
                "--model",
                "test.dxnn",
                "--video",
                "test.mp4",
                "--no-display",
            ],
        ):
            with patch("os.path.exists", return_value=True):
                args = module.parse_arguments()

        assert args.display is False
