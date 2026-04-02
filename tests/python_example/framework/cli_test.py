from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import load_module_from_file

from .config import ModelConfig

_SYS_ARGV = "sys.argv"
_TEST_DXNN = "test.dxnn"
_OS_PATH_EXISTS = "os.path.exists"
_TEST_MP4 = "test.mp4"


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

        module = load_module_from_file(str(script_path), script_name.replace(".py", ""))
        if module is None:
            pytest.skip(f"Failed to load module {script_name}")
        return module

    def _get_parse_fn(self, module):
        """Return parse_arguments if present, else parse_args (v3.0.0 scripts use parse_args)."""
        fn = getattr(module, 'parse_arguments', None) or getattr(module, 'parse_args', None)
        if fn is None:
            pytest.skip("No parse_arguments or parse_args function found in module")
        return fn

    def test_cli_help(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        with pytest.raises(SystemExit) as exc_info:
            with patch(_SYS_ARGV, [script_name, "--help"]):
                parse_fn()

        assert exc_info.value.code == 0

    def test_cli_missing_required_args(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        with pytest.raises(SystemExit) as exc_info:
            with patch(_SYS_ARGV, [script_name]):
                parse_fn()

        assert exc_info.value.code != 0

    def test_cli_unrecognized_argument(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        with pytest.raises(SystemExit) as exc_info:
            with patch(_SYS_ARGV, [script_name, "--invalid-option", "value"]):
                parse_fn()

        assert exc_info.value.code != 0

    def test_cli_image_mode(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        with patch(
            _SYS_ARGV, [script_name, "--model", _TEST_DXNN, "--image", "test.jpg"]
        ):
            with patch(_OS_PATH_EXISTS, return_value=True):
                args = parse_fn()

        assert hasattr(args, "model") and args.model == _TEST_DXNN
        assert hasattr(args, "image") and args.image == "test.jpg"

    def test_cli_video_mode(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        with patch(
            _SYS_ARGV, [script_name, "--model", _TEST_DXNN, "--video", _TEST_MP4]
        ):
            with patch(_OS_PATH_EXISTS, return_value=True):
                args = parse_fn()

        assert hasattr(args, "model") and args.model == _TEST_DXNN
        assert hasattr(args, "video") and args.video == _TEST_MP4

    def test_cli_camera_mode(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        with patch(_SYS_ARGV, [script_name, "--model", _TEST_DXNN, "--camera", "0"]):
            with patch(_OS_PATH_EXISTS, return_value=True):
                try:
                    with patch("sys.stderr"):
                        args = parse_fn()
                except SystemExit as e:
                    if e.code == 2:
                        pytest.skip(f"{script_name}: --camera argument not supported")
                    raise

        assert hasattr(args, "model") and args.model == _TEST_DXNN
        assert hasattr(args, "camera") and args.camera == 0

    def test_cli_rtsp_mode(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        rtsp_url = "rtsp://fake.url/stream"
        with patch(
            _SYS_ARGV, [script_name, "--model", _TEST_DXNN, "--rtsp", rtsp_url]
        ):
            with patch(_OS_PATH_EXISTS, return_value=True):
                try:
                    with patch("sys.stderr"):
                        args = parse_fn()
                except SystemExit as e:
                    if e.code == 2:
                        pytest.skip(f"{script_name}: --rtsp argument not supported")
                    raise

        assert hasattr(args, "model") and args.model == _TEST_DXNN
        assert hasattr(args, "rtsp") and args.rtsp == rtsp_url

    def test_cli_display_options(self, script_name: str):
        module = self._load_module(script_name)
        parse_fn = self._get_parse_fn(module)

        with patch(
            _SYS_ARGV, [script_name, "--model", _TEST_DXNN, "--video", _TEST_MP4]
        ):
            with patch(_OS_PATH_EXISTS, return_value=True):
                args = parse_fn()

        assert hasattr(args, "display")
        assert args.display is True

        with patch(
            _SYS_ARGV,
            [
                script_name,
                "--model",
                _TEST_DXNN,
                "--video",
                _TEST_MP4,
                "--no-display",
            ],
        ):
            with patch(_OS_PATH_EXISTS, return_value=True):
                args = parse_fn()

        assert args.display is False
