"""Regression tests for Q3 release issue fixes."""

import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_super_resolution_stream_path_does_not_use_fixed_20_tiles_width():
    """SR stream processing must preserve input size instead of forcing 20 tiles."""
    source = _read("src/python_example/common/runner/sync_runner.py")

    assert "20 tiles wide" not in source
    assert "tile_w * 20" not in source


def test_super_resolution_paths_use_padding_not_resize_for_tile_alignment():
    """Tile-boundary alignment should pad/crop, not geometrically resize frames."""
    expected_markers = {
        "src/python_example/common/runner/sync_runner.py": "cv2.copyMakeBorder",
        "src/python_example/common/runner/async_runner.py": "cv2.copyMakeBorder",
        "src/cpp_example/common/runner/sync_restoration_runner.hpp": "cv::copyMakeBorder",
        "src/cpp_example/common/runner/async_restoration_runner.hpp": "cv::copyMakeBorder",
    }

    for relpath, marker in expected_markers.items():
        assert marker in _read(relpath), f"{relpath} should use padding for SR tiles"


def test_cpp_async_sr_stream_path_warns_for_large_tile_count():
    """Async C++ SR stream processing should warn on very large tile counts."""
    source = _read("src/cpp_example/common/runner/async_restoration_runner.hpp")

    assert "tiles_count" in source or "tiles_total" in source
    assert "produces " in source
    assert "tiles; processing may be slow" in source


def test_python_image_only_wrappers_mark_stream_inputs_unsupported():
    """Embedding/ReID Python wrappers should mark stream inputs unsupported for runtime rejection."""
    for root in [
        ROOT / "src/python_example/embedding",
        ROOT / "src/python_example/reid",
    ]:
        for path in root.glob("*/*.py"):
            if path.name.startswith("__"):
                continue
            source = path.read_text(encoding="utf-8")
            if "parse_common_args(" not in source:
                continue
            assert "include_stream_inputs=False" in source, str(path.relative_to(ROOT))


def test_python_image_only_help_exposes_stream_options_for_parser_compatibility():
    """Embedding/ReID Python -h should parse stream flags and reject them at runtime."""
    scripts = [
        "src/python_example/embedding/arcface_mobilefacenet/arcface_mobilefacenet_sync.py",
        "src/python_example/reid/casvit_t/casvit_t_sync.py",
    ]
    for relpath in scripts:
        result = subprocess.run(
            [sys.executable, str(ROOT / relpath), "-h"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        for expected in ["--video", "--camera", "--rtsp"]:
            assert expected in result.stdout, f"{relpath} help does not expose {expected}"


def test_python_image_only_stream_input_keeps_guidance_message():
    """Image-only Python examples should reject stream input with actionable guidance."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/python_example/embedding/arcface_mobilefacenet/arcface_mobilefacenet_sync.py"),
            "-m",
            "assets/models/arcface_mobilefacenet.dxnn",
            "--video",
            "any.mp4",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "Video/camera input requires a detection crop pipeline" in result.stderr
    assert "--image" in result.stderr


class _ImageOnlyFactory:
    def get_task_type(self):
        return "embedding"


def _stream_args():
    return SimpleNamespace(
        model="dummy.dxnn",
        image=None,
        video="any.mp4",
        camera=None,
        rtsp=None,
        display=False,
        show_log=False,
        fast_postprocess=False,
    )


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("common.runner.sync_runner", "SyncRunner"),
        ("common.runner.async_runner", "AsyncRunner"),
    ],
)
def test_python_image_only_runners_reject_stream_before_no_input_hint(monkeypatch, module_name, class_name):
    """Image-only runner.run() should reject stream input before the no-image hint path."""
    module = __import__(module_name, fromlist=[class_name])
    runner_cls = getattr(module, class_name)
    runner = runner_cls(_ImageOnlyFactory())

    monkeypatch.setattr(module, "_check_dxrt_version", lambda: None)
    monkeypatch.setattr(module, "_apply_default_input", lambda args, factory: None)
    monkeypatch.setattr(module, "_validate_inputs", lambda args: None)
    monkeypatch.setattr(runner, "_init_engine", lambda *args, **kwargs: pytest.fail("engine should not initialize"))

    with pytest.raises(SystemExit) as excinfo:
        runner.run(_stream_args())

    assert excinfo.value.code == 1


def test_python_image_only_no_input_prints_hint_before_engine_init():
    """Image-only Python examples should print no-input hint before importing dx_engine."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/python_example/embedding/arcface_mobilefacenet/arcface_mobilefacenet_sync.py"),
            "-m",
            "assets/models/arcface_mobilefacenet.dxnn",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "takes image input only" in output
    assert "--image" in output
    assert "dx_engine" not in output


def test_cpp_embedding_runners_do_not_expose_stream_input_options():
    """Embedding/ReID C++ runners use embedding runners, so help must be image-only."""
    for relpath in [
        "src/cpp_example/common/runner/sync_embedding_runner.hpp",
        "src/cpp_example/common/runner/async_embedding_runner.hpp",
    ]:
        source = _read(relpath)
        for forbidden in ["video_path", "camera_index", "rtsp_url", "RTSP stream URL"]:
            assert forbidden not in source, f"{relpath} still exposes {forbidden}"


def test_cpp_embedding_runners_remove_stream_dead_paths():
    """Image-only embedding runners should not keep unreachable stream-processing code."""
    forbidden_markers = [
        "cv::VideoCapture",
        "cv::VideoWriter",
        "openVideoCapture",
        "processVideoFrame",
        "processVideoFrames",
        "initVideoWriter",
        "writeToVideo",
        "video_save_path",
        "autoDownloadVideos",
    ]
    for relpath in [
        "src/cpp_example/common/runner/sync_embedding_runner.hpp",
        "src/cpp_example/common/runner/async_embedding_runner.hpp",
    ]:
        source = _read(relpath)
        for marker in forbidden_markers:
            assert marker not in source, f"{relpath} still contains stream dead path {marker}"


def test_cpp_sync_embedding_save_mode_is_not_dump_mode():
    """Sync embedding should save images only for --save, not merely for --dump-tensors."""
    source = _read("src/cpp_example/common/runner/sync_embedding_runner.hpp")

    assert "processCount, args.no_display, args.saveMode," in source
    assert "if (!runDir.empty() && saveMode)" in source
    assert "savePath = dxapp::buildPerImageSavePath" in source


def test_release_sources_do_not_keep_model_specific_fast_postprocess_names():
    """Fast segmentation postprocess should use generic names, not model-only names."""
    forbidden = ["Pid" + "NetPostprocessor", "pid" + "net_argmax_scale", "PID" + "Net"]
    for root_name in ["src", "scripts", "config", "tests"]:
        for path in (ROOT / root_name).rglob("*"):
            if path.is_dir() or path.suffix in {".pyc", ".so", ".dll", ".pyd"}:
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for token in forbidden:
                assert token not in source, f"{path.relative_to(ROOT)} still contains {token}"


def test_generic_fast_segmentation_postprocessor_exists_in_both_languages():
    """The low-resolution argmax fast path should have a generic public name."""
    assert "FastSegmentationPostprocessor" in _read(
        "src/python_example/common/processors/fast_segmentation_postprocessor.py"
    )
    assert "FastSegmentationPostprocessor" in _read(
        "src/cpp_example/common/processors/segmentation_postprocessor.hpp"
    )


def test_add_model_exposes_generic_fast_segmentation_alias():
    """Model generation scripts should expose the generic fast segmentation alias."""
    add_model = _read("scripts/add_model.sh")
    dx_tool = _read("scripts/dx_tool.sh")

    assert "fast_segmentation" in add_model
    assert "FastSegmentationPostprocessor" in add_model
    assert "fast_segmentation" in dx_tool


def test_release_docs_cover_yolo_and_fast_segmentation_updates():
    """Release documentation should describe YOLO customization and generic fast segmentation."""
    mkdocs = _read("docs/mkdocs.yml")
    yolo_guide = _read("docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md")
    cpp_postprocess = _read("docs/source/docs/07_DX-APP_CPP_PostProcess_Overview.md")
    pybind_postprocess = _read("docs/source/docs/08_DX-APP_Pybind_PostProcess_Overview.md")
    dx_tool = _read("docs/source/docs/10_DX-APP_DX-Tool_Guide.md")
    source_structure = _read("docs/source/docs/11_DX-APP_Example_Source_Structure.md")

    assert "12_DX-APP_YOLO_Customizing_Guide.md" in mkdocs
    assert "YOLO Customizing Guide" in yolo_guide
    assert "FastSegmentationPostprocessor" in cpp_postprocess
    assert "FastSegmentationPostprocessor" in pybind_postprocess
    assert "fast_segmentation" in dx_tool
    assert "fast_segmentation_postprocessor.py" in source_structure
