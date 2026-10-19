"""Regression tests for Q3 release issue fixes."""

import re
import subprocess
import sys
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from test_helpers.proc import example_python, run_bounded


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
    # The C++ sync runner pads through srtiling::prepareLowRes, which the graph's
    # tiled-SR stage shares; the other three pad in place.
    expected_markers = {
        "src/python_example/common/runner/sync_runner.py": "cv2.copyMakeBorder",
        "src/python_example/common/runner/async_runner.py": "cv2.copyMakeBorder",
        "src/cpp_example/common/runner/sync_restoration_runner.hpp": "srtiling::prepareLowRes(",
        "src/cpp_example/common/runner/async_restoration_runner.hpp": "cv::copyMakeBorder",
    }

    for relpath, marker in expected_markers.items():
        assert marker in _read(relpath), f"{relpath} should use padding for SR tiles"

    helper = re.search(r"inline void prepareLowRes\(.*?\n\}",
                       _read("src/cpp_example/common/utility/sr_tiling.hpp"), re.S)
    assert helper, "sr_tiling.hpp should define prepareLowRes"
    assert "cv::copyMakeBorder" in helper.group(0), "prepareLowRes should pad the LR frame"
    assert "resize" not in helper.group(0), "prepareLowRes should pad, not resize"


def test_cpp_async_sr_stream_path_warns_for_large_tile_count():
    """Async C++ SR stream processing should warn on very large tile counts."""
    source = _read("src/cpp_example/common/runner/async_restoration_runner.hpp")

    # Assert the guard itself — a tile-count threshold wrapping the WARN — not the
    # name of the variable holding the count. The halo-aware tiling refactor
    # replaced the local `tiles_count` with `plans.size()` without changing the
    # threshold or the message.
    assert re.search(
        r"if\s*\(.*>\s*400\s*\)\s*\{[^}]*tiles; processing may be slow", source, re.S
    ), "async SR path must keep the >400-tile warning guard"
    assert "produces " in source


def test_python_image_only_wrappers_mark_stream_inputs_unsupported():
    """Embedding/ReID Python wrappers should mark stream inputs unsupported for runtime rejection."""
    for root in [
        ROOT / "src/python_example/face_recognition",
        ROOT / "src/python_example/image_classification/casvit",
    ]:
        # The guarantee moved from a source literal to the variant config: one family
        # entry script serves every variant, so it reads include_stream_inputs from
        # <variant>/config.json instead of hard-coding False. Assert the DATA now.
        # root may be a task dir (holding families) or a family dir itself.
        configs = sorted(root.glob("*/*/config.json")) or \
            sorted(root.glob("*/config.json"))
        assert configs, f"no variant configs under {root}"
        for cfg_path in configs:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert cfg["cli"]["include_stream_inputs"] is False, \
                f"{cfg_path.relative_to(ROOT)} must not register stream inputs"


def test_python_image_only_help_hides_stream_options():
    """Embedding/ReID Python -h must NOT expose stream flags.

    Image-only wrappers build their parser with ``include_stream_inputs=False``,
    so ``--video`` / ``--camera`` / ``--rtsp`` are never registered: they are
    absent from ``--help`` and ``--image`` remains available.
    """
    scripts = [
        "src/python_example/face_recognition/arcface/arcface_mobilefacenet_112x112/arcface_mobilefacenet_112x112_sync.py",
        "src/python_example/image_classification/casvit/casvit-t_224x224/casvit-t_224x224_sync.py",
    ]
    for relpath in scripts:
        result = run_bounded(
            [example_python(), str(ROOT / relpath), "-h"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "--image" in result.stdout, f"{relpath} help must expose --image"
        for forbidden in ["--video", "--camera", "--rtsp"]:
            assert forbidden not in result.stdout, f"{relpath} help still exposes {forbidden}"


def test_python_image_only_stream_input_rejected_by_argparse():
    """Image-only Python examples must reject a ``--video`` flag at the CLI level.

    Because the parser is built with ``include_stream_inputs=False``, ``--video``
    is an unknown option: argparse exits with code 2 and reports
    "unrecognized arguments" — inference is never reached.
    """
    result = run_bounded(
        [
            example_python(),
            str(ROOT / "src/python_example/face_recognition/arcface/arcface_mobilefacenet_112x112/arcface_mobilefacenet_112x112_sync.py"),
            "-m",
            "assets/models/arcface_mobilefacenet_112x112.dxnn",
            "--video",
            "any.mp4",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )

    assert result.returncode == 2, result.stderr
    assert "unrecognized arguments" in result.stderr
    assert "--video" in result.stderr


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
    result = run_bounded(
        [
            example_python(),
            str(ROOT / "src/python_example/face_recognition/arcface/arcface_mobilefacenet_112x112/arcface_mobilefacenet_112x112_sync.py"),
            "-m",
            "assets/models/arcface_mobilefacenet_112x112.dxnn",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
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


_GENERATED_CARD = "<!-- Generated by DX AI Studio"


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
            # So do the generated model cards; hand-written READMEs are still checked.
            if _GENERATED_CARD in source[:300]:
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
    # add_model.sh is a wrapper; the generation (and its postprocessor table) lives in add_model_variant.py.
    add_model = _read("scripts/add_model.sh") + _read("scripts/add_model_variant.py")
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


# ---------------------------------------------------------------------------
# The 3D task is spelled "3d_object_detection" everywhere: registry, example
# directory, variant config, and the task type both runners report. These tests
# hold that single spelling, and hold the guard that depends on it -- a LiDAR
# example must refuse a non-.bin input rather than push image pixels through a
# point-cloud network.
# ---------------------------------------------------------------------------

SFA3D_DIR = ("src/python_example/3d_object_detection/sfa3d/sfa3d_608x608")


def test_no_source_file_compares_against_a_short_form_task_name():
    """Only the full task name is compared at runtime, in both trees."""
    offenders = []
    for rel in ("src/python_example/common/runner/sync_runner.py",
                "src/python_example/common/runner/async_runner.py",
                "src/cpp_example/common/utility/common_util.hpp",
                "src/cpp_example/common/runner/sync_3d_object_detection_runner.hpp",
                "src/cpp_example/common/runner/async_3d_object_detection_runner.hpp",
                f"{SFA3D_DIR}/factory/sfa3d_608x608_factory.py"):
        path = ROOT / rel
        if not path.is_file():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            bare = line.strip()
            if bare.startswith(("#", "//", "*", "/*")):
                continue
            if '"3d_detection"' in line:
                offenders.append(f"{rel}:{n}: {bare}")
    assert not offenders, ("a short-form task name is still compared:\n  "
                           + "\n  ".join(offenders))


def test_cpp_factory_reports_the_canonical_task_name():
    src = _read("src/cpp_example/3d_object_detection/sfa3d/sfa3d_608x608/"
                "factory/sfa3d_608x608_factory.hpp")
    assert 'getTaskType() const override { return "3d_object_detection"; }' in src


def test_the_3d_task_declares_no_default_video():
    """SFA3D consumes LiDAR point clouds; there is no video form of that input."""
    import ast
    src = _read("src/python_example/common/runner/sync_runner.py")
    start = src.index("_DEFAULT_SAMPLE_VIDEO = {")
    table = src[start + len("_DEFAULT_SAMPLE_VIDEO = ") : src.index("\n}", start) + 2]
    # resolve the module-level _VID_* constants the table refers to
    env = {name: value for name, value in
           re.findall(r'^(_VID_\w+)\s*=\s*"([^"]+)"', src, re.M)}
    mapping = eval(table, {"__builtins__": {}}, env)  # noqa: S307 - literal table
    assert mapping["3d_object_detection"] is None
    assert "3d_detection" not in mapping


def test_a_non_bin_input_is_rejected_for_the_lidar_example():
    """The guard that makes the dead comparison matter.

    Runs the real entry script: an image path must be refused BEFORE inference,
    not silently fed to a point-cloud model.
    """
    script = ROOT / SFA3D_DIR / "sfa3d_608x608_sync.py"
    model = ROOT / "assets" / "models" / "sfa3d_608x608.dxnn"
    image = ROOT / "sample" / "img" / "sample_street.jpg"
    if not model.is_file():
        pytest.skip(f"{model.name} not downloaded")
    result = subprocess.run(
        [sys.executable, str(script), "-m", str(model), "-i", str(image),
         "--no-display"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT),
    )
    combined = result.stdout + result.stderr
    assert "LiDAR point-cloud .bin" in combined, (
        "a .jpg was accepted by the LiDAR example:\n" + combined[-1500:])
    assert result.returncode != 0, "the guard must exit non-zero"
