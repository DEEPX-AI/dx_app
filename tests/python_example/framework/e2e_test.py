import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from conftest import load_module_from_file

from .config import ModelConfig, TaskType
from .performance_collector import get_collector


MODEL_FALLBACKS = {
    "SCRFD500M.dxnn": ["SCRFD500M_1.dxnn", "SCRFD500M-1.dxnn", "SCRFD500M_PPU.dxnn"],
    "YOLOv5s_Face.dxnn": ["YOLOV5S_Face-1.dxnn"],
    "YOLOV5Pose640_1.dxnn": ["YOLOV5Pose_PPU.dxnn"],
    "YoloV5S.dxnn": ["YOLOV5S_6.dxnn", "YOLOV5S_1.dxnn", "YOLOV5S-1.dxnn"],
    "YoloV5S_PPU.dxnn": ["YOLOV5S_PPU.dxnn"],
    "yolov8n_seg.dxnn": ["YOLOV8N_SEG-1.dxnn"],
    "YOLOV10N.dxnn": ["YOLOV10N-1.dxnn"],
    "yolo26s.dxnn": ["yolo26s-1.dxnn"],
    "DeepLabV3PlusMobilenet.dxnn": ["DeepLabV3PlusMobileNetV2_2.dxnn"],
    # Naming differences not resolvable by case-insensitive matching alone
    "ArcFace_MobileFaceNet.dxnn": ["arcface_mobilefacenet.dxnn"],
    "FastDepth.dxnn": ["FastDepth_1.dxnn"],
    "YoloxSLeaky.dxnn": ["YoloX_S_Leaky.dxnn"],
    "DAMOYoloS.dxnn": ["DamoYoloS.dxnn"],
    "SSDMv1.dxnn": ["SSDMV1.dxnn"],
    # Completely renamed models
    "ZeroDCE.dxnn": ["zero_dce.dxnn"],
    "RetinaFace_Mobilenet0_25_640.dxnn": ["retinaface_mobilenet0.25_640.dxnn"],
}


class RunnerScriptProxy:
    """Proxy for v3.0.0 runner-pattern scripts that have no top-level class.

    Executes the script as a subprocess so that ``image_inference`` and
    ``stream_inference`` calls in E2ETestFramework work transparently.
    """

    def __init__(self, script_path: Path, model_path: str, src_root: Path):
        self._script_path = script_path
        self._model_path = model_path
        self._src_root = src_root

    def _run(self, input_flag: str, input_path: str, display: bool = False) -> str:
        cmd = [
            sys.executable,
            str(self._script_path),
            "--model", self._model_path,
            input_flag, input_path,
            "--loop", "1",
        ]
        if not display:
            cmd.append("--no-display")

        env = os.environ.copy()
        pythonpath = str(self._src_root)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (pythonpath + os.pathsep + existing if existing else pythonpath)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Script exited with code {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result.stdout

    def image_inference(self, image_path: str, display: bool = False) -> None:
        self._run("--image", image_path, display=display)

    def stream_inference(self, video_path: str, display: bool = False) -> None:
        output = self._run("--video", video_path, display=display)
        # Print to stdout so _capture_output() can collect metrics
        print(output, end="")


class E2ETestFramework:

    def __init__(self, model_config: ModelConfig):
        self.config = model_config

        tests_root = Path(__file__).parent.parent
        src_root = tests_root.parent.parent / "src" / "python_example"
        self._src_root = src_root
        self.model_path = src_root / self.config.base_path

        assets_root = tests_root.parent.parent / "assets" / "models"
        model_filename = (
            self.config.model_filename or f"{self.config.name.lower()}.dxnn"
        )
        self.test_model_path = str(self._resolve_model_path(assets_root, model_filename))

        self.test_image_path = str(
            tests_root.parent.parent / "sample" / "img" / "sample_kitchen.jpg"
        )
        self.test_video_path = str(
            tests_root.parent.parent / "assets" / "videos" / "dance-group.mov"
        )

        self.collector = get_collector()

    def _resolve_model_path(self, assets_root: Path, model_filename: str) -> Path:
        candidates = [model_filename, *MODEL_FALLBACKS.get(model_filename, [])]
        stem = Path(model_filename).stem

        if stem.endswith("-1"):
            candidates.extend([f"{stem[:-2]}.dxnn", f"{stem[:-2]}_1.dxnn"])
        if stem.endswith("_1"):
            candidates.extend([f"{stem[:-2]}.dxnn", f"{stem[:-2]}-1.dxnn"])

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            candidate_path = assets_root / candidate
            if candidate_path.exists():
                return candidate_path

        # Case-insensitive fallback: handles naming differences between
        # model configs and actual compiled .dxnn filenames.
        if assets_root.exists():
            lower_target = model_filename.lower()
            for entry in sorted(assets_root.iterdir()):
                if entry.name.lower() == lower_target:
                    return entry

        return assets_root / model_filename

    def _run_model_benchmark(self, use_ort: bool = True) -> Optional[float]:
        """Run run_model command to get run_model FPS"""
        cmd = ["run_model", "-m", self.test_model_path, "-l", "1000"]
        if use_ort:
            cmd.append("--use-ort")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse FPS from output: "FPS : 192.48"
            match = re.search(r"FPS\s*:\s*([\d.]+)", result.stdout)
            if match:
                return float(match.group(1))
        except Exception as e:
            print(f"Warning: Failed to run run_model benchmark: {e}")

        return None

    def collect_run_model_fps_if_needed(self):
        """Collect run_model FPS for both ORT configurations if not already cached"""
        # Check if we need to collect ORT ON max FPS
        if self.collector.get_run_model_fps(self.test_model_path, use_ort=True) is None:
            max_fps_ort_on = self._run_model_benchmark(use_ort=True)
            if max_fps_ort_on is not None:
                self.collector.set_run_model_fps(self.test_model_path, use_ort=True, max_fps=max_fps_ort_on)
                print(f"Collected run_model FPS (ORT ON): {max_fps_ort_on:.1f}")

        # Check if we need to collect ORT OFF max FPS
        if self.collector.get_run_model_fps(self.test_model_path, use_ort=False) is None:
            max_fps_ort_off = self._run_model_benchmark(use_ort=False)
            if max_fps_ort_off is not None:
                self.collector.set_run_model_fps(self.test_model_path, use_ort=False, max_fps=max_fps_ort_off)
                print(f"Collected run_model FPS (ORT OFF): {max_fps_ort_off:.1f}")

    def _clear_dx_modules(self):
        # Clear dx_engine/dx_postprocess mocks and also common.* modules
        # that may have been cached with mock dx_engine by pytest-cov's
        # early import phase.
        stale_keys = [
            k for k in list(sys.modules.keys())
            if k in ("dx_engine", "dx_postprocess")
            or k == "common" or k.startswith("common.")
            or k == "factory" or k.startswith("factory.")
        ]
        for key in stale_keys:
            sys.modules.pop(key, None)

    def _load_module(self, script_name: str):
        script_path = self.model_path / script_name

        if not script_path.exists():
            return None

        self._clear_dx_modules()

        # Ensure src/python_example is at the front of sys.path so that
        # 'common' resolves to src/python_example/common, not tests/common.
        src_root_str = str(self._src_root)
        original_sys_path = sys.path.copy()
        if src_root_str in sys.path:
            sys.path.remove(src_root_str)
        sys.path.insert(0, src_root_str)
        try:
            return load_module_from_file(str(script_path), script_name.replace(".py", ""))
        finally:
            sys.path[:] = original_sys_path

    def _create_model_instance(self, script_name: str):
        module = self._load_module(script_name)

        if module is None:
            pytest.skip(f"Failed to load module: {script_name}")

        cls = getattr(module, self.config.class_name, None)
        if cls is None:
            # v3.0.0 runner-pattern script: no top-level class.
            # Fall back to subprocess execution via RunnerScriptProxy.
            if getattr(module, "main", None) is not None:
                script_path = self.model_path / script_name
                return RunnerScriptProxy(script_path, self.test_model_path, self._src_root)
            pytest.skip(
                f"Class '{self.config.class_name}' not found in {script_name} "
                f"(v3.0.0 scripts use runner pattern instead of class-based inference)"
            )

        with patch("os.path.exists", return_value=True):
            return cls(self.test_model_path)

    def _capture_output(self, func, *args, **kwargs):
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout

        return captured_output.getvalue()

    def test_image_inference_real(self, script_name: str):
        display = os.getenv("E2E_DISPLAY", "0") == "1"

        if not Path(self.test_model_path).exists():
            pytest.skip(f"Model file not found: {self.test_model_path}")

        model = self._create_model_instance(script_name)

        try:
            if self.config.task == TaskType.CLASSIFICATION:
                model.image_inference(self.test_image_path)
            elif display:
                model.image_inference(self.test_image_path, display=True)
            else:
                model.image_inference(self.test_image_path, display=False)

        except Exception as e:
            pytest.fail(
                f"{script_name}: image_inference failed with {type(e).__name__}: {e}"
            )

    def test_stream_inference_real(self, script_name: str):
        display = os.getenv("E2E_DISPLAY", "0") == "1"

        if not Path(self.test_model_path).exists():
            pytest.skip(f"Model file not found: {self.test_model_path}")

        is_async = "async" in script_name.lower()

        if is_async and display:
            pytest.skip(
                "Async variants do not support display mode (cv2.imshow thread-safety issue)"
            )

        # Collect run_model FPS before running tests
        self.collect_run_model_fps_if_needed()

        task = Path(self.config.base_path).parts[0]
        model_name = self.config.name.lower()
        variant_name = script_name.replace(".py", "")
        test_file = Path(self.test_video_path).name

        # Determine which ORT configuration this variant uses
        is_ort_off = "ort_off" in variant_name.lower()
        use_ort = not is_ort_off
        run_model_fps = self.collector.get_run_model_fps(self.test_model_path, use_ort=use_ort)

        model = self._create_model_instance(script_name)

        try:
            if display:
                model.stream_inference(self.test_video_path, display=True)
            else:
                output = self._capture_output(
                    model.stream_inference, self.test_video_path, display=False
                )

                metrics = self.collector.parse_output(
                    output=output,
                    task=task,
                    model=model_name,
                    variant=variant_name,
                    test_file=test_file,
                    model_path=self.test_model_path,
                    video_path=self.test_video_path,
                )

                if metrics:
                    # Add run_model FPS to metrics
                    metrics.run_model_fps = run_model_fps
                    self.collector.add_result(metrics)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            pytest.fail(f"{script_name}: failed with {type(e).__name__}: {e}")
