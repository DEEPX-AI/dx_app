import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import load_module_from_file

from .config import ModelConfig, TaskType
from .performance_collector import get_collector


class E2ETestFramework:

    def __init__(self, model_config: ModelConfig):
        self.config = model_config

        tests_root = Path(__file__).parent.parent
        src_root = tests_root.parent.parent / "src" / "python_example"
        self.model_path = src_root / self.config.base_path

        assets_root = tests_root.parent.parent / "assets" / "models"
        model_filename = (
            self.config.model_filename or f"{self.config.name.lower()}.dxnn"
        )
        self.test_model_path = str(assets_root / model_filename)

        self.test_image_path = str(
            tests_root.parent.parent / "sample" / "img" / "1.jpg"
        )
        self.test_video_path = str(
            tests_root.parent.parent / "assets" / "videos" / "dance-group.mov"
        )

        self.collector = get_collector()

    def _run_model_benchmark(self, use_ort: bool = True) -> float:
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
        for module_name in ["dx_engine", "dx_postprocess"]:
            sys.modules.pop(module_name, None)

    def _load_module(self, script_name: str):
        script_path = self.model_path / script_name

        if not script_path.exists():
            return None

        self._clear_dx_modules()

        return load_module_from_file(str(script_path), script_name.replace(".py", ""))

    def _create_model_instance(self, script_name: str):
        module = self._load_module(script_name)

        with patch("os.path.exists", return_value=True):
            return getattr(module, self.config.class_name)(self.test_model_path)

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
