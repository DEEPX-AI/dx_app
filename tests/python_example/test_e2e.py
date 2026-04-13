"""
End-to-End tests for Python inference scripts.

Auto-discovers scripts from ``src/python_example/<task>/<model>/`` whose
filenames contain ``_sync`` or ``_async``, matches each to a ``.dxnn`` model
in ``assets/models/``, then runs them via **subprocess** — mirroring the
mechanism used by ``tests/cpp_example/test_e2e.py``.

Usage::

    pytest tests/python_example/test_e2e.py -v
    pytest tests/python_example/test_e2e.py -m e2e_image -v
    pytest tests/python_example/test_e2e.py -m e2e_stream -v
    pytest tests/python_example/test_e2e.py -k "alexnet" -v
"""

import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import PROJECT_ROOT, TASK_IMAGE_MAP, MODEL_IMAGE_OVERRIDE, E2E_SHORT_MODELS  # noqa: E402
from test_helpers.utils import discover_python_scripts, setup_environment, resolve_image_for_model  # noqa: E402


# ---------------------------------------------------------------------------
# Performance collection
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    task: str
    model: str
    variant: str
    test_file: str

    overall_fps: float
    read_fps: Optional[float] = None
    preprocess_fps: Optional[float] = None
    inference_fps: Optional[float] = None

    postprocess_fps: Optional[float] = None
    render_fps: Optional[float] = None

    total_frames: int = 0
    total_time: float = 0.0
    model_path: str = ""
    video_path: str = ""

    infer_inflight_avg: Optional[float] = None
    infer_inflight_max: Optional[int] = None


class PerformanceCollector:

    def __init__(self):
        self.results: Dict[str, List[PerformanceMetrics]] = defaultdict(list)

    def add_result(self, metrics: PerformanceMetrics):
        key = f"{metrics.task}/{metrics.model}"
        self.results[key].append(metrics)

    def parse_output(
        self,
        output: str,
        task: str,
        model: str,
        variant: str,
        test_file: str,
        model_path: str = "",
        video_path: str = "",
    ) -> Optional[PerformanceMetrics]:
        patterns = {
            "overall_fps": r"Overall FPS\s*:\s*([\d.]+)\s*FPS",
            "read_fps": r"Read\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS",
            "preprocess_fps": r"Preprocess\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS",
            "inference_fps": r"Inference\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS",
            "postprocess_fps": r"Postprocess\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS",
            "render_fps": r"Render\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS",
            "total_frames": r"Total Frames\s*:\s*(\d+)",
            "total_time": r"Total Time\s*:\s*([\d.]+)\s*s",
            "infer_inflight_avg": r"Inflight Avg\s*:\s*([\d.]+)",
            "infer_inflight_max": r"Inflight Max\s*:\s*(\d+)",
        }

        data: dict = {
            "task": task, "model": model, "variant": variant,
            "test_file": test_file, "model_path": model_path, "video_path": video_path,
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = match.group(1)
                data[key] = int(value) if key in ("total_frames", "infer_inflight_max") else float(value)

        if "overall_fps" not in data:
            return None
        return PerformanceMetrics(**data)

    def get_results_by_task_model(self, task: str, model: str) -> List[PerformanceMetrics]:
        return sorted(self.results[f"{task}/{model}"], key=lambda x: x.variant)

    def get_all_tasks(self) -> List[str]:
        return sorted(set(k.split("/")[0] for k in self.results))

    def get_models_by_task(self, task: str) -> List[str]:
        return sorted(set(k.split("/")[1] for k in self.results if k.startswith(f"{task}/")))

    def save_csv(self, filepath: str):
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Task", "Model", "Variant", "Model Path", "Video Path",
                "Total Frames", "Total Time (s)", "E2E FPS", "Read FPS",
                "Preprocess FPS", "Inference FPS",
                "Postprocess FPS", "Inflight Avg", "Inflight Max",
            ])
            for task in self.get_all_tasks():
                for model in self.get_models_by_task(task):
                    for m in self.get_results_by_task_model(task, model):
                        is_async = m.infer_inflight_avg is not None
                        pf = {"read": m.read_fps, "preprocess": m.preprocess_fps,
                              "inference": m.inference_fps, "postprocess": m.postprocess_fps}
                        valid_pf = {k: v for k, v in pf.items() if v is not None}
                        bn = min(valid_pf, key=valid_pf.get) if valid_pf else None
                        def _fmt(name, val, _is_async=is_async, _bn=bn):
                            if val is None:
                                return ""
                            return f"{val:.1f} *" if (_is_async and _bn == name) else f"{val:.1f}"
                        writer.writerow([
                            m.task, m.model, m.variant, m.model_path, m.video_path,
                            m.total_frames, f"{m.total_time:.1f}", f"{m.overall_fps:.1f}",
                            _fmt("read", m.read_fps), _fmt("preprocess", m.preprocess_fps),
                            _fmt("inference", m.inference_fps),
                            _fmt("postprocess", m.postprocess_fps),
                            f"{m.infer_inflight_avg:.1f}" if m.infer_inflight_avg is not None else "",
                            f"{m.infer_inflight_max}" if m.infer_inflight_max is not None else "",
                        ])

    def print_report(self):
        if not self.results:
            print("No performance data collected.")
            return
        sep = " | "
        headers = ["Variant", "E2E [FPS]", "Read [FPS]", "Preprocess [FPS]", "Inference [FPS]", "Postprocess [FPS]"]
        widths = [42, 12, 12, 17, 16, 17]
        total_w = sum(widths) + len(sep) * (len(widths) - 1)
        print("\n" + "=" * total_w)
        print(" E2E Performance Report")
        print("=" * total_w)
        for task in self.get_all_tasks():
            for model in self.get_models_by_task(task):
                results = self.get_results_by_task_model(task, model)
                if not results:
                    continue
                first = results[0]
                print(f"\n {task.replace('_', ' ').title()} - {model.upper()}")
                print(f"\n Model: {first.model_path}")
                print(f" Video: {first.video_path} ({first.total_frames} frames)")
                print("-" * total_w)
                print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
                print("-" * total_w)
                for r in results:
                    is_async = r.infer_inflight_avg is not None
                    pf = {"read": r.read_fps, "preprocess": r.preprocess_fps,
                          "inference": r.inference_fps, "postprocess": r.postprocess_fps}
                    valid_pf = {k: v for k, v in pf.items() if v is not None}
                    bn = min(valid_pf, key=valid_pf.get) if valid_pf else None
                    def _fmt(name, val, _is_async=is_async, _bn=bn):
                        if val is None:
                            return "-"
                        return f"{val:.1f} *" if (_is_async and _bn == name) else f"{val:.1f}"
                    row = [r.variant[:40], f"{r.overall_fps:.1f}",
                           _fmt("read", r.read_fps), _fmt("preprocess", r.preprocess_fps),
                           _fmt("inference", r.inference_fps), _fmt("postprocess", r.postprocess_fps)]
                    print(sep.join(cell.ljust(w) for cell, w in zip(row, widths)))
                print("-" * total_w)
        print("\n" + "=" * total_w)


_collector = PerformanceCollector()


def get_collector() -> PerformanceCollector:
    return _collector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TEST_VIDEO = PROJECT_ROOT / "assets" / "videos" / "dance-group.mov"

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _build_params() -> Tuple[List, List]:
    """Return (image_params, stream_params) each as list of pytest.param."""
    raw = discover_python_scripts(suffixes=("_sync", "_async"))
    image_params = []
    stream_params = []

    for task, model_name, sync_scripts, async_scripts, model_path in raw:
        img_rel = resolve_image_for_model(model_name, task)
        if img_rel is None:
            img_rel = TASK_IMAGE_MAP.get(task, "sample/img/sample_kitchen.jpg")
        image_path = PROJECT_ROOT / img_rel

        for script, mode in ([(s, "sync") for s in sync_scripts] + [(s, "async") for s in async_scripts]):
            exec_marker = pytest.mark.sync_exec if mode == "sync" else pytest.mark.async_exec
            model_marker = getattr(pytest.mark, model_name)
            marks = [exec_marker, model_marker]
            if model_name in E2E_SHORT_MODELS:
                marks.append(pytest.mark.e2e_short)
            param = pytest.param(script, model_path, image_path, id=script.stem, marks=marks)
            image_params.append(param)
            stream_param = pytest.param(script, model_path, id=script.stem, marks=marks)
            stream_params.append(stream_param)

    return image_params, stream_params


_IMAGE_PARAMS, _STREAM_PARAMS = _build_params()

if not _IMAGE_PARAMS and not _STREAM_PARAMS:
    pytest.skip(
        "No Python example scripts found in src/python_example/",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.e2e_image
@pytest.mark.parametrize("script,model,image", _IMAGE_PARAMS)
def test_image_inference_e2e(script: Path, model: Optional[Path], image: Path, loop_count):
    """Run Python script with --image and --no-display."""
    if model is None:
        pytest.skip(f"Model .dxnn not found for {script.stem}: run setup_sample_models.sh first")
    if not image.exists():
        pytest.skip(f"Test image not found: {image}")

    display = os.getenv("E2E_DISPLAY", "0") == "1"
    cmd = [
        sys.executable, str(script),
        "--model", str(model),
        "--image", str(image),
        "--loop", str(loop_count),
    ]
    if not display:
        cmd.append("--no-display")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=setup_environment(),
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name}: image inference timed out after 120s")

    assert result.returncode == 0, (
        f"image_inference FAILED (exit {result.returncode})\n"
        f"Script : {script}\nModel  : {model}\n"
        f"CMD    : {' '.join(cmd)}\n"
        f"stdout : {result.stdout[-2000:]}\nstderr : {result.stderr[-2000:]}"
    )

    # Collect performance metrics from stdout if FPS data is present
    task = script.parent.parent.name
    metrics = _collector.parse_output(
        result.stdout + result.stderr,
        task=task,
        model=script.parent.name,
        variant=script.stem,
        test_file=script.name,
        model_path=str(model),
        video_path=str(image),
    )
    if metrics is not None:
        _collector.add_result(metrics)


@pytest.mark.e2e
@pytest.mark.e2e_stream
@pytest.mark.parametrize("script,model", _STREAM_PARAMS)
def test_stream_inference_e2e(script: Path, model: Optional[Path], loop_count):
    """Run Python script with --video and --no-display."""
    if model is None:
        pytest.skip(f"Model .dxnn not found for {script.stem}: run setup_sample_models.sh first")
    if not _TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {_TEST_VIDEO}")

    display = os.getenv("E2E_DISPLAY", "0") == "1"
    if "_async" in script.stem and display:
        pytest.skip("Async variants do not support display mode")

    cmd = [
        sys.executable, str(script),
        "--model", str(model),
        "--video", str(_TEST_VIDEO),
        "--loop", str(loop_count),
    ]
    if not display:
        cmd.append("--no-display")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=setup_environment(),
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name}: stream inference timed out after 300s")

    assert result.returncode == 0, (
        f"stream_inference FAILED (exit {result.returncode})\n"
        f"Script : {script}\nModel  : {model}\n"
        f"CMD    : {' '.join(cmd)}\n"
        f"stdout : {result.stdout[-2000:]}\nstderr : {result.stderr[-2000:]}"
    )

    # Collect performance metrics from stdout if FPS data is present
    task = script.parent.parent.name
    metrics = _collector.parse_output(
        result.stdout + result.stderr,
        task=task,
        model=script.parent.name,
        variant=script.stem,
        test_file=script.name,
        model_path=str(model),
        video_path=str(_TEST_VIDEO),
    )
    if metrics is not None:
        _collector.add_result(metrics)
