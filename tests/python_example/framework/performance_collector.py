import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PerformanceMetrics:
    task: str
    model: str
    variant: str
    test_file: str

    overall_fps: float
    read_fps: float
    preprocess_fps: float
    inference_fps: float

    postprocess_fps: float = 0.0
    render_fps: float = 0.0

    total_frames: int = 0
    total_time: float = 0.0
    timestamp: str = ""
    model_path: str = ""
    video_path: str = ""

    infer_inflight_avg: Optional[float] = None
    infer_inflight_max: Optional[int] = None
    run_model_fps: Optional[float] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PerformanceCollector:

    def __init__(self):
        self.results: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.run_model_fps_cache: Dict[str, float] = {}

    def set_run_model_fps(self, model_path: str, use_ort: bool, max_fps: float):
        """Store run_model FPS for a model configuration"""
        key = f"{model_path}::{use_ort}"
        self.run_model_fps_cache[key] = max_fps

    def get_run_model_fps(self, model_path: str, use_ort: bool) -> Optional[float]:
        """Retrieve cached run_model FPS for a model configuration"""
        key = f"{model_path}::{use_ort}"
        return self.run_model_fps_cache.get(key)

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
            "infer_inflight_avg": r"Infer Inflight Avg\s*:\s*([\d.]+)",
            "infer_inflight_max": r"Infer Inflight Max\s*:\s*(\d+)",
        }

        data = {
            "task": task,
            "model": model,
            "variant": variant,
            "test_file": test_file,
            "model_path": model_path,
            "video_path": video_path,
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key in ["total_frames", "infer_inflight_max"]:
                    data[key] = int(value)
                else:
                    data[key] = float(value)

        if "overall_fps" not in data:
            return None

        return PerformanceMetrics(**data)

    def get_results_by_task_model(
        self, task: str, model: str
    ) -> List[PerformanceMetrics]:
        key = f"{task}/{model}"
        results = self.results[key]
        return sorted(results, key=lambda x: x.variant)

    def get_all_tasks(self) -> List[str]:
        return sorted(set(k.split("/")[0] for k in self.results.keys()))

    def get_models_by_task(self, task: str) -> List[str]:
        return sorted(
            set(
                k.split("/")[1] for k in self.results.keys() if k.startswith(f"{task}/")
            )
        )

    def save_csv(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "Task",
                    "Model",
                    "Variant",
                    "Model Path",
                    "Video Path",
                    "Total Frames",
                    "Total Time (s)",
                    "E2E FPS",
                    "Read FPS",
                    "Preprocess FPS",
                    "Inference FPS",
                    "run_model FPS",
                    "Postprocess FPS",
                    "Infer Inflight Avg",
                    "Infer Inflight Max",
                    "Timestamp",
                ]
            )

            for task in self.get_all_tasks():
                for model in self.get_models_by_task(task):
                    results = self.get_results_by_task_model(task, model)
                    for m in results:
                        # Find bottleneck - lowest FPS among pipeline stages (only for async)
                        pipeline_fps = {
                            "read": m.read_fps,
                            "preprocess": m.preprocess_fps,
                            "inference": m.inference_fps,
                            "postprocess": m.postprocess_fps,
                        }
                        bottleneck = min(pipeline_fps, key=pipeline_fps.get)

                        # Add asterisk to bottleneck only for async variants
                        is_async = m.infer_inflight_avg is not None
                        read_fps_str = f"{m.read_fps:.1f} *" if (is_async and bottleneck == "read") else f"{m.read_fps:.1f}"
                        preprocess_fps_str = f"{m.preprocess_fps:.1f} *" if (is_async and bottleneck == "preprocess") else f"{m.preprocess_fps:.1f}"
                        inf_fps_str = f"{m.inference_fps:.1f} *" if (is_async and bottleneck == "inference") else f"{m.inference_fps:.1f}"
                        postprocess_fps_str = f"{m.postprocess_fps:.1f} *" if (is_async and bottleneck == "postprocess") else f"{m.postprocess_fps:.1f}"

                        writer.writerow(
                            [
                                m.task,
                                m.model,
                                m.variant,
                                m.model_path,
                                m.video_path,
                                m.total_frames,
                                f"{m.total_time:.1f}",
                                f"{m.overall_fps:.1f}",
                                read_fps_str,
                                preprocess_fps_str,
                                inf_fps_str,
                                f"{m.run_model_fps:.1f}" if m.run_model_fps is not None else "",
                                postprocess_fps_str,
                                f"{m.infer_inflight_avg:.1f}" if m.infer_inflight_avg is not None else "",
                                f"{m.infer_inflight_max}" if m.infer_inflight_max is not None else "",
                                m.timestamp,
                            ]
                        )

    def print_report(self):
        if not self.results:
            print("No performance data collected.")
            return

        separator = " | "
        headers = [
            "Variant",
            "E2E [FPS]",
            "Read [FPS]",
            "Preprocess [FPS]",
            "Inference [FPS]",
            "Postprocess [FPS]",
        ]
        col_widths = [42, 12, 12, 17, 16, 17]
        total_width = sum(col_widths) + len(separator) * (len(col_widths) - 1)

        print("\n" + "=" * total_width)
        print(" E2E Performance Report")
        print("=" * total_width)

        for task in self.get_all_tasks():
            for model in self.get_models_by_task(task):
                results = self.get_results_by_task_model(task, model)
                if not results:
                    continue

                first = results[0]
                model_path = first.model_path
                video_path = first.video_path
                total_frames = first.total_frames

                print(f"\n {task.replace('_', ' ').title()} - {model.upper()}")
                print(f"\n Model: {model_path}")
                print(f" Video: {video_path} ({total_frames} frames)")
                
                # Show run_model FPS if available
                run_model_ort_on = self.get_run_model_fps(model_path, use_ort=True)
                run_model_ort_off = self.get_run_model_fps(model_path, use_ort=False)
                if run_model_ort_on is not None or run_model_ort_off is not None:
                    fps_info = []
                    if run_model_ort_on is not None:
                        fps_info.append(f"(ORT ON) {run_model_ort_on:.1f} FPS")
                    if run_model_ort_off is not None:
                        fps_info.append(f"(ORT OFF) {run_model_ort_off:.1f} FPS")
                    print(f"\n run_model FPS: {', '.join(fps_info)}\n")

                print("-" * total_width)
                header_row = separator.join(
                    h.ljust(w) for h, w in zip(headers, col_widths)
                )
                print(header_row)
                print("-" * total_width)

                for result in results:
                    # Find bottleneck - lowest FPS among pipeline stages (only for async)
                    pipeline_fps = {
                        "read": result.read_fps,
                        "preprocess": result.preprocess_fps,
                        "inference": result.inference_fps,
                        "postprocess": result.postprocess_fps,
                    }
                    bottleneck = min(pipeline_fps, key=pipeline_fps.get)

                    # Add asterisk to bottleneck only for async variants
                    is_async = result.infer_inflight_avg is not None
                    read_fps_str = f"{result.read_fps:.1f} *" if (is_async and bottleneck == "read") else f"{result.read_fps:.1f}"
                    preprocess_fps_str = f"{result.preprocess_fps:.1f} *" if (is_async and bottleneck == "preprocess") else f"{result.preprocess_fps:.1f}"
                    inf_fps_str = f"{result.inference_fps:.1f} *" if (is_async and bottleneck == "inference") else f"{result.inference_fps:.1f}"
                    postprocess_fps_str = f"{result.postprocess_fps:.1f} *" if (is_async and bottleneck == "postprocess") else f"{result.postprocess_fps:.1f}"

                    row = [
                        result.variant[:40],
                        f"{result.overall_fps:.1f}",
                        read_fps_str,
                        preprocess_fps_str,
                        inf_fps_str,
                        postprocess_fps_str,
                    ]

                    row_str = separator.join(
                        cell.ljust(w) for cell, w in zip(row, col_widths)
                    )
                    print(row_str)

                print("-" * total_width)

        print("\n" + "=" * total_width)


_global_collector = PerformanceCollector()


def get_collector() -> PerformanceCollector:
    return _global_collector
