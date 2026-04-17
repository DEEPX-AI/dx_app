"""
Performance data collector and reporter for C++ example E2E tests
"""
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test variant"""
    variant: str
    e2e_fps: float
    read_fps: float
    preprocess_fps: float
    inference_fps: float

    postprocess_fps: float = 0.0

    total_frames: int = 0
    total_time: float = 0.0
    infer_inflight_avg: Optional[float] = None
    infer_inflight_max: Optional[int] = None


class PerformanceCollector:
    """Singleton collector for performance data across tests"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.metrics: Dict[str, Dict[str, PerformanceMetrics]] = defaultdict(dict)
        self.model_info: Dict[str, Dict] = {}

    def add_metrics(self, model_group: str, variant: str, metrics: PerformanceMetrics):
        """Add performance metrics for an variant under a model group"""
        self.metrics[model_group][variant] = metrics

    def set_model_info(self, model_group: str, model_path: str, video_path: str, frames: int, task: str = ""):
        """Set model information for a group"""
        self.model_info[model_group] = {
            'model_path': model_path,
            'video_path': video_path,
            'total_frames': frames,
            'task': task,
        }

    def clear(self):
        """Clear all collected data"""
        self.metrics.clear()
        self.model_info.clear()

    def generate_report(self) -> str:
        """Generate performance report in the same format as Python example"""
        if not self.metrics:
            return ""

        lines = []
        lines.append("=" * 235)
        lines.append(" E2E Performance Report - C++ Executables")
        lines.append("=" * 235)
        lines.append("")

        # Group executables by model type
        model_groups = {
            'deeplabv3': 'Semantic Segmentation - DEEPLABV3',
            'scrfd': 'Object Detection - SCRFD',
            'scrfd_ppu': 'Ppu - SCRFD',
            'yolov5': 'Object Detection - YOLOV5',
            'yolov5_ppu': 'Ppu - YOLOV5_PPU',
            'yolov5face': 'Object Detection - YOLOV5FACE',
            'yolov5pose': 'Object Detection - YOLOV5POSE',
            'yolov5pose_ppu': 'Ppu - YOLOV5POSE_PPU',
            'yolov7': 'Object Detection - YOLOV7',
            'yolov7_ppu': 'Ppu - YOLOV7_PPU',
            'yolov8': 'Object Detection - YOLOV8',
            'yolov8seg': 'Instance Segmentation - YOLOV8SEG',
            'yolov9': 'Object Detection - YOLOV9',
            'yolov10': 'Object Detection - YOLOV10',
            'yolov11': 'Object Detection - YOLOV11',
            'yolov12': 'Object Detection - YOLOV12',
            'yolox': 'Object Detection - YOLOX',
        }

        for group_key in sorted(self.metrics.keys()):
            if group_key not in self.metrics or not self.metrics[group_key]:
                continue

            group_title = model_groups.get(group_key, group_key.upper())
            lines.append(f" {group_title}")
            lines.append("")

            if group_key in self.model_info:
                info = self.model_info[group_key]
                lines.append(f" Model: {info['model_path']}")
                lines.append(f" Video: {info['video_path']} ({info['total_frames']} frames)")
                lines.append("")

            # Table header
            lines.append("-" * 235)
            lines.append(f"{'Variant':<42} | {'E2E [FPS]':<12} | {'Read [FPS]':<12} | {'Preprocess [FPS]':<17} | {'Inference [FPS]':<16} | {'Postprocess [FPS]':<18} | {'Inflight Avg':<19} | {'Inflight Max'}")
            lines.append("-" * 235)

            # Sort variants
            sorted_variants = sorted(self.metrics[group_key].keys())

            for variant in sorted_variants:
                metrics = self.metrics[group_key][variant]
                is_async = metrics.infer_inflight_avg is not None

                pf = {
                    'Read': metrics.read_fps,
                    'Preprocess': metrics.preprocess_fps,
                    'Inference': metrics.inference_fps,
                    'Postprocess': metrics.postprocess_fps,
                }
                valid_pf = {k: v for k, v in pf.items() if v is not None}
                bn = min(valid_pf, key=valid_pf.get) if valid_pf else None

                def format_fps(fps: Optional[float], component: str, _is_async=is_async, _bn=bn) -> str:
                    if fps is None:
                        return "-"
                    fps_str = f"{fps:.1f}"
                    if _is_async and component == _bn:
                        fps_str += " *"
                    return fps_str

                e2e = f"{metrics.e2e_fps:.1f}" if metrics.e2e_fps is not None else "-"
                read = format_fps(metrics.read_fps, 'Read')
                preprocess = format_fps(metrics.preprocess_fps, 'Preprocess')
                inference = format_fps(metrics.inference_fps, 'Inference')
                postprocess = format_fps(metrics.postprocess_fps, 'Postprocess')

                inflight_avg = f"{metrics.infer_inflight_avg:.1f}" if metrics.infer_inflight_avg is not None else "-"
                inflight_max = str(metrics.infer_inflight_max) if metrics.infer_inflight_max is not None else "-"

                lines.append(f"{variant:<42} | {e2e:<12} | {read:<12} | {preprocess:<17} | {inference:<16} | {postprocess:<18} | {inflight_avg:<19} | {inflight_max}")

            lines.append("-" * 235)
            lines.append("")

        return "\n".join(lines)

    def generate_csv(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate CSV file with performance data

        Args:
            output_path: Path to output CSV file. If None, generates in tests/cpp_example/

        Returns:
            Path to the generated CSV file
        """
        if output_path is None:
            # Default to tests/cpp_example/performance_reports/performance_report_YYYYMMDD_HHMMSS.csv
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(__file__).parent / "performance_reports" / f"performance_report_{timestamp}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        csv_data = []

        # Prepare CSV rows
        for group_key in sorted(self.metrics.keys()):
            if group_key not in self.metrics or not self.metrics[group_key]:
                continue

            # Get model info
            model_path = ""
            video_path = ""
            total_frames = 0
            task = ""
            if group_key in self.model_info:
                info = self.model_info[group_key]
                model_path = info['model_path']
                video_path = info['video_path']
                total_frames = info['total_frames']
                task = info.get('task', '')

            # Add rows for each variant
            for variant in sorted(self.metrics[group_key].keys()):
                metrics = self.metrics[group_key][variant]
                is_async = metrics.infer_inflight_avg is not None
                pf = {
                    'Read': metrics.read_fps,
                    'Preprocess': metrics.preprocess_fps,
                    'Inference': metrics.inference_fps,
                    'Postprocess': metrics.postprocess_fps,
                }
                valid_pf = {k: v for k, v in pf.items() if v is not None}
                bn = min(valid_pf, key=valid_pf.get) if valid_pf else None

                def _fmt(fps, component, _is_async=is_async, _bn=bn):
                    if fps is None:
                        return ""
                    s = f"{fps:.1f}"
                    if _is_async and component == _bn:
                        s += " *"
                    return s

                csv_data.append({
                    'Task': task,
                    'Model': group_key,
                    'Variant': variant,
                    'Model Path': model_path,
                    'Video Path': video_path,
                    'Total Frames': total_frames,
                    'Total Time (s)': f"{metrics.total_time:.1f}",
                    'E2E FPS': f"{metrics.e2e_fps:.1f}" if metrics.e2e_fps is not None else "",
                    'Read FPS': _fmt(metrics.read_fps, 'Read'),
                    'Preprocess FPS': _fmt(metrics.preprocess_fps, 'Preprocess'),
                    'Inference FPS': _fmt(metrics.inference_fps, 'Inference'),
                    'Postprocess FPS': _fmt(metrics.postprocess_fps, 'Postprocess'),
                    'Inflight Avg': f"{metrics.infer_inflight_avg:.1f}" if metrics.infer_inflight_avg is not None else "",
                    'Inflight Max': f"{metrics.infer_inflight_max}" if metrics.infer_inflight_max is not None else "",
                })

        # Write CSV file
        if csv_data:
            fieldnames = [
                'Task', 'Model', 'Variant', 'Model Path', 'Video Path', 'Total Frames', 'Total Time (s)',
                'E2E FPS', 'Read FPS', 'Preprocess FPS', 'Inference FPS',
                'Postprocess FPS', 'Inflight Avg', 'Inflight Max',
            ]

            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

        return output_path


_collector = PerformanceCollector()


def get_collector() -> PerformanceCollector:
    """Get the global performance collector instance"""
    return _collector
