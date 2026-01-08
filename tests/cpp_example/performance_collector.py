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
    executable: str
    e2e_fps: Optional[float] = None
    read_fps: Optional[float] = None
    preprocess_fps: Optional[float] = None
    inference_fps: Optional[float] = None
    postprocess_fps: Optional[float] = None
    
    def get_bottleneck(self) -> Optional[str]:
        """Identify the bottleneck (slowest component)"""
        fps_map = {
            'E2E': self.e2e_fps,
            'Read': self.read_fps,
            'Preprocess': self.preprocess_fps,
            'Inference': self.inference_fps,
            'Postprocess': self.postprocess_fps,
        }
        
        # Filter out None values
        valid_fps = {k: v for k, v in fps_map.items() if v is not None}
        
        if not valid_fps:
            return None
        
        # Bottleneck is the component with lowest FPS
        return min(valid_fps.items(), key=lambda x: x[1])[0]


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
        
    def add_metrics(self, model_group: str, executable: str, metrics: PerformanceMetrics):
        """Add performance metrics for an executable"""
        self.metrics[model_group][executable] = metrics
    
    def set_model_info(self, model_group: str, model_path: str, video_path: str, frames: int):
        """Set model information for a group"""
        self.model_info[model_group] = {
            'model_path': model_path,
            'video_path': video_path,
            'frames': frames,
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
        lines.append("=" * 195)
        lines.append(" E2E Performance Report - C++ Executables")
        lines.append("=" * 195)
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
                lines.append(f" Video: {info['video_path']} ({info['frames']} frames)")
                lines.append("")
            
            # Table header
            lines.append("-" * 195)
            lines.append(f"{'Variant':<42} | {'E2E [FPS]':<12} | {'Read [FPS]':<12} | {'Preprocess [FPS]':<17} | {'Inference [FPS]':<16} | {'Postprocess [FPS]'}")
            lines.append("-" * 195)
            
            # Sort executables
            sorted_execs = sorted(self.metrics[group_key].keys())
            
            for executable in sorted_execs:
                metrics = self.metrics[group_key][executable]
                bottleneck = metrics.get_bottleneck()
                
                def format_fps(fps: Optional[float], component: str) -> str:
                    if fps is None:
                        return "-"
                    fps_str = f"{fps:.1f}"
                    if component == bottleneck:
                        fps_str += " *"
                    return fps_str
                
                e2e = format_fps(metrics.e2e_fps, 'E2E')
                read = format_fps(metrics.read_fps, 'Read')
                preprocess = format_fps(metrics.preprocess_fps, 'Preprocess')
                inference = format_fps(metrics.inference_fps, 'Inference')
                postprocess = format_fps(metrics.postprocess_fps, 'Postprocess')
                
                lines.append(f"{executable:<42} | {e2e:<12} | {read:<12} | {preprocess:<17} | {inference:<16} | {postprocess}")
            
            lines.append("-" * 195)
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
            # Default to tests/cpp_example/performance_report_YYYYMMDD_HHMMSS.csv
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(__file__).parent / f"performance_report_{timestamp}.csv"
        
        csv_data = []
        
        # Prepare CSV rows
        for group_key in sorted(self.metrics.keys()):
            if group_key not in self.metrics or not self.metrics[group_key]:
                continue
            
            # Get model info
            model_path = ""
            video_path = ""
            frames = ""
            if group_key in self.model_info:
                info = self.model_info[group_key]
                model_path = info['model_path']
                video_path = info['video_path']
                frames = info['frames']
            
            # Add rows for each executable
            for executable in sorted(self.metrics[group_key].keys()):
                metrics = self.metrics[group_key][executable]
                bottleneck = metrics.get_bottleneck() or ""
                
                csv_data.append({
                    'Model Group': group_key,
                    'Executable': executable,
                    'Model Path': model_path,
                    'Video Path': video_path,
                    'Frames': frames,
                    'E2E FPS': f"{metrics.e2e_fps:.1f}" if metrics.e2e_fps is not None else "",
                    'Read FPS': f"{metrics.read_fps:.1f}" if metrics.read_fps is not None else "",
                    'Preprocess FPS': f"{metrics.preprocess_fps:.1f}" if metrics.preprocess_fps is not None else "",
                    'Inference FPS': f"{metrics.inference_fps:.1f}" if metrics.inference_fps is not None else "",
                    'Postprocess FPS': f"{metrics.postprocess_fps:.1f}" if metrics.postprocess_fps is not None else "",
                    'Bottleneck': bottleneck,
                })
        
        # Write CSV file
        if csv_data:
            fieldnames = [
                'Model Group', 'Executable', 'Model Path', 'Video Path', 'Frames',
                'E2E FPS', 'Read FPS', 'Preprocess FPS', 'Inference FPS', 
                'Postprocess FPS', 'Bottleneck'
            ]
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        return output_path


def get_collector() -> PerformanceCollector:
    """Get the global performance collector instance"""
    return PerformanceCollector()
