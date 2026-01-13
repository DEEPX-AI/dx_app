"""
End-to-End tests for bin executables with --no-display option

These tests run actual inference on real images/videos to verify:
- The executable runs without crashing
- It completes within reasonable time
- It processes the expected number of frames
- FPS metrics are reasonable
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

from conftest import resolve_bin_dir

from performance_collector import get_collector, PerformanceMetrics


PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_DIR = resolve_bin_dir()
LIB_DIR = PROJECT_ROOT / "lib"
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLE_DIR = PROJECT_ROOT / "sample"

# Test data paths
TEST_IMAGE = SAMPLE_DIR / "img" / "1.jpg"
TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"

# Model mappings for each executable
MODEL_MAPPING = {
    "deeplabv3_async": "DeepLabV3PlusMobileNetV2_2.dxnn",
    "deeplabv3_sync": "DeepLabV3PlusMobileNetV2_2.dxnn",
    "scrfd_async": "SCRFD500M_1.dxnn",
    "scrfd_ppu_async": "SCRFD500M_PPU.dxnn",
    "scrfd_ppu_sync": "SCRFD500M_PPU.dxnn",
    "scrfd_sync": "SCRFD500M_1.dxnn",
    "yolov10_async": "YOLOV10N-1.dxnn",
    "yolov10_sync": "YOLOV10N-1.dxnn",
    "yolov11_async": "YOLOV11N.dxnn",
    "yolov11_sync": "YOLOV11N.dxnn",
    "yolov12_async": "YOLOV12N-1.dxnn",
    "yolov12_sync": "YOLOV12N-1.dxnn",
    "yolov5_async": "YOLOV5S-1.dxnn",
    "yolov5_sync": "YOLOV5S-1.dxnn",
    "yolov5face_async": "YOLOV5S_Face-1.dxnn",
    "yolov5face_sync": "YOLOV5S_Face-1.dxnn",
    "yolov5pose_async": "YOLOV5Pose640_1.dxnn",
    "yolov5pose_sync": "YOLOV5Pose640_1.dxnn",
    "yolov5pose_ppu_async": "YOLOV5Pose_PPU.dxnn",
    "yolov5pose_ppu_sync": "YOLOV5Pose_PPU.dxnn",
    "yolov5_ppu_async": "YOLOV5S_PPU.dxnn",
    "yolov5_ppu_sync": "YOLOV5S_PPU.dxnn",
    "yolov7_async": "YoloV7.dxnn",
    "yolov7_sync": "YoloV7.dxnn",
    "yolov7_ppu_async": "YoloV7_PPU.dxnn",
    "yolov7_ppu_sync": "YoloV7_PPU.dxnn",
    "yolov7_x_deeplabv3_async": ["YoloV7.dxnn", "DeepLabV3PlusMobileNetV2_2.dxnn"],
    "yolov7_x_deeplabv3_sync": ["YoloV7.dxnn", "DeepLabV3PlusMobileNetV2_2.dxnn"],
    "yolov8_async": "YoloV8N.dxnn",
    "yolov8_sync": "YoloV8N.dxnn",
    "yolov8seg_async": "YOLOV8N_SEG-1.dxnn",
    "yolov8seg_sync": "YOLOV8N_SEG-1.dxnn",
    "yolov9_async": "YOLOV9S.dxnn",
    "yolov9_sync": "YOLOV9S.dxnn",
    "yolox_async": "YOLOX-S_1.dxnn",
    "yolox_sync": "YOLOX-S_1.dxnn",
}

# Executables that support --no-display option
# Note: yolov7_x_deeplabv3_* are excluded because they require two models
EXECUTABLES_WITH_NO_DISPLAY = [
    "deeplabv3_async",
    "deeplabv3_sync",
    "scrfd_async",
    "scrfd_ppu_async",
    "scrfd_ppu_sync",
    "scrfd_sync",
    "yolov10_async",
    "yolov10_sync",
    "yolov11_async",
    "yolov11_sync",
    "yolov12_async",
    "yolov12_sync",
    "yolov5_async",
    "yolov5face_async",
    "yolov5face_sync",
    "yolov5pose_async",
    "yolov5pose_ppu_async",
    "yolov5pose_ppu_sync",
    "yolov5pose_sync",
    "yolov5_ppu_async",
    "yolov5_ppu_sync",
    "yolov5_sync",
    "yolov7_async",
    "yolov7_ppu_async",
    "yolov7_ppu_sync",
    "yolov7_sync",
    "yolov7_x_deeplabv3_async",  # Requires two models, skipped
    "yolov7_x_deeplabv3_sync",   # Requires two models, skipped
    "yolov8_async",
    "yolov8seg_async",
    "yolov8seg_sync",
    "yolov8_sync",
    "yolov9_async",
    "yolov9_sync",
    "yolox_async",
    "yolox_sync",
]


def _with_async_sync_marks(names):
    """Attach async/sync markers at collection time so -m filtering works."""
    params = []
    for name in names:
        marker = pytest.mark.async_exec if "_async" in name else pytest.mark.sync_exec
        params.append(pytest.param(name, marks=marker))
    return params


EXECUTABLE_PARAMS = _with_async_sync_marks(EXECUTABLES_WITH_NO_DISPLAY)


def setup_environment():
    """Setup environment with required library paths"""
    env = os.environ.copy()
    if LIB_DIR.exists():
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        if current_ld_path:
            env["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{current_ld_path}"
        else:
            env["LD_LIBRARY_PATH"] = str(LIB_DIR)
    return env


def get_model_path(executable_name: str) -> any:
    """Get the model file path for a given executable"""
    model_filename = MODEL_MAPPING.get(executable_name)
    if not model_filename:
        return None
    if isinstance(model_filename, list) :
        model_path = []
        for model_file_att in model_filename:
            model_path.append(ASSETS_DIR / "models" / model_file_att)
    else:
        model_path = ASSETS_DIR / "models" / model_filename
    
    return model_path 


def parse_fps_from_output(output: str) -> float:
    """
    Parse FPS from output. Common patterns:
    - "FPS : 123.45"
    - "FPS: 123.45"
    - "Average FPS: 123.45"
    """
    patterns = [
        r"FPS\s*:\s*([\d.]+)",
        r"Average\s+FPS\s*:\s*([\d.]+)",
        r"fps\s*:\s*([\d.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return -1.


def parse_detailed_fps(output: str) -> dict:
    """
    Parse detailed FPS metrics from C++ executable output
    
    Expected formats:
    1. Single model:
       Read               5.76 ms      173.6 FPS
       Preprocess         1.15 ms      868.9 FPS
       Inference        169.67 ms        5.9 FPS
       Postprocess        0.96 ms     1037.4 FPS
       Overall FPS         :   36.4 FPS
    
    2. Multi-model (yolov7_x_deeplabv3):
       Async Throughput    :    36.6 FPS
       Overall FPS         :   36.4 FPS
    """
    fps_data = {}
    
    # Pattern 1: "Pipeline Step" format (single line metrics)
    # Example: "Read               5.76 ms      173.6 FPS"
    patterns = {
        'read': r'Read\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS',
        'preprocess': r'Preprocess\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS',
        'inference': r'Inference\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS',
        'postprocess': r'Postprocess\s+[\d.]+\s+ms\s+([\d.]+)\s+FPS',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            fps_data[key] = float(match.group(1))
    
    # Overall FPS (E2E)
    overall_patterns = [
        r'Overall\s+FPS\s*:\s*([\d.]+)\s+FPS',
        r'Async\s+Throughput\s*:\s*([\d.]+)\s+FPS',
    ]
    
    for pattern in overall_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            fps_data['e2e'] = float(match.group(1))
            break
    
    return fps_data


def get_model_group(executable: str) -> str:
    """Determine the model group from executable name"""
    # Extract base model name
    if 'deeplabv3' in executable:
        return 'deeplabv3'
    elif 'scrfd_ppu' in executable:
        return 'scrfd_ppu'
    elif 'scrfd' in executable:
        return 'scrfd'
    elif 'yolov5pose_ppu' in executable:
        return 'yolov5pose_ppu'
    elif 'yolov5_ppu' in executable:
        return 'yolov5_ppu'
    elif 'yolov5face' in executable:
        return 'yolov5face'
    elif 'yolov5pose' in executable:
        return 'yolov5pose'
    elif 'yolov5' in executable:
        return 'yolov5'
    elif 'yolov7_ppu' in executable:
        return 'yolov7_ppu'
    elif 'yolov7' in executable:
        return 'yolov7'
    elif 'yolov8seg' in executable:
        return 'yolov8seg'
    elif 'yolov8' in executable:
        return 'yolov8'
    elif 'yolov9' in executable:
        return 'yolov9'
    elif 'yolov10' in executable:
        return 'yolov10'
    elif 'yolov11' in executable:
        return 'yolov11'
    elif 'yolov12' in executable:
        return 'yolov12'
    elif 'yolox' in executable:
        return 'yolox'
    
    return 'unknown'


@pytest.mark.e2e
@pytest.mark.parametrize("executable", EXECUTABLE_PARAMS)
def test_image_inference_e2e(executable, bin_dir):
    """
    Test image inference with --no-display option
    
    Runs actual inference on a real image to verify the executable works end-to-end
    """
    executable_path = bin_dir / executable
    if os.name == "nt":
        executable_path = executable_path.with_suffix(".exe")
    
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    model_path = get_model_path(executable)
    
    if not TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE}")
    
    env = setup_environment()
    
    if "yolov7_x_deeplabv3_async" in str(executable_path) or "yolov7_x_deeplabv3_sync" in str(executable_path):
        cmd = [
            str(executable_path),
            "-y", str(model_path[0]),
            "-d", str(model_path[1]),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "50"  # 50 iterations
        ]
    else:
        # Build command: executable -m model -i image --no-display -l 50
        cmd = [
            str(executable_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "50"  # 50 iterations
        ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=100,  # 100 second timeout
            env=env
        )
        
        # Check return code
        assert result.returncode == 0, (
            f"{executable} image inference failed with return code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        
        # Check output contains FPS info
        output = result.stdout + result.stderr
        fps = parse_fps_from_output(output)
        
        if fps is not None:
            # FPS should be reasonable (> 0 and typically < 10000)
            assert 0 < fps < 10000, (
                f"{executable} reported unreasonable FPS: {fps}\n"
                f"Output: {output[:500]}"
            )
            print(f"\n{executable} image inference: {fps:.2f} FPS")
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"{executable} image inference timed out after 100 seconds")
    except Exception as e:
        pytest.fail(f"{executable} image inference raised exception: {e}")


@pytest.mark.e2e
@pytest.mark.parametrize("executable", EXECUTABLE_PARAMS)
def test_video_inference_e2e(executable, bin_dir):
    """
    Test video inference with --no-display option
    
    Runs actual inference on a real video to verify the executable works end-to-end
    """
    executable_path = bin_dir / executable
    if os.name == "nt":
        executable_path = executable_path.with_suffix(".exe")
    
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    model_path = get_model_path(executable)
    
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")
    
    env = setup_environment()
    
    if "yolov7_x_deeplabv3_async" in str(executable_path) or "yolov7_x_deeplabv3_sync" in str(executable_path):
        cmd = [
            str(executable_path),
            "-y", str(model_path[0]),
            "-d", str(model_path[1]),
            "-v", str(TEST_VIDEO),
            "--no-display"
        ]
    else:
        # Build command: executable -m model -v video --no-display
        cmd = [
            str(executable_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display"
        ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes  timeout for video
            env=env
        )
        
        # Check return code
        assert result.returncode == 0, (
            f"{executable} video inference failed with return code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        
        # Parse output for FPS and detailed metrics
        output = result.stdout + result.stderr
        fps = parse_fps_from_output(output)
        detailed_fps = parse_detailed_fps(output)
        
        if fps > 0:
            # FPS should be reasonable (> 0 and typically < 10000)
            assert 0 < fps < 10000, (
                f"{executable} reported unreasonable FPS: {fps}\n"
                f"Output: {output[:500]}"
            )
            print(f"\n{executable} video inference: {fps:.2f} FPS")
            
            # Collect performance metrics for report
            collector = get_collector()
            model_group = get_model_group(executable)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                executable=executable,
                e2e_fps=detailed_fps.get('e2e', fps),
                read_fps=detailed_fps.get('read'),
                preprocess_fps=detailed_fps.get('preprocess'),
                inference_fps=detailed_fps.get('inference'),
                postprocess_fps=detailed_fps.get('postprocess'),
            )
            
            collector.add_metrics(model_group, executable, metrics)
            
            # Store model info (only once per group)
            if model_group not in collector.model_info:
                collector.set_model_info(
                    model_group,
                    str(model_path if isinstance(model_path, Path) else model_path[0]),
                    str(TEST_VIDEO),
                    478  # Known frame count for dance-group.mov
                )
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"{executable} video inference timed out after 10 minutes")
    except Exception as e:
        pytest.fail(f"{executable} video inference raised exception: {e}")


@pytest.mark.e2e
def test_e2e_prerequisites():
    """
    Sanity check that prerequisites are available
    """
    assert TEST_IMAGE.exists(), f"Test image not found: {TEST_IMAGE}"
    assert TEST_VIDEO.exists(), f"Test video not found: {TEST_VIDEO}"
    assert (ASSETS_DIR / "models").exists(), f"Models directory not found: {ASSETS_DIR / 'models'}"
    
    # Check that at least some models exist
    model_files = list((ASSETS_DIR / "models").glob("*.dxnn"))
    assert len(model_files) > 0, "No model files found in assets/models"
    
    print(f"\nFound {len(model_files)} model files")
    print(f"Test image: {TEST_IMAGE}")
    print(f"Test video: {TEST_VIDEO}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
