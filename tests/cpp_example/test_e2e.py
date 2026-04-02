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
import sys
from pathlib import Path
from typing import List, Optional, Union

import pytest

from conftest import resolve_bin_dir
from performance_collector import get_collector, PerformanceMetrics

# -- common module ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.constants import (  # noqa: E402
    ASSETS_DIR,
    MODELS_DIR,
    MULTI_MODEL_EXECUTABLES,
    PROJECT_ROOT,
    SAMPLE_DIR,
)
from common.utils import (  # noqa: E402
    normalize_model_name as _normalize_model_to_exe,
    setup_environment,
)

BIN_DIR = resolve_bin_dir()
LIB_DIR = PROJECT_ROOT / "lib"

# Test data paths
TEST_IMAGE = SAMPLE_DIR / "img" / "sample_kitchen.jpg"
TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"

# Multi-model executables expanded with sync/async suffix for E2E test use.
# The base map lives in common.constants.MULTI_MODEL_EXECUTABLES.
_MULTI_MODEL_E2E = {}
for _base, _pairs in MULTI_MODEL_EXECUTABLES.items():
    _MULTI_MODEL_E2E[f"{_base}_sync"] = _pairs
    _MULTI_MODEL_E2E[f"{_base}_async"] = _pairs


def _resolve_multi_model_paths(exe_name: str) -> Optional[List[Path]]:
    """Return all model paths for a multi-model executable, or None if any is missing."""
    flag_model_pairs = _MULTI_MODEL_E2E.get(exe_name)
    if flag_model_pairs is None:
        return None
    resolved = []
    for _flag, fname in flag_model_pairs:
        p = MODELS_DIR / fname
        if not p.exists():
            return None
        resolved.append(p)
    return resolved


def discover_test_cases() -> List[tuple]:
    """Auto-discover (executable_name, model_path_or_list) pairs.

    Strategy
    --------
    1. For every ``.dxnn`` in ``assets/models/``, normalise the stem to
       derive a candidate executable name prefix, then check ``bin/`` for
       ``{prefix}_async`` and ``{prefix}_sync``.
    2. Add multi-model executables whose all required model files exist.

    Any model without a matching binary (or binary without a model) is
    skipped automatically — no manual maintenance required.
    """
    cases = []
    seen_exes: set = set()

    # --- model-driven discovery -------------------------------------------
    for model_path in sorted(MODELS_DIR.glob("*.dxnn")):
        prefix = _normalize_model_to_exe(model_path.stem)
        for suffix in ("_async", "_sync"):
            exe_name = f"{prefix}{suffix}"
            if exe_name in seen_exes:
                continue
            if (BIN_DIR / exe_name).exists():
                cases.append((exe_name, model_path))
                seen_exes.add(exe_name)

    # --- multi-model executables ------------------------------------------
    for exe_name in _MULTI_MODEL_E2E:
        if exe_name in seen_exes:
            continue
        model_paths = _resolve_multi_model_paths(exe_name)
        if model_paths and (BIN_DIR / exe_name).exists():
            cases.append((exe_name, model_paths))
            seen_exes.add(exe_name)

    return sorted(cases, key=lambda x: x[0])


def _with_async_sync_marks(cases: list) -> list:
    """Attach async/sync pytest markers so ``-m async_exec / sync_exec`` works."""
    params = []
    for exe_name, model_path in cases:
        marker = pytest.mark.async_exec if "_async" in exe_name else pytest.mark.sync_exec
        params.append(pytest.param(exe_name, model_path, id=exe_name, marks=marker))
    return params


DISCOVERED_CASES = discover_test_cases()
EXECUTABLE_PARAMS = _with_async_sync_marks(DISCOVERED_CASES)


# setup_environment() is now imported from common.utils


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
        'total_frames': r"Total Frames\s*:\s*(\d+)",
        'total_time': r"Total Time\s*:\s*([\d.]+)\s*s",
        'infer_inflight_avg': r"Infer Inflight Avg\s*:\s*([\d.]+)",
        'infer_inflight_max': r"Infer Inflight Max\s*:\s*(\d+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key in ["total_frames", "infer_inflight_max"]:
                fps_data[key] = int(value)
            else:
                fps_data[key] = float(value)
    
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
    """Return a model group name for performance metrics grouping."""
    return re.sub(r'_(async|sync)$', '', executable)


@pytest.mark.e2e
@pytest.mark.parametrize("executable,model_path", EXECUTABLE_PARAMS)
def test_image_inference_e2e(executable, model_path, bin_dir, loop_count):
    """
    Test image inference with --no-display option.

    Executables and models are discovered automatically from ``bin/`` and
    ``assets/models/`` — no manual mapping maintenance required.
    """
    executable_path = bin_dir / executable
    if os.name == "nt":
        executable_path = executable_path.with_suffix(".exe")

    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")

    if not TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE}")

    env = setup_environment()

    # Heavy face models need reduced loop counts to fit within timeout:
    #   - TTA models (~125s/frame): -l 2 to stay within 300s timeout
    #   - W6 face models (~3-20s/frame): -l 5 to stay within 100s timeout
    #   - Regular face models (yolov7_face, yolov7s_face): -l 5 to stay within 100s timeout
    exe_lower = executable.lower()
    if "tta" in exe_lower:
        effective_loop = min(loop_count, 2)
    elif "face" in exe_lower:
        effective_loop = min(loop_count, 5)
    else:
        effective_loop = loop_count

    if isinstance(model_path, list):
        # Multi-model executable (e.g. yolov7_x_deeplabv3)
        flag_model_pairs = _MULTI_MODEL_E2E[executable]
        cmd = [str(executable_path)]
        for (flag, _fname), mpath in zip(flag_model_pairs, model_path):
            cmd += [flag, str(mpath)]
        cmd += ["-i", str(TEST_IMAGE), "--no-display", "-l", str(effective_loop)]
    else:
        cmd = [
            str(executable_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", str(effective_loop),
        ]

    # TTA models are significantly heavier (multiple forward passes per image)
    # and require a longer timeout, especially on aarch64 platforms.
    image_timeout = 300 if "tta" in exe_lower else 100

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=image_timeout,  # 100 seconds default, 300 for TTA models
            env=env,
            cwd=PROJECT_ROOT,
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
            # FPS should be non-negative (very slow models may report 0.0 FPS due to rounding)
            assert 0 <= fps, (
                f"{executable} reported negative FPS: {fps}\n"
                f"Output: {output[:500]}"
            )
            if fps >= 10000:
                print(f"\n[WARN] {executable} image inference: {fps:.2f} FPS (unusually high — likely no NPU or simulator mode)")
            else:
                print(f"\n{executable} image inference: {fps:.2f} FPS")
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"{executable} image inference timed out after {image_timeout} seconds")
    except Exception as e:
        pytest.fail(f"{executable} image inference raised exception: {e}")


def _build_video_cmd(executable_path, executable, model_path):
    """Build command list for video inference test."""
    if isinstance(model_path, list):
        flag_model_pairs = _MULTI_MODEL_E2E[executable]
        cmd = [str(executable_path)]
        for (flag, _fname), mpath in zip(flag_model_pairs, model_path):
            cmd += [flag, str(mpath)]
        cmd += ["-v", str(TEST_VIDEO), "--no-display"]
    else:
        cmd = [
            str(executable_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
        ]
    return cmd


@pytest.mark.e2e
@pytest.mark.parametrize("executable,model_path", EXECUTABLE_PARAMS)
def test_video_inference_e2e(executable, model_path, bin_dir):
    """
    Test video inference with --no-display option.

    Executables and models are discovered automatically from ``bin/`` and
    ``assets/models/``.
    """
    executable_path = bin_dir / executable
    if os.name == "nt":
        executable_path = executable_path.with_suffix(".exe")

    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")

    # Face models are too slow for full video processing in CI
    # (e.g. W6 face ~3-20s/frame, TTA ~125s/frame on aarch64).
    # Image tests with reduced loop counts already verify correctness.
    if "face" in executable.lower():
        pytest.skip(f"{executable}: face model too slow for video test in CI")

    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    env = setup_environment()
    cmd = _build_video_cmd(executable_path, executable, model_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes  timeout for video
            env=env,
            cwd=PROJECT_ROOT,
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
            # FPS should be positive
            assert fps > 0, (
                f"{executable} reported non-positive FPS: {fps}\n"
                f"Output: {output[:500]}"
            )
            if fps >= 10000:
                print(f"\n[WARN] {executable} video inference: {fps:.2f} FPS (unusually high — likely no NPU or simulator mode)")
            else:
                print(f"\n{executable} video inference: {fps:.2f} FPS")
            
            # Collect performance metrics for report
            collector = get_collector()
            model_group = get_model_group(executable)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                variant=executable,
                e2e_fps=detailed_fps.get('e2e', fps),
                read_fps=detailed_fps.get('read'),
                preprocess_fps=detailed_fps.get('preprocess'),
                inference_fps=detailed_fps.get('inference'),
                postprocess_fps=detailed_fps.get('postprocess'),
                total_frames=detailed_fps.get('total_frames'),
                total_time=detailed_fps.get('total_time'),
                infer_inflight_avg=detailed_fps.get('infer_inflight_avg'),
                infer_inflight_max=detailed_fps.get('infer_inflight_max')
            )
            
            collector.add_metrics(model_group, executable, metrics)
            
            # Store model info (only once per group)
            if model_group not in collector.model_info:
                collector.set_model_info(
                    model_group,
                    str(model_path if isinstance(model_path, Path) else model_path[0]),
                    str(TEST_VIDEO),
                    detailed_fps.get('total_frames'),
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
