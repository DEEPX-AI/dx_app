# Performance Patterns — dx_app

> FPS optimization techniques, profiling, and benchmarking for dx_app inference applications.

## Overview

dx_app performance depends on the full pipeline: preprocessing → NPU inference →
postprocessing → visualization. This document covers practical optimization techniques
organized by impact.

---

## High-Impact Optimizations

### 1. Sync → Async Runner (2-3x Throughput)

The single highest-impact optimization. AsyncRunner overlaps CPU work with NPU inference.

| Runner | Behavior | Typical FPS (YOLOv8n, 640x640) |
|---|---|---|
| SyncRunner | Sequential: preprocess → infer → postprocess | 25-35 fps |
| AsyncRunner | Pipelined: preprocess N+1 while inferring N | 60-90 fps |

```python
# Before: sync
from common.runner import SyncRunner
runner = SyncRunner(factory)

# After: async (2-3x improvement)
from common.runner import AsyncRunner
runner = AsyncRunner(factory)
```

**When to use SyncRunner:** Single image inference, debugging, or when frame ordering
is critical and you cannot tolerate the 1-frame display latency of async.

### 2. Python → C++ Postprocess (5-10x Postprocess Speed)

For detection models, C++ postprocess bindings (`dx_postprocess`) are 5-10x faster
than equivalent Python implementations.

| Variant | Postprocess Time (YOLOv8n) | Total FPS Impact |
|---|---|---|
| Python postprocess | 8-15 ms | Baseline |
| C++ postprocess | 1-2 ms | +20-40% total FPS |

```python
# Before: Python postprocess
runner = SyncRunner(factory)

# After: C++ postprocess
from dx_postprocess import YoloV8PostProcess
from dx_engine import InferenceOption

def on_engine_init(runner):
    use_ort = InferenceOption().get_use_ort()
    runner._cpp_postprocessor = YoloV8PostProcess(
        runner.input_width, runner.input_height, 0.25, 0.45, use_ort
    )
    runner._cpp_convert_fn = convert_cpp_detections

runner = SyncRunner(factory, on_engine_init=on_engine_init)
```

### 3. Async + C++ Postprocess (Best Combination)

Combining async runner with C++ postprocess gives the maximum Python throughput:

```python
runner = AsyncRunner(factory, on_engine_init=on_engine_init)
# Achieves: 80-120 fps for YOLOv8n @ 640x640
```

---

## Medium-Impact Optimizations

### 4. Resolution Tradeoff

Lower input resolution dramatically increases FPS at the cost of accuracy:

| Input Size | Relative FPS | Accuracy Impact |
|---|---|---|
| 640x640 | 1.0x (baseline) | Full accuracy |
| 416x416 | 2.0-2.5x | Moderate loss |
| 320x320 | 3.0-4.0x | Significant loss for small objects |

Choose a smaller model variant (e.g., nanodet_m at 320x320) rather than resizing a
640x640 model, since the model was optimized for its native resolution.

### 5. Model Selection

Choosing the right model family has more impact than micro-optimizations:

| Model | Input | Params | Relative Speed |
|---|---|---|---|
| nanodet_m | 320x320 | 0.9M | 4x faster |
| yolov5n | 640x640 | 1.9M | 2x faster |
| yolov8n | 640x640 | 3.2M | 1.5x faster |
| yolov8s | 640x640 | 11.2M | 1.0x (baseline) |
| yolov8m | 640x640 | 25.9M | 0.5x |

### 6. Batch Tuning

Batch inference can improve throughput for offline processing:

```python
option = InferenceOption()
option.set_batch_size(4)  # Process 4 frames at once
engine = InferenceEngine("model.dxnn", option)

# Input shape changes from (1, H, W, 3) to (4, H, W, 3)
batch_input = np.stack([frame1, frame2, frame3, frame4])
outputs = engine.infer(batch_input)
```

**Note:** Batch size must match the compiled `.dxnn` model. Some models are compiled
with batch=1 only. Check with `dxrt-cli --info model.dxnn`.

---

## 7-Field Metrics

SyncRunner and AsyncRunner both report 7 performance metrics:

| Field | Unit | What It Measures |
|---|---|---|
| `preprocess_time` | ms | cv2.resize, normalization, tensor creation |
| `inference_time` | ms | NPU execution (InferenceEngine.infer) |
| `postprocess_time` | ms | NMS, decoding, coordinate scaling |
| `visualize_time` | ms | cv2 drawing, imshow |
| `total_time` | ms | End-to-end per-frame time |
| `fps` | fps | 1000 / total_time (smoothed) |
| `frame_count` | count | Total frames processed |

### Reading Metrics

```python
runner.run(args)

# After run completes:
m = runner.metrics
print(f"Preprocess:  {m.preprocess_time:.1f} ms")
print(f"Inference:   {m.inference_time:.1f} ms")
print(f"Postprocess: {m.postprocess_time:.1f} ms")
print(f"Visualize:   {m.visualize_time:.1f} ms")
print(f"Total:       {m.total_time:.1f} ms")
print(f"FPS:         {m.fps:.1f}")
print(f"Frames:      {m.frame_count}")
```

### Identifying Bottlenecks

| If this is highest... | Bottleneck is... | Fix |
|---|---|---|
| `preprocess_time` | CPU image processing | Reduce resolution, optimize resize |
| `inference_time` | NPU speed | Use smaller model or lower resolution |
| `postprocess_time` | CPU postprocessing | Switch to C++ postprocess variant |
| `visualize_time` | Display overhead | Use `--no-display` for benchmarks |

---

## Profiler Integration

### Per-Layer NPU Profiling

```python
option = InferenceOption()
option.set_profiling(True)
engine = InferenceEngine("model.dxnn", option)

outputs = engine.infer(input_tensor)

# Get profiling data
profile = engine.get_profile()
for layer in profile:
    print(f"{layer.name}: {layer.time_ms:.2f} ms ({layer.percentage:.1f}%)")
```

### Python cProfile

```bash
python -m cProfile -s cumulative yolov8n_sync.py \
    --model yolov8n.dxnn --input test.mp4 --max-frames 100 --no-display
```

### Benchmark Script Pattern

```python
#!/usr/bin/env python3
"""Benchmark a model with consistent methodology."""

import time
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

def benchmark(model_path, num_warmup=10, num_iterations=100):
    option = InferenceOption()
    engine = InferenceEngine(model_path, option)
    shape = engine.get_input_shape()

    # Create dummy input
    dummy = np.random.rand(*shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        engine.infer(dummy)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        engine.infer(dummy)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = np.mean(times)
    std = np.std(times)
    fps = 1000.0 / avg

    print(f"Inference: {avg:.2f} +/- {std:.2f} ms ({fps:.1f} fps)")
    return avg, std, fps
```

---

## Memory Optimization

### NPU Memory

| Concern | Guideline |
|---|---|
| Model size | Larger models use more NPU memory |
| Multi-model | Each engine allocates NPU context (~10-50 MB overhead) |
| Batch size | Higher batch = more memory per inference |
| Max simultaneous | 3-4 models on DX-M1 (depends on model sizes) |

### Host Memory

| Concern | Guideline |
|---|---|
| Input tensors | 640x640x3 float32 = ~4.7 MB per frame |
| Output tensors | Varies (8400x84 float32 = ~2.7 MB for YOLOv8) |
| OpenCV frames | 1080p BGR = ~6 MB per frame |
| Async pipeline | 2-3 frames buffered = 2-3x memory |

### Reducing Memory

```python
# Use uint8 input if model supports it (avoids float32 conversion)
input_tensor = resized_frame  # Keep as uint8 if model input is UINT8

# Release frames promptly
del previous_frame

# Limit async queue depth
runner = AsyncRunner(factory, max_queue_depth=2)
```

---

## Performance Comparison Table

| Configuration | FPS (YOLOv8n, 640x640) | Notes |
|---|---|---|
| Python sync | 25-35 | Baseline |
| Python sync + C++ postprocess | 35-50 | +40% |
| Python async | 60-90 | 2-3x sync |
| Python async + C++ postprocess | 80-120 | Best Python |
| C++ sync | 100-140 | Native overhead minimal |
| C++ async | 130-180 | Best overall |

**Note:** Actual FPS depends on NPU hardware (DX-M1 vs DX-M1A (discontinued)), host CPU, input
source latency, and display overhead. Always benchmark with `--no-display` for
accurate NPU-only measurements.
