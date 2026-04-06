# Performance Patterns — dx_app

> FPS optimization techniques, profiling, and benchmarking for dx_app inference applications.
> All API references point to source files — verify there before using any call.

## ⚠️ Anti-Fabrication Notice

This file was audited and corrected. Previous versions contained fabricated API calls
that do not exist in the DEEPX SDK. If you are uncertain about an API, check the source
files listed below — do NOT invent methods. When in doubt, say "I need to verify this
against the source" rather than guessing.

---

## Source Files

| File | What it defines |
|---|---|
| `dx_rt/python_package/src/dx_engine/inference_engine.py` | `InferenceEngine`: `run()`, `run_async()`, `wait()`, `run_benchmark()`, `get_latency()`, `get_npu_inference_time()`, `get_input_tensors_info()` |
| `dx_rt/python_package/src/dx_engine/inference_option.py` | `InferenceOption`: `use_ort`, `devices`, `bound_option`, `buffer_count` |
| `dx_rt/python_package/src/dx_engine/configuration.py` | `Configuration`: profiler enable, thread config |
| `src/python_example/common/runner/sync_runner.py` | `SyncRunner`: 7 timing sums (`sum_read`, `sum_preprocess`, etc.) |
| `src/python_example/common/runner/async_runner.py` | `AsyncRunner`: 5+1 threads, 17 metrics fields, SafeQueue pipeline |

Always verify against these files before using any API call in this document.

---

## High-Impact Optimizations

### 1. Sync → Async Runner (2-3x Throughput)

The single highest-impact optimization. AsyncRunner uses a **5-worker-thread + main
thread** pipeline (6 threads total) to overlap CPU work with NPU inference.

| Runner | Behavior | Typical FPS (YOLOv8n, 640x640) |
|---|---|---|
| SyncRunner | Sequential: read → preprocess → infer → postprocess → render | 25-35 fps |
| AsyncRunner | 5+1 thread pipeline with SafeQueue (maxsize=4) between stages | 60-90 fps |

**AsyncRunner thread pipeline:**
```
read → reqid → output → render → display
(each stage is a worker thread; main thread coordinates)
```

```python
# Before: sync
from common.runner import SyncRunner
runner = SyncRunner(factory)

# After: async (2-3x improvement)
from common.runner import AsyncRunner
runner = AsyncRunner(factory)
```

**When to use SyncRunner:** Single image inference, debugging, or when frame ordering
is critical and you cannot tolerate the multi-frame display latency of async.

### 2. Python → C++ Postprocess (5-10x Postprocess Speed)

For detection models, C++ postprocess bindings (`dx_postprocess`) are 5-10x faster
than equivalent Python implementations. The C++ binding method is `postprocess()`
(not `process()`).

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

---

## Profiling and Benchmarking

### Built-in Benchmark (Preferred)

Use `engine.run_benchmark()` for consistent FPS measurement:

```python
from dx_engine import InferenceEngine, InferenceOption

option = InferenceOption()
engine = InferenceEngine("model.dxnn", option)

# Returns average FPS over num_loops iterations
fps = engine.run_benchmark(num_loops=100)
print(f"Average FPS: {fps:.1f}")
```

### Manual Timing

```python
# Single-inference latency (microseconds)
latency_us = engine.get_latency()
print(f"Total latency: {latency_us} us ({latency_us / 1000:.2f} ms)")

# NPU-only inference time (microseconds)
npu_time_us = engine.get_npu_inference_time()
print(f"NPU time: {npu_time_us} us ({npu_time_us / 1000:.2f} ms)")
```

### Getting Input Tensor Info

```python
# Correct way — returns List[Dict] with 'shape' key
tensors_info = engine.get_input_tensors_info()
for info in tensors_info:
    print(f"Input shape: {info['shape']}")

# ⚠️ WRONG: engine.get_input_shape() does NOT exist
```

### NPU Layer Profiling (via Configuration)

```python
from dx_engine import Configuration

config = Configuration()
config.set_enable(Configuration.ITEM.PROFILER, True)
config.set_enable(Configuration.ITEM.SHOW_PROFILE, True)

# ⚠️ WRONG: option.set_profiling(True) does NOT exist
# ⚠️ WRONG: engine.get_profile() does NOT exist
```

### Python cProfile

```bash
python -m cProfile -s cumulative yolov8n_sync.py \
    --model yolov8n.dxnn --input test.mp4 --max-frames 100 --no-display
```

### Benchmark Script Pattern (Corrected)

```python
#!/usr/bin/env python3
"""Benchmark a model with consistent methodology."""

import time
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

def benchmark(model_path, num_warmup=10, num_iterations=100):
    option = InferenceOption()
    engine = InferenceEngine(model_path, option)

    # Get input shape via correct API
    tensors_info = engine.get_input_tensors_info()
    shape = tensors_info[0]['shape']

    # Create dummy input
    dummy = np.random.rand(*shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        engine.run(dummy)

    # Benchmark — preferred: use built-in
    fps = engine.run_benchmark(num_loops=num_iterations, input_data=dummy)
    print(f"Built-in benchmark: {fps:.1f} fps")

    # Or manual timing
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        engine.run(dummy)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = np.mean(times)
    std = np.std(times)
    manual_fps = 1000.0 / avg

    print(f"Manual: {avg:.2f} +/- {std:.2f} ms ({manual_fps:.1f} fps)")
    return avg, std, manual_fps
```

---

## Metrics

### SyncRunner — 7 Timing Sums

SyncRunner accumulates timing into **sum fields** (not per-frame attribute accessors):

| Field | Type | What It Accumulates |
|---|---|---|
| `sum_read` | float | Total time reading/decoding frames |
| `sum_preprocess` | float | Total time for resize, normalization, tensor creation |
| `sum_inference` | float | Total time for `engine.run()` calls |
| `sum_postprocess` | float | Total time for NMS, decoding, coordinate scaling |
| `sum_render` | float | Total time for drawing overlays |
| `sum_save` | float | Total time for saving output frames |
| `sum_display` | float | Total time for cv2.imshow / display |

**Source:** `src/python_example/common/runner/sync_runner.py`

### AsyncRunner — 17 Metrics Fields

AsyncRunner has 17 metrics fields including inflight tracking across its 5+1 thread
pipeline. See `src/python_example/common/runner/async_runner.py` for the complete
list.

### Identifying Bottlenecks

| If this sum is disproportionately large... | Bottleneck is... | Fix |
|---|---|---|
| `sum_preprocess` | CPU image processing | Reduce resolution, optimize resize |
| `sum_inference` | NPU speed | Use smaller model or lower resolution |
| `sum_postprocess` | CPU postprocessing | Switch to C++ postprocess variant |
| `sum_render` / `sum_display` | Display overhead | Use `--no-display` for benchmarks |

---

## CPU MemoryOps Bottleneck Diagnosis

When dxcom compiles a model with preprocessing bake-in, some operations (transpose,
resize, expandDim) may remain as CPU MemoryOps. These run on a single CPU thread by
default, limiting throughput even when the async pipeline has high inflight count.

### Diagnosis via `run_model` Comparison

Compare NPU+CPU mode vs CPU-only (ONNX Runtime) mode to isolate the bottleneck:

```bash
# NPU + CPU ops (default) — measures full graph including CPU MemoryOps
run_model -m model.dxnn -t 5 -v

# CPU-only via ORT — bypasses NPU entirely, runs everything on CPU
run_model -m model.dxnn -t 5 -v --use-ort
```

| Result | Interpretation | Action |
|---|---|---|
| NPU+CPU FPS ≈ CPU-only FPS | CPU ops are the bottleneck | Enable `DXRT_DYNAMIC_CPU_THREAD=ON` |
| NPU+CPU FPS >> CPU-only FPS | NPU is the primary compute path | Normal — optimize via async/C++ postprocess |
| Low FPS + high inflight | Pipeline is buffered but CPU-bound | Enable `DXRT_DYNAMIC_CPU_THREAD=ON` |

### `DXRT_DYNAMIC_CPU_THREAD=ON` — Multi-Threaded CPU Ops

Enables multi-threaded execution of CPU MemoryOps in the DXNN inference graph.

```bash
export DXRT_DYNAMIC_CPU_THREAD=ON
python demo_async.py --model model.dxnn --video input.mp4
```

| Metric | Without (default) | With THREAD=ON |
|---|---|---|
| CPU queue load | 78.9% (saturated) | 28.4% |
| FPS (plant-seg example) | 0.6 fps | 1.4 fps (2.3x) |

**When to use**: Always enable when the compiled model has CPU MemoryOps
(check compiler log for skipped preprocessing ops, or DXRT verbose output
for `CPU TASK` queue load > 50%).

**Always add to `run.sh`** for models compiled with preprocessing bake-in:
```bash
#!/usr/bin/env bash
export DXRT_DYNAMIC_CPU_THREAD=ON
# ... rest of run.sh
```

### Identifying CPU MemoryOps in Compiled Models

1. **Compiler log**: Look for preprocessing ops marked as "skipped" or "not supported on NPU"
   (e.g., transpose, resize, expandDim)
2. **DXRT verbose output**: Run with `-v` flag and check for `CPU TASK` lines:
   ```
   [DXRT] CPU TASK [cpu_0] Inference Worker - Average Input Queue Load : 78.9%
   ```
   Queue load > 50% indicates CPU ops are a significant bottleneck.
3. **Input format change**: If DXNN input is NHWC uint8 while ONNX was NCHW float32,
   preprocessing was partially baked in and CPU MemoryOps likely exist.

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
| Async pipeline | Multiple frames buffered across SafeQueues (maxsize=4) |

### Reducing Memory

```python
# Use uint8 input if model supports it (avoids float32 conversion)
input_tensor = resized_frame  # Keep as uint8 if model input is UINT8

# Release frames promptly
del previous_frame
```

---

## Performance Comparison Table

| Configuration | FPS (YOLOv8n, 640x640) | Notes |
|---|---|---|
| Python sync | 25-35 | Baseline |
| Python sync + C++ postprocess | 35-50 | +40% |
| Python async (5+1 threads) | 60-90 | 2-3x sync |
| Python async + C++ postprocess | 80-120 | Best Python |
| C++ sync | 100-140 | Native overhead minimal |
| C++ async | 130-180 | Best overall |

**Note:** Actual FPS depends on NPU hardware (DX-M1 vs DX-M1A (discontinued)), host CPU, input
source latency, and display overhead. Always benchmark with `--no-display` for
accurate NPU-only measurements.

---

## ⚠️ Fabricated API Calls to Avoid

These calls appeared in previous versions of this file or may be hallucinated by LLMs.
**None of them exist in the DEEPX SDK.**

| Fabricated Call | Why It's Wrong | Correct Alternative |
|---|---|---|
| `engine.infer(data)` | Method does not exist | `engine.run(data)` |
| `engine.get_input_shape()` | Method does not exist | `engine.get_input_tensors_info()` → `info['shape']` |
| `engine.get_profile()` | Method does not exist | Use `Configuration` class with `PROFILER` and `SHOW_PROFILE` items |
| `option.set_batch_size(n)` | Method does not exist | Batch size is set at model compile time (dxcom), not at runtime |
| `option.set_profiling(True)` | Method does not exist | `Configuration().set_enable(Configuration.ITEM.PROFILER, True)` |
| `option.set_num_threads(n)` | Method does not exist | `Configuration().set_enable(Configuration.ITEM.CUSTOM_INTRA_OP_THREADS, ...)` |
| `runner.metrics.fps` | Wrong access pattern | SyncRunner uses `sum_*` timing fields, not named metric attributes |
| `runner.metrics.inference_time` | Wrong access pattern | Use `sum_inference` field |
| `runner.metrics.preprocess_time` | Wrong access pattern | Use `sum_preprocess` field |
| `AsyncRunner` as "2-thread overlap" | Wrong architecture | 5 worker threads + main thread = 6 total, SafeQueue pipeline |
