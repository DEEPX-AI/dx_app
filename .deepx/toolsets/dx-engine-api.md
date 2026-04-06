# DX Engine API Reference

> Reference-based guide to `dx_engine` — the core NPU inference interface for dx_app.
> This document provides an overview and points to source files for current API details.
> Do NOT rely on memorized signatures — always verify against the source files listed below.

## ⚠️ Anti-Fabrication Notice

**AI agents MUST verify every method name and signature against the actual source files
before generating code.** Previous versions of this document contained fabricated API
methods (`infer()`, `get_input_shape()`, `get_model_info()`, `get_output_shapes()`,
wrong `run_async()` return value name). These methods do not exist and will cause
`AttributeError` at runtime.

**Rule:** If you are unsure whether a method exists, read the source file. Never guess.

## Source Files

All API definitions live in the DX-RT source tree. Read these files for current signatures:

| Class | Source File |
|---|---|
| `InferenceEngine` | `dx_rt/python_package/src/dx_engine/inference_engine.py` |
| `InferenceOption` | `dx_rt/python_package/src/dx_engine/inference_option.py` |
| `Configuration` | `dx_rt/python_package/src/dx_engine/configuration.py` |
| `DeviceStatus` | `dx_rt/python_package/src/dx_engine/device_status.py` |
| `RuntimeEventDispatcher` | `dx_rt/python_package/src/dx_engine/runtime_event_dispatcher.py` |
| Python API docs | `dx_rt/docs/source/docs/10_02_Python_API_Reference.md` |

## Package Overview

Exported from `dx_engine`:

```python
from dx_engine import (
    InferenceEngine,       # Core inference class
    InferenceOption,       # Engine configuration
    Configuration,         # Runtime settings singleton
    DeviceStatus,          # Hardware diagnostics
    RuntimeEventDispatcher # Event handling
)
```

## InferenceEngine — Overview

> **Source of truth**: `dx_rt/python_package/src/dx_engine/inference_engine.py`
> Always read this file for exact signatures and parameter names.

### Construction

```python
from dx_engine import InferenceEngine, InferenceOption

# Option is OPTIONAL — defaults to None (creates default internally)
engine = InferenceEngine("model.dxnn")

# With explicit option
option = InferenceOption()
engine = InferenceEngine("model.dxnn", option)

# From memory buffer
import numpy as np
engine = InferenceEngine.from_buffer(memory_buffer, inference_option=None)

# Context manager supported (calls dispose() on exit)
with InferenceEngine("model.dxnn") as engine:
    outputs = engine.run(input_data)
```

### Method Categories

**Inference (synchronous):**

| Method | Returns | Notes |
|---|---|---|
| `run(input_data, output_buffers=None, user_args=None)` | `List[np.ndarray]` | Primary sync inference. **NOT `infer()`** |
| `run_multi_input(input_tensors: Dict[str, np.ndarray], ...)` | `List[np.ndarray]` | For multi-input models |
| `run_benchmark(num_loops, input_data=None)` | `float` | Returns FPS |
| `validate_device(input_data, device_id=0)` | — | Debug compile type only |

**Inference (asynchronous):**

| Method | Returns | Notes |
|---|---|---|
| `run_async(input_data, user_arg=None, output_buffer=None)` | `int` (**`job_id`**) | Single inference, NOT batch. Returns `job_id`, **NOT** `request_id` |
| `wait(job_id)` | `List[np.ndarray]` | Parameter is `job_id` (int) |
| `register_callback(callback)` | — | For async completion callbacks |

**Model information:**

| Method | Returns | Notes |
|---|---|---|
| `get_input_tensors_info()` | `List[Dict]` | Keys: `name`, `shape`, `dtype`, `elem_size`. **Dict access** (`info[0]['shape']`), NOT dot-access |
| `get_output_tensors_info()` | `List[Dict]` | Same keys as above |
| `get_input_tensor_count()` | `int` | Number of input tensors |
| `get_output_tensor_count()` | `int` | Number of output tensors |
| `get_input_tensor_names()` | `List[str]` | Input tensor names |
| `get_output_tensor_names()` | `List[str]` | Output tensor names |
| `get_input_size()` | `int` | Total input bytes |
| `get_output_size()` | `int` | Total output bytes |
| `get_input_tensor_sizes()` | `List[int]` | Per-tensor byte sizes |
| `get_output_tensor_sizes()` | `List[int]` | Per-tensor byte sizes |
| `has_dynamic_output()` | `bool` | — |
| `is_multi_input_model()` | `bool` | — |
| `is_ppu()` | `bool` | — |
| `get_compile_type()` | `str` | e.g. `"debug"` or `"release"` |
| `get_model_version()` | `str` | — |

**Performance metrics:**

| Method | Returns | Notes |
|---|---|---|
| `get_latency()` | `int` | Microseconds |
| `get_npu_inference_time()` | `int` | Microseconds |
| `get_latency_list()` | `List[int]` | Historical latency samples |
| `get_npu_inference_time_list()` | `List[int]` | Historical NPU time samples |
| `get_latency_mean()` | `float` | — |
| `get_npu_inference_time_mean()` | `float` | — |
| `get_latency_std()` | `float` | — |
| `get_npu_inference_time_std()` | `float` | — |

**Lifecycle:**

| Method | Notes |
|---|---|
| `dispose()` | Explicit resource release. Also called by `__exit__` in context manager |

### Verified Code Pattern

```python
from dx_engine import InferenceEngine
import numpy as np

with InferenceEngine("model.dxnn") as engine:
    # Query model shape via get_input_tensors_info() — returns List[Dict]
    info = engine.get_input_tensors_info()
    shape = info[0]['shape']   # Dict access, NOT dot-access
    dtype = info[0]['dtype']
    print(f"Input shape: {shape}")

    # Synchronous inference — use run(), NOT infer()
    input_data = np.zeros(shape, dtype=np.float32)
    outputs = engine.run(input_data)

    for i, out in enumerate(outputs):
        print(f"Output {i}: shape={out.shape}")
```

## InferenceOption

Configuration object passed to InferenceEngine constructor.

> **Source of truth**: `dx_rt/python_package/src/dx_engine/inference_option.py`
> and `dx_rt/docs/source/docs/10_02_Python_API_Reference.md`.
> Do NOT invent methods that are not listed here.

### Quick Start (Python)

```python
from dx_engine import InferenceOption

option = InferenceOption()

# Property syntax (preferred)
option.use_ort = True              # Enable ONNX Runtime fallback (CPU)
option.devices = [0]               # Use NPU device 0
option.bound_option = InferenceOption.BOUND_OPTION.NPU_ALL  # Use all 3 cores
option.buffer_count = 8            # Internal buffers (default: 6, range: 1-100)

# Setter syntax (equivalent)
option.set_use_ort(True)
option.set_devices([0])
option.set_bound_option(InferenceOption.BOUND_OPTION.NPU_ALL)
option.set_buffer_count(8)
```

### Properties (get/set)

| Property | Type | Default | Description |
|---|---|---|---|
| `use_ort` | `bool` | `False` | Use ONNX Runtime CPU backend instead of NPU. **Note:** Still requires `dxrtd` service running and NPU device initialization to succeed. ORT only affects the inference execution path, not device init. If `dxrtd` is down or firmware is incompatible, even ORT mode fails with `RuntimeError`. |
| `devices` | `List[int]` | `[]` (all) | NPU device IDs to use; empty = all available |
| `bound_option` | `BOUND_OPTION` | `NPU_ALL` | NPU core binding strategy |
| `buffer_count` | `int` | `6` | Internal buffers for pipelined inference (range: 1-100) |

### Setter / Getter Methods

| Method | Parameter | Description |
|---|---|---|
| `set_use_ort(bool)` | `bool` | Enable/disable ONNX Runtime CPU backend |
| `get_use_ort()` | — | Returns current ORT setting |
| `set_devices(List[int])` | `List[int]` | Set NPU device IDs (e.g., `[0]`, `[0, 1]`) |
| `get_devices()` | — | Returns current device list |
| `set_bound_option(BOUND_OPTION)` | `BOUND_OPTION` | Set NPU core binding |
| `get_bound_option()` | — | Returns current binding |
| `set_buffer_count(int)` | `int` | Set internal buffer count (1-100) |
| `get_buffer_count()` | — | Returns current buffer count |

### BOUND_OPTION Enum

Controls which NPU cores are used for inference:

| Value | Description |
|---|---|
| `NPU_ALL` | Use all 3 cores (default — maximum throughput) |
| `NPU_0` | Core 0 only |
| `NPU_1` | Core 1 only |
| `NPU_2` | Core 2 only |
| `NPU_01` | Cores 0 + 1 |
| `NPU_12` | Cores 1 + 2 |
| `NPU_02` | Cores 0 + 2 |

### ⚠️ Methods That Do NOT Exist (InferenceOption)

These methods were previously documented incorrectly. They will cause
`AttributeError` at runtime. **Never use them:**

| ❌ Fabricated Method | ✅ Correct Alternative |
|---|---|
| `set_device_id(int)` | `set_devices([int])` or `option.devices = [int]` |
| `set_num_threads(int)` | Does not exist — no equivalent |
| `set_batch_size(int)` | Does not exist — no equivalent |
| `set_profiling(bool)` | Use `dx_engine.Configuration` class instead |
| `set_log_level(int)` | Does not exist — no equivalent |

## Configuration

> **Source of truth**: `dx_rt/python_package/src/dx_engine/configuration.py`

Singleton for runtime settings. Uses an `ITEM` enum to select what to configure.

**Key methods:**

| Method | Returns | Notes |
|---|---|---|
| `get_version()` | `str` | DX-RT runtime version. **This is the correct way to get version** (NOT `InferenceEngine.get_runtime_version()`) |
| `get_driver_version()` | `str` | NPU driver version |
| `get_pcie_driver_version()` | `str` | PCIe driver version |
| `set_enable(item, enabled)` | — | Enable/disable a feature by `ITEM` enum |
| `set_attribute(item, attrib, value)` | — | Set a feature attribute |
| `set_fw_config_with_json(json_str)` | — | Configure firmware with a JSON string |

**ITEM enum values:**

| Value | Int | Notes |
|---|---|---|
| `DEBUG` | 1 | |
| `PROFILER` | 2 | |
| `SERVICE` | 3 | |
| `DYNAMIC_CPU_THREAD` | 4 | |
| `TASK_FLOW` | 5 | |
| `SHOW_THROTTLING` | 6 | |
| `SHOW_PROFILE` | 7 | |
| `SHOW_MODEL_INFO` | 8 | |
| `CUSTOM_INTRA_OP_THREADS` | 9 | |
| `CUSTOM_INTER_OP_THREADS` | 10 | |
| `NFH_ASYNC` | 11 | |
| `NFH_ACCELERATION` | 12 | Conditional — only present when NFH acceleration is compiled in |
| `CPU_OP_ACCELERATION` | 13 | Conditional — only present when CPU acceleration is compiled in |

See `dx_rt/python_package/src/dx_engine/configuration.py` for the complete ITEM enum and method list.

```python
from dx_engine import Configuration

config = Configuration()
version = config.get_version()
driver_version = config.get_driver_version()
pcie_version = config.get_pcie_driver_version()
```

## DeviceStatus

> **Source of truth**: `dx_rt/python_package/src/dx_engine/device_status.py`

Hardware diagnostics for NPU devices.

**Class methods (no instance needed):**

| Method | Returns | Notes |
|---|---|---|
| `DeviceStatus.get_device_count()` | `int` | Number of NPU devices. **This is the correct way to list devices** (NOT `InferenceEngine.list_devices()`) |
| `DeviceStatus.get_current_status(deviceId)` | `DeviceStatus` | Get status object for a specific device |

**Instance methods (on a DeviceStatus object):**

| Method | Returns | Notes |
|---|---|---|
| `get_id()` | `int` | Device ID |
| `get_temperature(ch)` | `int` | Temperature for channel |
| `get_npu_voltage(ch)` | `int` | Voltage for channel |
| `get_npu_clock(ch)` | `int` | Clock speed for channel |

```python
from dx_engine import DeviceStatus

count = DeviceStatus.get_device_count()
for i in range(count):
    status = DeviceStatus.get_current_status(i)
    print(f"Device {status.get_id()}: temp={status.get_temperature(0)}")
```

## RuntimeEventDispatcher

> **Source of truth**: `dx_rt/python_package/src/dx_engine/runtime_event_dispatcher.py`

Event handling for runtime events. Defines `LEVEL`, `TYPE`, and `CODE` enums.
Read the source file for available enum values and callback registration patterns.

## ⚠️ Methods That Do NOT Exist (InferenceEngine)

These methods have been fabricated in previous AI outputs. They will cause
`AttributeError` at runtime. **Never use them:**

| ❌ Fabricated Method | ✅ Correct Alternative |
|---|---|
| `infer(input_data)` | `run(input_data)` |
| `get_input_shape()` | `get_input_tensors_info()[0]['shape']` |
| `get_output_shapes()` | `[t['shape'] for t in get_output_tensors_info()]` |
| `get_model_info()` | No such method — use `get_model_version()`, `get_compile_type()`, etc. |
| `get_runtime_version()` | `Configuration().get_version()` |
| `list_devices()` | `DeviceStatus.get_device_count()` + `DeviceStatus.get_current_status(id)` |
| `predict(input_data)` | `run(input_data)` |
| `load(path)` / `load_model(path)` | Use constructor: `InferenceEngine(path)` |
| `close()` | `dispose()` or use context manager |
| `tensor.name` (dot-access on tensor info) | `tensor['name']` (dict key access) |
| `run_async()` returns `request_id` | Returns `job_id` — use `wait(job_id)` |

## Quick Start Example

```python
#!/usr/bin/env python3
"""Verified minimal InferenceEngine example."""

import cv2
import numpy as np
from dx_engine import InferenceEngine

def main():
    # 1. Create engine (option is optional)
    with InferenceEngine("yolov8n.dxnn") as engine:
        # 2. Query model input — returns List[Dict], use dict key access
        input_info = engine.get_input_tensors_info()
        shape = input_info[0]['shape']   # e.g. (1, 640, 640, 3)
        h, w = shape[1], shape[2]
        print(f"Model input: {w}x{h}")

        # 3. Preprocess
        image = cv2.imread("test.jpg")
        resized = cv2.resize(image, (w, h))
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        # 4. Synchronous inference — use run(), NOT infer()
        outputs = engine.run(input_tensor)

        # 5. Inspect outputs
        output_info = engine.get_output_tensors_info()
        for i, out in enumerate(outputs):
            print(f"Output '{output_info[i]['name']}': shape={out.shape}, dtype={out.dtype}")

        # 6. Performance metrics
        print(f"Latency: {engine.get_latency()} us")
        print(f"NPU time: {engine.get_npu_inference_time()} us")

if __name__ == "__main__":
    main()
```

### Async Pipeline Pattern

```python
from dx_engine import InferenceEngine
import numpy as np

with InferenceEngine("model.dxnn") as engine:
    input_info = engine.get_input_tensors_info()
    shape = input_info[0]['shape']

    # run_async returns job_id (int), NOT request_id
    job_id = engine.run_async(np.zeros(shape, dtype=np.float32))

    # ... do other work while NPU processes ...

    outputs = engine.wait(job_id)  # Parameter is job_id
```
