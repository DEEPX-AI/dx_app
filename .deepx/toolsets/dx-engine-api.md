# DX Engine API Reference

> InferenceEngine and InferenceOption — the core NPU inference interface for dx_app.

## Overview

`dx_engine` is the C++ shared library (with pybind11 Python bindings) that manages model
loading, tensor I/O, and NPU execution. All dx_app inference flows — Python and C++ — go
through InferenceEngine.

## InferenceEngine

### Constructor

```python
from dx_engine import InferenceEngine, InferenceOption

option = InferenceOption()
engine = InferenceEngine("path/to/model.dxnn", option)
```

```cpp
#include "dx_engine/inference_engine.h"
#include "dx_engine/inference_option.h"

dx::InferenceOption option;
dx::InferenceEngine engine("path/to/model.dxnn", option);
```

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `model_path` | `str` / `const char*` | Path to `.dxnn` model file |
| `option` | `InferenceOption` | Configuration for the engine |

**Raises:**
| Error | Code | When |
|---|---|---|
| `RuntimeError` | `DX_ERR_MODEL_NOT_FOUND` | `.dxnn` file does not exist |
| `RuntimeError` | `DX_ERR_MODEL_INVALID` | `.dxnn` file is corrupt or incompatible |
| `RuntimeError` | `DX_ERR_DEVICE_NOT_FOUND` | No NPU device detected |
| `RuntimeError` | `DX_ERR_DEVICE_BUSY` | NPU is locked by another process |

### infer() / run()

Synchronous inference. Accepts a preprocessed input tensor, returns output tensors.

```python
import numpy as np

# Prepare input: NHWC float32 tensor
input_tensor = np.zeros((1, 640, 640, 3), dtype=np.float32)
outputs = engine.infer(input_tensor)
# outputs: list[np.ndarray] — one array per output head
```

```cpp
std::vector<float> input_data(1 * 640 * 640 * 3, 0.0f);
auto outputs = engine.run({input_data});
// outputs: vector<vector<float>> — one vector per output head
```

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `input_data` | `np.ndarray` / `vector<float>` | Preprocessed input tensor |

**Returns:** List of output tensors (one per model output head).

### run_async() / wait()

Asynchronous inference for pipelined execution.

```python
request_id = engine.run_async(input_tensor)
# ... do other work (preprocess next frame) ...
outputs = engine.wait(request_id)
```

```cpp
int req_id = engine.run_async({input_data});
auto outputs = engine.wait(req_id);
```

**Parameters:**
| Method | Parameter | Type | Description |
|---|---|---|---|
| `run_async` | `input_data` | tensor | Input data (same as `infer()`) |
| `wait` | `request_id` | `int` | ID returned by `run_async()` |

### get_input_shape()

Returns the expected input tensor shape for the loaded model.

```python
shape = engine.get_input_shape()
# shape: (1, 640, 640, 3)  — (N, H, W, C) for NHWC models
```

```cpp
auto shape = engine.get_input_shape();
// shape: {1, 640, 640, 3}
```

**Returns:** Tuple/vector of integers representing `(batch, height, width, channels)`.

### get_output_shapes()

Returns shapes for all output heads.

```python
shapes = engine.get_output_shapes()
# shapes: [(1, 8400, 84), (1, 8400, 1)]  — example for YOLOv8
```

### get_input_tensors_info()

Returns detailed tensor metadata including name, shape, and data type.

```python
info = engine.get_input_tensors_info()
for tensor in info:
    print(f"Name: {tensor.name}, Shape: {tensor.shape}, DType: {tensor.dtype}")
```

```cpp
auto info = engine.get_input_tensors_info();
// info[0].shape -> {1, 640, 640, 3}
// info[0].name  -> "input"
// info[0].dtype -> DX_DTYPE_FLOAT32
```

### get_output_tensors_info()

Same as `get_input_tensors_info()` but for output tensors.

```python
out_info = engine.get_output_tensors_info()
for tensor in out_info:
    print(f"Output: {tensor.name}, Shape: {tensor.shape}")
```

### get_model_info()

Returns model metadata embedded in the `.dxnn` file.

```python
info = engine.get_model_info()
# info: {
#   "name": "yolov8n",
#   "task": "object_detection",
#   "input_size": [640, 640],
#   "format_version": 7,
#   "quantization": "INT8"
# }
```

## InferenceOption

Configuration object passed to InferenceEngine constructor.

```python
option = InferenceOption()
option.set_use_ort(True)          # Enable ONNX Runtime fallback (CPU)
option.set_device_id(0)           # Select NPU device index
option.set_num_threads(4)         # CPU threads for pre/postprocess
option.set_batch_size(1)          # Batch size (default: 1)
option.set_profiling(True)        # Enable inference profiling
```

```cpp
dx::InferenceOption option;
option.set_use_ort(true);
option.set_device_id(0);
option.set_num_threads(4);
option.set_batch_size(1);
option.set_profiling(true);
```

### Option Methods

| Method | Type | Default | Description |
|---|---|---|---|
| `set_use_ort(bool)` | bool | `false` | Use ONNX Runtime CPU backend instead of NPU |
| `get_use_ort()` | bool | — | Query current ORT setting |
| `set_device_id(int)` | int | `0` | Select NPU device (multi-device systems) |
| `set_num_threads(int)` | int | `4` | CPU thread count for ORT mode |
| `set_batch_size(int)` | int | `1` | Inference batch size |
| `set_profiling(bool)` | bool | `false` | Enable per-layer profiling |
| `set_log_level(int)` | int | `2` | Log verbosity (0=off, 1=error, 2=warn, 3=info, 4=debug) |

## Error Codes

| Code | Constant | Description |
|---|---|---|
| `-1` | `DX_ERR_GENERIC` | Unspecified error |
| `-2` | `DX_ERR_MODEL_NOT_FOUND` | Model file path does not exist |
| `-3` | `DX_ERR_MODEL_INVALID` | Model file is corrupt or version mismatch |
| `-4` | `DX_ERR_DEVICE_NOT_FOUND` | No NPU device detected on system |
| `-5` | `DX_ERR_DEVICE_BUSY` | NPU is in use by another process |
| `-6` | `DX_ERR_TENSOR_MISMATCH` | Input tensor shape does not match model expectation |
| `-7` | `DX_ERR_OUT_OF_MEMORY` | NPU or host memory allocation failed |
| `-8` | `DX_ERR_TIMEOUT` | Inference request timed out |
| `-9` | `DX_ERR_VERSION_MISMATCH` | DX-RT version incompatible with .dxnn format |

## Multi-Model Pattern

Running multiple models sequentially (e.g., detection + classification pipeline):

```python
from dx_engine import InferenceEngine, InferenceOption

option = InferenceOption()

# Load both models
det_engine = InferenceEngine("yolov8n.dxnn", option)
cls_engine = InferenceEngine("efficientnet_b0.dxnn", option)

# Run detection
det_outputs = det_engine.infer(frame)

# For each detection, crop and classify
for bbox in parse_detections(det_outputs):
    crop = extract_crop(frame, bbox)
    cls_outputs = cls_engine.infer(crop)
    label = parse_classification(cls_outputs)
```

**Important:** Each InferenceEngine instance holds an NPU context. The NPU handles
context switching automatically, but loading too many models simultaneously may cause
`DX_ERR_OUT_OF_MEMORY`.

## Device Management

### dxrt-cli

Command-line tool for NPU diagnostics:

```bash
# Check NPU device status
dxrt-cli -s
# Output:
#   Device 0: DX-M1 (ready)
#   Firmware: v3.2.1
#   Temperature: 42°C
#   Utilization: 0%

# List loaded models
dxrt-cli -m

# Show DX-RT version
dxrt-cli -v
# Output: DX-RT v3.0.0

# Reset NPU device
dxrt-cli --reset
```

### Python Device Check

```python
from dx_engine import InferenceEngine

# Quick device availability check
try:
    option = InferenceOption()
    engine = InferenceEngine("test.dxnn", option)
except RuntimeError as e:
    if "DEVICE_NOT_FOUND" in str(e):
        print("No NPU device — install DX-RT driver")
    elif "DEVICE_BUSY" in str(e):
        print("NPU busy — check for other running processes")
```

## Complete Sync Example

```python
#!/usr/bin/env python3
"""Minimal InferenceEngine sync example."""

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

def main():
    # 1. Configure
    option = InferenceOption()
    engine = InferenceEngine("yolov8n.dxnn", option)

    # 2. Query model
    h, w = engine.get_input_shape()[1:3]
    print(f"Model input: {w}x{h}")

    # 3. Read and preprocess
    image = cv2.imread("test.jpg")
    resized = cv2.resize(image, (w, h))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

    # 4. Infer
    outputs = engine.infer(input_tensor)

    # 5. Process results
    print(f"Got {len(outputs)} output tensors")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")

if __name__ == "__main__":
    main()
```

## Complete Async Example

```python
#!/usr/bin/env python3
"""Minimal InferenceEngine async pipeline example."""

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

def main():
    option = InferenceOption()
    engine = InferenceEngine("yolov8n.dxnn", option)
    h, w = engine.get_input_shape()[1:3]

    cap = cv2.VideoCapture("test.mp4")
    pending_id = None
    prev_outputs = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess current frame
        resized = cv2.resize(frame, (w, h))
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, 0)

        # Wait for previous inference (if any)
        if pending_id is not None:
            prev_outputs = engine.wait(pending_id)
            # ... postprocess prev_outputs on previous frame ...

        # Submit current frame (non-blocking)
        pending_id = engine.run_async(input_tensor)

    # Wait for final frame
    if pending_id is not None:
        final_outputs = engine.wait(pending_id)

    cap.release()

if __name__ == "__main__":
    main()
```

## Thread Safety

- `InferenceEngine.infer()` is **NOT** thread-safe. Do not call from multiple threads.
- `InferenceEngine.run_async()` / `wait()` are safe for single-producer patterns
  (one thread submits, same thread waits).
- For multi-threaded inference, create separate `InferenceEngine` instances per thread.
- `InferenceOption` is a value type — safe to copy across threads.

## Version Compatibility

| DX-RT Version | .dxnn Format | InferenceEngine API |
|---|---|---|
| 3.0.x | v7+ | Full API (infer, run_async, profiling) |
| 2.5.x | v6 | No profiling, no batch_size option |
| 2.0.x | v5 | Sync only, no run_async |
| < 2.0 | v4 | Not supported — upgrade required |
