# DX-RT Platform API — dx_app

> NPU device management, InferenceEngine lifecycle, version compatibility, and diagnostics.

## Overview

DX-RT is the runtime layer that manages DEEPX NPU hardware. dx_app applications use
DX-RT through the `dx_engine` library. This document covers platform-level concerns:
device detection, driver management, version compatibility, and error diagnostics.

---

## Device Management: dxrt-cli

`dxrt-cli` is the command-line tool for NPU device management.

### Device Status

```bash
dxrt-cli -s
```
Output:
```
DEEPX NPU Device Status
========================
Device 0: DX-M1 (ready)
  Firmware: v3.2.1
  Temperature: 42°C
  Utilization: 0%
  Memory: 512/1024 MB available
  Driver: dx_npu v3.0.0
```

### Version Check

```bash
dxrt-cli -v
```
Output:
```
DX-RT Runtime: v3.0.0
dx_engine: v3.0.0
dx_postprocess: v3.0.0
dx_compiler: v3.0.0
```

### Model Info

```bash
dxrt-cli --info model.dxnn
```
Output:
```
Model: yolov8n
Format: v7
Target: dx_m1
Input: [1, 640, 640, 3] UINT8
Output 0: [1, 8400, 84] FP16
Quantization: INT8 (PTQ)
Size: 6.2 MB
Compiled: 2025-01-15
```

### Device Reset

```bash
dxrt-cli --reset
```
Use when the NPU is stuck in a locked state (e.g., after a crashed process).

### List Loaded Models

```bash
dxrt-cli -m
```
Shows currently loaded models on the NPU.

---

## InferenceEngine Lifecycle

### Initialization

```python
from dx_engine import InferenceEngine, InferenceOption

# 1. Create option (lightweight, no device access)
option = InferenceOption()

# 2. Create engine (loads model, allocates NPU resources)
engine = InferenceEngine("model.dxnn", option)
# At this point: model is loaded on NPU, memory is allocated

# 3. Query model metadata
shape = engine.get_input_shape()
info = engine.get_model_info()
```

### Execution

```python
# 4. Run inference (sync)
outputs = engine.infer(input_tensor)

# Or async
req_id = engine.run_async(input_tensor)
outputs = engine.wait(req_id)
```

### Teardown

```python
# 5. Destructor releases NPU resources
del engine
# NPU context is freed, device becomes available
```

### Lifecycle Diagram

```
InferenceOption()  →  Lightweight config object
       ↓
InferenceEngine()  →  Model load + NPU allocation
       ↓
  .infer() / .run_async()  →  NPU execution
       ↓
  del engine / scope exit  →  NPU resource release
```

**Critical:** Always ensure the engine is properly destroyed. Use context managers
or RAII patterns in C++. Abnormal termination (kill -9, segfault) can leave the NPU
locked — use `dxrt-cli --reset` to recover.

---

## Version Compatibility Matrix

| DX-RT Version | .dxnn Format | dx_app Version | Python | Status |
|---|---|---|---|---|
| 3.0.x | v7 | v3.0.0 | 3.8-3.12 | Current |
| 2.5.x | v6 | v2.5.x | 3.8-3.10 | Deprecated |
| 2.0.x | v5 | v2.0.x | 3.8-3.9 | End of Life |
| 1.x | v4 | v1.x | 3.7-3.8 | Not Supported |

### Cross-Version Rules

1. **DX-RT 3.0.x can load v7, v6, and v5 models** (backward compatible)
2. **v7 models CANNOT run on DX-RT 2.x** (forward incompatible)
3. **Always match dx_app and DX-RT major versions** (3.x with 3.x)
4. **Recompile models when upgrading major versions** for best performance

### Version Check in Code

```python
from dx_engine import InferenceEngine

def check_runtime_version(min_version="3.0.0"):
    version = InferenceEngine.get_runtime_version()
    parts = [int(x) for x in version.split('.')]
    min_parts = [int(x) for x in min_version.split('.')]
    if parts < min_parts:
        raise RuntimeError(
            f"DX-RT {version} is too old. Minimum required: {min_version}"
        )
    return version
```

---

## Multi-Device Support

Systems with multiple NPU devices:

```python
# List available devices
devices = InferenceEngine.list_devices()
# devices: [{"id": 0, "type": "DX-M1", "status": "ready"},
#           {"id": 1, "type": "DX-M1", "status": "ready"}]

# Use specific device
option = InferenceOption()
option.set_device_id(1)  # Use device 1
engine = InferenceEngine("model.dxnn", option)

# Load different models on different devices
option_0 = InferenceOption()
option_0.set_device_id(0)
det_engine = InferenceEngine("yolov8n.dxnn", option_0)

option_1 = InferenceOption()
option_1.set_device_id(1)
cls_engine = InferenceEngine("efficientnet_b0.dxnn", option_1)
```

### Device Selection Strategy

| Scenario | Strategy |
|---|---|
| Single model | Default device 0 |
| Multi-model pipeline | Same device (context switching is fast) |
| Parallel independent models | Different devices (max throughput) |
| Large model (high memory) | Dedicated device |

---

## Error Codes

| Code | Constant | Description | Recovery |
|---|---|---|---|
| -1 | `DX_ERR_GENERIC` | Unspecified error | Check logs |
| -2 | `DX_ERR_MODEL_NOT_FOUND` | .dxnn file missing | Verify path, run setup.sh |
| -3 | `DX_ERR_MODEL_INVALID` | Corrupt or incompatible .dxnn | Recompile model |
| -4 | `DX_ERR_DEVICE_NOT_FOUND` | No NPU detected | Install driver, check hardware |
| -5 | `DX_ERR_DEVICE_BUSY` | NPU locked | Kill other processes, dxrt-cli --reset |
| -6 | `DX_ERR_TENSOR_MISMATCH` | Wrong input shape | Check get_input_shape() |
| -7 | `DX_ERR_OUT_OF_MEMORY` | NPU memory full | Reduce batch, unload other models |
| -8 | `DX_ERR_TIMEOUT` | Inference timeout | Check for hardware issues |
| -9 | `DX_ERR_VERSION_MISMATCH` | DX-RT/.dxnn mismatch | Upgrade DX-RT or recompile model |

### Error Handling Pattern

```python
from dx_engine import InferenceEngine, InferenceOption

try:
    option = InferenceOption()
    engine = InferenceEngine("model.dxnn", option)
    outputs = engine.infer(input_tensor)
except RuntimeError as e:
    error_msg = str(e)
    if "DEVICE_NOT_FOUND" in error_msg:
        print("No NPU device found. Check hardware connection.")
        print("Run: dxrt-cli -s")
    elif "DEVICE_BUSY" in error_msg:
        print("NPU is locked by another process.")
        print("Run: dxrt-cli --reset")
    elif "MODEL_NOT_FOUND" in error_msg:
        print("Model file not found. Run: ./setup.sh")
    elif "MODEL_INVALID" in error_msg:
        print("Model file is corrupt or incompatible.")
        print("Run: dxrt-cli --info model.dxnn")
    elif "TENSOR_MISMATCH" in error_msg:
        shape = engine.get_input_shape()
        print(f"Input shape mismatch. Expected: {shape}")
    else:
        raise
```

---

## Driver Diagnostics

### Check Driver Status

```bash
# Verify kernel module is loaded
lsmod | grep dx_npu

# Check device nodes
ls -la /dev/dx_npu*

# View driver logs
dmesg | grep dx_npu

# Full diagnostics
dxrt-cli --diag
```

### Common Driver Issues

| Symptom | Cause | Fix |
|---|---|---|
| `/dev/dx_npu0` missing | Driver not loaded | `sudo modprobe dx_npu` |
| Permission denied | User not in `dx` group | `sudo usermod -aG dx $USER` |
| Device not found after reboot | Module not in autoload | Add `dx_npu` to `/etc/modules` |
| Firmware mismatch | Driver/firmware version mismatch | Update driver package |

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Linux kernel | 5.4+ | 5.15+ |
| Python | 3.8 | 3.10 |
| NumPy | 1.20+ | 1.24+ |
| OpenCV | 4.5+ | 4.8+ |
| GCC (for build) | 9.0+ | 11+ |
| CMake | 3.14+ | 3.22+ |
