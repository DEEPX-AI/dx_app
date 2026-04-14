# DX-RT Platform API — dx_app

> NPU device management, InferenceEngine lifecycle, version compatibility, and diagnostics.
> For the full dx_engine Python API, see `.deepx/toolsets/dx-engine-api.md`.

## ⚠️ Anti-Fabrication Notice

This file was audited 2026-04. Every method call listed here has been verified
against `dx_engine` source code. If a method is **not listed in
`.deepx/toolsets/dx-engine-api.md`**, it does **not exist**. Do NOT hallucinate
convenience wrappers — always verify against the source files below.

---

## Source Files

| Class | Source |
|---|---|
| `InferenceEngine` | `dx_rt/python_package/src/dx_engine/inference_engine.py` |
| `InferenceOption` | `dx_rt/python_package/src/dx_engine/inference_option.py` |
| `Configuration` | `dx_rt/python_package/src/dx_engine/configuration.py` |
| `DeviceStatus` | `dx_rt/python_package/src/dx_engine/device_status.py` |

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

# 3. Query input tensor metadata
tensors_info = engine.get_input_tensors_info()
# Returns List[Dict] — each dict has keys: name, shape, dtype, elem_size
input_shape = tensors_info[0]['shape']    # e.g. [1, 640, 640, 3]
input_dtype = tensors_info[0]['dtype']    # e.g. 'uint8'
```

### Execution

```python
# 4. Run inference (sync) — method is run(), NOT infer()
outputs = engine.run(input_tensor)
```

### Teardown

```python
# 5. Explicit cleanup
engine.dispose()

# Or use context manager (preferred)
with InferenceEngine("model.dxnn", option) as engine:
    outputs = engine.run(input_tensor)
# NPU resources released automatically on exit
```

### Memory-Based Loading

```python
# Load from in-memory buffer instead of file path
with open("model.dxnn", "rb") as f:
    model_bytes = f.read()
engine = InferenceEngine.from_buffer(model_bytes, option)
```

### Lifecycle Diagram

```
InferenceOption()  →  Lightweight config object
       ↓
InferenceEngine()  →  Model load + NPU allocation
       ↓
  .run()           →  NPU execution (sync)
       ↓
  .dispose() / context manager exit  →  NPU resource release
```

**Critical:** Always ensure the engine is properly destroyed. Use context managers
or `dispose()`. Abnormal termination (kill -9, segfault) can leave the NPU
locked — use `dxrt-cli --reset` to recover.

---

## Device Discovery (Python)

Use `DeviceStatus` — NOT `InferenceEngine.list_devices()` (which does not exist).

```python
from dx_engine import DeviceStatus

# Get number of NPU devices
count = DeviceStatus.get_device_count()
print(f"Found {count} NPU device(s)")

# Query each device
for device_id in range(count):
    status = DeviceStatus.get_current_status(device_id)
    print(f"Device {status.get_id()}: "
          f"temp={status.get_temperature(0)}°C, "
          f"voltage={status.get_npu_voltage(0)}mV, "
          f"clock={status.get_npu_clock(0)}MHz")
```

---

## Version Check

Use `Configuration().get_version()` — NOT `InferenceEngine.get_runtime_version()`
(which does not exist).

```python
from dx_engine import Configuration

def check_runtime_version(min_version="3.0.0"):
    config = Configuration()  # singleton
    version = config.get_version()
    driver_version = config.get_driver_version()
    pcie_version = config.get_pcie_driver_version()

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
from dx_engine import InferenceEngine, InferenceOption, DeviceStatus

# Discover devices
count = DeviceStatus.get_device_count()

# Use specific device (property syntax — preferred)
option = InferenceOption()
option.devices = [1]  # Use device 1
engine = InferenceEngine("model.dxnn", option)

# Load different models on different devices
option_0 = InferenceOption()
option_0.set_devices([0])  # setter syntax also works
det_engine = InferenceEngine("yolov8n.dxnn", option_0)

option_1 = InferenceOption()
option_1.set_devices([1])
cls_engine = InferenceEngine("efficientnet_b0.dxnn", option_1)

# ⚠️ set_device_id(int) does NOT exist — always use set_devices(List[int])
# or option.devices = [int].  See dx-engine-api.md for the full API.
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
| -6 | `DX_ERR_TENSOR_MISMATCH` | Wrong input shape | Check `get_input_tensors_info()` |
| -7 | `DX_ERR_OUT_OF_MEMORY` | NPU memory full | Reduce batch, unload other models |
| -8 | `DX_ERR_TIMEOUT` | Inference timeout | Check for hardware issues |
| -9 | `DX_ERR_VERSION_MISMATCH` | DX-RT/.dxnn mismatch | Upgrade DX-RT or recompile model |

### Error Handling Pattern

```python
from dx_engine import InferenceEngine, InferenceOption

try:
    option = InferenceOption()
    engine = InferenceEngine("model.dxnn", option)
    outputs = engine.run(input_tensor)  # ← run(), NOT infer()
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
        info = engine.get_input_tensors_info()  # ← NOT get_input_shape()
        print(f"Input shape mismatch. Expected: {info[0]['shape']}")
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

---

## ⚠️ Methods That Do NOT Exist

These are commonly fabricated by LLMs. **None of them exist.**

| Fabricated Call | What To Use Instead |
|---|---|
| `engine.infer(tensor)` | `engine.run(tensor)` |
| `engine.get_input_shape()` | `engine.get_input_tensors_info()[0]['shape']` |
| `engine.get_model_info()` | No equivalent — use `dxrt-cli --info model.dxnn` |
| `InferenceEngine.get_runtime_version()` | `Configuration().get_version()` |
| `InferenceEngine.list_devices()` | `DeviceStatus.get_device_count()` + `DeviceStatus.get_current_status(id)` |
| `option.set_device_id(int)` | `option.set_devices([int])` or `option.devices = [int]` |
| `engine.run_async(tensor)` | Not verified — check source before using |
