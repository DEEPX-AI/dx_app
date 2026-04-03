# Common Pitfalls — dx_app

> Domain-tagged pitfalls for dx_app development. Read before every task.

---

## 1. [UNIVERSAL] model_registry.json Case Mismatch

**Symptom:** `KeyError: 'YoloV8n'` when querying model_registry.json.

**Cause:** Model names in `config/model_registry.json` are all lowercase with
underscores (e.g., `yolov8n`). Using mixed-case names like `YoloV8n`, `YOLOV8N`,
or `YOLOv8n` will not match.

**Fix:** Always use lowercase model names when querying the registry. Normalize
user input with `.lower()` before lookup:
```python
model_name = user_input.lower()
info = registry.get(model_name)
```

---

## 2. [DX_APP] AsyncRunner Frame Order Inversion

**Symptom:** Visualization shows postprocess results drawn on the wrong frame.
Bounding boxes appear shifted or lag behind the actual objects.

**Cause:** AsyncRunner pipelines preprocessing of frame N+1 while NPU processes
frame N. If the postprocess callback modifies the frame buffer in-place without
copying, the result is drawn on frame N+1 instead of frame N.

**Fix:** Always copy the frame before passing it to the postprocessor:
```python
# In the postprocess callback
display_frame = original_frame.copy()
draw_detections(display_frame, detections)
```
Alternatively, use the built-in `IVisualizer.draw()` method which handles copying
internally.

---

## 3. [DX_APP] dx_postprocess Not Installed → ImportError

**Symptom:** `ImportError: No module named 'dx_postprocess'` when running a
`*_cpp_postprocess` variant.

**Cause:** The pybind11 bindings for C++ postprocessing have not been compiled.
The `dx_postprocess` module is built by `./build.sh` and installed as a shared
library (`dx_postprocess.so`).

**Fix:** Build from the dx_app root:
```bash
./build.sh
# Verify:
python -c "import dx_postprocess; print('OK')"
```
If using a virtual environment, make sure `./build.sh` was run with the venv
activated.

---

## 4. [PPU] PPU Models Need Dedicated Postprocessors

**Symptom:** Garbled detection results, nonsensical bounding boxes, or crash
when using standard YOLOv5/v8 postprocessors with PPU models.

**Cause:** PPU (Pre/Post Processing Unit) models have a different output tensor
format. The standard YOLO postprocessors expect raw grid outputs, but PPU models
output pre-decoded detections in a PPU-specific format.

**Fix:** Use the dedicated `PPUPostProcess` binding:
```python
from dx_postprocess import PPUPostProcess
pp = PPUPostProcess(input_w, input_h, score_thresh, nms_thresh, use_ort)
```
Never use `YoloV5PostProcess` or `YoloV8PostProcess` with PPU models.

---

## 5. [DX_APP] OBB Detection: score_threshold Only (No NMS)

**Symptom:** `TypeError` or unexpected argument error when passing `nms_threshold`
to `Yolo26OBBPostProcess`.

**Cause:** Oriented Bounding Box (OBB) detection models do not use NMS. The
`Yolo26OBBPostProcess` constructor accepts `score_threshold` only:
```python
# WRONG:
pp = Yolo26OBBPostProcess(w, h, 0.25, 0.45, use_ort)  # Too many args
# RIGHT:
pp = Yolo26OBBPostProcess(w, h, 0.25, use_ort)
```

**Fix:** Pass only `score_threshold` to OBB postprocessors. Do not pass
`nms_threshold`.

---

## 6. [UNIVERSAL] Headless Mode: Check DISPLAY Before imshow

**Symptom:** `cv2.error: The function is not implemented` or
`cannot open display` error on headless servers/containers.

**Cause:** `cv2.imshow()` requires a display server (X11/Wayland). Remote SSH
sessions and Docker containers typically don't have one.

**Fix:** Check for display availability or use `--no-display`:
```python
import os
if os.environ.get("DISPLAY") is None:
    args.no_display = True
```
Or run with the `--no-display` flag:
```bash
python yolov8n_sync.py --model model.dxnn --input video.mp4 --no-display
```

---

## 7. [UNIVERSAL] DX-RT Version < 3.0.0 Silent Failures

**Symptom:** InferenceEngine loads successfully but produces all-zero outputs,
or `get_model_info()` returns empty/partial data. No error message.

**Cause:** `.dxnn` format v7 (required by dx_app v3.0.0) is not fully supported
by older DX-RT versions. The older runtime may load the model header but fail
to properly execute the inference graph.

**Fix:** Verify DX-RT version before running:
```bash
dxrt-cli -v
# Must show: DX-RT v3.0.0 or higher
```
```python
from dx_engine import InferenceEngine
# Check version programmatically
version = InferenceEngine.get_runtime_version()
major = int(version.split('.')[0])
if major < 3:
    raise RuntimeError(f"DX-RT {version} is too old. Upgrade to 3.0.0+")
```

---

## 8. [DX_APP] config.json score_threshold Alias Confusion

**Symptom:** Changing `score_threshold` in config.json has no effect, or CLI
`--score-threshold` doesn't override the config file value.

**Cause:** The config.json field is `score_threshold` (with underscore), but
the CLI flag is `--score-threshold` (with hyphen). argparse converts hyphens
to underscores in the namespace (`args.score_threshold`). However, if the
config loading code uses a different key name, one may shadow the other.

**Fix:** Ensure consistent naming:
- In `config.json`: use `"score_threshold"` (underscore)
- On CLI: use `--score-threshold` (hyphen, argparse standard)
- In code: access as `args.score_threshold` (underscore)

The precedence is: CLI > config.json > default. If CLI is not provided,
config.json takes effect. If neither is provided, the default (0.25) is used.

---

## 9. [UNIVERSAL] SIGINT Handler Missing → Zombie Processes

**Symptom:** Ctrl+C during inference leaves the process hanging. NPU device
stays locked. Subsequent runs fail with `DX_ERR_DEVICE_BUSY`.

**Cause:** No `SIGINT` handler installed. The default Python behavior raises
`KeyboardInterrupt` which may not properly release the NPU context, leaving
the device in a locked state.

**Fix:** Always use the framework's built-in signal handling:
```python
# SyncRunner and AsyncRunner handle SIGINT automatically.
# For custom scripts, install a handler:
import signal

running = True

def handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handler)
```
If a zombie process has locked the NPU:
```bash
dxrt-cli --reset
```

---

## 10. [DX_APP] 4-Variant Naming Mismatch Causes Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'factory'` when running a
variant script, even though the factory exists.

**Cause:** The 4 Python variants must follow strict naming:
```
<model>_sync.py
<model>_async.py
<model>_sync_cpp_postprocess.py
<model>_async_cpp_postprocess.py
```
The factory directory must be `factory/` with `__init__.py` that exports the
factory class. If the factory module name or the sys.path hack is wrong,
imports fail.

**Fix:** Verify:
1. Factory exists at `<model_dir>/factory/<model>_factory.py`
2. `factory/__init__.py` contains `from .<model>_factory import <Model>Factory`
3. Each variant script has the standard sys.path insertion:
   ```python
   _module_dir = Path(__file__).parent
   _v3_dir = _module_dir.parent.parent
   for _path in [str(_v3_dir), str(_module_dir)]:
       if _path not in sys.path:
           sys.path.insert(0, _path)
   ```
4. The variant filename matches the expected pattern exactly.
