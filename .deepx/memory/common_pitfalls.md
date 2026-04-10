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

---

## 11. [UNIVERSAL] dx-runtime Not Installed — Build or Import Fails

**Symptom:** `ImportError: No module named 'dx_engine'` or
`cmake: cannot find -ldx_engine` when building or running dx_app.

**Cause:** dx-runtime (dx_rt) is not installed or is at an incompatible version.
dx_app depends on dx_rt for both Python SDK (`dx_engine` module) and C++ build
(`libdx_engine.so`).

**Fix:**
1. Run the sanity check: `bash ../../scripts/sanity_check.sh --dx_rt`
2. If FAIL, install dx-runtime:
   ```bash
   bash ../../install.sh --target=dx_rt,dx_rt_npu_linux_driver,dx_fw --skip-uninstall --venv-reuse
   ```
3. Then rebuild dx_app: `./install.sh && ./build.sh`
4. Verify: `python -c "import dx_engine; print('OK')"`

---

## 12. [UNIVERSAL] Wrong Sample Image for Task Type

**Symptom:** Smoke test or validation runs but produces zero detections,
garbled results, or meaningless output — even though the model is correct.

**Cause:** Using a generic or mismatched sample image for the model's AI task.
For example, running a face detection model on `sample_dog.jpg` (no faces),
or running a super resolution model on a low-resolution JPEG (wrong format).

**Fix:** Always select sample images that match the model's task type:

| Task | Correct Sample Images |
|---|---|
| object_detection | `sample/img/sample_dog.jpg`, `sample/img/sample_horse.jpg` |
| face_detection | `sample/img/sample_face.jpg`, `sample/img/sample_crowd.jpg` |
| pose_estimation | `sample/img/sample_people.jpg`, `sample/img/sample_crowd.jpg` |
| hand_landmark | `sample/img/sample_hand.jpg` |
| obb_detection | `sample/dota8_test/P0177.png`, `sample/dota8_test/P0284.png` |
| segmentation | `sample/img/sample_street.jpg`, `sample/img/sample_parking.jpg` |
| classification | `sample/ILSVRC2012/0.jpeg`, `sample/ILSVRC2012/1.jpeg` |
| super_resolution | `sample/img/sample_superresolution.png` |
| image_enhancement | `sample/img/sample_lowlight.jpg`, `sample/img/sample_dark_room.jpg` |
| image_denoising | `sample/img/sample_denoising.jpg` |

See `config/README.md` for the authoritative task→image mapping.

---

## 13. [DX_APP] yolov26 Registry Key ≠ Python Postprocessor Class Name

**Symptom:** Agent generates `Yolo26Postprocessor` or `YOLOv26Postprocessor` — class
does not exist. Or generates a custom postprocessor that parses the wrong output format,
resulting in zero detections or garbled bounding boxes.

**Cause:** `model_registry.json` uses the registry key `"yolov26"` in the `postprocessor`
field. Agents may naively convert this to a class name like `Yolo26Postprocessor`. However,
YOLO26 models use the YOLOv8-compatible end-to-end output format `[1, 300, 6]` =
`[x1, y1, x2, y2, score, class_id]`, so they share the `YOLOv8Postprocessor` class.

**Fix:** Always use the **Registry Key → Python Class** mapping table:

| Registry Key | Correct Python Class |
|---|---|
| `yolov26` | `YOLOv8Postprocessor` |
| `yolov5` | `YOLOv5Postprocessor` |
| `yolov8` | `YOLOv8Postprocessor` |
| `yolov10` | `YOLOv10Postprocessor` |
| `yolov11` | `YOLOv11Postprocessor` |

For C++ bindings (`dx_postprocess`), yolo26 has its own `Yolo26PostProcess` class.
Only the Python postprocessor is shared with YOLOv8.

**Validation:** After generating a factory, cross-check the postprocessor import against
this mapping. See `dx-validate.md` Level 5 Check 5 for the automated cross-check script.

---

## 14. [DX_APP] Existing Example Ignored → Wrong Postprocessor Selected

**Symptom:** Agent generates new code for a model that already has a working example in
`src/python_example/<task>/<model>/`, but uses a different (incorrect) postprocessor.

**Cause:** Agent skipped the "Search Existing Examples" step and selected the postprocessor
based on model name heuristics or registry key alone, without checking the established
working pattern.

**Fix:** **Always search for existing examples first** before generating new code:
```bash
ls src/python_example/<task>/<model>/factory/ 2>/dev/null
grep "Postprocessor" src/python_example/<task>/<model>/factory/*_factory.py
```
If a working example exists, use the same postprocessor class. The existing example is
the ground truth for correct component selection.

---

## 15. [DX_APP] Zero Detections Pass Validation — Missing Output Accuracy Check

**Symptom:** Agent declares "all validation checks passed" but the generated app produces
zero detections on a known-good sample image.

**Cause:** The 5-Level Validation Pyramid only checked exit code and log messages at
Level 4 (Smoke Test). It did not verify that the model actually produced meaningful
output (detection count > 0, valid bbox coordinates, valid class IDs).

**Fix:** Always run Level 5 (Output Accuracy) validation after Level 4:
1. Detection count > 0 on task-appropriate sample image
2. Bbox coordinates within image bounds
3. Confidence scores in [0.0, 1.0]
4. Class IDs in valid range
5. Postprocessor-model family cross-check

See `dx-validate.md` Level 5 for the complete validation scripts.

---

## 16. [DX_APP] Skipping Cross-Validation with Precompiled Reference Model

**Symptom:** Agent declares output correct but the generated app actually produces
wrong results. The bug goes undetected because validation only checks plausibility
(non-zero detections, valid ranges) without comparing against a known-good baseline.

**Cause:** Level 5 validates that output "looks reasonable" but does NOT compare
against reference output from a precompiled model or existing verified example.
A model could pass all plausibility checks while producing completely different
(but structurally valid) detections from the correct output.

**Fix:** When a precompiled reference DXNN exists in `assets/models/` for the same
model, or an existing verified example exists in `src/python_example/`, ALWAYS run
the Level 5.5 cross-validation differential diagnosis:
1. **Test A**: Run generated app with precompiled model — isolates app code vs model
2. **Test B**: Compare against existing verified example with `--verbose`
3. **Test C**: Cross-model swap — run existing app with new model

See `dx-validate.md` Level 5.5 for the full Differential Diagnosis Decision Matrix.

---

## 17. [UNIVERSAL] Missing Deployment Artifacts — setup.sh, run.sh, session.log

**Symptom**: Generated app only has Python source files but no deployment scripts.
Running `./setup.sh` directly fails with NumPy/OpenCV version conflicts, or
`./run.sh` shows only `--model is required` with no guidance on where models are.

**Root cause**: Agent generates core app files but skips deployment artifacts,
or generates them without venv detection logic and without real file paths.

**Fix**: Every session directory MUST include these mandatory artifacts:
1. `setup.sh` — Environment setup script with **mandatory venv detection** (see below)
2. `run.sh` — One-command inference launcher with **real model + sample image paths**
3. `README.md` — Session summary with **runnable example commands** (no `/path/to/` placeholders)
4. `session.json` — Build metadata (model, task, variant, timestamp)
5. `session.log` — Actual command execution output captured via `tee`

### setup.sh MUST include venv detection (CRITICAL)

Without venv activation, `pip install` and `python` use the system Python, which often
has incompatible NumPy versions (e.g., NumPy 2.x vs OpenCV compiled with NumPy 1.x).

**Required logic** in setup.sh:
```bash
# Search upward for dx-runtime shared venv (preferred)
RUNTIME_VENV=""
_search="$SCRIPT_DIR"
for _i in 1 2 3 4 5; do
    _search="$(dirname "$_search")"
    [ -d "$_search/venv-dx-runtime" ] && RUNTIME_VENV="$_search/venv-dx-runtime" && break
done
# Activate shared venv, or create local .venv/ as fallback
if [ -n "$RUNTIME_VENV" ]; then
    source "$RUNTIME_VENV/bin/activate"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    python3 -m venv "$SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
fi
```

### run.sh and README MUST use real paths (CRITICAL)

Never use `/path/to/<model>.dxnn` or `input.jpg` placeholders. Always use:
- **Model**: `../../assets/models/<model>.dxnn` (precompiled) or relative dx-compiler path
- **Image**: Task-appropriate sample from `../../sample/img/` (see Task-Aware Sample Image table)
- **Video**: `../../assets/videos/dogs.mp4` or similar

**Prevention**: Check the setup.sh/run.sh templates in `dx-build-python-app.md` and the
deliverables list before claiming completion.
