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
Alternatively, use the built-in `IVisualizer.visualize()` method which handles copying
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

**Fix:** Use the dedicated model-specific PPU postprocessor bindings:
```python
from dx_postprocess import YOLOv5PPUPostProcess    # for YOLOv5 PPU models
from dx_postprocess import YOLOv7PPUPostProcess    # for YOLOv7 PPU models
from dx_postprocess import YOLOv5PosePPUPostProcess  # for YOLOv5-Pose PPU models
from dx_postprocess import SCRFDPPUPostProcess     # for SCRFD PPU models
```
There is no generic `PPUPostProcess` class — each model family has its own dedicated
PPU postprocessor. Never use `YoloV5PostProcess` or `YoloV8PostProcess` with PPU models.

---

## 5. [DX_APP] OBB Detection: score_threshold Only (No NMS)

**Symptom:** `TypeError` or unexpected argument error when passing `nms_threshold`
to `OBBPostProcess`.

**Cause:** Oriented Bounding Box (OBB) detection models do not use NMS. The
`OBBPostProcess` constructor accepts only 3 parameters — `(input_w, input_h, score_threshold)`:
```python
# WRONG:
pp = OBBPostProcess(w, h, 0.25, 0.45, use_ort)  # Too many args
# RIGHT:
pp = OBBPostProcess(w, h, 0.25)
```

**Fix:** Pass only `(input_w, input_h, score_threshold)` to OBB postprocessors. Do not pass
`nms_threshold` or `is_ort_configured`.

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
python yolov8n_sync.py --model model.dxnn --video video.mp4 --no-display
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
from dx_engine import Configuration
# Check version programmatically
version = Configuration().get_version()
major = int(version.split('.')[0])
if major < 3:
    raise RuntimeError(f"DX-RT {version} is too old. Upgrade to 3.0.0+")
```

---

## 8. [DX_APP] config.json score_threshold — Config Only, No CLI Flag

**Symptom:** User tries to override score threshold via CLI with `--score-threshold`
but the flag is not recognized by `parse_common_args()`.

**Cause:** There is no `--score-threshold` CLI flag in the dx_app argument parser.
Score threshold is configured exclusively through `config.json` (the `"score_threshold"`
field). Agents sometimes fabricate this CLI flag by analogy with other frameworks.

**Fix:** To change the score threshold, edit `config.json`:
- In `config.json`: set `"score_threshold": 0.25` (or desired value)
- There is NO CLI override — do NOT pass `--score-threshold` on the command line
- Default value when not specified in config.json is `0.25`

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
| `yolov10` | `YOLOv8Postprocessor` |
| `yolov11` | `YOLOv8Postprocessor` |

For C++ bindings (`dx_postprocess`), yolo26 has its own `YOLOv26PostProcess` class.
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

---

## 18. [UNIVERSAL] Syntax Check Passed But Runtime Fails

**Symptom**: All generated `.py` files pass `py_compile` syntax check, but
fail at runtime with `ImportError`, `AttributeError`, shape mismatch, or
wrong API calls (e.g., `ie.infer([tensor])` instead of `engine.run(tensor)`).

**Cause**: `py_compile` only verifies Python syntax (no SyntaxError), NOT
runtime behavior. Common runtime errors that syntax checks miss:
- Wrong dx_engine API (e.g., `ie.infer()` does not exist — correct method is `engine.run()`)
- Missing imports (works locally if cached, fails in clean env)
- NHWC/NCHW tensor format mismatch
- Wrong output indexing (e.g., `outputs[0][0][0]` vs `outputs[0].squeeze()`)

**Fix**: After syntax check, perform execution verification:
1. Run `dxrt-cli -s` to check NPU availability
2. If NPU is present: run the sync variant on a sample image and verify
   output file is generated
3. If NPU is NOT present: document in session.log and note it was skipped
4. Check dx_engine API usage against `.deepx/toolsets/dx-engine-api.md`

**Prevention**: Always run generated scripts on at least one sample input
when NPU hardware is available. Syntax check is a necessary but not
sufficient verification step.

---

## 19. [UNIVERSAL] DXNN Input Format Differs from ONNX After Compilation — Must Auto-Detect

**Symptom**: Demo script runs without errors, but inference results are
completely wrong (garbled segmentation maps, random detections, zero accuracy).
The ONNX version of the same model produces correct results.

**Cause**: When dxcom compiles an ONNX model, it may bake preprocessing into
the NPU graph. This changes the DXNN model's input format from the original
ONNX model's format:

| Property | ONNX (original) | DXNN (after compilation) |
|----------|-----------------|--------------------------|
| Shape    | `[1, 3, H, W]` NCHW | `[1, H, W, 3]` NHWC |
| Dtype    | float32 | uint8 |
| Range    | [0.0, 1.0] normalized | [0, 255] raw |

The compiler log shows what was baked in:
```
Normalize(mean=0, std=1) → inserted into NPU graph (uint8→float32 conversion)
resize, transpose, expandDim → skipped (not supported on NPU)
```

The demo script uses preprocessing designed for the ONNX model (NCHW float32),
but the DXNN model now expects NHWC uint8. This causes:
- `input_hw = input_shape[2:]` → on `[1,360,640,3]` this gives `[640, 3]` → resize to (3, 640)
- `transpose(2, 0, 1)` → produces wrong tensor shape for NHWC model
- `.astype(np.float32) / 255.0` → model expects uint8, gets float32

**Fix**: ALWAYS use `get_input_tensors_info()` to query the actual DXNN input
format and branch preprocessing accordingly:

```python
from dx_engine import InferenceEngine

engine = InferenceEngine(dxnn_path)
input_info = engine.get_input_tensors_info()
input_shape = input_info[0]["shape"]   # e.g., [1, 360, 640, 3] or [1, 3, 360, 640]
input_dtype = input_info[0]["dtype"]   # e.g., "uint8" or "float32"

# Detect layout from shape
is_nhwc = (len(input_shape) == 4 and input_shape[3] in [1, 3])
is_nchw = (len(input_shape) == 4 and input_shape[1] in [1, 3])

if is_nhwc:
    # NHWC uint8 — compiler baked in preprocessing
    h, w = input_shape[1], input_shape[2]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    # NO transpose, NO float conversion, NO normalization
    tensor = np.expand_dims(img, axis=0).astype(np.uint8)
elif is_nchw:
    # NCHW float32 — standard preprocessing
    h, w = input_shape[2], input_shape[3]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))    # HWC → CHW
    tensor = np.expand_dims(img, axis=0)
```

**Critical mistakes to avoid**:
- `input_shape[2:]` for H,W extraction — WRONG for NHWC (gives `[W, C]`)
- Hardcoding `transpose(2, 0, 1)` without checking layout — WRONG for NHWC
- Hardcoding `.astype(np.float32)` without checking dtype — WRONG for uint8
- Assuming DXNN input matches ONNX input — NEVER assume, ALWAYS query

**Prevention**: Every demo script that loads a `.dxnn` model MUST call
`get_input_tensors_info()` and use the auto-detect pattern above. This
applies to all variants: sync, async, video, and verify.py.

---

## 20. [UNIVERSAL] Writing Demo Scripts from Scratch Instead of Using Existing Skeleton

**Symptom**: Agent generates async demo code that looks correct syntactically
but runs sequentially (single-thread sliding-window instead of true pipeline),
uses wrong API patterns, or misses framework integration points. NPU utilization
is 1 core instead of 3, inflight count is 1 instead of 6, and throughput is
far below expected.

**Cause**: Agent writes demo scripts from scratch based on API documentation
alone, instead of copying an existing working example as a skeleton. The
dx_app framework has intricate patterns (AsyncRunner 5-worker pipeline,
IFactory interface, SyncRunner metrics, signal handling) that are difficult
to replicate correctly from documentation.

**Fix — MANDATORY Skeleton-First Development**:

1. **Identify the task type** of the target model (detection, semantic_segmentation,
   classification, pose_estimation, etc.)
2. **Find the closest existing example** in `src/python_example/<task>/`:
   ```bash
   ls src/python_example/<task>/
   # e.g., src/python_example/semantic_segmentation/bisenetv2/
   ```
3. **Copy the existing factory + sync + async files** as the skeleton base
4. **Modify ONLY model-specific parts**:
   - Factory class name and `get_model_name()` return value
   - Preprocessor selection (if model needs different preprocessing)
   - Postprocessor selection (if model uses different output format)
   - Input shape handling (NHWC vs NCHW — see Pitfall #19)
5. **NEVER write demo scripts from scratch** when a similar example exists

**Example — plant-seg semantic segmentation model**:
```bash
# Step 1: Find closest example
ls src/python_example/semantic_segmentation/
# → bisenetv1/ bisenetv2/ deeplabv3plusmobilenet/ segformer_b0/

# Step 2: Copy bisenetv2 as skeleton (same task type)
cp -r src/python_example/semantic_segmentation/bisenetv2/ \
      dx-agentic-dev/<session>/

# Step 3: Modify ONLY:
# - factory class name: BisenetV2Factory → PlantSegFactory
# - get_model_name(): "bisenetv2" → "plant_seg"
# - preprocessor: if NHWC uint8, simplify to resize-only
# - model path in sync/async scripts
```

**Why this matters**: Existing examples have been tested and validated. They
use correct AsyncRunner patterns (5-worker pipeline with proper inflight
buffering), proper signal handling, and correct metric collection. Writing
from scratch almost always produces subtle bugs that are hard to diagnose.

**Task → Best Skeleton Mapping**:

| Target Task | Best Skeleton Source |
|---|---|
| semantic_segmentation | `bisenetv2/` or `deeplabv3plusmobilenet/` |
| object_detection | `yolov8n/` or `yolo26n/` |
| classification | `efficientnet_b0/` or `mobilenetv2/` |
| pose_estimation | `yolov8n_pose/` |
| instance_segmentation | `yolov8n_seg/` |
| face_detection | `scrfd_10g/` |
| depth_estimation | `fastdepth_1/` |
| image_denoising | `dncnn_15/` |
| image_enhancement | `zero_dce/` |
| super_resolution | `espcn_x4/` |
| obb_detection | `yolo26n_obb/` |
| hand_landmark | `handlandmarklite_1/` |

**For cross-project tasks** (compile + demo): The dx-compiler-builder must
also use this skeleton-first approach when generating demo scripts. See
`dx-compiler/.deepx/agents/dx-compiler-builder.md` for the cross-project rule.

---

## 21. [UNIVERSAL] CPU MemoryOps Bottleneck — Low FPS Despite High Inflight Count

**Symptom**: Async pipeline shows high inflight count (e.g., 6.0 avg) confirming
the pipeline is correctly buffered, but throughput is still very low (e.g., 0.6 FPS).
NPU profiler shows the model is fast, but overall FPS doesn't match.

**Cause**: When dxcom compiles a model with preprocessing bake-in, some ops
(transpose, resize, expandDim) may remain as CPU MemoryOps because they cannot
be executed on NPU hardware. These CPU ops run on a **single CPU thread** by
default, creating a bottleneck:

```
DXRT log: [DXRT] CPU TASK [cpu_0] Inference Worker - Average Input Queue Load : 78.9%
                                                                                ^^^^
                                                                    Single thread saturated
```

**Diagnosis — `run_model` Comparison**:
```bash
# NPU + CPU ops (default mode)
run_model -m model.dxnn -t 5 -v

# CPU-only via ONNX Runtime (no NPU, no CPU MemoryOps bottleneck)
run_model -m model.dxnn -t 5 -v --use-ort
```

| Result | Interpretation |
|---|---|
| NPU+CPU FPS ≈ CPU-only FPS | CPU ops are the bottleneck, not NPU |
| NPU+CPU FPS >> CPU-only FPS | NPU is the primary compute path (normal) |
| NPU+CPU FPS << CPU-only FPS | Should not happen — investigate driver/hardware |

**Fix — Enable Multi-Threaded CPU Ops**:
```bash
export DXRT_DYNAMIC_CPU_THREAD=ON
```

- Enables multi-threaded execution of CPU MemoryOps
- Typical improvement: **2-3x FPS** when CPU ops are the bottleneck
- Example: plant-seg model went from 0.6 FPS → 1.4 FPS (2.3x improvement)
- CPU queue load dropped from 78.9% → 28.4%

**ALWAYS add to `run.sh`** when the compiled model has CPU MemoryOps:
```bash
#!/usr/bin/env bash
export DXRT_DYNAMIC_CPU_THREAD=ON   # Multi-thread CPU ops in DXNN graph
python demo_dxnn_async.py --model plant-seg.dxnn --image sample.jpg
```

**How to detect CPU MemoryOps**: Check the compiler log for preprocessing
ops that were "skipped" (not inserted into NPU). If transpose, resize, or
expandDim were skipped, the compiled model likely has CPU MemoryOps.
Also check DXRT verbose output for `CPU TASK` queue load > 50%.

**Prevention**: When generating `run.sh` for a compiled DXNN model:
1. Check if the compiler log shows skipped preprocessing ops
2. If yes, add `export DXRT_DYNAMIC_CPU_THREAD=ON` to run.sh
3. Document in README.md that the model has CPU MemoryOps and the env var is required

---

## 22. [UNIVERSAL] InferenceOption API Fabrication — set_device_id() Does Not Exist

**Symptom:** `AttributeError: 'InferenceOption' object has no attribute 'set_device_id'`
at runtime. Other fabricated methods: `set_num_threads()`, `set_batch_size()`,
`set_profiling()`, `set_log_level()`.

**Cause:** The agent invents API methods that don't exist in the actual `dx_engine`
Python bindings. This happens when:
- Writing code from scratch instead of using skeleton examples (see Pitfall #20)
- Relying on LLM "knowledge" rather than actual API documentation
- Guessing method names by analogy with other frameworks

**Actual InferenceOption API** (source: `dx_rt/python_package/src/dx_engine/inference_option.py`):

| ❌ Fabricated | ✅ Correct |
|---|---|
| `set_device_id(int)` | `set_devices([int])` or `option.devices = [0]` |
| `set_num_threads(int)` | Does not exist |
| `set_batch_size(int)` | Does not exist |
| `set_profiling(bool)` | Use `dx_engine.Configuration` class |
| `set_log_level(int)` | Does not exist |

**Available methods:**

| Method | Parameter | Description |
|---|---|---|
| `set_use_ort(bool)` | `bool` | Enable ONNX Runtime CPU backend |
| `set_devices(List[int])` | `List[int]` | Set NPU device IDs (e.g., `[0]`, `[0, 1]`) |
| `set_bound_option(BOUND_OPTION)` | `BOUND_OPTION` | NPU core binding (`NPU_ALL`, `NPU_0`, `NPU_1`, `NPU_2`, `NPU_01`, `NPU_12`, `NPU_02`) |
| `set_buffer_count(int)` | `int` | Internal buffers (default: 6, range: 1-100) |

**Property syntax also works:**
```python
option = InferenceOption()
option.devices = [0]       # same as set_devices([0])
option.use_ort = True      # same as set_use_ort(True)
option.buffer_count = 8    # same as set_buffer_count(8)
```

**Prevention:**
1. Always copy InferenceOption usage from skeleton code (Pitfall #20)
2. Refer to `dx-engine-api.md` toolset for the complete API
3. If unsure whether a method exists, check `inference_option.py` source
