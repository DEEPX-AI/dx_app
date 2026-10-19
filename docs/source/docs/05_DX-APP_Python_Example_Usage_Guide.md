# DX-APP Python Usage Guide

This guide explains how to navigate and use the refactored Python example tree in DX-APP.

---

## Overview

The Python examples are located under `src/python_example/` and are organized by:  

- **task**    
- **model family**    
- **variant** (the `.dxnn` stem; each folder has sync, async, and C++ post-process entries)

All examples share a common runtime layer under `src/python_example/common/` that provides base interfaces, processors, runners, input sources, visualizers, and utilities. This is the Python counterpart of `src/cpp_example/common/` — both languages implement the same 7-module factory-based architecture. Each variant directory contains thin entry-point scripts and a factory that wires shared components together.  

Representative task directories include:  

- `image_classification/` — EfficientNet, AlexNet, ResNet, MobileNet, CasViT, etc.  
- `object_detection/` — YOLOv3/v5/v7/v8/v9/v10/v11/v12, YOLO26, YOLOX, NanoDet, DAMOYOLO, SSD. PPU variants live under families such as `yolo_ppu/`  
- `face_detection/` — SCRFD, YOLOv5Face, YOLOv7Face, RetinaFace  
- `pose_estimation/` — YOLOv8-Pose  
- `semantic_segmentation/` — BiSeNet, DeepLabV3+, SegFormer  
- `instance_segmentation/` — YOLOv8Seg, YOLOv26Seg  
- `depth_estimation/` — FastDepth, SCDepthV3  
- `face_recognition/` — ArcFace  
- `oriented_object_detection/` — YOLOv26OBB  
- `image_denoising/`, `low_light_enhancement/`, `super_resolution/`  
- `hand_landmark/` — Hand landmark estimation  
- `person_attribute/` — Attribute recognition (DeepMAR)  
- `person_reid/` — Person re-identification (RepVGG)  
- `face_landmark/` — Face alignment / 3D landmark (3DDFA v2)  

For the full repository-level structure, refer to [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md).  

---

## Architecture & Design Pattern

### Architecture Strategy

**Shared Runtime Layer** (`common/`)  

The `common/` directory is the engine behind all Python examples:  

| Module | Role |
|--------|------|
| `common/base/` | Abstract interfaces: `IFactory`, `IProcessor`, `IVisualizer`, `IInputSource` |
| `common/config/` | `ModelConfig` — loads `config.json` (input size, labels, thresholds) |
| `common/processors/` | Shared processors covering all model families |
| `common/runner/` | `SyncRunner`, `AsyncRunner`, `run_dir`, `verify_serialize`, `sr_tiling`, `args` — generic execution engines with built-in profiling |
| `common/inputs/` | Input source abstraction: image, video, camera, RTSP |
| `common/visualizers/` | Task-specific visualizers (detection, segmentation, pose, etc.) |
| `common/utility/` | Labels, preprocessing, profiling, drawing helpers |

**Factory Pattern & Model Registry**  

Each model directory has a `factory/{model}_factory.py` that implements `IFactory`:  

```python
from common.processors import YOLOv5Postprocessor
from common.visualizers import DetectionVisualizer

class Yolov9sFactory(IFactory):
    def create_postprocessor(self):
        return YOLOv5Postprocessor(self.config)
    def create_visualizer(self):
        return DetectionVisualizer(self.config)
```

The entry-point script simply delegates to the runner:  

```python
from common.runner import SyncRunner
runner = SyncRunner(factory)
runner.run()
```

Model Registry (`config/model_registry.json`)  

A JSON registry stores per-model metadata (task, postprocessor type, input dimensions, thresholds). The `scripts/add_model.sh` tool reads this registry to auto-generate factory files, config.json, and all entry-point scripts — enabling zero-code model onboarding.  

### Directory & File Pattern

Each task directory contains model families. Each variant is a subdirectory named after the `.dxnn` stem and holds that variant's factory, `config.json`, and thin entry scripts.

```text
src/python_example/object_detection/yolov8/
└── yolov8-n_640x640/
    ├── factory/
    │   └── yolov8-n_640x640_factory.py
    ├── config.json
    ├── yolov8-n_640x640_sync.py
    ├── yolov8-n_640x640_async.py
    ├── yolov8-n_640x640_sync_cpp_postprocess.py
    └── yolov8-n_640x640_async_cpp_postprocess.py
```

---

## Execution Framework

### Execution Variants

**Pure Python Flow** (`*_sync.py`, `*_async.py`)  

Use these when you want:  

- easier logic inspection — post-processing is readable Python in `common/processors/`  
- Python-first experimentation  
- simpler debugging during algorithm development  

**C++ Post-process Flow** (`*_cpp_postprocess.py`)  

Use these when you want:  

- faster post-processing — uses C++ via pybind11 (`dx_postprocess`)  
- better alignment with shared C++ decode logic  
- more realistic performance validation  

### CLI Interface

All Python examples use `argparse` via `common/runner/args.py` and share a consistent interface:  

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--model` | `-m` | string | Path to `.dxnn` model file (auto-downloaded if missing) |
| `--image` | `-i` | string | Input image file or directory |
| `--video` | `-v` | string | Input video file |
| `--camera` | `-c` | int | Camera device index |
| `--rtsp` | `-r` | string | RTSP stream URL |
| `--save` | `-s` | flag | Save rendered output to a run directory |
| `--save-dir` | — | string | Base output directory (default: `artifacts/python_example/`) |
| `--dump-tensors` | — | flag | Dump input/output tensors to `.npy` files |
| `--loop` | `-l` | int | Inference repeat count (default: 1; bare `--loop` = 2) |
| `--no-display` | — | flag | Disable visualization window |
| `--show-log` | — | flag | Enable verbose log output (default: quiet) |
| `--config` | — | string | Model config JSON path (auto-detected if omitted) |
| `--fast-postprocess` | — | flag | Opt-in faster postprocessing variant where available (standard path is the default) |
| `--output` | `-o` | string | Output file path (restoration/depth/SR only) |
| `--help` | `-h` | — | Show usage |

- **Input source:** `--image`, `--video`, `--camera`, and `--rtsp` form a mutually exclusive group. If none is specified, a **default sample image** is automatically selected based on the task type.

!!! note "NOTE"

    **Image-only tasks:** 11 tasks accept `--image` input only; `--video`, `--camera` and `--rtsp`
    are rejected at runtime with an error. The authoritative list is
    `_IMAGE_ONLY_TASKS` in `src/python_example/common/runner/sync_runner.py`.

    `embedding`, `reid`, `attribute_recognition`, `face_recognition`, `face_attribute`, `person_attribute`, `person_reid`, `image_retrieval`, `visual_place_recognition` need a crop of a pre-detected subject, or a committed gallery to rank
    against. Running a single embedding model on a raw stream without a preceding
    detector would not produce valid results.

    `object_pose_estimation`, `3d_object_detection` do not take a video frame at all: `3d_object_detection` (SFA3D) consumes a
    LiDAR point cloud (`sample/kitti/velodyne/*.bin`) and refuses any other file
    type, and DOPE scores the pose of a static object from a single image.

---

## Getting Started (Workflow)

**Step 1. Prepare assets**  

```bash
./setup.sh
```

**Step 2. Build shared libraries**  

```bash
./build.sh
```

**Step 3. Run a Python example**  

```bash
python src/python_example/object_detection/yolov9/yolov9-s_640x640/yolov9-s_640x640_sync.py --model assets/models/yolov9-s_640x640.dxnn --image sample/img/sample_kitchen.jpg
python src/python_example/object_detection/yolov9/yolov9-s_640x640/yolov9-s_640x640_async_cpp_postprocess.py --model assets/models/yolov9-s_640x640.dxnn --video assets/videos/dance-group.mov
```

---

## Advanced Operations & Debugging

### Runtime Features

**Auto-Download**

When a specified model file is not found locally, the runner automatically attempts to download it via `setup_sample_models.sh`. If a `--video` file is missing, `setup_sample_videos.sh` is invoked. If the download fails, a clear error message with manual download instructions is displayed.

**Default Input Fallback**

If no input source is provided, the runner automatically selects a default sample image appropriate for the task type (e.g., `sample/img/sample_street.jpg` for object detection). A log message indicates which default was applied:
```
[DXAPP] [INFO] No input specified. Using default sample: sample/img/sample_street.jpg
```

**Signal Handling**  

Both runners install SIGINT/SIGTERM handlers (`common/runner/interrupts.py`), mirroring the C++ runners: the first Ctrl-C or a SIGTERM stops gracefully (the video writer is finalized and the summary printed); a later Ctrl-C, more than 200 ms after the first, terminates at once; repeats within 200 ms count as one request; SIGTERM never escalates. Python runs signal handlers between bytecodes, so during one long native call the escalation waits until the call returns. The async pipeline uses a SENTINEL chain to propagate the stop through its queues.  

The handlers are installed only while a runner's `run()` executes and the previous handlers are restored when it returns, so importing the runner modules changes nothing and each `run()` starts with a fresh first request. A stop outside the stream loop (during setup or in image mode) prints `Interrupted by user.` and exits with status 0 instead of a `KeyboardInterrupt` traceback.  

**Output order**  

The async runner waits for dxrt's results in submit order (one wait worker and FIFO queues between the workers), so saved output and `DXAPP_VERIFY` records follow the input order and equal the sync runner's frame by frame. The live window also shows frames in input order, but when nothing is saved it is a best-effort preview: it draws at most about 60 frames a second and, when the window falls behind, drops the oldest waiting frame instead of slowing inference (`offer_latest` in `common/runner/async_runner.py`).  

**Output Management** (`--save`)  

When `--save` is enabled, a timestamped directory is created (e.g., `artifacts/python_example/{model}-image-{name}-{timestamp}/`) containing `run_info.txt`, saved images/video, and optional tensor dumps.

**Headless Mode**

When `DISPLAY`/`WAYLAND_DISPLAY` environment variables are absent, `cv2.imshow()` is automatically skipped. Use `--no-display` for explicit headless operation.  

**Model Configuration** (`--config`)  

Runtime parameters (thresholds, top-k, etc.) come from `<task>/<family>/<variant>/config.json` via `load_variant_config`, with alias normalization (`score_threshold` → `conf_threshold`). The thin script in that folder fixes the variant. A single-model extract may keep `config.json` beside the entry script.  

**Fast Postprocessing** (`--fast-postprocess`)  

An opt-in, performance-oriented postprocessing path. By default the runner always uses the standard postprocessor; passing `--fast-postprocess` selects a faster variant **only for the model families that provide one** — other models silently keep the standard path, so the flag is always safe to add. The optimization skips work the standard path performs *before* thresholding (e.g. running `sigmoid` over the full anchor grid, taking `argmax` over every anchor, or upsampling each mask to the full input resolution).

There are two accuracy tiers:

| Tier | Output vs. standard path | Model families |
|------|--------------------------|----------------|
| **Exact** | Byte-identical (verified by parity unit tests) | Object detection — YOLOv5 / YOLOv7, EfficientDet |
| **Approximate** | Differs only at sub-pixel mask boundaries (binary masks agree at high IoU) | Instance segmentation (YOLOv8-seg), YOLACT, SegFormer semantic segmentation |

- **Exact tier (detection):** the fast path gates anchors *before* the expensive decode. Because objectness/score gating is monotonic and the survivors are decoded with the same formulas, the detections are bit-for-bit identical to the standard path — it can be enabled with no accuracy cost.
- **Approximate tier (segmentation):** the fast path crops and resizes masks at prototype (or output) resolution instead of upsampling each mask to the full input resolution first. This changes only sub-pixel boundary interpolation; on the profiled workloads the measured speedup is roughly 2.3x for instance segmentation and 4.9x for SegFormer-style semantic segmentation (the exact figure depends on the model, resolution, and host CPU).

**When to enable.** The benefit is largest when postprocessing dominates end-to-end latency — typically high-resolution feature grids or large anchor×class counts (dense detection heads, many-class detectors, high-resolution segmentation). Profile your own workload rather than assuming a fixed factor.

**When to keep the default.** The standard path is always the default and remains the accuracy reference. For the approximate (segmentation) tier, keep the standard path when exact mask boundaries matter.

```bash
# Object detection (exact tier) — identical results, faster decode
python src/python_example/object_detection/yolov7/yolov7_640x640/yolov7_640x640_sync.py \
    --model assets/models/yolov7_640x640.dxnn --image sample/img/sample_street.jpg --fast-postprocess
```

### Verification & Diagnostics

**Numerical Verification** (`DXAPP_VERIFY`)  

Set `DXAPP_VERIFY=1` to serialize post-processing results. Each run writes `logs/verify/{model}.json` (override the directory with `DXAPP_VERIFY_DIR`) holding the last frame, and `{model}.frames.jsonl` with one record per frame (`"frame": 0, 1, …` in input order). Writes are serialized and the JSON is replaced atomically, so a reader never sees a half-written file. Every runner dumps its real results, including tiled super-resolution (statistics of the stitched output), 3D detection and YOLOPv2 (boxes plus drivable/lane mask statistics); SuperPoint records carry each frame's keypoints and the descriptor size. The `*_cpp_postprocess` scripts dump too (raw arrays as `output_stats` / `output_stats_list`, for the scripts whose post-processing returns arrays). `{model}.json` holds the last frame.  

Frame numbers count per process, so with `--loop N` on a video the records of later loops continue the numbering.  

**Tensor Dump for Debugging** (`--dump-tensors`)  

Dumps raw input/output tensors as `.npy` files. On exception, tensors and a `reason.txt` are auto-dumped for debugging.  

**Model Validation** (optional)  

```bash
# Run NPU inference for all supported models
bash scripts/validate_models.sh --lang py
```

### Environment Variables Reference

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | Save visualization to the specified file path (no `--save` required) |
| `DXAPP_VERIFY` | When `1`, dump JSON verification data |
| `DXAPP_VERIFY_DIR` | Directory for `DXAPP_VERIFY` output (default `logs/verify`) |
| `DXAPP_PROGRESS` | When `1`, print `[PROGRESS] frames=N fps=F avg_fps=A read_ms=… pre_ms=… infer_ms=… post_ms=…` once a second and at exit (front ends) |

---

## Supplementary Information

### Component Relationships

The `*_cpp_postprocess.py` variants depend on the shared Python binding exposed from `src/bindings/python/dx_postprocess/`.  

See also: [DX-APP Pybind PostProcess Overview](08_DX-APP_Pybind_PostProcess_Overview.md)  

### Developer Resources

- For contributor workflows, use [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md)
- For test execution, use [DX-APP Python Example Tests](06_DX-APP_Python_Example_Test.md)
- For repository layout details, use [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)

---
