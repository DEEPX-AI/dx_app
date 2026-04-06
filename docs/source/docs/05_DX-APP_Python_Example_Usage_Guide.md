# DX-APP Python Usage Guide

This guide explains how to navigate and use the refactored Python example tree in DX-APP.

---

## Overview

The Python examples are located under `src/python_example/` and are organized by:

1. **task**
2. **model family**
3. **execution and post-processing variant**

All examples share a common runtime layer under `src/python_example/common/` that provides base interfaces, processors, runners, input sources, visualizers, and utilities. This is the Python counterpart of `src/cpp_example/common/` — both languages implement the same 7-module factory-based architecture. Each model directory contains only thin entry-point scripts and a factory that wires shared components together.

Representative task directories include:

- `classification/` — EfficientNet, AlexNet, ResNet, MobileNet, etc.
- `object_detection/` — YOLOv3/v5/v7/v8/v9/v10/v11/v12, YOLO26, YOLOX, NanoDet, DAMOYOLO, SSD
- `face_detection/` — SCRFD, YOLOv5Face, YOLOv7Face, RetinaFace
- `pose_estimation/` — YOLOv8-Pose
- `semantic_segmentation/` — BiSeNet, DeepLabV3+, SegFormer
- `instance_segmentation/` — YOLOv8Seg, YOLOv26Seg
- `depth_estimation/` — FastDepth, SCDepthV3
- `embedding/` — ArcFace
- `obb_detection/` — YOLOv26OBB
- `image_denoising/`, `image_enhancement/`, `super_resolution/`
- `hand_landmark/` — Hand landmark estimation
- `ppu/` — PPU-accelerated variants (YOLOv5/v7/v8/v9/v10/v11/v12/SCRFD/Pose)
- `attribute_recognition/` — Attribute recognition (DeepMAR)
- `reid/` — Person re-identification (CasViT)

For the full repository-level structure, refer to [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md).

---

## Architecture

### Shared Runtime Layer (`common/`)

The `common/` directory is the engine behind all Python examples:

| Module | Role |
|--------|------|
| `common/base/` | Abstract interfaces: `IFactory`, `IProcessor`, `IVisualizer`, `IInputSource` |
| `common/config/` | `ModelConfig` — loads `config.json` (input size, labels, thresholds) |
| `common/processors/` | 35 shared post-processors covering all model families |
| `common/runner/` | `SyncRunner`, `AsyncRunner`, `run_dir`, `verify_serialize`, `args` — generic execution engines with built-in profiling |
| `common/inputs/` | Input source abstraction: image, video, camera, RTSP |
| `common/visualizers/` | 10 task-specific visualizers (detection, segmentation, pose, etc.) |
| `common/utility/` | Labels, preprocessing, profiling, drawing helpers |

### Factory Pattern

Each model directory has a `factory/{model}_factory.py` that implements `IFactory`:

```python
from common.processors import YOLOv5Postprocessor
from common.visualizers import DetectionVisualizer

class Yolov9sFactory(IFactory):
    def create_processor(self):
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

### Model Registry (`config/model_registry.json`)

A JSON registry stores per-model metadata (task, postprocessor type, input dimensions, thresholds). The `scripts/add_model.sh` tool reads this registry to auto-generate factory files, config.json, and all entry-point scripts — enabling zero-code model onboarding.

---

## Directory Pattern

Each model family has its own directory with a consistent structure:

```text
src/python_example/object_detection/yolov9s/
├── config.json                           # Model-specific runtime settings
├── factory/
│   └── yolov9s_factory.py                # Wires shared processor + visualizer
├── yolov9s_sync.py                       # Pure Python synchronous
├── yolov9s_async.py                      # Pure Python asynchronous
├── yolov9s_sync_cpp_postprocess.py       # Synchronous + C++ binding
└── yolov9s_async_cpp_postprocess.py      # Asynchronous + C++ binding
```

---

## Variant Selection Guide

### Pure Python variants (`*_sync.py`, `*_async.py`)

Use these when you want:

- easier logic inspection — post-processing is readable Python in `common/processors/`
- Python-first experimentation
- simpler debugging during algorithm development

### `*_cpp_postprocess.py` variants

Use these when you want:

- faster post-processing — uses C++ via pybind11 (`dx_postprocess`)
- better alignment with shared C++ decode logic
- more realistic performance validation

---

## Typical Workflow

### 1. Prepare assets

```bash
./setup.sh
```

### 2. Build shared libraries and bindings

```bash
./build.sh
```

### 3. Run a Python example

```bash
python src/python_example/object_detection/yolov9s/yolov9s_sync.py --model assets/models/YoloV9S.dxnn --image sample/img/sample_kitchen.jpg
python src/python_example/object_detection/yolov9s/yolov9s_async_cpp_postprocess.py --model assets/models/YoloV9S.dxnn --video assets/videos/dance-group.mov
```

---

## CLI Arguments

All Python examples use `argparse` via `common/runner/args.py` and share a consistent interface:

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--model` | `-m` | string (required) | Path to `.dxnn` model file |
| `--image` | `-i` | string | Input image file or directory |
| `--video` | `-v` | string | Input video file |
| `--camera` | `-c` | int | Camera device index |
| `--rtsp` | `-r` | string | RTSP stream URL |
| `--save` | `-s` | flag | Save rendered output to a run directory |
| `--save-dir` | — | string | Base output directory (default: `artifacts/python_example/`) |
| `--dump-tensors` | — | flag | Dump input/output tensors to `.npy` files |
| `--loop` | `-l` | int | Inference repeat count (default: 1; bare `--loop` = 2) |
| `--no-display` | — | flag | Disable visualization window |
| `--verbose` | — | flag | Enable verbose log output (default: quiet) |
| `--config` | — | string | Model config JSON path (auto-detected if omitted) |
| `--output` | `-o` | string | Output file path (restoration/depth/SR only) |
| `--help` | `-h` | — | Show usage |

> **Input source:** `--image`, `--video`, `--camera`, and `--rtsp` form a mutually exclusive group.

---

## Advanced Features

### Signal Handling

Both `SyncRunner` and `AsyncRunner` use `stop_event` (`threading.Event`) for graceful Ctrl+C shutdown. The async pipeline uses a SENTINEL chain to propagate stop signals through all queues.

### Run Directory (`--save`)

When `--save` is enabled, a timestamped directory is created (e.g., `artifacts/python_example/{model}-image-{name}-{timestamp}/`) containing `run_info.txt`, saved images/video, and optional tensor dumps.

### Numerical Verification (`DXAPP_VERIFY`)

Set `DXAPP_VERIFY=1` to serialize all post-processing results to `logs/verify/{model}.json`. Use `scripts/verify_inference_output.py` to validate correctness against task-specific rules.

### Tensor Dump (`--dump-tensors`)

Dumps raw input/output tensors as `.npy` files. On exception, tensors and a `reason.txt` are auto-dumped for debugging.

### Model Config (`--config`)

Runtime parameters (thresholds, top-k, etc.) are loaded via `_FactoryConfigMixin` with alias normalization (`score_threshold` → `conf_threshold`). If omitted, `config.json` is auto-detected adjacent to the model or script.

### Headless Mode

When `DISPLAY`/`WAYLAND_DISPLAY` environment variables are absent, `cv2.imshow()` is automatically skipped. Use `--no-display` for explicit headless operation.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | Save visualization to the specified file path (no `--save` required) |
| `DXAPP_VERIFY` | When `1`, dump JSON verification data |

### 4. Validate all models (optional)

```bash
# Run NPU inference + numerical verification for all supported models
bash scripts/validate_models.sh --numerical --lang py
```

---

## Relationship to `dx_postprocess`

The `*_cpp_postprocess.py` variants depend on the shared Python binding exposed from `src/bindings/python/dx_postprocess/`.

See also:

- [DX-APP Pybind PostProcess Overview](08_DX-APP_Pybind_PostProcess_Overview.md)

---

## Developer Notes

If you are modifying or adding Python examples, also review:

- [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md) — model onboarding, validation, benchmarking
- [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md) — detailed `common/` architecture
- [DX-APP Python Example Tests](06_DX-APP_Python_Example_Test.md) — test framework

---

## Next Steps

- For contributor workflows, use [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md)
- For test execution, use [DX-APP Python Example Tests](06_DX-APP_Python_Example_Test.md)
- For repository layout details, use [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)
