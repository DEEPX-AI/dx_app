# DX-APP Example Source Structure

This document is intended for contributors and maintainers who need to understand or extend the refactored DX-APP example layout.

---

## Purpose of the `src/` Layout

The `src/` directory is the implementation root for DX-APP examples and shared runtime logic.

Its structure is designed to:

- separate C++ and Python example implementations clearly
- organize examples by **AI task** and then by **model family**
- share post-processing logic across languages
- make onboarding of new models predictable
- support testing, validation, and benchmarking workflows

---

## Top-Level Structure

```text
src/
├── cpp_example/          # C++ examples organized by task/model
│   └── common/           # ← C++ shared runtime layer
├── python_example/       # Python examples organized by task/model
│   └── common/           # ← Python shared runtime layer
├── postprocess/          # C++ post-processing (consumed by pybind11 bindings)
├── bindings/             # pybind11 bridge exposing src/postprocess/ to Python
└── utility/              # Shared utility code used by build flow
```

### Directory roles

- **`src/cpp_example/`**: end-to-end C++ example applications — each model directory contains thin entry-point files that delegate to the shared `common/` layer
- **`src/cpp_example/common/`**: shared C++ runtime layer — base interfaces (4 hpp), processors (44 hpp), runners (24 hpp), visualizers (12 hpp), input sources (5 hpp), config (1 hpp), and utilities (8 hpp)
- **`src/python_example/`**: end-to-end Python example applications — same factory-based delegation pattern as C++
- **`src/python_example/common/`**: shared Python runtime layer — base interfaces (4 py), processors (35 py), runners (5 py), visualizers (10 py), input sources (5 py), config (1 py), and utilities (7 py)
- **`src/postprocess/`**: C++ post-processing libraries consumed by the **pybind11 bindings** (`src/bindings/`). This enables `*_cpp_postprocess` variants to use C++ decode logic from Python.
- **`src/bindings/`**: pybind11 bridge exposing `src/postprocess/` to Python as the `dx_postprocess` package
- **`src/utility/`**: common support code shared across the build flow

Both `cpp_example/common/` and `python_example/common/` share the same 7-module structure: `base/`, `config/`, `processors/`, `runner/`, `inputs/`, `visualizers/`, `utility/`. They are language-specific implementations of the same architectural pattern.

---

## Task-First Organization

Both `cpp_example/` and `python_example/` follow the same task-first structure.

Representative task directories include:

- `classification/`
- `object_detection/`
- `face_detection/`
- `pose_estimation/`
- `semantic_segmentation/`
- `instance_segmentation/`
- `depth_estimation/`
- `embedding/`
- `image_denoising/`
- `image_enhancement/`
- `super_resolution/`
- `obb_detection/`
- `ppu/`
- `hand_landmark/`
- `attribute_recognition/`
- `reid/`

This makes it easy to answer both questions below:

- what task does this example belong to?
- where should a new model example be added?

---

## Model-First Subdirectories Inside Each Task

Within each task directory, examples are split again by model family.

Examples:

```text
src/cpp_example/object_detection/yolov9s/
src/cpp_example/object_detection/yolov8/
src/python_example/object_detection/yolov9s/
src/python_example/object_detection/ssdmv1/
```

This gives each model family its own isolated workspace for:

- config files
- source files
- factory helpers
- model-specific wiring

---

## Variant Naming Rules

### C++ variants

A typical C++ model directory contains:

```text
config.json
factory/
<model>_sync.cpp
<model>_async.cpp
```

Example:

```text
src/cpp_example/object_detection/yolov9s/
├── config.json
├── factory/
├── yolov9s_sync.cpp
└── yolov9s_async.cpp
```

Common C++ variant patterns:

- `*_sync.cpp`: sequential execution path
- `*_async.cpp`: pipelined or threaded execution path
- task/model-specific additional variants when required

### Python variants

A typical Python model directory contains:

```text
config.json
factory/
<model>_sync.py
<model>_async.py
<model>_sync_cpp_postprocess.py
<model>_async_cpp_postprocess.py
```

Example:

```text
src/python_example/object_detection/yolov9s/
├── config.json
├── factory/
├── yolov9s_sync.py
├── yolov9s_async.py
├── yolov9s_sync_cpp_postprocess.py
└── yolov9s_async_cpp_postprocess.py
```

Common Python variant patterns:

- `*_sync.py`: Python-only synchronous path
- `*_async.py`: Python-only asynchronous path
- `*_sync_cpp_postprocess.py`: synchronous path using shared C++ post-processing bindings
- `*_async_cpp_postprocess.py`: asynchronous path using shared C++ post-processing bindings

---

## Shared Post-Processing Layer

### C++ and Python Shared Runtime (`common/`)

Both `cpp_example/` and `python_example/` contain a `common/` directory with the same 7-module architecture:

| Module | C++ | Python | Role |
|--------|-----|--------|------|
| `base/` | 4 interfaces (.hpp) | 4 interfaces (.py) | `IFactory`, `IProcessor`, `IVisualizer`, `IInputSource` |
| `config/` | `model_config.hpp` | `model_config.py` | Loads `config.json` (input size, labels, thresholds) |
| `processors/` | 44 header files | 35 Python files | Shared processors for all model families |
| `runner/` | 24 runner headers | 5 runner files | Sync/Async execution engines with profiling |
| `inputs/` | 5 source headers | 5 source files | Image, Video, Camera, RTSP input abstraction |
| `visualizers/` | 12 visualizer headers | 10 visualizer files | Task-specific result rendering |
| `utility/` | 8 utility headers | 7 utility files | Labels, preprocessing, profiling, drawing, run_dir, queue, verify |

This is the core architectural pattern of DX-APP: model directories are thin wrappers (factory + entry points) that delegate all heavy logic to their respective `common/` layer.

### Pybind11 Post-Processing Bridge (`src/postprocess/` + `src/bindings/`)

`src/postprocess/` contains C++ post-processing implementations that are **not** used by `cpp_example/common/processors/` directly. Instead, they are consumed by the pybind11 bindings under `src/bindings/python/dx_postprocess/`.

This bridge enables the `*_cpp_postprocess.py` Python variants to use C++ decode logic for higher performance, while the pure Python variants (`*_sync.py`, `*_async.py`) use `python_example/common/processors/` natively.

| Variant | Post-processing source |
|---------|----------------------|
| `*_sync.cpp` / `*_async.cpp` | `cpp_example/common/processors/` |
| `*_sync.py` / `*_async.py` | `python_example/common/processors/` |
| `*_sync_cpp_postprocess.py` / `*_async_cpp_postprocess.py` | `src/postprocess/` via pybind11 |

---

## Shared Runtime Layer (`common/`)

Both `src/cpp_example/common/` and `src/python_example/common/` implement the same shared runtime architecture. Each provides base interfaces, configuration loading, input sources, processors, runners, visualizers, and utilities — so individual model directories only need thin entry points and a factory.

### C++ Common Structure

```text
src/cpp_example/common/
├── base/                          # Abstract interfaces (.hpp)
│   ├── i_factory.hpp              #   IFactory — assembles processor + visualizer + runner
│   ├── i_processor.hpp            #   IProcessor — postprocess(outputs, meta) → results
│   ├── i_visualizer.hpp           #   IVisualizer — draw(frame, results) → frame
│   └── i_input_source.hpp         #   IInputSource — image/video/camera/RTSP abstraction
├── config/
│   └── model_config.hpp           # ModelConfig — loads config.json
├── processors/                    # 44 shared processors
│   ├── yolov5_postprocessor.hpp
│   ├── yolov8_postprocessor.hpp
│   ├── scrfd_postprocessor.hpp
│   ├── nanodet_postprocessor.hpp
│   ├── damoyolo_postprocessor.hpp
│   ├── ssd_postprocessor.hpp
│   ├── segmentation_postprocessor.hpp
│   ├── instance_seg_postprocessor.hpp
│   ├── depth_postprocessor.hpp
│   └── ...
├── runner/                        # 24 task-specific runner pairs
│   ├── sync_detection_runner.hpp  #   Sequential detection loop
│   ├── async_detection_runner.hpp #   Pipelined detection loop
│   ├── sync_classification_runner.hpp
│   ├── async_classification_runner.hpp
│   ├── sync_segmentation_runner.hpp
│   ├── async_segmentation_runner.hpp
│   └── ...                        #   12 sync + 12 async = 24 total
├── inputs/                        # 5 input source headers
│   ├── camera_source.hpp
│   ├── image_source.hpp
│   ├── video_source.hpp
│   ├── rtsp_source.hpp
│   └── input_factory.hpp
├── visualizers/                   # 12 task-specific visualizers
│   ├── detection_visualizer.hpp
│   ├── classification_visualizer.hpp
│   ├── segmentation_visualizer.hpp
│   ├── face_visualizer.hpp
│   ├── pose_visualizer.hpp
│   └── ...
└── utility/                       # 8 utility headers
    ├── common_util.hpp
    ├── labels.hpp
    ├── preprocessing.hpp
    ├── profiling.hpp
    ├── run_dir.hpp
    ├── safe_queue.hpp
    ├── verify_serialize.hpp
    └── visualization.hpp
```

C++ runners are **task-specific**: each task type has a dedicated sync/async runner pair (e.g., `sync_detection_runner`, `async_detection_runner`). This provides optimal performance for each task's specific data flow.

### Python Common Structure

```text
src/python_example/common/
├── base/                  # Abstract interfaces (.py)
│   ├── i_factory.py       #   IFactory — assembles processor + visualizer + runner
│   ├── i_processor.py     #   IProcessor — postprocess(outputs, meta) → results
│   ├── i_visualizer.py    #   IVisualizer — draw(frame, results) → frame
│   └── i_input_source.py  #   IInputSource — image/video/camera/RTSP abstraction
├── config/
│   └── model_config.py    # ModelConfig — loads config.json (input size, labels, thresholds)
├── processors/            # 35 shared processors
│   ├── yolo_postprocessor.py           # YOLOv5/v7/v8/v9/v10/v11/v12/YOLOX
│   ├── face_postprocessor.py           # SCRFD, YOLOv5Face, YOLOv7Face
│   ├── segmentation_postprocessor.py   # BiSeNet, DeepLabV3+, SegFormer
│   ├── instance_seg_postprocessor.py   # YOLOv8Seg, YOLOv26Seg
│   ├── obb_postprocessor.py            # YOLOv26OBB
│   ├── pose_postprocessor.py           # YOLOv8-Pose
│   ├── ppu_postprocessor.py            # PPU variants (YOLOv5/v7/SCRFD/Pose)
│   ├── classification_postprocessor.py # EfficientNet, AlexNet, etc.
│   ├── depth_postprocessor.py          # FastDepth, MiDaS
│   ├── nanodet_postprocessor.py        # NanoDet
│   ├── ssd_postprocessor.py            # SSD MobileNet
│   ├── damoyolo_postprocessor.py       # DAMOYOLO
│   ├── embedding_postprocessor.py      # CLIP, ArcFace
│   ├── restoration_postprocessor.py    # DnCNN, Zero-DCE
│   ├── nms_utils.py                    # Shared NMS / box utilities
│   ├── letterbox_preprocessor.py       # Shared letterbox preprocessing
│   └── ...
├── runner/                # 5 generic runner files
│   ├── sync_runner.py     # SyncRunner — sequential Pre→Infer→Post→Display loop
│   ├── async_runner.py    # AsyncRunner — pipelined multi-thread runner
│   ├── args.py            # Unified CLI argument parser (--model, --image, --video, etc.)
│   ├── run_dir.py         # Directory-based batch runner
│   └── verify_serialize.py # Serialize results to JSON for numerical verification
├── inputs/                # 5 input source files
│   ├── image_source.py
│   ├── video_source.py
│   ├── camera_source.py
│   ├── rtsp_source.py
│   └── input_factory.py
├── visualizers/           # 10 task-specific visualizers
│   ├── detection_visualizer.py
│   ├── classification_visualizer.py
│   ├── segmentation_visualizer.py
│   ├── face_visualizer.py
│   ├── pose_visualizer.py
│   ├── instance_seg_visualizer.py
│   ├── obb_visualizer.py
│   └── ...
└── utility/                       # 7 utility files
    ├── common_util.py     # General utilities
    ├── labels.py          # COCO / ImageNet label constants
    ├── preprocessing.py   # Shared resize/normalize/letterbox
    ├── profiling.py       # Stage-wise latency profiler
    ├── safe_queue.py      # Thread-safe queue for async pipeline
    ├── skeleton.py        # Pose skeleton definitions
    └── visualization.py   # Drawing helpers (boxes, text, masks)
```

Python runners are **generic**: `SyncRunner` and `AsyncRunner` work for all task types via the factory pattern. This provides simplicity and uniform usage across all models.

### Key Difference: C++ vs Python Runners

| Aspect | C++ (`cpp_example/common/runner/`) | Python (`python_example/common/runner/`) |
|--------|-----------------------------------|-----------------------------------------|
| Runner count | 24 (12 sync + 12 async) | 5 (2 runners + args + run_dir + verify) |
| Dispatch | Task-specific runner per category | Generic runner for all tasks |
| Example | `sync_detection_runner.hpp` | `sync_runner.py` |

### How Model Directories Connect to `common/`

#### C++ model directory

```text
src/cpp_example/object_detection/yolov9s/
├── config.json
├── factory/
│   └── yolov9s_factory.hpp      # Assembles processor + visualizer from common/
├── yolov9s_sync.cpp              # Entry point → sync_detection_runner
└── yolov9s_async.cpp             # Entry point → async_detection_runner
```

#### Python model directory

```text
src/python_example/object_detection/yolov9s/
├── config.json
├── factory/
│   └── yolov9s_factory.py                # Assembles processor + visualizer from common/
├── yolov9s_sync.py                       # Entry point → SyncRunner
├── yolov9s_async.py                      # Entry point → AsyncRunner
├── yolov9s_sync_cpp_postprocess.py       # Entry point → SyncRunner + C++ binding
└── yolov9s_async_cpp_postprocess.py      # Entry point → AsyncRunner + C++ binding
```

The factory imports shared components from `common/`:

```cpp
// C++ factory example
#include "common/processors/yolov8_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
```

```python
# Python factory example
from common.processors import YOLOv5Postprocessor
from common.visualizers import DetectionVisualizer
```

The entry-point script/program delegates to the runner:

```python
# Python
from common.runner import SyncRunner
runner = SyncRunner(factory)
runner.run()
```

This means adding a new model typically requires only a `config.json` and a factory file — the shared infrastructure handles everything else.

---

## Model Registry and Auto-Generation

### `config/model_registry.json`

The model registry is a JSON array that serves as the single source of truth for model metadata:

```json
{
  "model_name": "yolov9s",
  "dxnn_file": "YoloV9S.dxnn",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "input_width": 640,
  "input_height": 640,
  "config": { "conf_threshold": 0.5, "num_classes": 80 },
  "supported": true
}
```

### `scripts/add_model.sh`

The `add_model.sh` script reads an entry from the registry and auto-generates:

- `config.json` with correct input dimensions and thresholds
- `factory/{model}_factory.py` wired to the correct processor and visualizer
- 4 entry-point scripts (sync/async × python/cpp_postprocess)
- C++ equivalents under `src/cpp_example/`

This enables onboarding a new model with zero manual code in most cases.

---

## Numerical Verification Framework

DX-APP includes an automated verification pipeline that validates model outputs after inference:

- **`scripts/validate_models.sh --numerical`**: runs all supported models through NPU inference and checks output correctness
- **`scripts/verify_inference_output.py`**: task-specific validators (14 types) that check bounding boxes, class IDs, confidence ranges, segmentation masks, depth maps, etc.
- **`scripts/inference_verify_rules.json`**: configurable thresholds per task type
- **`common/runner/verify_serialize.py`**: serializes postprocess results to JSON for comparison

This framework catches regressions such as broken post-processing, incorrect model configurations, or NPU output changes.

---

## How This Structure Connects to Tests

The source tree and the test tree are related, but not identical.

### Shared Test Infrastructure (`tests/common/`)

`tests/common/` provides shared constants and utilities used by both C++ and Python test suites:

- `constants.py` — paths, timeout values, suffix patterns (e.g., `_sync`, `_async`)
- `utils.py` — helper functions for executable/script discovery, process execution, result validation

### C++ tests

C++ tests are organized into four categories:

| Category | File | Description |
|----------|------|-------------|
| CLI Help | `test_cli_help.py` | Validates `--help` output for all executables |
| CLI Basic | `test_cli_basic.py` | Validates `--model` argument handling |
| E2E | `test_e2e.py` | End-to-end inference with image/video inputs |
| Visualization | `test_visualization.py` | Validates output image generation with `--save` |
| Feature: Save Mode | `test_save_mode.py` | Tests `--save` / `--save-dir` behavior |
| Feature: Dump Tensors | `test_dump_tensors.py` | Tests `--dump-tensors` output |
| Feature: Verify | `test_verify.py` | Tests `DXAPP_VERIFY` environment variable |
| Feature: Multi-Loop | `test_multi_loop.py` | Tests `--loop` repeated execution |
| Feature: Signal | `test_signal_handling.py` | Tests SIGINT/SIGTERM graceful shutdown |

All tests auto-discover executables from `bin/` using `tests/common/utils.py`.

### Python tests

- Python example tests are driven by centralized configuration under `tests/python_example/framework/`
- `test_visualization.py` validates output image generation across all task types
- adding a new source directory alone does **not** automatically guarantee full test coverage

Relevant files include:

- `tests/python_example/framework/config.py`
- `tests/python_example/framework/performance_collector.py`
- `tests/python_example/test_visualization.py`
- `tests/python_example/<task>/` (14 task directories)
- `config/test_models.conf`

### Model validation

- `scripts/validate_models.sh` runs registry-driven validation across all supported models
- `scripts/validate_models.sh --numerical` additionally performs numerical verification using `verify_inference_output.py`
- `config/model_registry.json` is the primary reference for which models are validated

This means source layout, test coverage, and registry entries must be updated together when onboarding new examples.

---

## Contributor Rules for Adding a New Example

When adding a new example, keep the following rules:

1. choose the correct task directory first
2. create a dedicated model directory under that task
3. follow existing variant naming conventions
4. place shared decode logic in `src/cpp_example/common/processors/` (C++) or `src/python_example/common/processors/` (Python) when appropriate
5. add pybind11 bindings in `src/postprocess/` + `src/bindings/` only if the new flow needs `*_cpp_postprocess.py` variants
6. register the model in `config/model_registry.json` if it should be part of the standard validation flow
7. update validation/test registration where required

### Contributor checklist

When a new example is intended to become part of the maintained repository flow, confirm all of the following:

- source files are placed under the correct task/model directory
- naming follows the current variant convention
- required shared post-processing logic exists or is added under `src/cpp_example/common/processors/` (C++) or `src/python_example/common/processors/` (Python)
- model assets can be prepared through the standard setup flow
- validation passes
- test registration is updated if automated coverage is required
- user-facing or contributor-facing docs are updated when the structure meaningfully changes

---

## Recommended Onboarding Flow

When you add or refactor a model example:

1. create or update the source layout under `src/`
2. prepare the required model assets
3. build the repository
4. validate the example structure
5. run relevant tests

Typical related commands:

```bash
./setup.sh
./build.sh --clean
./scripts/dx_tool.sh validate
./scripts/validate_models.sh --numerical --lang py
./run_tc.sh --cpp --cli
./run_tc.sh --python
```

If the example is intended to be part of the standard repository workflow, also review the relevant documentation pages so that user and contributor guidance stays aligned with the current structure.

---

## What This Structure Replaces

The refactored source layout replaces the older demo-centric organization and establishes `src/cpp_example/` and `src/python_example/` as the primary example roots.

For current contributor work, treat the `src/` tree as the canonical location for example development.

---

## See Also

- `src/cpp_example/`
- `src/cpp_example/common/`
- `src/python_example/`
- `src/python_example/common/`
- `src/postprocess/` (pybind11 source)
- `src/bindings/python/dx_postprocess/`
- `config/model_registry.json`
- `scripts/validate_models.sh`
- `scripts/verify_inference_output.py`
- `tests/cpp_example/`
- `tests/python_example/`
- `docs/10_DX-APP_DX-Tool_Guide.md`
