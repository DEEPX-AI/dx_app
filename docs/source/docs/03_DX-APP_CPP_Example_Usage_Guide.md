# DX-APP C++ Usage Guide

This guide explains how to navigate and use the refactored C++ example tree in DX-APP.

---

## Overview

The C++ examples are located under `src/cpp_example/` and are organized by:

1. **task**
2. **model family**
3. **execution variant**

All examples share a common runtime layer under `src/cpp_example/common/` that provides base interfaces, processors, runners, input sources, visualizers, and utilities. This is the C++ counterpart of `src/python_example/common/` — both languages implement the same 7-module factory-based architecture. Each model directory contains thin entry-point source files and a factory that wires shared components together.

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
- `hand_landmark/`
- `obb_detection/`
- `ppu/`

For the full repository-level structure, refer to [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md).

---

## Architecture

### Shared Runtime Layer (`common/`)

The `common/` directory is the engine behind all C++ examples:

| Module | Contents | Role |
|--------|----------|------|
| `common/base/` | 4 interfaces (.hpp) | `IFactory`, `IProcessor`, `IVisualizer`, `IInputSource` |
| `common/config/` | `model_config.hpp` | Loads `config.json` (input size, labels, thresholds) |
| `common/processors/` | 42 post-processors | Shared decode logic for all model families |
| `common/runner/` | 24 runner headers | 12 sync + 12 async task-specific runner pairs |
| `common/inputs/` | 5 source headers | Image, Video, Camera, RTSP input abstraction |
| `common/visualizers/` | 11 visualizers | Task-specific result rendering |
| `common/utility/` | 8 utility headers | Labels, preprocessing, profiling, run_dir, signal_handler, verify_serialize |

Unlike Python's generic `SyncRunner`/`AsyncRunner`, C++ runners are **task-specific**: each task type has a dedicated sync/async pair (e.g., `sync_detection_runner.hpp`, `async_detection_runner.hpp`).

### Factory Pattern

Each model directory has a `factory/{model}_factory.hpp` that implements `IFactory`:

```cpp
// factory/yolov9s_factory.hpp
#include "common/processors/yolov8_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"

class YOLOv9sFactory : public IFactory {
    IProcessor* createProcessor() override { return new YOLOv8Postprocessor(config); }
    IVisualizer* createVisualizer() override { return new DetectionVisualizer(); }
};
```

The entry-point delegates to a task-specific runner:

```cpp
// yolov9s_sync.cpp
auto factory = YOLOv9sFactory(config);
SyncDetectionRunner runner(factory);
runner.run();
```

---

## Directory Pattern

Each model family usually has its own directory.

Example:

```text
src/cpp_example/object_detection/yolov9s/
├── config.json
├── factory/
│   └── yolov9s_factory.hpp    # Assembles processor + visualizer from common/
├── yolov9s_sync.cpp            # Entry point → sync_detection_runner
└── yolov9s_async.cpp           # Entry point → async_detection_runner
```

Common files:

- `config.json`: model-specific runtime settings
- `factory/`: factory header wiring shared `common/` components
- `*_sync.cpp`: synchronous execution example (via task-specific sync runner)
- `*_async.cpp`: asynchronous execution example (via task-specific async runner)

---

## Execution Variants

### `*_sync.cpp`

Use this variant when you want:

- simpler control flow
- easier step-by-step debugging
- single-image or low-complexity usage examples

### `*_async.cpp`

Use this variant when you want:

- higher throughput
- better overlap of pipeline stages
- real-time image/video processing patterns

---

## Typical Workflow

### 1. Prepare assets

```bash
./setup.sh
```

### 2. Build the repository

```bash
./build.sh
```

### 3. Run a C++ example

```bash
./bin/yolov9s_sync -m assets/models/YoloV9S.dxnn -i sample/img/sample_kitchen.jpg
./bin/yolov9s_async -m assets/models/YoloV9S.dxnn -v assets/videos/dance-group.mov
```

---

## CLI Arguments

All C++ examples use `cxxopts` for argument parsing and share a consistent interface:

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--model_path` | `-m` | string (required) | Path to `.dxnn` model file |
| `--image_path` | `-i` | string | Input image file or directory |
| `--video_path` | `-v` | string | Input video file |
| `--camera_index` | `-c` | int | Camera device index |
| `--rtsp_url` | `-r` | string | RTSP stream URL |
| `--save` | `-s` | bool | Save rendered output to a run directory |
| `--save-dir` | — | string | Base output directory (default: `artifacts/cpp_example`) |
| `--dump-tensors` | — | bool | Dump input/output tensors to `.bin` files |
| `--loop` | `-l` | int | Inference repeat count (default: auto) |
| `--no-display` | — | bool | Disable visualization window, output FPS only |
| `--show-log` | — | bool | Enable verbose log output (default: quiet) |
| `--config` | — | string | Model config JSON path (auto-detected if omitted) |
| `--help` | `-h` | — | Show usage |

> **Input source:** Exactly one of `--image_path`, `--video_path`, `--camera_index`, or `--rtsp_url` must be specified.

---

## Advanced Features

### Signal Handling

All runners install SIGINT/SIGTERM handlers (`installSignalHandlers()`). Pressing Ctrl+C triggers a graceful shutdown with clean resource release.

### Run Directory (`--save`)

When `--save` is enabled, a timestamped directory is created (e.g., `artifacts/cpp_example/{model}-image-{name}-{timestamp}/`) containing `run_info.txt`, saved images/video, and optional tensor dumps.

### Numerical Verification (`DXAPP_VERIFY`)

Set `DXAPP_VERIFY=1` to serialize all post-processing results to `logs/verify/{model}.json`. Use `scripts/verify_inference_output.py` to validate correctness.

### Tensor Dump (`--dump-tensors`)

Dumps raw input/output tensors as `.bin` files. On exception, tensors and a `reason.txt` are auto-dumped for debugging.

### Model Config (`--config`)

Runtime parameters (thresholds, top-k, etc.) can be customized per-model via `config.json`. If omitted, auto-detected adjacent to the model file.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | Save visualization to the specified file path |
| `DXAPP_VERIFY` | When `1`, dump JSON verification data |

---

## Relationship to `src/postprocess/` and Pybind11

`src/postprocess/` contains C++ post-processing implementations that are **not** used by `cpp_example/common/processors/` directly. Instead, they are consumed by the pybind11 bindings (`src/bindings/python/dx_postprocess/`) to enable `*_cpp_postprocess.py` variants in Python.

The C++ examples rely on their own shared processors in `src/cpp_example/common/processors/`.

See also:

- [DX-APP C++ Post-processing Overview](07_DX-APP_CPP_PostProcess_Overview.md)

---

## Developer Notes

If you are modifying or adding C++ examples, also review:

- [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md)
- [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)
- [DX-APP C++ Example Tests](04_DX-APP_CPP_Example_Test.md)

---

## Next Steps

- For contributor workflows, use [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md)
- For test execution, use [DX-APP C++ Example Tests](04_DX-APP_CPP_Example_Test.md)
- For repository layout details, use [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)
