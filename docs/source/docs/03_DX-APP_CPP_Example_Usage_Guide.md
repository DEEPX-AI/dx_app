# DX-APP C++ Usage Guide

This guide explains how to navigate and use the refactored C++ example tree in DX-APP.

---

## Overview

The C++ examples are located under `src/cpp_example/` and are organized by:

- **task**  
- **model family**  
- **execution variant**  

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
- `attribute_recognition/`  
- `reid/`  
- `face_alignment/`  

For the full repository-level structure, refer to [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md).

---

## Architecture & Design Pattern

### Architecture Strategy

**Shared Runtime Layer (`common/`)**  

The `common/` directory is the engine behind all C++ examples:

| Module | Contents | Role |
|--------|----------|------|
| `common/base/` | 4 interfaces (.hpp) | `IFactory`, `IProcessor`, `IVisualizer`, `IInputSource` |
| `common/config/` | `model_config.hpp` | Loads `config.json` (input size, labels, thresholds) |
| `common/processors/` | 44 processors | Shared decode logic for all model families |
| `common/runner/` | 24 runner headers | 12 sync + 12 async task-specific runner pairs |
| `common/inputs/` | 5 source headers | Image, Video, Camera, RTSP input abstraction |
| `common/visualizers/` | 12 visualizers | Task-specific result rendering |
| `common/utility/` | 8 utility headers | Labels, preprocessing, profiling, run_dir, signal_handler, verify_serialize |

Unlike Python's generic `SyncRunner`/`AsyncRunner`, C++ runners are **task-specific**: each task type has a dedicated sync/async pair (e.g., `sync_detection_runner.hpp`, `async_detection_runner.hpp`).

**Factory Pattern Implementation**  

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

### Directory Pattern & File Pattern 

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

## Execution Framework

### Execution Variants

**Synchronous Flow (`*_sync.cpp`)**  

Use this variant when you want:

- simpler control flow  
- easier step-by-step debugging  
- single-image or low-complexity usage examples  

 **Asynchronous Flow (`*_async.cpp`)**  

Use this variant when you want:

- higher throughput  
- better overlap of pipeline stages  
- real-time image/video processing patterns  

### CLI Interface

All C++ examples use `cxxopts` for argument parsing and share a consistent interface:

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--model_path` | `-m` | string | Path to `.dxnn` model file (auto-downloaded if missing) |
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

- **Input source:** `--image_path`, `--video_path`, `--camera_index`, and `--rtsp_url` form a mutually exclusive group. If none is specified, a **default sample image** is automatically selected based on the task type.

!!! note "NOTE"
    **Image-only tasks:** `embedding`, `reid`, and `attribute_recognition` tasks accept `--image_path` input only. `--video_path`, `--camera_index`, and `--rtsp_url` are not supported for these tasks because meaningful inference requires a crop of a pre-detected subject (face or person). Running a single embedding model on a raw video stream without a preceding detector would not produce valid results.

---

## Getting Started (Workflow)

**Step 1. Prepare assets**  

```bash
./setup.sh
```

**Step 2. Build the repository**  

```bash
./build.sh
```

**Step 3. Run a C++ example**  

```bash
./bin/yolov9s_sync -m assets/models/YoloV9S.dxnn -i sample/img/sample_kitchen.jpg
./bin/yolov9s_async -m assets/models/YoloV9S.dxnn -v assets/videos/dance-group.mov
```

---

## Advanced Operations & Debugging

### Runtime Features

**Auto-Download**

When a specified model file is not found locally, the runner automatically attempts to download it via `setup_sample_models.sh`. If a `--video` file is missing, `setup_sample_videos.sh` is invoked. If the download fails, a clear error message with manual download instructions is displayed.

**Default Input Fallback**

If no input source is provided, the runner automatically selects a default sample image appropriate for the task type (e.g., `sample/img/sample_street.jpg` for object detection). A log message indicates which default was applied.

**Signal Handling**  

All runners install SIGINT/SIGTERM handlers (`installSignalHandlers()`). Pressing Ctrl+C triggers a graceful shutdown with clean resource release.

**Output Management (`--save`)**  

When `--save` is enabled, a timestamped directory is created (e.g., `artifacts/cpp_example/{model}-image-{name}-{timestamp}/`) containing `run_info.txt`, saved images/video, and optional tensor dumps.

**Configuration Management (`--config`)**  

Runtime parameters (thresholds, top-k, etc.) can be customized per-model via `config.json`. If omitted, auto-detected adjacent to the model file.

### Verification & Diagnostics

**Numerical Verification (`DXAPP_VERIFY`)**  

Set `DXAPP_VERIFY=1` to serialize all post-processing results to `logs/verify/{model}.json`. Use `scripts/verify_inference_output.py` to validate correctness.

**Tensor Dump for Debugging (`--dump-tensors`)**  

Dumps raw input/output tensors as `.bin` files. On exception, tensors and a `reason.txt` are auto-dumped for debugging.

### Environment Variables Reference

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | Save visualization to the specified file path |
| `DXAPP_VERIFY` | When `1`, dump JSON verification data |

---

## Supplementary Information

### Component Relationships

`src/postprocess/` contains C++ post-processing implementations that are **not** used by `cpp_example/common/processors/` directly. Instead, they are consumed by the pybind11 bindings (`src/bindings/python/dx_postprocess/`) to enable `*_cpp_postprocess.py` variants in Python.  

The C++ examples rely on their own shared processors in `src/cpp_example/common/processors/`.  

See also: [DX-APP C++ Post-processing Overview](07_DX-APP_CPP_PostProcess_Overview.md)  

### Developer Resources

- For contributor workflows, use [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md)  
- For test execution, use [DX-APP C++ Example Tests](04_DX-APP_CPP_Example_Test.md)  
- For repository layout details, use [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)  

---
