# DX-APP C++ Usage Guide

This guide explains how to navigate and use the refactored C++ example tree in DX-APP.

---

## Overview

The C++ examples are located under `src/cpp_example/` and are organized by:

- **task**  
- **model family**  
- **variant** (the `.dxnn` stem; sync and async are separate entry files inside that folder)

All examples share a common runtime layer under `src/cpp_example/common/` that provides base interfaces, processors, runners, input sources, visualizers, and utilities. This is the C++ counterpart of `src/python_example/common/` — both languages implement the same 7-module factory-based architecture. Each variant directory contains thin entry-point source files and a factory that wires shared components together.  

Representative task directories include:

- `image_classification/`  
- `object_detection/` — PPU models are a family here, for example `yolo_ppu/`  
- `face_detection/`  
- `pose_estimation/`  
- `semantic_segmentation/`  
- `instance_segmentation/`  
- `depth_estimation/`  
- `face_recognition/`  
- `image_denoising/`  
- `low_light_enhancement/`  
- `super_resolution/`  
- `hand_landmark/`  
- `oriented_object_detection/`  
- `person_attribute/`  
- `person_reid/`  
- `face_landmark/`  

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
| `common/processors/` | Postprocessor / preprocessor headers | Shared decode logic for all model families |
| `common/runner/` | Runner headers | A dedicated sync/async runner pair per task type |
| `common/inputs/` | 5 source headers | Image, Video, Camera, RTSP input abstraction |
| `common/visualizers/` | Visualizer headers | Task-specific result rendering |
| `common/utility/` | Utility headers | Labels, preprocessing, profiling, run_dir, colorspace, verify_serialize |

Unlike Python's generic `SyncRunner`/`AsyncRunner`, C++ runners are **task-specific**: each task type has a dedicated sync/async pair (e.g., `sync_detection_runner.hpp`, `async_detection_runner.hpp`).

**Factory Pattern Implementation**  

Each variant directory has a `factory/<variant>_factory.hpp` that implements the task factory:

```cpp
// object_detection/yolov9/yolov9-s_640x640/yolov9-s_640x640_sync.cpp
auto factory = std::make_unique<dxapp::v_yolov9_s_640x640::Yolov9Factory>();
dxapp::SyncDetectionRunner<dxapp::v_yolov9_s_640x640::Yolov9Factory> runner(std::move(factory));
return runner.run(argc, argv);
```

### Directory Pattern & File Pattern 

Each task directory contains model families. Each variant directory is named after the `.dxnn` stem and holds that variant's factory, `config.json`, and sync/async entries.

Example:

```text
src/cpp_example/object_detection/yolov9/
└── yolov9-s_640x640/
    ├── factory/
    │   └── yolov9-s_640x640_factory.hpp
    ├── yolov9-s_640x640_sync.cpp
    ├── yolov9-s_640x640_async.cpp
    └── config.json
```

The CMake target and `bin/` executable use the same stem: `yolov9-s_640x640_sync`.

Common files:

- `<task>/<family>/<variant>/config.json`: that variant's runtime settings
- `<variant>/factory/<variant>_factory.hpp`: factory header wiring shared `common/` components
- `<variant>_sync.cpp`: synchronous entry for that variant
- `<variant>_async.cpp`: asynchronous entry for that variant  

---

## Execution Framework

### Execution Variants

**Synchronous Flow (`xx_sync.cpp`)**  

Use this variant when you want:

- simpler control flow  
- easier step-by-step debugging  
- single-image or low-complexity usage examples  

**Asynchronous Flow (`xx_async.cpp`)**  

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
| `--drop-frames` | — | bool | Preview drops stale frames instead of pacing the pipeline (default: off, every frame is shown) |
| `--show-log` | — | bool | Enable verbose log output (default: quiet) |
| `--config` | — | string | Model config JSON path (auto-detected if omitted) |
| `--help` | `-h` | — | Show usage |

- **Input source:** `--image_path`, `--video_path`, `--camera_index`, and `--rtsp_url` form a mutually exclusive group. If none is specified, a **default sample image** is automatically selected based on the task type.
- **Raspberry Pi camera:** on a Raspberry Pi, `--camera_index N` opens the CSI camera through a libcamera GStreamer pipeline (`libcamerasrc`) and camera `N` is camera `N` of `rpicam-hello --list-cameras`; a USB webcam node keeps the V4L2 path. `install.sh` installs `gstreamer1.0-libcamera` when it detects a Pi camera.

!!! note "NOTE"
    **Image-only tasks:** 11 tasks accept `--image_path` input only; `--video_path`, `--camera_index` and `--rtsp_url`
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

**Step 2. Build the repository**  

```bash
./build.sh
```

**Step 3. Run a C++ example**  

```bash
./bin/yolov9-s_640x640_sync -m assets/models/yolov9-s_640x640.dxnn -i sample/img/sample_kitchen.jpg
./bin/yolov9-s_640x640_async -m assets/models/yolov9-s_640x640.dxnn -v assets/videos/dance-group.mov
```

---

## Advanced Operations & Debugging

### Runtime Features

**Auto-Download**

When a specified model file is not found locally, the runner automatically attempts to download it via `setup_sample_models.sh`. If a `--video` file is missing, `setup_sample_videos.sh` is invoked. If the download fails, a clear error message with manual download instructions is displayed.

**Default Input Fallback**

If no input source is provided, the runner automatically selects a default sample image appropriate for the task type (e.g., `sample/img/sample_street.jpg` for object detection). A log message indicates which default was applied.

**Signal Handling**  

All C++ binaries - every single-model runner (`installSignalHandlers()`) and `multi_model_graph_sync`/`multi_model_graph_async` - install SIGINT/SIGTERM handlers. Pressing Ctrl+C triggers a graceful shutdown with clean resource release (the graph CLI finishes the frames in flight in every stream and finalizes `--report`/`--output`/`--events`). A later Ctrl+C, more than 200 ms after the first, terminates the process at once (for a shutdown that is itself stuck); repeats within 200 ms count as the same request. SIGTERM is always graceful, however often it arrives.

**Preview window.** The window opens at half the screen size with the frame's aspect ratio kept (`DXAPP_SCREEN_W`/`DXAPP_SCREEN_H`, else `xdpyinfo`, else 1920x1080), the same rule as the Python runners' `show_image()`. Every frame is shown by default: a window slower than the NPU paces the pipeline, so measure throughput with `--no-display`. `--drop-frames` selects a depth-1 preview that discards stale frames (the sync preview is then capped at 10 fps, the async one at 60 fps, and the summary prints `dropped (stale)`).

**Output order (async runners)**  

Async runners get results back from dxrt's completion threads in any order. Each frame gets an index when it is submitted, and the display thread releases frames through a reorder buffer (`common/utility/frame_reorder.hpp`), so display, save and `DXAPP_VERIFY` output (written where the buffer releases a frame) follow the input order and equal the sync runner's frame by frame. The runner's one postprocessor is shared by dxrt's callback threads and serialized (`common/processors/serialized_postprocessor.hpp`). SuperPoint's tracker is updated by its visualizer, after the reorder buffer, so it also sees frames in input order.

- At the end of a video the runner waits until every submitted frame has reached the display thread and, with `--save`, has been written. If nothing progresses for 5 s it prints `[DXAPP] [WARN] Output frames stopped draining; saved video may be truncated.` (without `--save`: `... the last frames may be missing from the display and DXAPP_VERIFY records.`) and stops.
- On an interrupted run (Ctrl+C, SIGTERM, or closing the window) that wait ends at once, and the frames the reorder buffer already holds are still released in index order.
- The reorder buffer holds at most twice the runner's in-flight depth. If a frame never comes back from dxrt, the lowest buffered frame is released once that cap is passed, so the display cannot stall; from then on output order is no longer guaranteed.

`DXRT_TASK_MAX_LOAD=40` (a dxrt setting) raises dxrt's I/O buffer count from its default (about 7 jobs in flight) to the runners' 40-frame pipeline depth (instance segmentation keeps its own limit of 4). For a large model dxrt may allocate fewer buffers when NPU memory is short and logs `Buffer count reduced from 40 to N due to NPU memory limit (...)`. If a large model such as `yolopv2_384x640` fails to start at 40 (dxrt reports `Dynamic IPC task init failed` before the first frame), lower the value, for example to `DXRT_TASK_MAX_LOAD=20`. The limit depends on the device's free NPU memory, which other processes on the NPU share.

**Output Management (`--save`)**  

When `--save` is enabled, a timestamped directory is created (e.g., `artifacts/cpp_example/{model}-image-{name}-{timestamp}/`) containing `run_info.txt`, saved images/video, and optional tensor dumps.

**Configuration Management (`--config`)**  

Runtime parameters (thresholds, top-k, etc.) live in `<task>/<family>/<variant>/config.json`, next to that variant's entry. A single-model extract may keep `config.json` beside the entry.

### Verification & Diagnostics

**Numerical Verification (`DXAPP_VERIFY`)**  

Set `DXAPP_VERIFY=1` to serialize post-processing results. Each run writes `logs/verify/{model}.json` (override the directory with `DXAPP_VERIFY_DIR`) holding the last frame, and `{model}.frames.jsonl` with one record per frame (`"frame": 0, 1, …` in input order). Writes are serialized and the JSON is replaced atomically, so a reader never sees a half-written file. Every runner dumps its real results, including tiled super-resolution (statistics of the stitched output), 3D detection and YOLOPv2 (boxes plus drivable/lane mask statistics); SuperPoint records carry each frame's keypoints.

Frame numbers count per process, so with `-l N` on a video the records of later loops continue the numbering. `{model}` is the model file's stem (for example `yolopv2_384x640.json` for `yolopv2_384x640.dxnn`).

**Tensor Dump for Debugging (`--dump-tensors`)**  

Dumps raw input/output tensors as `.bin` files. On exception, tensors and a `reason.txt` are auto-dumped for debugging.

### Environment Variables Reference

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | Save visualization to the specified file path |
| `DXAPP_VERIFY` | When `1`, dump JSON verification data |
| `DXAPP_VERIFY_DIR` | Directory for `DXAPP_VERIFY` output (default `logs/verify`) |
| `DXAPP_PROGRESS` | When `1`, print `[PROGRESS] frames=N fps=F avg_fps=A read_ms=… pre_ms=… infer_ms=… post_ms=…` once a second and at exit (front ends); multi_model_graph prints `node_ms=<id>:<ms>,…` instead of the stage fields |
| `DXAPP_ROOT` | The root that repository-relative paths (`sample/...`, a model's `config.json`, galleries) are read against. Default: the repository the program was built from; an exported graph project uses the folder holding its `.dxapp-root` (its `run.sh` sets this variable) |

---

## Exporting a graph as a standalone project

`python3 scripts/export_graph_project.py --graph <graph.json> --out <dir>` writes a multi-model graph as a CMake project that needs only the DEEPX runtime, CMake, a C++17 compiler and OpenCV 4: `./setup.sh` builds it, `./run.sh` runs it and writes `results/`. Each "User logic" node becomes a C++ class stub in `user_logic/` that passes data through until you edit it. See `src/cpp_example/multi_model_graph/README.md`, "Take a graph away".

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
