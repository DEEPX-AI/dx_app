# DX-APP Architecture and Usage Reference

This chapter is the reference for the whole repository: how it is laid out, how an example is built from the shared runtime layer, the command-line options and environment variables every example shares, and the runtime features built into all of them.  

To run a first example, follow [DX-APP Quick Start](01a_DX-APP_Quick_Start.md). Requirements and installation are in [DX-APP Installation and Build](02_DX-APP_Installation_and_Build.md).  

---

## Core Concepts & Architecture

### Repository Layout

The project is structured to separate core logic from language-specific implementations.  

```text
dx_app/
├── src/
│   ├── cpp_example/            # C++ end-to-end examples (organized across 29 tasks)
│   │   ├── multi_model_graph/  # Multi-model graph CLI and sample graph JSON files
│   │   └── common/             # ← Shared C++ runtime layer
│   │       ├── base/           #   Abstract interfaces (task factories such as IDetectionFactory, IPreprocessor, IPostprocessor, IVisualizer, IInputSource)
│   │       ├── processors/     #   Shared processors
│   │       ├── runner/         #   Task-specific sync/async runner pairs
│   │       ├── inputs/         #   Image/Video/Camera/RTSP input sources
│   │       ├── visualizers/    #   Task-specific visualizers
│   │       ├── config/         #   ModelConfig loader
│   │       ├── utility/        #   Labels, preprocessing, profiling, run_dir, interrupt handling, verify_serialize
│   │       ├── graph/          #   Multi-model graph engine
│   │       ├── registry/       #   Model registry used by the graph engine
│   │       ├── trackers/       #   IoU tracker
│   │       └── third_party/    #   Header-only third-party libraries (nlohmann_json)
│   ├── python_example/         # Python end-to-end examples (organized across 29 tasks)
│   │   └── common/             # ← Shared Python runtime layer
│   │       ├── base/           #   Abstract interfaces (task factories such as IDetectionFactory, IPreprocessor, IPostprocessor, IVisualizer, IInputSource)
│   │       ├── processors/     #   Shared processors
│   │       ├── runner/         #   SyncRunner, AsyncRunner, run_dir, verify_serialize, sr_tiling, args
│   │       ├── inputs/         #   Image/Video/Camera/RTSP input sources
│   │       ├── visualizers/    #   Task-specific visualizers
│   │       ├── config/         #   ModelConfig loader
│   │       ├── trackers/       #   IoU tracker
│   │       └── utility/        #   Labels, preprocessing, profiling
│   ├── postprocess/            # C++ post-processing (consumed by pybind11 bindings)
│   ├── utility/                # Shared support code used by build flow
│   └── bindings/
│       └── python/
│           ├── dx_postprocess/ # pybind11 bindings wrapping src/postprocess/
│           └── dx_graph/       # pybind11 module running multi-model graphs
├── config/
│   ├── model_registry.json     # Model registry — single source of truth
│   ├── test_models.conf        # Test model configuration
│   └── README.md               # Config directory documentation
├── scripts/                    # Developer tools, validation, and helper scripts
├── tests/                      # pytest-based test suites
│   ├── test_helpers/           #   Shared test constants & utilities
│   ├── cpp_example/            #   C++ tests (CLI, E2E, visualization, features)
│   └── python_example/         #   Python tests (unit, CLI, E2E, visualization)
├── assets/                     # Downloaded models/videos (via setup.sh)
├── build.sh                    # Top-level build script
├── run_tc.sh                   # Unified test runner for example tests
├── install.sh                  # Dependency and OpenCV installer
└── docs/                       # Detailed documentation
```

For contributor-oriented layout details, refer to [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md).  

!!! note "User vs Contributor Guidance"  

      This chapter is primarily a user-facing reference. If you are extending examples, onboarding new models, or maintaining the repository structure, use the contributor-oriented documents in *DX-APP Developer Guide*.  

### Pipeline Architecture

**C++ Application Templates** (`src/cpp_example/`)  

These templates provide high-performance, production-ready references for building applications using the DX-RT C++ API.  

The C++ tree is organized by **task → model family → variant**, with a shared `common/` layer providing base interfaces, shared processors, task-specific runner pairs, visualizers, and input abstraction. Each model directory delegates to `common/` via the factory pattern. For details, refer to [DX-APP C++ Usage Guide](03_DX-APP_CPP_Example_Usage_Guide.md) and [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md).

**Pipeline Architecture**  

Each template follows a self-contained pipeline designed for modularity  

- **Step 1. Input:** Image, Video, Camera, or RTSP stream (via `common/inputs/`).  
- **Step 2. Pre-process:** Resizing and normalization (via the preprocessors in `common/processors/`, which use the letterbox helpers in `common/utility/`).  
- **Step 3. Inference:** Execution on the NPU via **DX-RT**.  
- **Step 4. Post-process:** Call to shared C++ processors in `common/processors/` (e.g., NMS, box scaling).  
- **Step 5. Output:** Result rendering via `common/visualizers/` (Display) or storage (Save).  

### Post-processing Libraries (`src/postprocess/`)

These libraries transform raw NPU output tensors into structured, actionable data. They are **consumed by the pybind11 bindings** (`src/bindings/python/dx_postprocess/`) to enable `*_cpp_postprocess.py` variants in Python.  

!!! note "NOTE"  

    The C++ examples under `src/cpp_example/` do **not** link the `src/postprocess/` libraries; they use their own shared processors in `src/cpp_example/common/processors/`. The exceptions are four modules (`sfa3d`, `superpoint`, `dope`, `yolopv2`), whose sources are compiled directly into the C++ examples and are also exposed through the pybind11 bindings.  

**Module Structure**  

The library is organized into model-specific subdirectories (e.g., `yolov5/, yolov8/, deeplabv3/`), each containing  

- `*_postprocess.h`: Defines the post-processing class (e.g., `YOLOv5PostProcess`) and standard result structures (e.g., `YOLOv5Result`).  
- `*_postprocess.cpp`: Contains the optimized implementation for decoding, coordinate scaling, and filtering.  
- `CMakeLists.txt`: Present in the modules listed in `src/postprocess/CMakeLists.txt`; `./build.sh --all --make_so` builds those modules as standalone shared libraries (`lib/libdxapp_<module>_postprocess.so`).  

**Functional Responsibilities**  

The libraries handle the heavy computational load required after the inference stage  

- **Tensor Decoding:** Converting raw NPU buffer outputs into human-readable results such as bounding boxes, confidence scores, and class IDs.  
- **Advanced Geometry:** Extracting keypoints for pose estimation or skeletons.  
- **Mask Generation:** Processing multi-dimensional tensors into segmentation masks.  
- **Filtering & Optimization:** Applying algorithms like **Non-Maximum Suppression (NMS)** and threshold-based filtering to remove redundant detections.  

**Cross-Language Integration**  

- **For Python Developers:** The `*_cpp_postprocess.py` variants use these C++ libraries via the `dx_postprocess` pybind11 module, achieving near-native performance.  
- **For C++ Developers:** The C++ examples use their own shared processors in `src/cpp_example/common/processors/`, which are compiled and linked directly.  

---

## Usage Reference

### CLI Reference

All C++ and Python examples share a consistent set of command-line arguments.

**Common Arguments**  

| Flag | C++ | Python | Description |
|------|-----|--------|-------------|
| `-m` / `--model` | `-m` / `--model_path` | `-m` / `--model` | Path to `.dxnn` model file. Optional: if omitted, the example's default model is used and auto-downloaded when missing |
| `-i` / `--image` | `-i` / `--image_path` | `-i` / `--image` | Input image file or directory |
| `-v` / `--video` | `-v` / `--video_path` | `-v` / `--video` | Input video file |
| `-c` / `--camera` | `-c` / `--camera_index` | `-c` / `--camera` | Camera device index |
| `-r` / `--rtsp` | `-r` / `--rtsp_url` | `-r` / `--rtsp` | RTSP stream URL |
| `-l` / `--loop` | `-l` / `--loop` (default: auto) | `-l` / `--loop` (default: 1; bare `--loop` = 2) | Inference repeat count |
| `--no-display` | `--no-display` | `--no-display` | Disable visualization window |
| `-s` / `--save` | `-s` / `--save` | `-s` / `--save` | Save rendered output to run directory |
| `--save-dir` | `--save-dir` | `--save-dir` | Base output directory (default: `artifacts/cpp_example` under the current directory for C++, `artifacts/python_example/` under the dx_app root for Python) |
| `--dump-tensors` | `--dump-tensors` | `--dump-tensors` | Dump raw input/output tensors to files |
| `--show-log` | `--show-log` | `--show-log` | Enable verbose log output (default: quiet) |
| `--drop-frames` | `--drop-frames` | — | C++ only: preview drops stale frames instead of pacing the pipeline (not available in the classification sync runner or the embedding runners) |
| `--config` | `--config` | `--config` | Model config JSON path. Python: auto-detected if omitted (`config.json` next to the model, then next to the script). C++: read only when given; otherwise the factory's built-in defaults are used |
| `-h` / `--help` | `-h` / `--help` | `-h` / `--help` | Show usage |

- **Input Source Rule:** `--image`, `--video`, `--camera`, and `--rtsp` form a mutually exclusive group. If none is specified, a **default sample image** is automatically selected based on the task type. Image-only examples (embedding, re-ID, attribute recognition) accept only `--image`.

**Environment Variables**  

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | When set to a file path, saves the visualization output to that path (no `--save` required) |
| `DXAPP_VERIFY` | When set to `1`, dumps post-processing results to `logs/verify/{model}.json` (last frame) and `{model}.frames.jsonl` (every frame, in input order) for numerical verification |
| `DXAPP_VERIFY_DIR` | Directory for `DXAPP_VERIFY` output (default `logs/verify`) |
| `DXAPP_PROGRESS` | When set to `1`, prints `[PROGRESS] frames=N fps=F avg_fps=A read_ms=R pre_ms=P infer_ms=I post_ms=Q` at most once a second and once at exit, for front ends such as DX AI Studio: `fps` is the last second's rate, `avg_fps` counts from the first frame (start-up and model load excluded), `*_ms` are the stage averages over the same second (the final line: run averages). The multi_model_graph binaries print `node_ms=<id>:<ms>,...` (time per graph node) instead of the `*_ms` stage fields |

### C++ Templates 

To help developers optimize for specific hardware targets, templates are provided in two execution patterns  

- **Synchronous (`xx_sync.cpp`):**

    : **Logic:** A single-threaded, sequential loop (**Input → Inference → Output**).  
    : **Use Case:** Best for single-image processing and simplified debugging.  

- **Asynchronous (`xx_async.cpp`):**

    : **Logic:** Uses multi-threading and the `RunAsync()` API to overlap stages. While the NPU performs inference on Frame **N**, the CPU simultaneously handles pre-processing for Frame **N+1** and post-processing for Frame **N-1**.  
    : **Use Case:** Essential for maximizing **FPS** on live video streams and ensuring high NPU utilization.  

### Python Integration (Bindings & Examples)  

DX-APP provides a unified environment that combines the rapid development of Python with the high performance of native C++.  

**High-Performance C++ Bindings** (`dx_postprocess`)  

To eliminate post-inference bottlenecks, DX-APP provides optimized C++ logic exposed via `pybind11`.  

- **Key Capabilities:** Handles CPU-intensive tasks such as NMS (Non-Maximum Suppression), tensor decoding, and mask generation at native speeds.  
- **Scope:** Used by the `*_cpp_postprocess.py` variants. The C++ examples use their own shared processors under `src/cpp_example/common/processors/`.  
- **Installation:** Automatically compiled during `./build.sh`.  
    : **Manual install:** `cd src/bindings/python/dx_postprocess && python3 -m pip install .`  

For detailed usage examples and API references, please refer to the documentation in 
[**Section. DX-APP Python Post-processing**](08_DX-APP_Pybind_PostProcess_Overview.md)

**Application Examples** (`src/python_example/`)  

These templates utilize `dx_engine` (for inference) and `dx_postprocess` (for acceleration). Users can choose from four variants depending on their performance requirements.  

The Python tree is organized by **task → model family → variant**, with a shared `common/` layer providing base interfaces, shared processors, generic sync/async runners, visualizers, and input abstraction — the same factory-based architecture as the C++ side. For structure and contributor-facing rules, refer to [DX-APP Python Usage Guide](05_DX-APP_Python_Example_Usage_Guide.md) and [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md).  

**Task-Based Structure**    

Templates are categorized by task across multiple task directories. All examples share the `common/` runtime layer for processors, runners, and visualizers. Representative tasks:  

- **Image Classification:** EfficientNet, AlexNet, ResNet, MobileNet, etc.  
- **Object Detection:** YOLOv5/v7/v8/v9/v10/v11/v12/v13, YOLOX, NanoDet, DAMOYOLO, SSD (PPU variants under `yolo_ppu/`)  
- **Face Detection:** SCRFD, YOLOv5Face, YOLOv7Face, RetinaFace  
- **Pose Estimation:** YOLOv8-Pose  
- **Segmentation:** BiSeNet, DeepLabV3+, SegFormer, YOLOv8Seg  
- **Depth Estimation, Face Recognition, Oriented Object Detection, Image Denoising, Low-Light Enhancement, Super Resolution, Hand Landmark, Person Attribute, Person Re-ID**  

**Functional Variants**  

| **Variant** | **Post-processing** | **Threading Model** | **Recommendation** | 
|----|----|----|----|
| `*_sync.py` | Pure Python | Synchronous | Learning & Logic Debugging | 
| `*_async.py` | Pure Python | Asynchronous | Basic performance optimization | 
| `*_sync_cpp_postprocess.py` | C++ Binding | Synchronous | Accelerating heavy CPU tasks | 
| `*_async_cpp_postprocess.py` | C++ Binding | Asynchronous | Maximum FPS (Recommended) | 

---

## Advanced Features

DX-APP includes several production-oriented features built into all templates.


**Auto-Download**

When an example (C++ or Python) runs without `-m`, the runner uses its default model from `config/model_registry.json` and, if the file is missing, downloads it via `setup_sample_models.sh`. If the download fails, an error message with manual download instructions is displayed. An explicit `-m` path that does not exist is an error (no download). Missing input files, including videos, are not downloaded; run `./setup.sh` or `./setup_sample_videos.sh` first.

**Default Input Fallback**

If no input source (`--image`, `--video`, `--camera`, `--rtsp`) is provided, the runner automatically selects a **default sample image** appropriate for the task type. A log message indicates which default was applied.

**Signal Handling**  

C++ runners and the `multi_model_graph_*` CLI register SIGINT/SIGTERM handlers for graceful shutdown. Pressing Ctrl+C during inference cleanly exits, releasing all resources: a runner prints `"Interrupted by user"`, and the graph CLI finishes the frames in flight in every stream and finalizes `--report`/`--output`/`--events`. A later Ctrl+C, more than 200 ms after the first, terminates the process at once (for a shutdown that is itself stuck); repeats within 200 ms count as the same request. SIGTERM is always graceful, however often it arrives. Python runners behave the same way (`common/runner/interrupts.py`); see the [Python usage guide](05_DX-APP_Python_Example_Usage_Guide.md).  

**Run Directory** (`--save` / `--save-dir`)  

When `--save` is enabled, a timestamped run directory is created:  

```text
artifacts/cpp_example/
  {model}_sync-image-{name}-{YYYYMMDD-HHMMSS}/
    run_info.txt        # Metadata (script, model, input paths)
    output.jpg          # Saved visualization (image mode)
    output.mp4          # Saved visualization (video mode)
    dump_tensors/       # (if --dump-tensors) raw tensor files
```

Video, camera and RTSP runs are named `{model}_sync-stream-{name}-...`, and a directory input `...-image-dir-...`. The Python runners use the same pattern under `artifacts/python_example/`, prefixed with the script name; they save images as `{image}_result.jpg` and dump tensors to `tensors/` as `.npy` files.  

**Numerical Verification** (`DXAPP_VERIFY`)  

A complete pipeline for serializing inference results for inspection and debugging:  

- (1) Set `DXAPP_VERIFY=1` before running any example  
- (2) Post-processing results are serialized to `logs/verify/{model}.json` (last frame) and `{model}.frames.jsonl` (every frame, in input order)  
- (3) Supports every result type (Detection, Classification, Pose, Segmentation, Panoptic, 3D, SuperPoint keypoints, …)  

**Tensor Dump** (`--dump-tensors`)  

Dumps raw input/output tensors for debugging. On exception, tensors are auto-dumped (C++: sync runners, `.bin` files; Python: `.npy` files plus a `reason.txt`).  

**Model Config** (`--config`)  

Runtime parameters (score threshold, NMS threshold, top-k) live in `<task>/<family>/<variant>/config.json`, next to that variant's entry. The Python examples load it automatically. A C++ example reads it only when you pass `--config <path>`; otherwise it uses the defaults compiled into its factory. A single-model extract may keep `config.json` beside the entry script.  

**Version Compatibility**  

Every example in this release, C++ and Python, needs **DX-RT 3.5.0 or later**: the DX Model Zoo 2_5_0 files that `./setup.sh` downloads are `.dxnn` container v9, and DX-RT 3.5.0 is the first runtime that loads them. Check the runtime before you run an example:  

```bash
dxcli --version   # the first line must be DXRT v3.5.0 or later
```

The runners also check versions when they start, but the C++ and Python checks differ:  

| Language | What the runner checks | What an older DX-RT shows |
|----------|------------------------|---------------------------|
| C++ | DX-RT ≥ 3.5.0 and model format ≥ v7 (`minversionforRTandCompiler` in `common/utility/common_util.hpp`), once the engine is created | `DXRT library version is too low. (required: >= 3.5.0, current: <version>)`, or DX-RT's own error on the `.dxnn` if loading fails first |
| Python | DX-RT ≥ 3.0.0 only, and a warning for a model format older than v7 | DX-RT 3.0.0 to 3.4.x pass the check, and the example then fails while loading the model, with DX-RT's own error (for example `Model file format version 9 is not supported`) |

!!! warning "Python examples on DX-RT 3.0.0 to 3.4.x"  

    The Python runners do not stop on DX-RT 3.0.0 to 3.4.x, and the error you see when the model loads comes from DX-RT, not from DX-APP. If a Python example fails while loading a model, check `dxcli --version` first and upgrade DX-RT to 3.5.0 or later. `import dx_engine` fails with `dx_engine and DX-RT versions do not match` when the `dx_engine` package and the runtime differ in major.minor version.  

The multi-model graph checks the `.dxnn` container itself before it loads a model and names the required DX-RT version (see [DX-APP Multi-Model Graph](06a_DX-APP_Multi_Model_Graph.md)). The requirements are listed in *Software Requirements on Linux* in [DX-APP Installation and Build](02_DX-APP_Installation_and_Build.md).

**Headless Mode**  

C++ and Python runners detect the absence of `DISPLAY`/`WAYLAND_DISPLAY` and skip the preview window automatically. Use `--no-display` for explicit headless operation.  

---
