# DX-APP Overview

**DX-APP** is a production-ready suite of application templates designed to accelerate the development of AI services on **DEEPX NPUs**. It bridges the gap between raw model deployment and high-performance application engineering.  

**Key Features & Objectives**  

- **Rapid Deployment:** Ready-to-run examples across multiple AI task categories — Classification, Object Detection, Face Detection, Pose Estimation, Semantic/Instance Segmentation, Depth Estimation, OBB Detection, Embedding, and more.
-	**Dual-Language Flexibility:** High-performance **C++** for production and **Python** for rapid prototyping, each with their own shared runtime layer (`src/cpp_example/common/` for C++, `src/python_example/common/` for Python).  
- **Hardware Acceleration:** Native support for **PPU-enabled models** and **Async templates** that overlap pipeline stages to maximize FPS.  
- **Modular Design:** Clean, task-oriented templates that serve as reusable blueprints for custom commercial applications.  

**Reference Documentation**  

For deeper technical specifications, refer to the [`docs/source/docs/`](./docs/source/docs/) directory  

| # | Document | Description |
|---|----------|-------------|
| 01 | [DXNN Application Overview](./docs/source/docs/01_DXNN_Application_Overview.md) | SDK architecture, DX-APP features & core design |
| 01a | [Quick Start](./docs/source/docs/01a_DX-APP_Quick_Start.md) | From a fresh installation to the first inference, in five steps |
| 02 | [Installation and Build](./docs/source/docs/02_DX-APP_Installation_and_Build.md) | Prerequisites, build steps (Linux/Windows) |
| 03 | [C++ Example Usage Guide](./docs/source/docs/03_DX-APP_CPP_Example_Usage_Guide.md) | C++ template structure & execution guide |
| 04 | [C++ Example Tests](./docs/source/docs/04_DX-APP_CPP_Example_Test.md) | C++ test framework & coverage |
| 05 | [Python Example Usage Guide](./docs/source/docs/05_DX-APP_Python_Example_Usage_Guide.md) | Python template structure & execution guide |
| 06 | [Python Example Tests](./docs/source/docs/06_DX-APP_Python_Example_Test.md) | Python test framework (pytest) |
| 06a | [Multi-Model Graph](./docs/source/docs/06a_DX-APP_Multi_Model_Graph.md) | Several models in one pipeline from a graph JSON: usage, graph format, logic nodes |
| 07 | [C++ Post-processing](./docs/source/docs/07_DX-APP_CPP_PostProcess_Overview.md) | C++ post-processing library design |
| 08 | [Python Post-processing](./docs/source/docs/08_DX-APP_Pybind_PostProcess_Overview.md) | pybind11 bindings (dx_postprocess) |
| 09 | [Architecture and Usage Reference](./docs/source/docs/09_DX-APP_Project_Overview.md) | Repository layout, CLI reference, advanced features |
| 10 | [DX Tool Guide](./docs/source/docs/10_DX-APP_DX-Tool_Guide.md) | Developer tooling (dx_tool.sh) |
| 11 | [Example Source Structure](./docs/source/docs/11_DX-APP_Example_Source_Structure.md) | Source tree conventions & contributor guide |
| 12 | [YOLO Customizing Guide](./docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md) | YOLO model onboarding and customization workflow |
| 13 | [Agent-Driven Development Guide](./docs/source/docs/13_DX-APP_Agent_Driven_Development.md) | Agent architecture, skills, routing, validation, and troubleshooting |
| — | [Appendix: Third-Party License](./docs/source/docs/Appendix_Third_Party_License.md) | License information for third-party models & datasets |
| — | [Change Log](./docs/source/docs/Appendix_Change_Log.md) | Version history |

---

# Architectural Overview

DX-APP is engineered to maximize NPU throughput while minimizing CPU-side bottlenecks.  

## Unified Post-Processing Engine

To ensure consistency and speed, all model-specific decoding (NMS, box scaling, mask generation) is implemented in optimized C++ libraries.  

- **Cross-Language Parity:** These modules are exposed to Python via `pybind11` (`dx_postprocess`), ensuring Python developers achieve C++-level performance.  
- **Logic Standardization:** Identical decoding logic across both environments guarantees consistent inference results.  

## Execution Paradigms: Sync vs. Async

Templates are provided in two variants to help developers optimize for their specific use cases  

- **Synchronous (Sync):** Sequential execution (**Pre → Inference → Post**). Best for single-image analysis and simplified debugging.  
- **Asynchronous (Async):** A multi-threaded design using `RunAsync()` to overlap stages. While the NPU processes Frame **N**, the CPU prepares Frame **N+1** and post-processes Frame **N-1**. This is critical for maximizing **FPS** on real-time video or RTSP streams.  

## Performance Profiling & Bottleneck Analysis

Every application template in DX-APP—regardless of the language (C++/Python) or execution paradigm (Sync/Async)—is equipped with a built-in performance profiler. Upon completion, the console outputs a **Performance Summary** that serves as a critical tool for application tuning.  

**Key Metrics Collected**  

- **Stage Latency:** Precise timing for each stage of the pipeline  
      : **Pre-processing:** Image decoding, resizing, and normalization  
      : **NPU Inference:** Pure execution time on the DEEPX NPU via DX-RT  
      : **Post-processing:** Result decoding (NMS, box scaling, etc.)  
      : **Display/I/O:** Time taken to render or save the output  
- **End-to-End Throughput (FPS):** The overall frames per second achieved by the entire system.  

**Strategic Objectives**  

- **Bottleneck Identification:** Instantly determine if the system is limited by CPU-side tasks (Pre/Post-processing) or NPU throughput. For instance, if post-processing latency is high in a Python script, you can strategically switch to the **C++ Binding** (`dx_postprocess`) variant.  
- **Architectural Benchmarking:** Quantitatively validate how much performance is gained by moving from a **Synchronous** to an **Asynchronous** design.  
- **Resource Optimization:** Help developers balance NPU utilization and CPU overhead to find the "sweet spot" for their specific hardware and commercial use case.  
 
---

# Repository Layout & Installation 

This section guides you through the environment setup and the initial build process required to run DX-APP.  

## Repository Layout

> **Model cards.** Every example folder has a `README.md` describing its model -- what it is, accuracy
> and speed with their source, downloads and how to run it -- and each Python example folder also has
> `model.html` (the same card, viewable offline) and `result.webp`. [MODELS.md](MODELS.md) lists all of
> them by task. They are generated by DX AI Studio
> (`python -m dx_modelzoo.tools.export_dx_app_cards --dx-app <this checkout>`); do not edit them by hand.

> The registry lists **497 variants** (497 rows, one per variant) across 29 AI tasks,
> and **all 497 have a downloadable `.dxnn`**.

> **2_5_0 needs DX-RT 3.5.0.** Those `.dxnn` files are container **format version 9**,
> which DX-RT 3.4.2 refuses (`Model file format version 9 is not supported`) while
> 3.5.0 parses it. `scripts/sweep_npu_inference.py` probes the runtime once per
> container version and reports `UNSUPPORTED_FORMAT` on an older runtime rather
> than a failure of the example code.

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

For contributor-oriented layout details, refer to [DX-APP Example Source Structure](./docs/source/docs/11_DX-APP_Example_Source_Structure.md).

!!! note "User vs Contributor Guidance"
      This README is primarily a user-facing overview. If you are extending examples, onboarding new models, or maintaining the repository structure, use the contributor-oriented documents linked from this page.

## Prerequisites

Before building the templates, ensure your system meets the following hardware and software requirements.  

**A. DEEPX Runtime (DX-RT) and NPU Drivers**  

To utilize NPU acceleration, you **must** install the kernel-mode drivers and the user-space runtime library  

- **DEEPX NPU Linux Driver:** Required for low-level NPU communication. [Github Repository](https://github.com/DEEPX-AI/dx_rt_npu_linux_driver)  
- **DX-RT (Runtime & Tools):** The core library for model inference and hardware management. [Github Repository](https://github.com/DEEPX-AI/dx_rt)  

Versions this release needs:

- **OS**: Ubuntu 20.04 / 22.04 / 24.04 (x64) and Debian 12 / 13 (x64)  
- **DEEPX M1 Runtime Lib Version (DX-RT)**: v3.5.0 or higher  
- **NPU Device Driver**: v2.7.0 or higher (RT and PCIe driver). The runtime's own minimum driver, firmware and `.dxnn` versions are printed by `dxcli --version`.  
- **dx_engine** (Python): 3.5.x, matching the runtime's major.minor. `import dx_engine` raises `dx_engine and DX-RT versions do not match` otherwise.  

Tested with DX-RT 3.5.0, NPU driver 2.7.0 (RT and PCIe) and firmware 2.7.4 on a DX-M1 M.2 card (x86_64).

**B. Development Toolchain and Libraries**  

The following tools are required to compile the C++ templates and the Python dx_postprocess bindings.  

**B-a.** Build System  

- **CMake:** 3.14 or higher, with make or ninja 
- **Compiler:** C++14-compatible (GCC 7.5 or later, Clang)

**B-b.** Core Libraries  

- **OpenCV**: 4.2.0 or higher (**4.5.5 recommended**), for image I/O and visualization  
- **Python**: 3.8 or higher with pip, for the CMake configure step (it generates the graph model registry), the Python examples and the pybind11 modules    

## Development Workflow Overview

The process from environment setup to running your first AI application is divided into three main phases. For detailed commands and execution steps, please refer to the [**Section. Quick Start Guide**](#quick-start-guide).  

- **Hardware & Driver Verification:** Ensure the NPU is recognized by the system using the `dxrt-cli` tool.  
- **Asset & Dependency Preparation:** Install required libraries (OpenCV, Build tools) via `./install.sh` and prepare models/videos via `./setup.sh`. Model assets are fetched through the current [DX-ModelZoo](https://developer.deepx.ai/modelzoo/)-based setup flow.  
- **Build & Execution:** Compile the source code using `./build.sh` and run the generated binaries or Python scripts located in the `bin/` or `src/python_example/` directories.  

For contributor workflows such as model onboarding, validation, filtered execution, and benchmarking, refer to [DX Tool Guide](./docs/source/docs/10_DX-APP_DX-Tool_Guide.md).

---

# Multi-Model Graph

The multi-model runtime is the node graph: `bin/multi_model_graph_sync`, `bin/multi_model_graph_async`, and the Python module `dx_graph`. Sources are [`src/cpp_example/multi_model_graph/`](./src/cpp_example/multi_model_graph/README.md) and [`src/bindings/python/dx_graph/`](./src/bindings/python/dx_graph/README.md).

A graph is a JSON file of registry models wired by frame, ROI and image hand-off edges. It runs the 492 models [`docs/graph_models.md`](./docs/graph_models.md) lists as running in a graph, on images, videos, cameras and RTSP streams, several at once. Head pose and package volume are cpu nodes (`dms_headpose.json`, `logistics_volume.json`). An ROI crop uses `PaddedCropRect` in `common/utility/roi_crop.hpp`. Details: the [multi-model graph README](./src/cpp_example/multi_model_graph/README.md).

The graph binaries and `dx_graph` are built by `./build.sh --all` (the default `--minimal` build does not build them); `./build.sh --all --python_exec <python>` builds `dx_graph` for that interpreter into `bin/python/dx_graph`.

**Model files.** `./setup.sh` downloads all 497 registered models: 493 from DX Model Zoo 2_5_0, whose `.dxnn` files are container v9, which needs DX-RT 3.5.0 (see *Repository Layout* above), and 4 that 2_5_0 does not publish from 2_4_0 (`beit-l-p16_384x384`, `levit-128s_224x224`, `vit-b-p16_384x384`, `vit-t-p16_224x224`). The graph runtime checks the container before loading; on a DX-RT older than 3.5.0 it stops with `.dxnn container v9 needs DX-RT >= 3.5.0, but this runtime is <version>. Use the v8 file (dxnn/2_4_0) or upgrade DX-RT.`, and `--check` reports it and exits 1. The graphs were validated on DX-RT 3.4.1 with v8 files and on DX-RT 3.5.0 with both the v8 and the v9 files: the golden reports (9 sample graphs) are byte-identical between `multi_model_graph_sync` and `_async` on either store, and the v8 ones equal the DX-RT 3.4.1 results.

---

# CLI Reference

All C++ and Python examples share a consistent set of command-line arguments.

## Common Arguments

| Flag | C++ | Python | Description |
|------|-----|--------|-------------|
| `-m` / `--model` | `-m` | `--model` | Path to `.dxnn` model file (auto-downloaded if missing). Without it, an example runs its own variant's model, `assets/models/<variant>.dxnn` |
| `-i` / `--image` | `-i` | `--image` | Input image file or directory |
| `-v` / `--video` | `-v` | `--video` | Input video file |
| `-c` / `--camera` | `-c` | `--camera` | Camera device index |
| `-r` / `--rtsp` | `-r` | `--rtsp` | RTSP stream URL |
| `-l` / `--loop` | `-l` (default: auto) | `--loop` (default: 1) | Inference repeat count |
| `--no-display` | `--no-display` | `--no-display` | Disable visualization window |
| `--drop-frames` | `--drop-frames` | — | C++ preview drops stale frames instead of pacing the pipeline (default: off, every frame is shown) |
| `--show-log` | `--show-log` | `--show-log` | Enable verbose log output (default: quiet) |
| `-s` / `--save` | `--save` | `--save` | Save rendered output to run directory |
| `--save-dir` | `--save-dir` | `--save-dir` | Base output directory (default: `artifacts/`) |
| `--dump-tensors` | `--dump-tensors` | `--dump-tensors` | Dump raw input/output tensors to files |
| `--config` | `--config` | `--config` | Model config JSON path (auto-detected if omitted) |
| `-h` / `--help` | `-h` | `-h` | Show usage |

> **Input Source Rule:** `--image`, `--video`, `--camera`, and `--rtsp` form a mutually exclusive group. If none is specified, a **default sample image** is automatically selected based on the task type (e.g., `sample/img/sample_street.jpg` for object detection).

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | When set to a file path, saves the visualization output to that path (no `--save` required) |
| `DXAPP_VERIFY` | When set to `1`, dumps post-processing results to `logs/verify/{model}.json` (last frame) and `{model}.frames.jsonl` (every frame, in input order) for numerical verification |
| `DXAPP_VERIFY_DIR` | Directory for `DXAPP_VERIFY` output (default `logs/verify`) |
| `DXAPP_PROGRESS` | When set to `1`, prints `[PROGRESS] frames=N fps=F avg_fps=A read_ms=R pre_ms=P infer_ms=I post_ms=Q` at most once a second and once at exit, for front ends such as DX AI Studio: `fps` is the last second's rate, `avg_fps` counts from the first frame (start-up and model load excluded), `*_ms` are the stage averages over the same second (the final line: run averages). The multi_model_graph binaries print `node_ms=<id>:<ms>,...` (time per graph node) instead of the `*_ms` stage fields |

---

# Advanced Features

DX-APP includes several production-oriented features built into all templates.

## Signal Handling

All runners and the `multi_model_graph_sync`/`multi_model_graph_async` CLI register SIGINT/SIGTERM handlers for graceful shutdown. Pressing Ctrl+C during inference cleanly exits, releasing all resources: a runner prints `"Interrupted by user"`, and the graph CLI finishes the frames in flight in every stream and finalizes `--report`/`--output`. A later Ctrl+C, more than 200 ms after the first, terminates the process at once (for a shutdown that is itself stuck); repeats within 200 ms count as the same request. SIGTERM is always graceful, however often it arrives.

## Run Directory (`--save` / `--save-dir`)

When `--save` is enabled, a timestamped run directory is created:
```text
artifacts/cpp_example/
  {model}_sync-image-{name}-{YYYYMMDD-HHMMSS}/
    run_info.txt        # Metadata (script, model, input paths)
    output.jpg          # Saved visualization (image mode)
    output.mp4          # Saved visualization (video mode)
    dump_tensors/       # (if --dump-tensors) raw tensor files
```

## Numerical Verification (`DXAPP_VERIFY`)

Serialize post-processing results to JSON for inspection and debugging:
1. Set `DXAPP_VERIFY=1` before running any example
2. Post-processing results are serialized to `logs/verify/{model}.json` (last frame) and `{model}.frames.jsonl` (every frame, in input order); `DXAPP_VERIFY_DIR` overrides the directory
3. Supports every result type (Detection, Classification, Pose, Segmentation, Panoptic, 3D, SuperPoint keypoints, …)

## Tensor Dump (`--dump-tensors`)

Dumps raw input/output tensors for debugging. On exception, tensors are auto-dumped with a `reason.txt` file. C++ outputs `.bin` files; Python outputs `.npy` files.

## Model Config (`--config`)

Runtime parameters (score threshold, NMS threshold, top-k) live in `<task>/<family>/<variant>/config.json`, next to that variant's entry. A single-model extract may keep `config.json` beside the entry script.

## Version Compatibility

The runners check the runtime and the model before they run:
- **C++:** DX-RT library ≥ 3.5.0 and compiled model format ≥ v7 (`minversionforRTandCompiler`, `common/utility/common_util.hpp`), checked once the engine is created. An older runtime or model stops the example with an error and a hint.
- **Python:** DX-RT library ≥ 3.0.0 (an older one exits with an error) and a warning for a model format older than v7. `import dx_engine` itself refuses a DX-RT of another major.minor. The Model Zoo 2_5_0 files still need DX-RT ≥ 3.5.0 (see *Prerequisites*).

## Auto-Download

When running any example (C++ or Python), if the specified model file is not found locally, the runner automatically attempts to download it via `setup_sample_models.sh`. Similarly, if a `--video` file is missing, `setup_sample_videos.sh` is invoked automatically. If the download fails, a clear error message is displayed with manual download instructions.

## Default Input Fallback

If no input source (`--image`, `--video`, `--camera`, `--rtsp`) is provided, the runner automatically selects a **default sample image** appropriate for the task type. For example, object detection tasks default to `sample/img/sample_street.jpg`, face detection to `sample/img/sample_face.jpg`, and so on. A log message indicates which default was applied:
```
[DXAPP] [INFO] No input specified. Using default sample: sample/img/sample_street.jpg
```
This allows the simplest possible execution — just specify the model:
```bash
python src/python_example/object_detection/yolov7/yolov7_640x640/yolov7_640x640_sync.py --model assets/models/yolov7_640x640.dxnn
```

## Headless Mode

Python runners detect the absence of `DISPLAY`/`WAYLAND_DISPLAY` and skip `cv2.imshow()` automatically. Use `--no-display` for explicit headless operation in both C++ and Python.

---

# C++ Application Templates (src/cpp_example/)

These templates provide high-performance, production-ready references for building applications using the DX-RT C++ API.  

The refactored C++ tree is organized by **task → model family → variant**, with a shared `common/` layer providing base interfaces, shared processors, task-specific runners, visualizers, and input abstraction. Each model directory delegates to `common/` via the factory pattern. For details, refer to [DX-APP C++ Usage Guide](./docs/source/docs/03_DX-APP_CPP_Example_Usage_Guide.md) and [DX-APP Example Source Structure](./docs/source/docs/11_DX-APP_Example_Source_Structure.md).

**Pipeline Architecture**  

Each template follows a self-contained pipeline designed for modularity  

- **Step 1. Input:** Image, Video, Camera, or RTSP stream (via `common/inputs/`).  
- **Step 2. Pre-process:** Resizing and normalization (via `common/utility/`).  
- **Step 3. Inference:** Execution on the NPU via **DX-RT**.  
- **Step 4. Post-process:** Call to shared C++ processors in `common/processors/` (e.g., NMS, box scaling).  
- **Step 5. Output:** Result rendering via `common/visualizers/` (Display) or storage (Save).  

**Design Variants**  

To help developers optimize for specific hardware targets, templates are provided in two execution patterns  

- **Synchronous (`*_sync.cpp`):**  
      : **Logic:** A single-threaded, sequential loop (**Input → Inference → Output**).  
      : **Use Case:** Best for single-image processing and simplified debugging.  

- **Asynchronous (`*_async.cpp`):**  
      : **Logic:** Uses multi-threading and the `RunAsync()` API to overlap stages. While the NPU performs inference on Frame **N**, the CPU simultaneously handles pre-processing for Frame **N+1** and post-processing for Frame **N-1**.  
      : **Use Case:** Essential for maximizing **FPS** on live video streams and ensuring high NPU utilization.  

---

# Post-processing Libraries (src/postprocess/)

These libraries transform raw NPU output tensors into structured, actionable data. They are **consumed by the pybind11 bindings** (`src/bindings/python/dx_postprocess/`) to enable `*_cpp_postprocess.py` variants in Python.  

> **Note:** The C++ examples under `src/cpp_example/` do **not** use `src/postprocess/` directly. They have their own shared processors in `src/cpp_example/common/processors/`. The `src/postprocess/` library exists specifically for the pybind11 bridge.

**Module Structure**  

The library is organized into model-specific subdirectories (e.g., `yolov5/, yolov8/, deeplabv3/`), each containing  

- `*_postprocess.h`: Defines the post-processing class (e.g., `YOLOv5PostProcess`) and standard result structures (e.g., `YOLOv5Result`).  
-	`*_postprocess.cpp`: Contains the optimized implementation for decoding, coordinate scaling, and filtering.  
-	`CMakeLists.txt`: Facilitates the compilation of these modules into reusable shared libraries.  

**Functional Responsibilities**  

The libraries handle the heavy computational load required after the inference stage  

-	**Tensor Decoding:** Converting raw NPU buffer outputs into human-readable results such as bounding boxes, confidence scores, and class IDs.  
-	**Advanced Geometry:** Extracting keypoints for pose estimation or skeletons.  
-	**Mask Generation:** Processing multi-dimensional tensors into segmentation masks.  
-	**Filtering & Optimization:** Applying algorithms like **Non-Maximum Suppression (NMS)** and threshold-based filtering to remove redundant detections.  

**Cross-Language Integration**  

- **For Python Developers:** The `*_cpp_postprocess.py` variants use these C++ libraries via the `dx_postprocess` pybind11 module, achieving near-native performance.  
- **For C++ Developers:** The C++ examples use their own shared processors in `src/cpp_example/common/processors/`, which are compiled and linked directly.  

---

# Python Integration (Bindings & Examples)

DX-APP provides a unified environment that combines the rapid development of Python with the high performance of native C++.  

## High-Performance C++ Bindings (`dx_postprocess`)

To eliminate post-inference bottlenecks, DX-APP provides optimized C++ logic exposed via `pybind11`.  

- **Key Capabilities:** Handles CPU-intensive tasks such as NMS (Non-Maximum Suppression), tensor decoding, and mask generation at native speeds.  
- **Unified Logic:** Shares the exact same decoding logic as the C++ examples, ensuring consistent inference results across all platforms.  
- **Installation:** Automatically compiled during `./build.sh`.  
    - **Manual install:** `cd src/bindings/python/dx_postprocess && python3 -m pip install .`  

For detailed usage examples and API references, please refer to the documentation in 
[**Section. DX-APP Python Post-processing**](./src/bindings/python/dx_postprocess/README.md)

## Application Examples (`src/python_example/`)

These templates utilize `dx_engine` (for inference) and `dx_postprocess` (for acceleration). Users can choose from four variants depending on their performance requirements.  

The refactored Python tree is organized by **task → model family → variant**, with a shared `common/` layer providing base interfaces, shared processors, generic sync/async runners, visualizers, and input abstraction — the same factory-based architecture as the C++ side. For structure and contributor-facing rules, refer to [DX-APP Python Usage Guide](./docs/source/docs/05_DX-APP_Python_Example_Usage_Guide.md) and [DX-APP Example Source Structure](./docs/source/docs/11_DX-APP_Example_Source_Structure.md).

**Task-Based Structure**  

Templates are categorized by task across multiple task directories. All examples share the `common/` runtime layer for processors, runners, and visualizers. Representative tasks:  

- **Classification:** EfficientNet, AlexNet, ResNet, MobileNet, etc.  
- **Object Detection:** YOLOv5/v7/v8/v9/v10/v11/v12, YOLOX, NanoDet, DAMOYOLO, SSD  
- **Face Detection:** SCRFD, YOLOv5Face, YOLOv7Face, RetinaFace  
- **Pose Estimation:** YOLOv8-Pose  
- **Segmentation:** BiSeNet, DeepLabV3+, SegFormer, YOLOv8Seg  
- **Image Retrieval / Visual Place Recognition / Person Re-ID:** CLIP RN50, EigenPlaces, PP-ShiTuV2, RepVGG-A0 -- each ranks a query descriptor against a gallery built on the NPU (`sample/gallery/*.bin`, one shared format both example trees read -- see `scripts/build_gallery_database.py`). A relative gallery or thumbnail path that does not exist from the working directory is read against the repository root, in both trees, so these examples rank and draw their matches the same from any working directory  
- **Image Matting:** PP-Matting HRNet-W48 (continuous alpha matte, not a class map)  
- **Depth, Embedding, OBB, Denoising, Enhancement, Super Resolution, Hand Landmark, Attribute Recognition, PPU**  

Functional Variants  

| **Variant** | **Post-processing** | **Threading Model** | **Recommendation** | 
|----|----|----|----|
| `*_sync.py` | Pure Python | Synchronous | Learning & Logic Debugging | 
| `*_async.py` | Pure Python | Asynchronous | Basic performance optimization | 
| `*_sync_cpp_postprocess.py` | C++ Binding | Synchronous | Accelerating heavy CPU tasks | 
| `*_async_cpp_postprocess.py` | C++ Binding | Asynchronous | Maximum FPS (Recommended) | 

---
<a name="quick-start-guide"></a>
# Quick Start Guide

Follow these steps to transition from a fresh installation to your first successful inference on DEEPX NPU.  

**Step 1. Environment Setup & Verification**  

First, verify that the NPU driver and DX-RT are correctly installed. This is a mandatory prerequisite.  
```bash
# Verify hardware connection and driver status
dxrt-cli -s
```

!!! warning "Caution: Prerequisite Check"  
      If the command above fails, you **must** manually install the NPU Drivers and DX-RT before continuing with DX-APP setup. Refer to the installation and build documentation under `docs/source/docs/`.  


Once hardware is verified, install the necessary toolchain and system libraries.  
```bash
# Install Build tools, CMake, and OpenCV
./install.sh --all
```

**Step 2. Asset Acquisition**  

Download the required models and sample media files.  
```bash
# Interactive mode (default) — select categories and models from a menu
./setup.sh

# Non-interactive — download all models automatically without prompts
./setup.sh --all

# Preview what would be downloaded (no actual download)
./setup.sh --dry-run

# Download only a specific category (name as shown by --list)
./setup.sh --category "Object Detection"

# Download specific models by name
./setup.sh --models yolov8n yolov9s efficientnet_lite0
```

**`setup.sh` Options**

| Option | Description |
|--------|-------------|
| `--all` | Download all models non-interactively |
| `--dry-run` | List models that would be downloaded without downloading |
| `--list` | List available models without downloading |
| `--workers=<N>` | Parallel download threads (default: 4) |
| `--category=<name>` | Download models of a specific category only |
| `--models <m1> [m2...]` | Download specific models by name |
| `--no-json` | Skip JSON metadata file downloads |
| `--manifest=<path>` | Use an alternate manifest JSON file |
| `--force` | Force overwrite if files already exist |
| `--verbose` | Enable verbose logging |

- **Models:** Saved to `assets/models/`. By default, an interactive menu lets you select which model categories and models to download. Use `--all` to skip the menu and download everything automatically.
- **Media:** Saved to `assets/videos/`.

For most users, `./setup.sh` is the only required entry point for asset preparation.

If you are maintaining examples rather than only consuming them, review [DX Tool Guide](./docs/source/docs/10_DX-APP_DX-Tool_Guide.md).

**Step 3. Compilation**  

Build the C++ binaries and the Python dx_postprocess bindings simultaneously.  
```bash
# Standard build
./build.sh

# For a clean rebuild, use: ./build.sh --clean

# Build specific targets only (faster incremental builds)
./build.sh --target yolov9-s_640x640_sync yolov9-s_640x640_async

# List all available build targets
./build.sh --target list
```

- **Output:** Binaries are located in `bin/`, and shared libraries are in their respective build folders.  

**Step 4. Execution Examples**  

The quickest way to explore the AI task categories (26 demo tasks) is the unified interactive demo script:

```bash
# Interactive — select task, mode, and input type from menus
./run_demo.sh

# Non-interactive — run a specific task directly
./run_demo.sh --task 0 --mode 1 --input 2   # YOLOv7, C++ sync, image
./run_demo.sh --task 0 --mode 2 --input 1   # YOLOv7, C++ async, video
./run_demo.sh --show-log                    # Enable verbose logs
```

| Option | Description |
|--------|-------------|
| `--task NUM` | Pre-select task (0–25) |
| `--mode NUM` | Pre-select mode (1=cpp_sync, 2=cpp_async, 3=py_sync, …) |
| `--input NUM` | Pre-select input (1=video, 2=image) |
| `--show-log` | Enable verbose log output (default: quiet) |

Shipped multi-model graphs have their own checker:

```bash
./run_multi_model_demo.sh                           # menu
./run_multi_model_demo.sh --check                   # every sample graph, no NPU
./run_multi_model_demo.sh --all                     # --check, then each scenario
./run_multi_model_demo.sh --scenario hand_cascade
```

| Option | Description |
|--------|-------------|
| `--scenario ID\|NUM` | `hand_cascade`, `logistics_volume`, `dms`, or `0`–`2` |
| `--async` | Use `multi_model_graph_async` |
| `--check` | Validate every `src/cpp_example/multi_model_graph/*.json` |
| `--all` | `--check`, then each scenario with `multi_model_graph_sync` |

**Demo Task ↔ Model Reference**

Each of the 26 demo tasks uses exactly one model. When you run `run_demo.sh`, any missing models and videos are **automatically downloaded** — no manual `setup.sh` required.

| # | Demo Task | Model File | Group |
|--:|-----------|-----------|-------|
| 0 | Object Detection         (YOLOv7) | `yolov7_640x640.dxnn` | Detection |
| 1 | Object Detection         (YOLOv11N) | `yolo11-n_640x640.dxnn` | Detection |
| 2 | Face Detection           (SCRFD500M) | `scrfd-500m_640x640.dxnn` | Detection |
| 3 | OBB Detection            (YOLO26N-OBB) | `yolo26-n-obb_1024x1024.dxnn` | Detection |
| 4 | Pose Estimation          (YOLOv8s-Pose) | `yolov8-s-pose_640x640.dxnn` | Pose & Landmark |
| 5 | Hand Landmark            (HandLandmarkLite) | `mediapipe-hands-lite_224x224.dxnn` | Pose & Landmark |
| 6 | Face Alignment           (3DDFA-V2-MobileNetV1) | `3ddfa-v2_mobilenetv1_120x120.dxnn` | Pose & Landmark |
| 7 | Instance Segmentation    (YOLOv8N-Seg) | `yolov8-n-seg_640x640.dxnn` | Segmentation |
| 8 | Semantic Segmentation    (DeepLabV3+MobileNet) | `deeplabv3plus_mobilenetv1_512x512.dxnn` | Segmentation |
| 9 | Classification           (ResNet50) | `resnet50_224x224.dxnn` | Classification |
| 10 | Depth Estimation         (YOLO26-Depth-S) | `yolo26-depth-s_768x768.dxnn` | Depth Estimation |
| 11 | Image Denoising          (DnCNN-50) | `dncnn-50_512x512.dxnn` | Image Restoration |
| 12 | Super Resolution         (ESPCN-X4) | `espcn-x4_17x17.dxnn` | Image Restoration |
| 13 | Image Enhancement        (Zero-DCE) | `zerodce_400x600.dxnn` | Image Restoration |
| 14 | Embedding                (ArcFace) | `arcface_mobilefacenet_112x112.dxnn` | Recognition |
| 15 | Attribute Recognition    (DeepMAR-ResNet50) | `deepmar_resnet50_224x224.dxnn` | Recognition |
| 16 | PPU Pipeline             (YOLOv7-PPU) | `yolov7_640x640_ppu.dxnn` | PPU |
| 17 | Keypoint Detection       (SuperPoint) | `superpoint_480x640.dxnn` | Keypoint & Pose |
| 18 | Object Pose Estimation   (DOPE) | `dope-hope-ketchup_480x640.dxnn` | Keypoint & Pose |
| 19 | Panoptic Driving         (YOLOPv2) | `yolopv2_384x640.dxnn` | Driving & 3D |
| 20 | 3D Object Detection      (SFA3D) | `sfa3d_608x608.dxnn` | Driving & 3D |
| 21 | Hand Detection           (MediaPipe Palm) | `mediapipe-hand-detector_192x192.dxnn` | Hand Detection |
| 22 | Image Retrieval          (CLIP RN50) | `clip-img_resnet50_224x224_openai.dxnn` | Retrieval & Matting |
| 23 | Visual Place Recognition (EigenPlaces R18) | `eigenplaces-resnet18_512x512.dxnn` | Retrieval & Matting |
| 24 | Person Re-ID             (RepVGG-A0) | `repvgg-a0-reid_256x128.dxnn` | Retrieval & Matting |
| 25 | Image Matting            (PP-Matting HRNet-W48) | `ppmatting-hrnet-w48-composition_512x512.dxnn` | Retrieval & Matting |

Sizes are not listed here because they move with each DX Model Zoo release; `./setup.sh --demo-models` downloads exactly this set and reports each file. The
sample video pack is a separate ~1.1 GB download (`./setup_sample_videos.sh`).

To download only specific demo models without running the demo:
```bash
./setup.sh --models YoloV7 SCRFD500M ResNet50
```

> **TIP — Running other models**  
> `run_demo.sh` showcases 26 representative models. To run or benchmark **all 497 registered variants**,
> use the **example runner** or the **DX Model Tool**:
>
> ```bash
> # Interactive — 6-stage guided menu (language, category, model filter, etc.)
> scripts/run_examples.sh
> ./scripts/dx_tool.sh run          # same interactive menu
>
> # Interactive benchmark with performance report
> ./scripts/dx_tool.sh bench
>
> # Non-interactive — pass options directly
> scripts/run_examples.sh --lang cpp --category face_detection --filter scrfd
> ./scripts/dx_tool.sh run --lang cpp --category face_detection --filter scrfd
> ./scripts/dx_tool.sh bench --lang both --filter yolov8 --loops 5
> ```
>
> Run `./scripts/dx_tool.sh help` for all available commands (add/delete/search/validate models, etc.).

Alternatively, run individual binaries or scripts directly:

**Simplest Execution (auto-download model + default sample image)**
```bash
# Just specify the model — everything else is automatic
python src/python_example/object_detection/yolov7/yolov7_640x640/yolov7_640x640_sync.py --model assets/models/yolov7_640x640.dxnn
# → Model auto-downloaded if missing
# → Default sample image auto-selected for the task
```

C++ Implementation (High Performance)  
```bash
# Static Image Inference (Synchronous)
./bin/yolov9-s_640x640_sync \
-m assets/models/yolov9-s_640x640.dxnn \
-i sample/img/sample_kitchen.jpg

# Video Stream Inference (Asynchronous)
./bin/yolov9-s_640x640_async \
-m assets/models/yolov9-s_640x640.dxnn \
-v assets/videos/dance-group.mov
```

Python Implementation (Rapid Prototyping)  
```bash
# Python Baseline (Synchronous)
python src/python_example/object_detection/yolov9/yolov9-s_640x640/yolov9-s_640x640_sync.py \
   --model assets/models/yolov9-s_640x640.dxnn \
   --image sample/img/sample_kitchen.jpg

# Python Optimized (Asynchronous + C++ Post-processing)
python src/python_example/object_detection/yolov9/yolov9-s_640x640/yolov9-s_640x640_async_cpp_postprocess.py \
  --model assets/models/yolov9-s_640x640.dxnn \
    --video assets/videos/dance-group.mov 
```

**Output and Analysis**  

Following execution, a window will render results (`boxes/masks`), and the console will output a **Performance Summary** (`Latency/FPS`). For additional usage details, refer to [DX-APP C++ Usage Guide](./docs/source/docs/03_DX-APP_CPP_Example_Usage_Guide.md) and [DX-APP Python Usage Guide](./docs/source/docs/05_DX-APP_Python_Example_Usage_Guide.md).  

---

# Storage-Constrained Platforms (RPi, Edge Devices)

When running on platforms with limited storage (e.g., Raspberry Pi 5, embedded boards), use the following strategies to minimize disk usage.

**Disk Usage Summary**

| Asset | Count | Size |
|-------|------:|-----:|
| Demo models | 26 | varies by Model Zoo release (`./setup.sh --demo-models --dry-run` lists them) |
| All registered models | 497 | several GB |
| Sample videos | 20 | ~1.1 GB |
| Sample images (bundled) | — | ~5 MB |

**Strategy 1 — Let `run_demo.sh` handle it**  
Just run `./run_demo.sh`. It automatically downloads **only the 26 demo models** on first run. No need to run `setup.sh --all`.

**Strategy 2 — Download only what you need**  
```bash
# Download a single category
./setup.sh --category "Face Detection"     # ~30 MB for 15 models

# Download specific models by name
./setup.sh --models SCRFD500M YOLOV11N     # ~9 MB total

# Preview before downloading
./setup.sh --list                          # List all available models
./setup.sh --dry-run                       # Show what would be downloaded
```

**Strategy 3 — Skip video downloads**  
Videos (~1.1 GB) are much larger than models. If storage is tight:
- Use `--input 2` (image mode) with `run_demo.sh` — images are bundled in the repo and require no extra download
- Run demos directly with `--image` flag instead of `--video`

**Strategy 4 — Clean up after testing**  
```bash
# Remove all downloaded models
rm -rf assets/models/*

# Remove all downloaded videos
rm -rf assets/videos/*
```

---

# Third-Party License Notice

Sample models (`.dxnn`) and dataset images included in DX-APP are provided for **evaluation and development purposes only** and are **not licensed for commercial deployment**.

- **Models** are compiled from third-party open-source projects (e.g., AGPL-3.0, GPL-3.0, non-commercial research licenses). Commercial use requires obtaining licenses from the original model providers or using your own commercially licensed models.
- **Datasets** (e.g., ImageNet, DOTA, COCO, Pascal VOC) are subject to their respective license terms, most of which restrict usage to non-commercial research and education.

For full details, see [Appendix: Third-Party License Notice](./docs/source/docs/Appendix_Third_Party_License.md).
