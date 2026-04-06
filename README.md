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
| 02 | [Installation and Build](./docs/source/docs/02_DX-APP_Installation_and_Build.md) | Prerequisites, build steps (Linux/Windows) |
| 03 | [C++ Example Usage Guide](./docs/source/docs/03_DX-APP_CPP_Example_Usage_Guide.md) | C++ template structure & execution guide |
| 04 | [C++ Example Tests](./docs/source/docs/04_DX-APP_CPP_Example_Test.md) | C++ test framework & coverage |
| 05 | [Python Example Usage Guide](./docs/source/docs/05_DX-APP_Python_Example_Usage_Guide.md) | Python template structure & execution guide |
| 06 | [Python Example Tests](./docs/source/docs/06_DX-APP_Python_Example_Test.md) | Python test framework (pytest) |
| 07 | [C++ Post-processing](./docs/source/docs/07_DX-APP_CPP_PostProcess_Overview.md) | C++ post-processing library design |
| 08 | [Python Post-processing](./docs/source/docs/08_DX-APP_Pybind_PostProcess_Overview.md) | pybind11 bindings (dx_postprocess) |
| 09 | [Project Overview](./docs/source/docs/09_DX-APP_Project_Overview.md) | Repository layout, CLI reference, advanced features |
| 10 | [DX Tool Guide](./docs/source/docs/10_DX-APP_DX-Tool_Guide.md) | Developer tooling (dx_tool.sh) |
| 11 | [Example Source Structure](./docs/source/docs/11_DX-APP_Example_Source_Structure.md) | Source tree conventions & contributor guide |
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

The project is structured to separate core logic from language-specific implementations.  
```text
dx_app/
├── src/
│   ├── cpp_example/            # C++ end-to-end examples (137 models across 16 tasks)
│   │   └── common/             # ← Shared C++ runtime layer
│   │       ├── base/           #   Abstract interfaces (IFactory, IProcessor, ...)
│   │       ├── processors/     #   44 shared processors (42 post + 2 pre)
│   │       ├── runner/         #   24 task-specific sync/async runner pairs
│   │       ├── inputs/         #   Image/Video/Camera/RTSP input sources
│   │       ├── visualizers/    #   12 task-specific visualizers
│   │       ├── config/         #   ModelConfig loader
│   │       └── utility/        #   Labels, preprocessing, profiling, run_dir, signal_handler, verify_serialize
│   ├── python_example/         # Python end-to-end examples (137 models across 16 tasks)
│   │   └── common/             # ← Shared Python runtime layer
│   │       ├── base/           #   Abstract interfaces (IFactory, IProcessor, ...)
│   │       ├── processors/     #   35 shared post-processors
│   │       ├── runner/         #   SyncRunner, AsyncRunner, run_dir, verify_serialize, args
│   │       ├── inputs/         #   Image/Video/Camera/RTSP input sources
│   │       ├── visualizers/    #   10 task-specific visualizers
│   │       ├── config/         #   ModelConfig loader
│   │       └── utility/        #   Labels, preprocessing, profiling
│   ├── postprocess/            # C++ post-processing (consumed by pybind11 bindings)
│   ├── utility/                # Shared support code used by build flow
│   └── bindings/
│       └── python/
│           └── dx_postprocess/ # pybind11 bindings wrapping src/postprocess/
├── config/
│   ├── model_registry.json     # Model registry — single source of truth
│   └── test_models.conf        # Test model configuration
├── scripts/                    # Developer tools, validation, and helper scripts
├── tests/                      # pytest-based test suites
│   ├── common/                 #   Shared test constants & utilities
│   ├── cpp_example/            #   C++ tests (CLI, E2E, visualization, features)
│   ├── python_example/         #   Python tests (unit, integration, CLI, E2E, visualization)
│   └── test_visualization_result/  #   Generated visualization images (gitignored)
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

**B. Development Toolchain and Libraries**  

The following tools are required to compile the C++ templates and the Python dx_postprocess bindings.  

**B-a.** Build System  

- **CMake:** Version 3.14 or higher.  
- **Compiler:** C++14-compatible (GCC 7.5+, Clang, etc.).  
- **Build Utility:** make or ninja.  

**B-b.** Core Libraries  

-	**OpenCV:** Version 4.2.0 or higher (**4.5.5 recommended**). This is used for image I/O and pre/post-processing visualization.  
-	**Python Environment:** Python 3.8 or higher and pip are required for Python-based examples and pybind11 integration.  

## Development Workflow Overview

The process from environment setup to running your first AI application is divided into three main phases. For detailed commands and execution steps, please refer to the [**Section. Quick Start Guide**](#quick-start-guide).  

- **Hardware & Driver Verification:** Ensure the NPU is recognized by the system using the `dxrt-cli` tool.  
- **Asset & Dependency Preparation:** Install required libraries (OpenCV, Build tools) via `./install.sh` and prepare models/videos via `./setup.sh`. Model assets are fetched through the current [DX-ModelZoo](https://developer.deepx.ai/modelzoo/)-based setup flow.  
- **Build & Execution:** Compile the source code using `./build.sh` and run the generated binaries or Python scripts located in the `bin/` or `src/python_example/` directories.  

For contributor workflows such as model onboarding, validation, filtered execution, and benchmarking, refer to [DX Tool Guide](./docs/source/docs/10_DX-APP_DX-Tool_Guide.md).

---

# CLI Reference

All C++ and Python examples share a consistent set of command-line arguments.

## Common Arguments

| Flag | C++ | Python | Description |
|------|-----|--------|-------------|
| `-m` / `--model` | `-m` (required) | `--model` (required) | Path to `.dxnn` model file |
| `-i` / `--image` | `-i` | `--image` | Input image file or directory |
| `-v` / `--video` | `-v` | `--video` | Input video file |
| `-c` / `--camera` | `-c` | `--camera` | Camera device index |
| `-r` / `--rtsp` | `-r` | `--rtsp` | RTSP stream URL |
| `-l` / `--loop` | `-l` (default: auto) | `--loop` (default: 1) | Inference repeat count |
| `--no-display` | `--no-display` | `--no-display` | Disable visualization window |
| `--show-log` / `--verbose` | `--show-log` | `--verbose` | Enable verbose log output (default: quiet) |
| `-s` / `--save` | `--save` | `--save` | Save rendered output to run directory |
| `--save-dir` | `--save-dir` | `--save-dir` | Base output directory (default: `artifacts/`) |
| `--dump-tensors` | `--dump-tensors` | `--dump-tensors` | Dump raw input/output tensors to files |
| `--config` | `--config` | `--config` | Model config JSON path (auto-detected if omitted) |
| `-h` / `--help` | `-h` | `-h` | Show usage |

> **Input Source Rule:** Exactly one of `--image`, `--video`, `--camera`, or `--rtsp` must be specified.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DXAPP_SAVE_IMAGE` | When set to a file path, saves the visualization output to that path (no `--save` required) |
| `DXAPP_VERIFY` | When set to `1`, dumps post-processing results to `logs/verify/{model}.json` for numerical verification |

---

# Advanced Features

DX-APP includes several production-oriented features built into all templates.

## Signal Handling

All runners register SIGINT/SIGTERM handlers for graceful shutdown. Pressing Ctrl+C during inference prints `"Interrupted by user"` and cleanly exits, releasing all resources.

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

A complete verification pipeline for validating inference correctness:
1. Set `DXAPP_VERIFY=1` before running any example
2. Post-processing results are serialized to `logs/verify/{model}.json`
3. Run `scripts/verify_inference_output.py` to validate against task-specific rules
4. Supports all 12 result types (Detection, Classification, Pose, Segmentation, etc.)

## Tensor Dump (`--dump-tensors`)

Dumps raw input/output tensors for debugging. On exception, tensors are auto-dumped with a `reason.txt` file. C++ outputs `.bin` files; Python outputs `.npy` files.

## Model Config (`--config`)

Runtime parameters (score threshold, NMS threshold, top-k) can be tuned per-model via `config.json`. If not specified, the runner auto-detects `config.json` adjacent to the model or script.

## Version Compatibility

All runners verify:
- **DX-RT library** ≥ 3.0.0
- **Compiled model format** ≥ v7

Incompatible versions produce a clear error message before exit.

## Headless Mode

Python runners detect the absence of `DISPLAY`/`WAYLAND_DISPLAY` and skip `cv2.imshow()` automatically. Use `--no-display` for explicit headless operation in both C++ and Python.

---

# C++ Application Templates (src/cpp_example/)

These templates provide high-performance, production-ready references for building applications using the DX-RT C++ API.  

The refactored C++ tree is organized by **task → model family → variant**, with a shared `common/` layer providing base interfaces, 44 processors, 24 task-specific runners, 12 visualizers, and input abstraction. Each model directory delegates to `common/` via the factory pattern. For details, refer to [DX-APP C++ Usage Guide](./docs/source/docs/03_DX-APP_CPP_Example_Usage_Guide.md) and [DX-APP Example Source Structure](./docs/source/docs/11_DX-APP_Example_Source_Structure.md).

**Pipeline Architecture**  

Each template follows a self-contained pipeline designed for modularity  

- **Step 1. Input:** Image, Video, Camera, or RTSP stream (via `common/inputs/`).  
- **Step 2. Pre-process:** Resizing and normalization (via `common/utility/`).  
- **Step 3. Inference:** Execution on the NPU via **DX-RT**.  
- **Step 4. Post-process:** Call to shared C++ processors in `common/processors/` (e.g., NMS, box scaling).  
- **Step 5. Output:** Result rendering via `common/visualizers/` (Display) or storage (Save).  

**Design Variants**  

To help developers optimize for specific hardware targets, templates are provided in two execution patterns  

- **Synchronous (`*_sync.cpp`): * Logic:** A single-threaded, sequential loop (**Input → Inference → Output**).  
      : **Use Case:** Best for single-image processing and simplified debugging.  

- **Asynchronous (`*_async.cpp`): * Logic:** Uses multi-threading and the `RunAsync()` API to overlap stages. While the NPU performs inference on Frame **N**, the CPU simultaneously handles pre-processing for Frame **N+1** and post-processing for Frame **N-1**.  
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
- **Installation:** -Automatically compiled during `./build.sh`.  
    - **Manual install:** `cd src/bindings/python/dx_postprocess && pip install`.  

For detailed usage examples and API references, please refer to the documentation in 
[**Section. DX-APP Python Post-processing**](./src/bindings/python/dx_postprocess/README.md)

## Application Examples (`src/python_example/`)

These templates utilize `dx_engine` (for inference) and `dx_postprocess` (for acceleration). Users can choose from four variants depending on their performance requirements.  

The refactored Python tree is organized by **task → model family → variant**, with a shared `common/` layer providing base interfaces, 35 processors, generic sync/async runners, 10 visualizers, and input abstraction — the same factory-based architecture as the C++ side. For structure and contributor-facing rules, refer to [DX-APP Python Usage Guide](./docs/source/docs/05_DX-APP_Python_Example_Usage_Guide.md) and [DX-APP Example Source Structure](./docs/source/docs/11_DX-APP_Example_Source_Structure.md).

**Task-Based Structure**  

Templates are categorized by task across multiple task directories. All examples share the `common/` runtime layer for processors, runners, and visualizers. Representative tasks:  

- **Classification:** EfficientNet, AlexNet, ResNet, MobileNet, etc.  
- **Object Detection:** YOLOv5/v7/v8/v9/v10/v11/v12, YOLOX, NanoDet, DAMOYOLO, SSD  
- **Face Detection:** SCRFD, YOLOv5Face, YOLOv7Face, RetinaFace  
- **Pose Estimation:** YOLOv8-Pose  
- **Segmentation:** BiSeNet, DeepLabV3+, SegFormer, YOLOv8Seg  
- **Depth, Embedding, OBB, Denoising, Enhancement, Super Resolution, Hand Landmark, Attribute Recognition, Re-ID, PPU**  

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

# Download only a specific category
./setup.sh --category=object_detection

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
| `--manifest=<path>` | Use an alternate manifest JSON file (e.g., for air-gapped environments) |
| `--force` | Force overwrite if files already exist |
| `--verbose` | Enable verbose logging |

- **Models:** Saved to `assets/models/`. By default, an interactive menu lets you select which model categories and models to download. Use `--all` to skip the menu and download everything automatically.  
- **Media:** Saved to `assets/videos/`.  
- **Air-gapped environments:** Provide a local manifest JSON via `--manifest=/path/to/manifest.json`.

For most users, `./setup.sh` is the only required entry point for asset preparation.

If you are maintaining examples rather than only consuming them, review [DX Tool Guide](./docs/source/docs/10_DX-APP_DX-Tool_Guide.md).

**Step 3. Compilation**  

Build the C++ binaries and the Python dx_postprocess bindings simultaneously.  
```bash
# Standard build
./build.sh

# For a clean rebuild, use: ./build.sh --clean
```

- **Output:** Binaries are located in `bin/`, and shared libraries are in their respective build folders.  

**Step 4. Execution Examples**  

The quickest way to explore all 16 AI task categories (17 demos) is the unified interactive demo script:

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
| `--task NUM` | Pre-select task (0–16) |
| `--mode NUM` | Pre-select mode (1=cpp_sync, 2=cpp_async, 3=py_sync, …) |
| `--input NUM` | Pre-select input (1=video, 2=image) |
| `--show-log` | Enable verbose log output (default: quiet) |

> **TIP — Running other models**  
> `run_demo.sh` showcases 17 representative models. To run or benchmark **all 130+ registered models**,
> use the **DX Model Tool**:
>
> ```bash
> # Interactive — select category, language, filter from menus
> ./scripts/dx_tool.sh run
>
> # Interactive benchmark with performance report
> ./scripts/dx_tool.sh bench
>
> # Non-interactive — pass options directly
> ./scripts/dx_tool.sh run --lang cpp --category face_detection --filter scrfd
> ./scripts/dx_tool.sh bench --lang both --filter yolov8 --loops 5
> ```
>
> Run `./scripts/dx_tool.sh help` for all available commands (add/delete/search/validate models, etc.).

Alternatively, run individual binaries or scripts directly:

C++ Implementation (High Performance)  
```bash
# Static Image Inference (Synchronous)
./bin/yolov9s_sync \
-m assets/models/YoloV9S.dxnn \
-i sample/img/sample_kitchen.jpg

# Video Stream Inference (Asynchronous)
./bin/yolov9s_async \
-m assets/models/YoloV9S.dxnn \
-v assets/videos/dance-group.mov
```

Python Implementation (Rapid Prototyping)  
```bash
# Python Baseline (Synchronous)
python src/python_example/object_detection/yolov9s/yolov9s_sync.py \
   --model assets/models/YoloV9S.dxnn \
   --image sample/img/sample_kitchen.jpg

# Python Optimized (Asynchronous + C++ Post-processing)
python src/python_example/object_detection/yolov9s/yolov9s_async_cpp_postprocess.py \
  --model assets/models/YoloV9S.dxnn \
    --video assets/videos/dance-group.mov 
```

**Output and Analysis**  

Following execution, a window will render results (`boxes/masks`), and the console will output a **Performance Summary** (`Latency/FPS`). For additional usage details, refer to [DX-APP C++ Usage Guide](./docs/source/docs/03_DX-APP_CPP_Example_Usage_Guide.md) and [DX-APP Python Usage Guide](./docs/source/docs/05_DX-APP_Python_Example_Usage_Guide.md).  

---
