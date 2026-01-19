# DX-APP Overview

**DX-APP** is a production-ready suite of application templates designed to accelerate the development of AI services on **DEEPX NPUs**. It bridges the gap between raw model deployment and high-performance application engineering.  

**Key Features & Objectives**  

- **Rapid Deployment:** Ready-to-run demos for Classification (`EfficientNet`), Detection (`YOLO series, SCRFD`), and Segmentation (`DeepLabv3, YOLOv8seg`).  
-	**Dual-Language Flexibility:** High-performance **C++** for production and **Python** for rapid prototyping, sharing a **unified C++ post-processing engine**.  
- **Hardware Acceleration:** Native support for **PPU-enabled models** and **Async templates** that overlap pipeline stages to maximize FPS.  
- **Modular Design:** Clean, task-oriented templates that serve as reusable blueprints for custom commercial applications.  

**Reference Documentation**  

For deeper technical specifications, refer to the `docs/` directory  

- **Application Overview:** [01_DXNN_Application_Overview.md](./docs/source/docs/01_DXNN_Application_Overview.md)  
- **Installation & Build Guide:** [02_DX-APP_Installation_and_Build.md](./docs/source/docs/02_DX-APP_Installation_and_Build.md)  

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
│   ├── cpp_example/          # C++ end to end examples (sync/async, by task/model)
│   ├── python_example/       # Python end to end examples
│   ├── postprocess/          # C++ post processing libraries (shared C++/Python)
│   └── bindings/
│       └── python/
│           └── dx_postprocess/  # pybind11 bindings for C++ post processing
├── assets/                   # Downloaded models/videos (via setup.sh)
├── scripts/                  # Helper scripts to run demos
├── build.sh                  # Top level build script
├── install.sh                # Dependency and OpenCV installer
└── docs/                     # Detailed documentation
```

## Prerequisites

Before building the templates, ensure your system meets the following hardware and software requirements.  

**A. DEEPX Runtime (DX-RT) and NPU Drivers**  

To utilize NPU acceleration, you **must** install the kernel-mode drivers and the user-space runtime library  

- **DEEPX NPU Linux Driver:** Required for low-level NPU communication. [Github Repository](https://github.com/DEEPX-AI/dx_rt_npu_linux_driver>)  
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
- **Asset & Dependency Preparation:** Install required libraries (OpenCV, Build tools) via `./install.sh` and download optimized `.dxnn` models using `./setup.sh`.  
- **Build & Execution:** Compile the source code using `./build.sh` and run the generated binaries or Python scripts located in the `bin/` or `src/python_example/` directories.  

---

# C++ Application Templates (src/cpp_example/)

These templates provide high-performance, production-ready references for building applications using the DX-RT C++ API.  

**Pipeline Architecture**  

Each template follows a self-contained pipeline designed for modularity  

- **Step 1. Input:** Image, Video, Camera, or RTSP stream.  
- **Step 2. Pre-process:** Resizing and normalization.  
- **Step 3. Inference:** Execution on the NPU via **DX-RT**.  
- **Step 4. Post-process:** Call to shared C++ classes in `src/postprocess/` (e.g., NMS, box scaling).  
- **Step 5. Output:** Result rendering (Display) or storage (Save).  

**Design Variants**  

To help developers optimize for specific hardware targets, templates are provided in two execution patterns  

- **Synchronous (`*_sync.cpp`): * Logic:** A single-threaded, sequential loop (**Input → Inference → Output**).  
      : **Use Case:** Best for single-image processing and simplified debugging.  

- **Asynchronous (`*_async.cpp`): * Logic:** Uses multi-threading and the `RunAsync()` API to overlap stages. While the NPU performs inference on Frame **N**, the CPU simultaneously handles pre-processing for Frame **N+1** and post-processing for Frame **N-1**.  
      : **Use Case:** Essential for maximizing **FPS** on live video streams and ensuring high NPU utilization.  

---

# Post-processing Libraries (src/postprocess/)

These libraries transform raw NPU output tensors into structured, actionable data. By centralizing this logic in C++, DX-APP ensures high-performance execution and consistency across all application variants.  

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

A major architectural advantage of DX-APP is the portability of these libraries  

- **For C++ Developers:** Modules are integrated directly by including the relevant headers in the application source.  
- **For Python Developers:** The same C++ logic is accessed via the `dx_postprocess` pybind11 module, eliminating the performance overhead typically associated with Python-based decoding.  

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

**Task-Based Structure**  

Templates are categorized by task. Within each model folder (e.g., yolov9/), you will find the following scripts  

- **Classification:** `EfficientNet`  
- **Object Detection:** `YOLOv5/7/8/9, SCRFD`, etc.  
- **Segmentation:** `DeepLabv3, YOLOv8seg`  

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
    If the command above fails, you **must** manually install the NPU Drivers and DX-RT as described in [**Section. DX-APP C++ Examples - Prerequisites**](./src/cpp_example/README.md).  


Once hardware is verified, install the necessary toolchain and system libraries.  
```bash
# Install Build tools, CMake, and OpenCV
./install.sh --all
```

**Step 2. Asset Acquisition**  

Download the pre-compiled NPU models and sample media files required for the demos.  
```bash
# Fetch models and videos
./setup.sh
```

- **Models:** Saved to `assets/models/`  
- **Media:** Saved to `assets/videos/` or `assets/images/`  

**Step 3. Compilation**  

Build the C++ binaries and the Python dx_postprocess bindings simultaneously.  
```bash
# Standard build
./build.sh

# For a clean rebuild, use: ./build.sh --clean
```

- **Output:** Binaries are located in `bin/`, and shared libraries are in their respective build folders.  

**Step 4. Execution Examples (YOLOv9)**  

Test the NPU performance using the YOLOv9 object detection template.  

C++ Implementation (High Performance)  
```bash
# Static Image Inference (Synchronous)
./bin/yolov9_sync \
-m assets/models/YOLOV9S.dxnn \
-i sample/img/1.jpg

# Video Stream Inference (Asynchronous)
./bin/yolov9_async \
-m assets/models/YOLOV9S.dxnn \
-v assets/videos/dance-group.mov
```

Python Implementation (Rapid Prototyping)  
```bash
# Python Baseline (Synchronous)
python src/python_example/object_detection/yolov9/yolov9_sync.py \
   --model assets/models/YOLOV9S.dxnn \
   --image sample/img/1.jpg

# Python Optimized (Asynchronous + C++ Post-processing)
python src/python_example/object_detection/yolov9/yolov9_async_cpp_postprocess.py \
  --model assets/models/YOLOV9S.dxnn \
    --video assets/videos/dance-group.mov 
```

**Output and Analysis**  

Following execution, a window will render results (`boxes/masks`), and the console will output a **Performance Summary** (`Latency/FPS`) as described in [**Section. DX-APP C++ Examples**](./src/cpp_example/README.md).  

---
