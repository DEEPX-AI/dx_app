## Introduction

C++ examples are the starting point for building **DEEPX NPU–based AI applications in C++**. Each example is **self-contained end-to-end pipeline** that demonstrate how to transform raw input into real-time inference results.  

The examples are designed with the following technical goals.  

- **Modular Pipeline Design:** Each example follows a standardized flow: **Input → Pre-process → Inference → Post-process → Visualization**. By isolating the post-processing into **a Shared C++ Library** (`src/postprocess/`), we ensure that the core logic is reusable and consistent across both C++ and Python environments.  

- **Performance Optimization Patterns:** The guide introduces two distinct execution variants—**Synchronous (Sync)** and **Asynchronous (Async)**. These allow you to compare sequential processing against multi-threaded scheduling, helping you understand how to maximize NPU utilization and minimize latency.  

- **Flexible Ingestion Backends:** Beyond basic inference, the examples showcase advanced input handling using **GStreamer, OpenCV,** and **V4L2**. This provides a practical foundation for integrating various sources like high-resolution cameras, RTSP network streams, and local video files.  

- **Profiling-Driven Development:** Every example includes built-in telemetry. By analyzing the **Performance Summary**, you can identify pipeline bottlenecks and fine-tune your application for production-level throughput (**FPS**).  

- **Extensible Templates:** These examples are not just demos but **practical application templates**. You can easily adapt them by swapping the model (`.dxnn`), modifying pre/post-processing parameters, or integrating custom input/output logic to meet your specific project requirements.  

---

## Prerequisites

Before building and running the C++ examples, make sure the following environment is prepared.

- **DEEPX Runtime(DX-RT)**  
     : Purpose: Required runtime to execute `.dxnn` models on the NPU.  
     : Setup: The `dxrt` libraries **must** be installedand their path added to the environment `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/dxrt/lib`.  

- **Build Tools**  
     : CMake: Version 3.16 or later  
     : Compiler: C++14-compatible compiler (GCC, Clang)  
     : Build Tool: `Make` or `Ninj`  

- **Dependencies**  
     : OpenCV: Required for image/video I/O and visualization  
      **NOTE.** Ensure the development headers are installed (e.g., `libopencv-dev` on Ubuntu).  

- **Assets (Models and Data)**  
    To download the required sample models and test videos, run the setup script from the project root.  
    
```bash
./setup.sh
```

The assets will be available in the following directories  

- `assets/models/`: Contains pre-compiled `.dxnn` files.  
- `assets/videos/`: Contains sample video files for testing.  

---

## Example Structure

The C++ examples are organized in a **Task → Model → Variant** hierarchy.  

**Hierarchy Overview**  

- **Task (Top Level)**: Examples are grouped by functional categories  
     : e.g: `classification`, `object_detection`, `semantic_segmentation`, `instance_segmentation`, `ppu`, etc.  
- **Model (Second Level)**: Each task contains sub-folders for specific model architecture  
     : e.g., `object_detection/yolov5/`, `object_detection/yolov7/`, `classification/efficientnet/`.  
- **Variants(File Level)**: Individual C++ source files within the model folder represent different execution pipelines  
     : `*_sync.cpp`: **Synchronous** implementation (simpler logic, sequential processing)  
     : `*_async.cpp`: **Asynchronous** implementation (optimized for throughput using multi-threading)  

```text
src/cpp_example/
├── classification/
│   └── efficientnet_sync/
│       └── efficientnet_sync.cpp
│   └── efficientnet_async/
│       └── efficientnet_async.cpp
├── object_detection/
│   ├── yolov5/
│   │   ├── yolov5_sync.cpp
│   │   └── yolov5_async.cpp
│   ├── yolov7/
│   │   ├── yolov7_sync.cpp
│   │   └── yolov7_async.cpp
│   └── ... (scrfd, yolov8, yolov9, yolox 등)
├── semantic_segmentation/
│   └── deeplabv3/
│       └── ...
├── instance_segmentation/
│   └── yolov8seg/
│       └── yolv8seg_sync.cpp
├── ppu/
│   └── yolov5_ppu/
│       ├── yolov5_ppu_sync.cpp
│       └── yolov5_ppu_async.cpp
└── input_source_process_example/
    └── image/, video/, camera/, rtsp/ ...
```

---

## Understanding Example Variants : Sync vs. Async

For each model, DEEPX provides two primary pipeline styles. Choosing the right implementation depends on your development stage and throughput goals.  

**`sync` Examples (`*_sync.cpp`)**  

The synchronous variant follows a linear, sequential execution model.  

- **Flow**: **Read → Pre-process → Inference → Post-process → Display/Save**  
- **Mechanism**: Uses **a single thread**. The CPU waits for the NPU to complete each `InferenceEngine.Run()` call before proceeding.  
- **Best Use Cases**  
     : Initial prototyping and debugging  
     : Learning the basic end-to-end API flow  
     : Processing single images where throughput is not a priority  

**`async` Examples (`*_async.cpp`)**  

The asynchronous variant optimizes performance by decoupling stages into parallel tasks.  

- **Flow**: Each stage (Read, Pre-process, Inference, etc.) runs in its own thread, connected by a SafeQueue  
- **Mechanism**: Uses `InferenceEngine.RunAsync()` to submit multiple in-flight requests. While the NPU processes Frame N, the CPU handles pre/post-processing for other frames.  
- **Key Features**: Includes an `ASYNC_BUFFER_SIZE` to manage the pipeline depth and maximize NPU utilization.  
- **Best Use Cases**  
     : Production-level continuous streams (RTSP, Cameras)  
     : High-performance benchmarking (provides detailed latency/FPS metrics)  

---

## Building and Running Examples

This section describes how to compile the C++ source code and execute the examples using various input modes and command-line options.  

### Build Process

You can compile all C++ examples simultaneously from the project root using the provided build script. This script automates the CMake configuration and compilation.  

**Step 1. Execute the Build Script**  

```bash
# From the project root (dx_app/)
./build.sh
```

**Step 2. Verify Output Binaries**  

After a successful build, executable binaries are generated in the bin/ directory.  

- Location: `{project_root}/bin/`  
- Format: `[model_name]_[variant]` (e.g., `yolov5_sync, yolov9_async`)  

```text
bin/
├── yolov5_sync
├── yolov5_async
├── efficientnet_sync
├── efficientnet_async
└── ...
```

!!! note "NOTE"  
    The exact set of binaries may vary depending on the target platform and specific build options.  

### Command-Line Options  

Each example binary supports multiple input modes. Unlike Python examples, a single binary handles all modes (Image, Video, Camera, RTSP) via CLI flags.  

!!! warning "Important"  
    You **must** specify exactly one input source per execution. Providing multiple sources will result in an error.  

| **Category** | **Option** | **Description** | 
|----|----|----|
| **Required** | `-m, --model_path` | Path to the .dxnn model file. (e.g., `-m assets/models/yolov5.dxnn`) | 
| **Input Source** | `-i, --image_path` | Path to a single image file (JPG, PNG, etc.) | 
|                  | `-v, --video_path` | Path to a video file (MP4, MOV, AVI, etc.) | 
|                  | `-c, --camera_index` | Local camera index (e.g., `0` for `/dev/video0`) | 
|                  | `-r, --rtsp_url` | RTSP stream URL for network cameras | 
| **Execution** | `-l, --loop`| Number of inference iterations. Default is `1` | 
| **Output** | `--no-display` | Disable GUI windows. Recommended for headless/embedded systems | 
|            | `-s, --save_video` | Save processed frames to a video file | 
| **Help** | `-h, --help` | Print usage and all supported options for the specific binary | 


### Usage Examples

The following examples are common ways to execute the examples using different input sources and pipeline variants.  

**Example 1: Single Image Inference (Sync)**  

Processes a single image and displays the result.  

```bash
./bin/yolov9_sync \
  -m assets/models/YOLOV9S.dxnn \
  -i assets/images/bus.jpg
```

**Example 2: Video File Inference (Sync)**  

Reads frames from a video file and processes them sequentially.  

```bash
./bin/yolov9_sync \
  -m assets/models/YOLOV9S.dxnn \
  -v assets/videos/dance-group.mov
```

**Example 3: Live Camera Stream (Async)**  

Performs high-performance inference on a local camera feed (e.g., `/dev/video0`).  

```bash
./bin/yolov9_async \
  -m assets/models/YOLOV9S.dxnn \
  -c 0
```

**Example 4: RTSP Network Stream (Async)**  

Connects to a network camera or streaming server for real-time analysis.  

```bash
./bin/yolov9_async \
  -m assets/models/YOLOV9S.dxnn \
  -r rtsp://path/to/rtsp_stream
```

### Verification and Performance Summary

After each run, you can verify the results as follows.  

- **Visualization**: A window displays the processed video with detection boxes (unless `--no-display` is used).  
- **Performance Summary**: The console prints metrics to help you compare Sync vs. Async behavior.  
     : **Sync**: Total time and average latency per frame.  
     : **Async**: Inference FPS and NPU In-flight Avg/Max utilization.  

---

## Input Source Processing Examples

The folder `src/cpp_example/input_source_process_example` contains focused samples that show how to ingest frames from different input sources and feed them to **DX-RT (DEEPX Runtime)**.  


### Directory Layout

Each directory contains standalone examples focusing on specific backends to handle different data streams.  

- `camera/`: GStreamer (`camera_gstreamer`), OpenCV (`camera_opencv`), low-level V4L2 mmap (`camera_v4l2`) capture  
- `rtsp/`: GStreamer (`rtsp_gstreamer`), OpenCV (`rtsp_opencv`) RTSP reception  
- `video/`: GStreamer (`video_gstreamer`), OpenCV (`video_opencv`) file decode  
- `image/`: OpenCV single-image repeated inference (`image_opencv`)  


### Build Instructions  

The project includes a root `CMakeLists.txt` for a single-pass build of all sub-samples.  

```bash
# Navigate to the example directory 
cd src/cpp_example/input_source_process_example 
mkdir -p build && cd build 

# Configure and build 
cmake .. 
make -j 

# Binaries are installed to: {project_root}/bin
```

### Backend Implementation Comparison

The following table summarizes the technical approach of each provided sample.  

| **Source** | **Backend** | **Implementation Highlights** | 
|----|----|----|
| **Camera** | **GStreamer** | Uses `v4l2src → videoconvert → appsink`. Efficient BGR frame copying | 
|            | **OpenCV** | Uses `cv::VideoCapture`. Writes directly into preallocated input tensor buffers | 
|            | **V4L2** | Minimalist low-level `mmap` capture (MJPEG/YUYV). Ideal for resource-constrained systems. | 
| **RTSP** | **GStreamer** | Dynamic pad linking (`rtspsrc → avdec_h264`). High stability for network streams | 
|          | **OpenCV** | Fast prototyping. Includes FPS tuning and buffer management | 
| **Video** | **GStreamer** | `filesrc → decodebin`. Robust handling of EOS (End of Stream) and quit signals | 
|           | **OpenCV** | Sequential iteration with progress logging every 100 frames. | 
| **Image** | **OpenCV** | Repeatedly queues a single image to demonstrate `RunAsync() / Wait()` cycles | 

### Execution Guide

**Common Options**  

All binaries share consistent CLI flags for ease of use.  

- `-m, --model_path` (Required): Path to the `.dxnn` model file  
- `Resizing`: Uses `--input_width / --input_height` or model metadata to perform hardware-accelerated resizing before inference  
- Execution Flow: Most samples demonstrate the **Async** flow  
     : Step 1. Submit frame via `RunAsync()`.  
     : Step 2. Gather results using `Wait()` from a thread-safe queue.  


**Usage Examples**  

Run these commands from the project root after building.  

GStreamer Camera Capture  
```bash
./bin/camera_gstreamer -m assets/models/yolov5s.dxnn -d /dev/video0
```

OpenCV RTSP Stream  
```bash
./bin/rtsp_opencv -m assets/models/yolov5s.dxnn -r rtsp://user:pass@host/stream1
```

GStreamer Video File Decode  
```bash
./bin/video_gstreamer -m assets/models/yolov5s.dxnn -v assets/videos/dogs.mp4
```

Static Image Loop (Benchmarking)  
```bash
/bin/image_opencv -m assets/models/yolov5s.dxnn -i assets/images/bus.jpg
```

---

## Performance Measurement and Tuning

Each example includes a **Performance Summary** printed upon exit. Use this telemetry to identify bottlenecks and optimize NPU utilization.  
 
**Interpreting the Performance Summary**  

The report breaks down performance by pipeline stage.  

Sample Output  
```text
==================================================
               PERFORMANCE SUMMARY                
==================================================
Pipeline Step   Avg Latency     Throughput     
--------------------------------------------------
Read               1.44 ms      692.1 FPS
Preprocess         0.61 ms     1646.9 FPS
Inference         40.42 ms      164.1 FPS*
Postprocess        0.56 ms     1781.1 FPS
Display            7.10 ms      140.8 FPS
--------------------------------------------------
* Actual throughput via async inference
--------------------------------------------------
Infer Completed     :    478
Infer Inflight Avg  :    6.0
Infer Inflight Max  :      8
--------------------------------------------------
Total Frames        :    478
Total Time          :    3.5 s
Overall FPS         :   136.5 FPS
==================================================
```

**Bottleneck Analysis & Optimization**  

- **Bottleneck Detection:** The async summary makes it easy to identify the pipeline bottleneck. The stage with the **highest latency** (and lowest throughput) is the primary factor limiting your **Overall FPS**. In the example above, the Inference stage (40.42 ms) is the main bottleneck.  
- **Sync vs. Async Comparison:** Run both variants on the same input source to compare performance. In streaming scenarios, the **Async** variant typically achieves a higher **Overall FPS** by overlapping CPU tasks with NPU inference, leading to superior NPU utilization.  


**Conclusion**  

Use these examples as a starting point to  

- Understand the pipeline by tracing the source code.  
- Compare variants to evaluate NPU performance impact.  
- Customize input handling and processing logic to match your specific application requirements.  

---
