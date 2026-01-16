# C++ Examples Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Example Structure](#3-example-structure)
4. [Understanding Example Variants](#4-understanding-example-variants)
   - 4.1. [Pipeline and NPU Processing: Sync vs. Async](#41-pipeline-and-npu-processing-sync-vs-async)
5. [Execution Modes and Options](#5-execution-modes-and-options)
6. [Building and Running the Examples](#6-building-and-running-the-examples)
   - 6.1. [Build](#61-build)
   - 6.2. [Run: Object Detection (YOLOv9)](#62-run-object-detection-yolov9)
7. [Performance Measurement and Tuning](#7-performance-measurement-and-tuning)

---

## 1. Introduction

C++ examples are the starting point for building **DEEPX NPU–based AI applications in C++**. Each example is a **self-contained end-to-end pipeline** using a real model and is designed with the following goals:

- **Per‑model, single‑file pipeline**
  - Each C++ example focuses on a single model and covers the full flow: Input → Pre-process → Inference → (post-process call) → Display/Save.
  - The example code is intentionally minimal and focused on one task, so it is easy to read, modify, and reuse.

- **Shared C++ post-processing library**
  - The **post-processing algorithms themselves are not implemented inside the example files**.
  - They live under `src/postprocess/` as reusable C++ classes (e.g., `YOLOv5PostProcess`, `YOLOv9PostProcess`).
  - C++ examples simply include and call these classes, and the **same C++ post-processors are exposed to Python via pybind11**, so C++ and Python examples share identical post-processing logic.

- **Learning optimization patterns**
  - For the same model, you get multiple variants such as **sync / async** pipelines.
  - All variants share the same post-processing classes, but differ in pipeline structure, threading, and asynchronous scheduling.
  - By building and running these variants and comparing their performance logs, you can intuitively understand the impact of different designs.

- **Practical application templates**
  - You can copy an example file and adapt it by changing the input source, pre-processing, post-processing parameters, and visualization/saving logic.
  - Because the core model and post-processing are already wired, it is straightforward to evolve the example into your own application.

---

## 2. Prerequisites

Before building and running the C++ examples, make sure the following environment is prepared:

- **DX-RT (DEEPX Runtime)**
  - Required runtime to execute `.dxnn` models on the NPU.
  - The `dxrt` libraries must be installed and discoverable via `LD_LIBRARY_PATH`.

- **Build tools**
  - CMake 3.16 or later
  - A C++14-compatible compiler (GCC, Clang, ...)
  - Make or Ninja

- **Required libraries**
  - OpenCV (for image/video I/O and visualization)

- **Models and sample input data**
  - Run `dx_app/setup.sh` once from the project root to download sample models and videos into `assets/models` and `assets/videos`.

  ```bash
  ./setup.sh
  ```

---

## 3. Example Structure

C++ examples are organized in a **Task → Model → Variant** hierarchy.

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

- **Task directories** 
  - Grouped by task type such as `classification`, `object_detection`, `semantic_segmentation`, `instance_segmentation`, `ppu`, etc.
- **Model directories**
  - Under each task you will find sub-folders per model (e.g., `yolov5`, `yolov7`, `efficientnet`).
- **Variants**
  - Within a model folder there are one or more C++ example files such as `*_sync.cpp` and `*_async.cpp`, which differ in pipeline design (synchronous vs. asynchronous).

---

## 4. Understanding Example Variants

For each model, examples are provided for different **pipeline styles**, primarily synchronous vs. asynchronous.

### 4.1. Pipeline and NPU Processing: Sync vs. Async

#### `sync` examples (`*_sync.cpp`)

- **Pipeline structure**
  - Read → Pre-process → Inference → Post-process → Display/Save runs **sequentially in a single thread**.
- **NPU execution**
  - Uses `InferenceEngine.Run()` to process **one request at a time**.
- **When to use**
  - Best for understanding the basic end-to-end flow and for debugging.
  - Works well for single-image runs and simple video/stream processing where maximum throughput is not critical.

#### `async` examples (`*_async.cpp`)

- **Pipeline structure**
  - Read / Pre-process / Inference / Post-process / Display are **split across multiple threads and internal queues**.
  - For example, the YOLOv5 async sample uses a `SafeQueue`, separate post-processing and display threads, and an `ASYNC_BUFFER_SIZE` to keep the NPU busy.
- **NPU execution**
  - Uses `InferenceEngine.RunAsync()` to submit **multiple in-flight requests** to the NPU.
  - This helps maximize utilization on multi-core NPUs.
- **When to use**
  - Recommended for continuous streams (video files, cameras, RTSP) where high end-to-end FPS is desired.
  - The examples print detailed metrics such as average latency per stage, **inference throughput (FPS\*)**, `Infer Completed`, and `Infer Inflight Avg/Max` to help analyze NPU utilization.

---

## 5. Execution Modes and Options

Each C++ example binary supports **multiple input modes** (image / video / camera / RTSP) controlled by command-line options. Unlike the Python examples, there are no separate functions per mode; instead, you choose the mode via CLI flags.

### 5.1. Common options

- `-m, --model_path <path>`
  - **Required.** Path to the `.dxnn` model file to use for inference.
  - Example: `-m assets/models/YOLOV9S.dxnn`

- `-l, --loop <N>`
  - Number of inference iterations to run. Default is `1`.
  - Useful for performance measurements (average latency, FPS) on the same input.

- `--no-display`
  - Do not show any result windows; only **FPS and performance logs** are printed.
  - Recommended for headless/embedded environments when you only care about throughput.

- `-s, --save_video`
  - When using video/camera/RTSP input, save the processed output to a video file.

- `-h, --help`
  - Print usage and all supported options for the given example binary.

> Note: The exact help text may vary slightly between binaries (e.g., `yolov5_sync`, `yolov9_async`), but the core options are consistent.

### 5.2. Input source options

In a single execution, you must choose **exactly one** input source. If you pass more than one, the examples will print an error and exit (see `yolov5_sync.cpp`).

- `-i, --image_path <path>`
  - Use a single image file as input (jpg, png, jpeg, ...).
  - The example reads the image once and runs the configured number of inference loops.

- `-v, --video_path <path>`
  - Use a video file (mp4, mov, avi, ...) as input.
  - Frames are read in a loop and processed sequentially (sync) or via asynchronous queues (async).

- `-c, --camera_index <index>`
  - Use a local camera device as input. Example: `-c 0` for the default camera.
  - Performs real-time inference on live camera frames.

- `-r, --rtsp_url <url>`
  - Use an RTSP stream URL as input.
  - Intended for network cameras or streaming servers.

---

## 6. Building and Running the Examples

### 6.1. Build

You can build all C++ examples from the project root using the provided script:

```bash
# From the project root (dx_app/)
./build.sh
```

After a successful build, the example binaries are created under `bin/`:

```text
bin/
├── yolov5_sync
├── yolov5_async
├── efficientnet_sync
├── efficientnet_async
└── ...
```

> Note: The exact set of binaries may vary depending on the target platform and build options.

### 6.2. Run: Object Detection (YOLOv9)

The YOLOv9 examples illustrate how to run sync vs. async pipelines with different input sources.

```bash
# 1) Image inference (Sync)
./bin/yolov9_sync \
  -m assets/models/YOLOV9S.dxnn \
  -i sample/img/1.jpg

# 2) Video inference (Sync)
./bin/yolov9_sync \
  -m assets/models/YOLOV9S.dxnn \
  -v assets/videos/dance-group.mov

# 3) Camera stream inference (Async)
./bin/yolov9_async \
  -m assets/models/YOLOV9S.dxnn \
  -c 0

# 4) RTSP stream inference (Async)
./bin/yolov9_async \
  -m assets/models/YOLOV9S.dxnn \
  -r rtsp://path/to/rtsp
```

After each run:

- The processed video is displayed in a window unless you pass `--no-display`.
- A **Performance Summary** is printed to the console so you can directly compare sync vs. async behavior.

---

## 7. Performance Measurement and Tuning

- **Using the profiling output**
  - Each example includes timers and a summary function that report average latency per stage.
      ```
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
  - The async examples additionally make it easy to see which stage in the pipeline (Preprocess / Inference / Postprocess / Display) is the bottleneck.

- **Comparing sync vs. async**
  - Run both `*_sync` and `*_async` variants on the same input source and compare `Overall FPS`.
  - In most streaming scenarios, the async variant should achieve higher `Overall FPS` thanks to parallelism and better NPU utilization.

Use this guide as a starting point to open each C++ example, understand the pipeline, and then customize input handling, pre-processing, post-processing, and output logic to match your application requirements.