## Introduction

The Python examples provide a professional guide for developing high-performance AI applications on the **DEEPX NPU**. These standalone, end-to-end scripts bridge the gap between initial prototyping and hardware-optimized deployment.  

The examples are built upon four core pillars  

- **Transparent Pipelines:** Each script is model-specific, showcasing the complete lifecycle — **Input → Pre-processing → Inference → Post-processing** — without complex abstractions.  
- **Optimization Variants:** Users can compare **Synchronous (Sync) vs. Asynchronous (Async)** execution and Native Python vs. C++ Accelerated post-processing to observe significant performance gains.  
- **Bottleneck Analysis:** Integrated **Performance Summaries** provide metrics like **FPS, Latency, and Inflight Avg**, helping developers identify if their system is limited by the NPU, CPU contention, or display logic.  
- **Automated Benchmarking:** An included **E2E testing framework** automates performance reporting across all variants, allowing users to establish hardware-specific baselines and optimization strategies.  

---

## Prerequisites

Follow these steps to configure your environment. These prerequisites ensure you have the necessary assets, core NPU libraries, and optimization modules required to execute the Python examples.  

### Download Assets

The examples require pre-compiled model files (`.dxnn`) and sample video streams. The setup.sh script automates the download and placement of these files into the `assets/` directory.  

```bash
# (Run from the dx_app/ path)
./setup.sh
```

!!! note "NOTE"  
    Models are stored in `assets/models` and videos in `assets/videos`.  
    
### Install `dx_engine` (Core Runtime)

`dx_engine` is the essential Python interface for the **DEEPX NPU**. Built on `pybind11`, it provides high-performance Python bindings to the DX-RT C++ API.  

To install `dx_engine` as part of the full SDK  

```bash
# Install along with the full dx_rt build
git clone https://github.com/DEEPX-AI/dx_rt
cd dx_rt
./install.sh --all
./build.sh
```

### Install `dx_postprocess` (Optional Optimization)

The `dx_postprocess` library wraps C++ post-processing logic for use in Python. While Python-native post-processing is available, this library is recommended for performance-critical applications.  

- **Standard Examples**: Use native Python post-processing.  
- `*_cpp_postprocess` **Examples**: Require `dx_postprocess` for hardware-accelerated logic.  

```bash
# (Run from the dx_app/ path)
# Method 1: Install along with the full dx_app build
./build.sh

# Method 2: Standalone installation
./src/bindings/python/dx_postprocess/install.sh
```

### Install Python Dependencies

Finally, install the standard utility libraries (such as `NumPy` and `OpenCV`) required for data manipulation and image rendering  

```bash
# (Run from the dx_app/ path)
pip install -r src/python_example/requirements.txt
```

---

## Understanding Execution Variants

The Python examples are structured hierarchically by task and model. This organization allows for side-by-side comparison of different execution logic (Synchronous vs. Asynchronous) and post-processing implementations.  

```text
python_example/
├── classification/
│   └── efficientnet/
│       └── efficientnet_sync.py
├── object_detection/
│   ├── yolov5/
│   │   ├── yolov5_sync.py
│   │   ├── yolov5_sync_cpp_postprocess.py
│   │   ├── yolov5_async.py
│   │   └── yolov5_async_cpp_postprocess.py
│   └── ... (other models)
├── semantic_segmentation/
│   └── ...
└── ppu/
    └── ...
```

**Organization Logic**  

- **Categorization by Task:** Top-level directories (e.g., `classification, object_detection`) group models by their functional AI task.  
- **Model-Specific Subdirectories:** Each model family (e.g., `yolov5, efficientnet`) is isolated to prevent dependency overlaps and simplify navigation.  
- **Specialized PPU Folder:** The ppu/ directory contains examples utilizing models compiled specifically for the **DEEPX PPU (Post-Processing Unit)**. These bypass host-side CPU post-processing entirely.  

### Pipeline Architecture: Sync (Sequential) vs. Async (Parallel)

The choice between Synchronous and Asynchronous execution determines how the CPU manages the NPU workload and data flow.  

**Synchronous`sync`**  

- **Workflow:** Executes tasks sequentially (**Pre-process → Inference → Post-process**) in a single thread.  
- **NPU Interaction:** Uses `dx_engine.InferenceEngine.run()`, which blocks until the NPU returns a result for a single request.  
- **Best For:** Debugging, initial prototyping, and scenarios where latency for a single frame is more important than total throughput.  

**Asynchronous `async`**  

- **Workflow:** Decouples pipeline stages into separate parallel threads.  
- **NPU Interaction:** Uses `dx_engine.InferenceEngine.run_async()`. This enables the NPU to process multiple requests in a queue, drastically reducing idle time between inferences.  
- **Best For:** Real-time video streams and camera feeds where high FPS and maximum hardware utilization are required.  

### Implementation Logic: Native Python vs. C++ Accelerated

While the core inference always runs on the NPU, the **Post-processin** (e.g., NMS for object detection) typically runs on the CPU.  

**Native Python Implementation**  

- **Filename:** `(Default/No suffix)`  
- **Logic:** Written entirely in Python for maximum readability and ease of modification.  
- **Consideration:** Ideal for logic development, but may become a performance bottleneck on low-power embedded CPUs.  

**C++ Accelerated Implementation**  

- **Filename:** `*_cpp_postprocess.py`  
- **Logic:** Uses the `dx_postprocess` library. This calls high-performance C++ classes (the same ones used in our C++ SDK) via `pybind11`.  
- **Advantages:**  
     : **Lower Latency:** Accelerates computationally heavy operations like Non-Maximum Suppression (NMS).  
     : **Resource Efficiency:** Reduces CPU contention. In CPU-bound environments (embedded boards), lowering CPU usage for post-processing frees up cycles for the NPU driver, leading to higher overall throughput.  

### Summary Comparison: Choosing the Right Variant for Your Environment

| **Variant** | **Logic Location** | **Threading** | **Recommended Environment** | 
|----|----|----|----|
| **Sync + Python** | Python Script | Single | Desktop / PC for debugging |
| **Async + Python** | Python Script | Multi | General throughput testing |
| **Async + C++** | Compiled C++ | Multi | Production / Embedded NPU deployment |

---

## Running the Examples (Practical Guide)

All Python examples support two primary execution modes: **Image Inference** (single frame) and **Stream Inference** (continuous video). Upon completion, a performance report is generated to help you identify pipeline bottlenecks.  

### Image Inference: Validating Accuracy

Designed for validating model accuracy and verifying the environment setup.  

- **Behavior:** Processes a single input image and visualizes results.  
- **Artifacts:** By default, results are shown in a GUI window. Use `--no-display` to suppress the window and save the output to `artifacts/python_example/<task>/`.  
- **Scope:** `async` variants do not support this mode, as they are optimized for multi-frame throughput.  
- **Metric Report:** Provides a **Latency Summary** for each serial step.  

Example output  

```text
===================================
IMAGE PROCESSING SUMMARY      
===================================
Pipeline Step       Latency
-----------------------------------
Read                2.04 ms
Preprocess          1.15 ms
Inference          39.50 ms
Postprocess         2.63 ms
Display             0.36 ms
-----------------------------------
Total Time      :   45.7 ms
===================================
```

### Stream Inference: Performance Comparison

Stream Inference is designed for the continuous processing of video data. It manages the pipeline for decoding, preprocessing, model inference, and results visualization in real-time.  

**A. Functional Overview**  

- **Input Support:** Video files (.mp4, .avi), RTSP network streams, and live camera feeds (USB/MIPI).  
- **Scope:** Optimized for Object Detection and Tracking. Standard Classification tasks are optimized for discrete image inputs and do not support `stream_inference`.  
- **Termination:** Execution ends when the video reaches the EOF (End of Frame) or when the user presses `Esc` or `q`.  

**B. Performance Summaries**  

Upon termination, a Performance Summary is printed to the console to help identify hardware utilization and pipeline bottlenecks.  

Key Metrics  

- **Avg Latency:** Average time (ms) to complete a specific pipeline step.  
- **Throughput (FPS):** Frames processed per second by a specific step.  
     : **Sync Mode:** Calculated as `1000ms / Avg Latency`.  
     : **Async Mode (Inference):** Calculated via the **Time Window** method to account for parallel NPU execution.  

```text  
Actual throughput = Number of processed frames / (Last completion time - First frame submission time)
```

**C. Comparison: Synchronous vs. Asynchronous Execution**  

**C-a.** Synchronous (Sync) Mode  

Steps are processed sequentially. The CPU waits for each frame to clear one stage before moving to the next.  

Example Sync Output  

```text
==================================================
            PERFORMANCE SUMMARY                
==================================================
Pipeline Step   Avg Latency     Throughput     
--------------------------------------------------
Read                1.28 ms      783.1 FPS
Preprocess          0.40 ms     2505.8 FPS
Inference          28.27 ms       35.4 FPS
Postprocess         5.89 ms      169.8 FPS
Display            23.19 ms       43.1 FPS
--------------------------------------------------
Total Frames    :    300
Total Time      :   17.7 s
Overall FPS     :   16.9 FPS
==================================================
```

Metric Explanation (Sync)  

- In the provided sync example, **Overall FPS (16.9)** is the inverse of the **sum** of all latencies (`1.28 + 0.40 + 28.27 + 5.89 + 23.19 = 59.03ms`).  
- The slowest steps (Inference and Display) have a compounding effect, dragging down the total speed.  

**C-b.** Asynchronous (Async) Mode  

The NPU processes multiple frames simultaneously by utilizing its internal command queue, decoupling NPU capacity from CPU pipeline management.  

Example Async Output  
```text
==================================================
            PERFORMANCE SUMMARY                
==================================================
Pipeline Step   Avg Latency     Throughput     
--------------------------------------------------
Read                1.95 ms      512.9 FPS
Preprocess          1.10 ms      906.2 FPS
Inference          67.29 ms      101.1 FPS*
Postprocess         2.22 ms      450.9 FPS
Display             16.54 ms       60.5 FPS
--------------------------------------------------
* Actual throughput via async inference
--------------------------------------------------
Infer Completed     :    300
Infer Inflight Avg  :    5.9
Infer Inflight Max  :      7
--------------------------------------------------
Total Frames        :    300
Total Time          :    5.1 s
Overall FPS         :   59.3 FPS
==================================================
```

Advanced Async Metrics  

- Avg Latency (67.29 ms): Includes NPU computation plus time spent waiting in the `InferenceEngine` queue.  
- Inference Throughput (101.1 FPS): The NPU’s actual processing speed, independent of queue wait times.  
- Infer Inflight Avg (5.9): Average frames submitted to the engine.  
     :Interpretation: If your NPU has 3 cores, an Inflight Avg of 5.9 means 3 frames are being processed while ~2.9 are queued. This is an **ideal state**, ensuring the NPU is never idle.  

**D. Identifying Pipeline Bottlenecks**  

The overall throughput is governed by the **Slowest Stage** principle.  

- **Observation:** The NPU achieves **101.1 FPS**, but the **Overall FPS** is only **59.3**.  
- **Diagnosis:** The **Display stage (60.5 FPS)** is the bottleneck. The system cannot output frames faster than the display logic permits.  
- **Recommendation:** To measure peak NPU performance, run the script with the `--no-display` flag.  

### Expanding Sources: : RTSP and Cameras

Swap the `--video` argument for `--rtsp` or `--camera` for different inputs.  

```bash
# RTSP Input
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --rtsp rtsp://{YOUR_RTSP_URL}

# Camera Input (Camera 0)
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --camera 0
```

---

## Hands-on Examples: Benchmarking YOLOv9

This section walks you through executing the YOLOv9 examples to observe performance differences between **Sync vs. Async** modes and **Python vs. C++** post-processing.  

### Running Sync vs. Async

This basic synchronous example infers a single image. Use it to compare the latency of post-processing implementations.  

```bash
# Python Post-processing
python src/python_example/object_detection/yolov9/yolov9_sync.py --model assets/models/YOLOV9S.dxnn --image sample/img/1.jpg

# C++ Post-processing
python src/python_example/object_detection/yolov9/yolov9_sync_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --image sample/img/1.jpg
```

**Expected Outcome**    

- A detection result window will appear.  
- The terminal summary will show **significantly lower latency** for the Postprocess step in the C++ version.  

### Comparing Python vs. C++ Post-processing

Stream inference measures the throughput of a continuous data pipeline.  

**Parallelism Efficiency (Sync vs. Async)**  

Run the same video through both modes to compare the  `Overall FPS`.  

```bash
# Sync Stream Inference
python src/python_example/object_detection/yolov9/yolov9_sync.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov

# Async Stream Inference
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov
```

Analysis: **Async** mode typically yields higher Overall FPS because  

- (1) Pipeline Parallelism: Concurrent execution of stages (`Read, Preprocess, Inference, Postprocess`) minimizes CPU idle time.  
- (2) NPU Core Utilization: The `run_async()` API allows the NPU to process multiple requests in parallel across all available cores.  


**Post-Processing & CPU Contention Analysis**  

Use the `--no-display` flag to observe the raw computational potential of the NPU.  

```bash
# Async + Python Post-processing
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov --no-display

# Async + C++ Post-processing
python src/python_example/object_detection/yolov9/yolov9_async_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov --no-display
```

- Execution Result: C++ implementation typically shows higher `Postprocess` throughput. However, **Overall FPS improvement occurs only if the post-processing stage was the bottleneck.**  

When C++ Post-processing Improves Overall FPS:  

- (1) **Direct Pipeline Bottleneck:** When `Postprocess` throughput is the lowest value in the pipeline.  
- (2) **Indirect Bottleneck (CPU Contention):** In embedded environments, high Python CPU usage can "starve" the `Read` or `Preprocess` stages. C++ reduces this load, allowing the CPU to feed data to the NPU faster (indicated by an increased `Infer Inflight Avg`).  

---

## Advanced: Automated Benchmarking

You can use the pytest-based **End-to-End (E2E) testing framework** to measure and compare the performance of all example variants automatically.  

### Overview of the E2E Framework

The E2E tests serve two purposes  

- **Validation:** Verifies that all example scripts (`Image/Stream`) function correctly using actual .dxnn models and media assets.  
- **Automated Reporting:** During the `stream_inference` tests, the framework captures the **Performance Summary** output from every variant, aggregates the data, and generates a comparative report.  

### Installation and Batch Execution

- **Step 1. Install Test Dependencies:** Navigate to the root directory (`dx_app/`) and install the required packages  
- **Step 2. Run the Benchmarks:** Navigate to the test directory and use pytest with the e2e marker  

### Analyzing Automated CSV Reports for System Optimization

After the tests complete, the framework generates reports in two formats  

- **Console Output:** A summarized E2E Performance Report is printed directly to the terminal for immediate review.  
- **CSV Reports:** Detailed logs are saved to: `tests/python_example/performance_reports/`.  

By reviewing the generated .csv files, you can identify  

- **Best-Performing Combination:** Determine which variant (e.g., `Async + C++ Post-processing`) provides the highest throughput for your specific hardware.  
- **Environment Baselining:** Establish a performance baseline for your current computing environment to track how future software or model updates affect speed.  
- **Optimization Strategy:** Use the data to decide whether to prioritize CPU optimization (C++) or pipeline management (Async) for your production application.  

For a deeper dive into the testing architecture, refer to the [tests/python_example/README.md](/tests/python_example/README.md).   

---
