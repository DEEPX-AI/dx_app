# Python Examples Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
   - 2.1. [Download Assets](#21-download-assets)
   - 2.2. [Install `dx_engine` Python Library](#22-install-dx_engine-python-library)
   - 2.3. [Install `dx_postprocess` Python Library](#23-install-dx_postprocess-python-library)
   - 2.4. [Install Dependencies](#24-install-dependencies)
3. [Example Structure](#3-example-structure)
4. [Understanding Example Variants](#4-understanding-example-variants)
   - 4.1. [Pipeline and NPU Processing: Sync vs. Async](#41-pipeline-and-npu-processing-sync-vs-async)
   - 4.2. [Post-processing Implementation: Python vs. C++](#42-post-processing-implementation-python-vs-c)
5. [Core Features and Execution Modes](#5-core-features-and-execution-modes)
   - 5.1. [Image Inference](#51-image-inference-image_inference)
   - 5.2. [Stream Inference](#52-stream-inference-stream_inference)
6. [Running the Examples](#6-running-the-examples)
   - 6.1. [Image Inference](#61-image-inference)
   - 6.2. [Stream Inference and Performance Comparison](#62-stream-inference-and-performance-comparison)
   - 6.3. [Using Other Stream Sources](#63-using-other-stream-sources)
7. [Advanced: Automated Performance Report Generation for All Examples](#7-advanced-bulk-performance-report-generation)

---

## 1. Introduction

The Python examples provide a starting guide for users who want to develop AI applications utilizing the DEEPX NPU. Each example is a standalone, end-to-end script based on a real model, designed with the following goals:

-   **Model-Specific, Clear Pipelines**: Each example is an independent script tailored for a single model. It showcases the model's specific pipeline (Input -> Pre-process -> Inference -> Post-process) without complex branching or abstraction. This helps users grasp the operational principles with minimal code and easily extend it for their own applications.

-   **Learning Various Optimization Techniques**: For the same model, we provide multiple variants for processing, such as synchronous/asynchronous and Python/C++ post-processing. Users can compare these examples to naturally learn performance optimization techniques.

-   **Providing a Practical Development Foundation**: Based on these examples, users can gain practical insights to design a better structure for their applications and expand their code.

---

## 2. Prerequisites

Before running the examples, you need to set up the necessary environment by following these steps.

### 2.1. Download Assets

Model (`.dxnn`) and video files are required to run the examples. After executing the [`setup.sh`](../../setup.sh) script, sample models and videos will be downloaded to the `assets/models` and `assets/videos` paths, respectively.
```bash
# (Run from the dx_app/ path)
./setup.sh
```

### 2.2. Install `dx_engine` Python Library

`dx_engine` is the core Python library for running `.dxnn` model inference on the DEEPX NPU. It is part of the [DX-RT](https://github.com/DEEPX-AI/dx_rt) (DEEPX Runtime) SDK and provides Python bindings to the DX-RT C++ API via `pybind11`, enabling NPU acceleration in Python environments. All Python examples operate based on `dx_engine`.

You can install `dx_engine` using one of the following methods.

```bash
# Install along with the full dx_rt build
git clone https://github.com/DEEPX-AI/dx_rt
cd dx_rt
./install.sh --all
./build.sh
```

### 2.3. Install `dx_postprocess` Python Library

[`dx_postprocess`](..//bindings/python/dx_postprocess/) is a library that wraps the [post-processing classes](../postprocess) used in [C++ Examples](../cpp_example/) with `pybind11`, making them available in Python. It is used when performance optimization of Python-based post-processing is needed. Examples with `'_cpp_postprocess'` in their filenames use this library.

You can install `dx_postprocess` using one of the following methods:

```bash
# (Run from the dx_app/ path)
# Method 1: Install along with the full dx_app build
./build.sh

# Method 2: Standalone installation
./src/bindings/python/dx_postprocess/install.sh
```

### 2.4. Install Dependencies
Install dependency packages like `Numpy` and `OpenCV` required to run the Python examples. Run the following command to install the packages listed in the [`requirements.txt`](requirements.txt) file.

```bash
# (Run from the dx_app/ path)
pip install -r src/python_example/requirements.txt
```

---

## 3. Example Structure

The Python examples are designed with the following structure to help users easily find the desired example and intuitively compare the differences between models and variants.

```
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

-   **Folders by Task**: Folders are organized by AI Task, such as `classification`, `object_detection`, and `semantic_segmentation`. The `ppu` folder contains specialized examples that use `.dxnn` models compiled with the PPU (Post-Processing Unit) option.
-   **Folders by Model**: Under each Task folder, there are subfolders for each model, like `yolov5` and `efficientnet`.
-   **Example Variants**: Within a single model folder, there are multiple variants of the example script, differing by pipeline processing method (sync/async) and post-processing implementation (Python/C++).

---

## 4. Understanding Example Variants

Each model provides several variants based on the **combination of pipeline processing and post-processing implementation**, allowing users to directly compare and experience how different script implementation approaches affect processing performance.


### 4.1. Pipeline and NPU Processing: Sync vs. Async

**sync examples** and **async examples** have fundamental differences not only in their pipeline structure but also in how they use the NPU.

#### **`sync` examples**
Examples with `'_sync'` in their filenames.

-   **Pipeline**: Each step of the pipeline is **processed sequentially in a single thread**.
-   **NPU Processing**: The `Inference` step calls the `dx_engine.InferenceEngine.run()` API, which **processes only one inference request at a time**.
-   **Advantage**: The simple structure makes it easy to understand the basic principles of operation.

#### **`async` examples**
Examples with `'_async'` in their filenames.

-   **Pipeline**: Each step of the pipeline is **processed in parallel in separate threads**.
-   **NPU Processing**: The `Inference` step calls the `dx_engine.InferenceEngine.run_async()` API. This API processes multiple inference requests asynchronously, minimizing NPU idle time and achieving high inference throughput.
-   **Advantage**: When processing continuous data streams (e.g., video, camera), it effectively utilizes hardware resources to achieve high end-to-end FPS.

### 4.2. Post-processing Implementation: Python vs. C++

#### **Python Post-process examples**
The default examples without any special suffix in their filenames.
-   **Implementation**: The post-processing logic is **implemented directly in the Python script**.
-   **Advantage**: Easy to understand and debug the entire pipeline using only Python code.

#### **C++ Post-process examples**
Examples with `'_cpp_postprocess'` in their filenames.
-   **Implementation**: 
[`dx_postprocess`](../bindings/python/dx_postprocess/) is a library that wraps the [post-processing classes](../postprocess) used in [C++ Examples](../cpp_example/) with `pybind11`, making them available in Python. `'_cpp_postprocess'` examples import this library and directly call the C++ implemented post-processing functions instead of the original Python logic.
-   **Advantage**: Can be helpful for performance optimization in the following situations:
    1.  **Accelerating post-processing operations**: When the Python-implemented post-processing operation itself is slow and becomes a bottleneck, the C++ implementation directly accelerates it.
    2.  **Reducing CPU resource contention**: In CPU-constrained environments (like embedded boards), the high CPU usage of Python post-processing can interfere with other pipeline stages. C++ post-processing reduces the CPU load, improving system-wide efficiency and contributing to better NPU throughput.

---

## 5. Core Features and Execution Modes

All examples take an image or a stream as input, perform inference, and print a performance summary to the console upon completion.

### 5.1. Image Inference

-   **Function**: Performs inference on a single image and processes the result.
    -   By default, it visualizes the result image on the screen. (Classification models are an exception)
    -   With the `--no-display` option, it saves the result image to the `artifacts/python_example/<task>/` folder instead of displaying it.
-   **Unsupported Examples**: `async` examples are designed for stream processing performance improvement and do not provide the `image_inference` feature.
-   **Performance Report**: After execution, it prints an **`Image Processing Summary`** to the console, showing the time taken for each processing step.
    -   With the `--no-display` option, the `Display` item is excluded from the output.

    Example output:

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

### 5.2. Stream Inference

-   **Function**: Takes continuous frames (from video, RTSP, or camera) as input and performs inference on the stream.
-   **Input Modes**:
    -   **Video Mode**: Input from a video file.
    -   **RTSP Mode**: Input from a network stream via an RTSP URL.
    -   **Camera Mode**: Input from a system-connected camera.
-   **Unsupported Examples**: The Classification Task is specialized for image-based classification and does not provide the `stream_inference` feature.
-   **Performance Report**: When execution ends (video finishes, or `Esc`/`q` key is pressed), it prints a **`Performance Summary`** to the console, including **overall throughput (Overall FPS)** and other metrics.
    -   **Avg Latency**: Average processing time for each pipeline step (milliseconds)
    -   **Throughput**: Frames per second (FPS) that each step can process
        -   **Default calculation**: `1000ms / Avg Latency`
        -   **`async` example's Inference Throughput**: Since the NPU processes multiple frames simultaneously, the actual throughput is measured using the **Time Window** method
            ```
            Actual throughput = Number of processed frames / (Last completion time - First frame submission time)
            ```

    -   **Additional metrics (`async` examples only)**:
        -   `Infer Completed`: Total number of completed inferences
        -   `Infer Inflight Avg`: Average number of frames submitted to `InferenceEngine`
            - Lower values: Insufficient workload supplied to NPU (other stages are bottleneck)
            - Higher values: NPU is being effectively utilized
        -   `Infer Inflight Max`: Maximum number of frames submitted to `InferenceEngine`
    -   With the `--no-display` option, the `Display` item is excluded from the output.

    Example output (sync):

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
    
    **Metric Explanation**:
    - In `sync` examples, all steps are processed sequentially, so each step's `Throughput` is calculated as `1000ms / Avg Latency`.
    - In this example, `Overall FPS`(16.9) is much lower than individual step `Throughput` values.
    - This is because all step latencies accumulate sequentially: `1.28 + 0.40 + 28.27 + 5.89 + 23.19 = 59.03ms ≈ 16.9 FPS`

    Example output (async):

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
    
    **Metric Explanation**:
    - **`Avg Latency` vs `Throughput` (Inference step)**:
        - `Avg Latency`(67.29ms): Average time from `run_async()` call to `wait()` completion for individual frames. This includes **`InferenceEngine` internal queue wait time**.
        - `Throughput`(101.1 FPS*): NPU's **actual throughput** measured using Time Window method. Represents frames per second processed by NPU, regardless of queue wait time.
    
    - **Understanding `Infer Inflight Avg`(5.9)**:
        - Average number of frames submitted to `InferenceEngine`.
        - For example, in DX-M1 with 3 NPU cores, only 3 frames can be **processed simultaneously**.
        - `Inflight Avg = 5.9` means on average, 3 frames were being processed by NPU and approximately 2.9 frames were **waiting in `InferenceEngine`'s internal queue**.
        - **When `Inflight Avg` exceeds the NPU core count**, the NPU is effectively utilized without idle time, with sufficient work queued - an ideal state.
        - This queue wait time increases `Avg Latency`, but the NPU maintains high throughput by utilizing all cores.
    
    - In this example, the `Display` step (60.5 FPS) becomes the overall pipeline bottleneck, limiting `Overall FPS` to 59.3 FPS, while the Inference step achieved 101.1 FPS throughput.

---

## 6. Running the Examples

This section walks you through running the [`object_detection/yolov9`](object_detection/yolov9) examples to directly observe key features and performance differences between variants.

### 6.1. Image Inference

This is the most basic `sync` example, inferring a single image. Use the two examples below to compare the performance difference based on the post-processing implementation (Python/C++).

```bash
# Python Post-processing
python src/python_example/object_detection/yolov9/yolov9_sync.py --model assets/models/YOLOV9S.dxnn --image sample/img/1.jpg

# C++ Post-processing
python src/python_example/object_detection/yolov9/yolov9_sync_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --image sample/img/1.jpg
```

-   **Execution Result**:
    - An image with object detection results is displayed on the screen.
    - The `Image Processing Summary` is printed to the terminal.
    - You can see that the `Postprocess` step's Latency is lower in the C++ post-processing example than in the Python one.

### 6.2. Stream Inference and Performance Comparison

Now, let's use a stream input to compare the performance of `sync` vs. `async` and Python vs. C++ post-processing.

#### 6.2.1. Sync vs. Async Performance Comparison

Run the same video with both `sync` and `async` examples and compare the `Overall FPS`.

```bash
# Sync Stream Inference
python src/python_example/object_detection/yolov9/yolov9_sync.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov

# Async Stream Inference
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov
```

-   **Execution Result**:
    - Generally, the `Overall FPS` of the `async` example is measured higher than the `sync` example. This is due to the combined effect of two optimizations in the `async` example:
    1.  **Pipeline Parallelism**: Each pipeline stage (`Preprocess`, `Inference`, `Postprocess`, etc.) runs concurrently in a separate thread, minimizing CPU idle time.
    2.  **Improved NPU Throughput**: The `dx_engine.InferenceEngine.run_async()` API allows multiple inference requests to be sent to the NPU in advance and processed in parallel, thus effectively utilizing NPU cores.

#### 6.2.2. Python PP vs. C++ PP Performance Comparison

Use the `--no-display` option with the `async` examples to compare pure computation performance and see the effect of post-processing optimization.

```bash
# Async + Python Post-processing
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov --no-display

# Async + C++ Post-processing
python src/python_example/object_detection/yolov9/yolov9_async_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov --no-display
```

-   **Execution Result**:
    -   The `Postprocess` step's throughput is typically higher in the C++ post-processing example.
    -   However, **the improvement in `Overall FPS` depends on where the bottleneck occurs in the pipeline**
    -   C++ post-processing can improve `Overall FPS` in the following situations:
    
        1. **When post-processing is a direct bottleneck**
            - When the `Postprocess` step's `Throughput` is significantly lower than other steps
            - C++ post-processing directly accelerates it
        
        2. **When CPU contention causes an indirect bottleneck** (especially in embedded environments)
            - Python post-processing's high CPU occupancy interferes with `Read` and `Preprocess` stages
            - When `Infer Inflight Avg` is measured low (insufficient data supply to NPU)
            - C++ post-processing resolves CPU contention to improve overall pipeline performance

### 6.3. Using Other Stream Sources

You can apply the examples to other stream sources by using the `--rtsp` or `--camera` arguments instead of `--video`.

```bash
# RTSP Input
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --rtsp rtsp://{YOUR_RTSP_URL}

# Camera Input (Camera 0)
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --camera 0
```

---

## 7. Advanced: Automated Performance Report Generation for All Examples

The tutorial in Section 6 compared performance by running `yolov9` examples one by one. If you want to **measure the performance of all supported example variants at once and compare the results in a table**, you can use the `pytest`-based End-to-End (E2E) tests.

-   **Core Function**:
    -   The `e2e` tests verify the end-to-end functionality of all example scripts by running both `image_inference` and `stream_inference` with **actual `.dxnn` models and data (images, videos)**.
    -   During this process, it **collects Performance Summary output from the `stream_inference` tests** and saves `.csv` report files summarizing the performance of each model's example variants to the `tests/python_example/performance_reports/` folder, and prints a summarized E2E Performance Report to the console.

-   **How to Run**:
    1.  First, install the necessary dependencies for the tests.
        ```bash
        # (Run from the dx_app/ path)
        pip install -r tests/python_example/requirements.txt
        ```
    2.  Navigate to the `tests/python_example` directory and run the `e2e` tests with the following command.
        ```bash
        # (Run from the dx_app/ path)
        cd tests/python_example

        # Run all e2e tests
        pytest -m e2e
        
        # Or run e2e tests for a specific model (e.g., yolov9)
        pytest -m "e2e and yolov9"
        ```

-   **Using the Results**:
    -   With the generated `.csv` reports, you can see at a glance which processing approach combination yields the best performance for each model in your current computing environment and formulate an optimization strategy for your own application.
    -   For more details on the test framework, please refer to [`tests/python_example/README.md`](../../tests/python_example/README.md).
