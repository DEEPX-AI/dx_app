## DX-APP: DEEPX NPU Application Templates

**DX-APP** is a collection of ready‑to‑run application templates for **DEEPX NPUs**, covering both **C++** and **Python**.  
It helps you quickly experience NPU performance and serves as a practical starting point for building your own AI applications.

Key goals:

- **Fast first run**: Minimal setup to run classification, object detection, and segmentation demos on DEEPX devices.
- **Reusable templates**: Clean, per‑task examples that you can copy and adapt for your own use cases.
- **Shared core logic**: Model‑specific **C++ post‑processing** is shared between C++ and Python (via `dx_postprocess`).

For detailed step‑by‑step guides (installation, templates, demos), see the documents under `docs/`:

- Overview: [01_DXNN_Application_Overview.md](./docs/source/docs/01_DXNN_Application_Overview.md)
- Installation & Build: [02_DX-APP_Installation_and_Build.md](./docs/source/docs/02_DX-APP_Installation_and_Build.md)

---

## 1. Features at a Glance

DX-APP provides:

- **End‑to‑end demo applications**
  - Image classification (EfficientNet)
  - Object detection (YOLOv5/7/8/8‑OBB/9, YOLOX, YOLOv5Face, SCRFD, YOLOv5Pose)
  - Semantic / instance segmentation (DeepLabv3, YOLOv8seg)
  - PPU‑enabled variants for selected models

- **Both C++ and Python pipelines**
  - C++ examples: high‑performance, production‑oriented reference.
  - Python examples: faster iteration and easier experimentation.

- **Shared C++ post‑processing**
  - Model‑specific decoding (boxes, scores, keypoints, masks) implemented once in C++.
  - Reused by C++ examples and exposed to Python via the `dx_postprocess` binding.
  - Enables apples‑to‑apples comparison between **pure Python** and **C++ post‑processing**.

- **Scalable pipeline designs**
  - Sync and async variants that show how to maximize NPU utilization and end‑to‑end FPS on DEEPX hardware.

---

## 2. Prerequisites

Before building or running DX-APP, prepare the following environment.

### 2.1. DX-RT (DEEPX Runtime) and Drivers

To run `.dxnn` models on the NPU, you must install both the **DX-RT runtime** and the **NPU drivers**.

- **DX-RT (runtime and tools)**  
  <https://github.com/DEEPX-AI/dx_rt>

- **DEEPX NPU Linux Driver**  
  <https://github.com/DEEPX-AI/dx_rt_npu_linux_driver>

Follow the installation guides in those repositories for your target platform.  
After installation, you can verify that DX-RT and the drivers are correctly installed with:

```shell
dxrt-cli -s
```

### 2.2. Toolchain and Libraries

- **Build tools**
  - CMake ≥ 3.14
  - C++14‑compatible compiler (GCC, Clang, …)
  - Make or Ninja

- **OpenCV**
  - OpenCV ≥ 4.2.0 (4.5.5 recommended).

- **Python (for Python examples and bindings)**
  - Python 3.8 or higher
  - `pip` for installing required packages

You can install these dependencies with:

```shell
./install.sh --all
```

---

## 3. Repository Layout

At a high level:

```text
dx_app/
├── src/
│   ├── cpp_example/          # C++ end‑to‑end examples (sync/async, by task/model)
│   ├── python_example/       # Python end‑to‑end examples
│   ├── postprocess/          # C++ post‑processing libraries (shared C++/Python)
│   └── bindings/
│       └── python/
│           └── dx_postprocess/  # pybind11 bindings for C++ post‑processing
├── assets/                   # Downloaded models/videos (via setup.sh)
├── scripts/                  # Helper scripts to run demos
├── build.sh                  # Top‑level build script
├── install.sh                # Dependency and OpenCV installer
└── docs/                     # Detailed documentation
```

The next sections briefly describe each major component.

---

## 4. C++ Examples ([`src/cpp_example/`](/src/cpp_example/))

C++ examples demonstrate how to build **DEEPX NPU–based applications in C++**.  
Each example is a **self‑contained, per‑model pipeline**:

- **Pipeline structure**
  - Input → Pre‑process → Inference (DX-RT) → Post‑process (C++ class) → Display/Save
  - Post‑processing is **not** inlined; it calls shared classes from [`src/postprocess/`](/src/postprocess/).

- **Organization**
  - Task / use‑case oriented layout:

    ```text
    src/cpp_example/
    ├── classification/                             # Image classification (EfficientNet)
    ├── object_detection/                           # Single‑stream object detection (YOLO, SCRFD, ...)
    ├── semantic_segmentation/                      # Semantic segmentation (DeepLabv3)
    ├── instance_segmentation/                      # Instance segmentation (YOLOv8seg)
    ├── ppu/                                        # PPU‑enabled variants
    ├── input_source_process_example/               # Camera / image / video / RTSP input patterns
    ├── multi_channel_process_example/              # Multi‑channel object detection pipelines
    └── object_detection_x_semantic_segmentation/   # Combined detection + segmentation pipelines
    ```

- **Variants**
  - **Sync vs. Async**
    - `*_sync.cpp`: single-threaded, step-by-step pipeline (input → inference → post-processing → display).
    - `*_async.cpp`: uses multiple threads and `RunAsync()` to overlap pre/post-processing with inference and keep the NPU busy, especially for streams (video/RTSP/camera).
  - **Post-processing**
    - All variants call shared C++ post-processing classes from [`src/postprocess/`](/src/postprocess/), rather than inlining decoding logic in each example.

- **What you learn**
  - How to wire real `.dxnn` models into a C++ pipeline.
  - How sync vs. async designs affect NPU utilization and end‑to‑end FPS.
  - How to plug in shared C++ post‑processing classes.
  - How to interpret the printed **performance summary** to see:
    - which stage (pre‑processing, inference, post‑processing, display) is the bottleneck, and
    - how design choices (sync vs. async, number of threads) change latency and overall FPS.

For details, see [`src/cpp_example/README.md`](/src/cpp_example/README.md).

---

## 5. Post-processing Libraries ([`src/postprocess/`](/src/postprocess/))

This directory hosts the **C++ post‑processing libraries** used by both C++ and Python examples.

- **Per‑model modules**
  - Each subfolder (e.g. `yolov5/`, `yolov7/`, `deeplabv3/`) provides:
    - `*_postprocess.h`: a C++ post‑processing class (e.g. `YOLOv5PostProcess`) and result structs (e.g. `YOLOv5Result`).
    - `*_postprocess.cpp`: implementation of decoding, NMS, etc.
    - `CMakeLists.txt`: builds a reusable shared library.

- **Responsibilities**
  - Take DX-RT inference outputs (tensors) and convert them into **human‑readable results**:
    - bounding boxes, scores, class IDs
    - keypoints/skeletons
    - segmentation masks
  - Apply post‑processing such as NMS and filtering.

- **Shared between languages**
  - C++ examples `#include` these headers directly.
  - Python examples access the same logic via [`dx_postprocess`](/src/bindings/python/dx_postprocess/).

For a deeper description, see [`src/postprocess/README.md`](/src/postprocess/README.md).

---

## 6. Python Bindings: `dx_postprocess` ([`src/bindings/python/dx_postprocess/`](/src/bindings/python/dx_postprocess/))

`dx_postprocess` exposes the C++ post‑processing modules to Python via **pybind11**.

- **What it provides**
  - Python classes mirroring the C++ post‑processors, e.g.:

    ```python
    from dx_postprocess import YOLOv9PostProcess
    ```

  - High‑performance, C++‑implemented post‑processing callable from Python.

- **Why it matters**
  - Lets you:
    - accelerate Python pipelines by offloading heavy post‑processing to C++, and
    - compare **pure Python vs. C++ post‑processing** on identical decoding logic.

- **Installation**
  - As part of the full build:

    ```shell
    ./build.sh
    ```

  - Or standalone:

    ```shell
    cd src/bindings/python/dx_postprocess
    pip install .
    ```

See [`src/bindings/python/dx_postprocess/README.md`](/src/bindings/python/dx_postprocess/README.md) for details and usage examples.

---

## 7. Python Examples ([`src/python_example/`](/src/python_example/))

Python examples provide **easy‑to‑read, end‑to‑end scripts** built on top of **DX-RT’s Python bindings (`dx_engine`)**.

- **Structure**

  ```text
  src/python_example/
  ├── classification/
  ├── instance_segmentation/
  ├── object_detection/
  ├── ppu/
  ├── semantic_segmentation/
  └── utils/
  ```

  - Per‑model folders (e.g. `yolov5/`, `yolov9/`) contain multiple variants:
    - `*_sync.py` / `*_async.py`
    - `*_sync_cpp_postprocess.py` / `*_async_cpp_postprocess.py`

- **Variants**
  - **Sync vs. Async**
    - `run()` vs. `run_async()` in `dx_engine.InferenceEngine`.
    - `*_sync.py`: single-threaded, step-by-step pipeline (input → inference → post-processing).
    - `*_async.py`: uses additional threads and `run_async()` to overlap pre/post-processing with inference and keep the NPU busy, especially for streams (video/RTSP/camera).
  - **Python vs. C++ post‑processing**
    - Default scripts implement post‑processing in Python.
    - `'_cpp_postprocess'` scripts use the `dx_postprocess` binding to call C++ so that Python pipelines can reuse the same C++ decoding logic as the C++ examples.

- **Performance focus**
  - All examples print a **performance summary** (latency per stage, overall FPS).
  - This lets you:
    - identify which stage is the bottleneck (pre‑processing, inference, post‑processing, display), and
    - compare how different pipeline designs — **sync vs. async** and **Python vs. C++ post‑processing** — change latency and overall FPS for the same model and input.

See [`src/python_example/README.md`](/src/python_example/README.md) for a complete guide.

---

## 8. Quick Start

### 8.1. Install dependencies

From the project root:

```shell
./install.sh --all
```

Ensure DX-RT (drivers + libraries) is already installed as described in section 2.

### 8.2. Download sample models and videos

```shell
./setup.sh
# Downloads models to assets/models and videos to assets/videos
```

### 8.3. Build DX-APP

```shell
./build.sh

# For a clean build:
# ./build.sh --clean
```

On success, C++ example binaries and required libraries are placed under `bin/` and the appropriate build directories. Python binding `dx_postprocess` is also built.

### 8.4. Run example applications

#### 8.4.1 Quick demo launchers

From the project root, you can quickly try multiple demos via helper scripts:

```shell
# C++ demo launcher
./run_demo.sh

# Python demo launcher
./run_demo_python.sh
```

Just run the script and either:
- type a menu number, or
- wait for the timeout to use the default option (`0`: YOLOv7).

#### 8.4.2 C++ (YOLOv9 object detection)

```shell
# Image inference (sync)
./bin/yolov9_sync \
  -m assets/models/YOLOV9S.dxnn \
  -i sample/img/1.jpg

# Video inference (async)
./bin/yolov9_async \
  -m assets/models/YOLOV9S.dxnn \
  -v assets/videos/dance-group.mov
```

#### 8.4.3 Python (YOLOv9 object detection)

```shell
# Python sync example
python3 src/python_example/object_detection/yolov9/yolov9_sync.py \
  --model assets/models/YOLOV9S.dxnn \
  --image sample/img/1.jpg

# Python async + C++ post-processing
python3 src/python_example/object_detection/yolov9/yolov9_async_cpp_postprocess.py \
  --model assets/models/YOLOV9S.dxnn \
  --video assets/videos/dance-group.mov
```

After each run you’ll see a performance summary in the console, and (unless disabled) a window showing the processed results.

