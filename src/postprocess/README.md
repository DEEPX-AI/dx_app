# Post-processing Libraries Overview

This directory contains the **C++ post-processing libraries** used by both:

- C++ examples under [`src/cpp_example/`](../cpp_example/)
- Python examples via the [`dx_postprocess`](../bindings/python/dx_postprocess/) pybind11 module

Each post-processing module implements **model-specific decoding logic** (e.g., YOLO box/score decoding, NMS, keypoint extraction, segmentation mask decoding) in optimized C++ so that:

- C++ examples can simply call a C++ class instead of re-implementing algorithms.
- Python examples can reuse the **exact same** post-processing logic via the `dx_postprocess` module, making it easy to compare **pure Python post-processing vs. C++ post-processing**

---

## 1. Directory Structure

Post-processing code is organized **by model** as follows:

```text
src/postprocess/
├── CMakeLists.txt
├── README.md                      # (this file)
├── deeplabv3/
│   ├── CMakeLists.txt
│   ├── deeplabv3_postprocess.cpp
│   └── deeplabv3_postprocess.h
├── scrfd/
│   ├── scrfd_postprocess.cpp
│   ├── scrfd_postprocess.h
├── scrfd_ppu/
│   ├── scrfd_ppu_postprocess.cpp
│   └── scrfd_ppu_postprocess.h
├── yolov5/
│   ├── yolov5_postprocess.cpp
│   └── yolov5_postprocess.h
├── yolov5_ppu/
│   ├── yolov5_ppu_postprocess.cpp
│   └── yolov5_ppu_postprocess.h
├── yolov5face/
│   ├── yolov5face_postprocess.cpp
│   └── yolov5face_postprocess.h
├── yolov5pose/
│   ├── yolov5pose_postprocess.cpp
│   └── yolov5pose_postprocess.h
├── yolov5pose_ppu/
│   ├── yolov5pose_ppu_postprocess.cpp
│   └── yolov5pose_ppu_postprocess.h
├── yolov7/
│   ├── yolov7_postprocess.cpp
│   └── yolov7_postprocess.h
├── yolov7_ppu/
│   ├── yolov7_ppu_postprocess.cpp
│   └── yolov7_ppu_postprocess.h
├── yolov8/
│   ├── yolov8_postprocess.cpp
│   └── yolov8_postprocess.h
├── yolov8seg/
│   ├── yolv8seg_postprocess.cpp
│   └── yolv8seg_postprocess.h
├── yolov9/
│   ├── yolov9_postprocess.cpp
│   └── yolov9_postprocess.h
├── yolov10/
│   ├── yolov10_postprocess.cpp
│   └── yolov10_postprocess.h
├── yolov11/
│   ├── yolov11_postprocess.cpp
│   └── yolov11_postprocess.h
├── yolov12/
│   ├── yolov12_postprocess.cpp
│   └── yolov12_postprocess.h
└── yolox/
    ├── yolox_postprocess.cpp
    └── yolox_postprocess.h
```

Each subdirectory typically provides:

- A **header** (`*_postprocess.h`) declaring:
  - a C++ post-processing class (e.g. `YOLOv5PostProcess`), and
  - the corresponding result data structures (e.g. `YOLOv5Result`) used to hold decoded outputs.
- A **source file** (`*_postprocess.cpp`) implementing the algorithms.
- A **CMakeLists.txt** to build that post-processing module into a shared library.

---

## 2. What Each Library Does

At a high level, each module takes **inference results** (DX-RT tensors) and converts them into **human‑readable results**.

Typical responsibilities include:

- Decoding model output tensors into:
  - bounding boxes
  - class scores
  - keypoints / skeletons
  - segmentation masks
- Applying **NMS (Non‑Maximum Suppression)** or other filtering
- Producing a convenient C++ data structure used by examples for:
  - visualization (drawing boxes, keypoints, masks)

Examples:

- `yolov5_postprocess`  
  Decodes YOLOv5 detection heads, runs NMS, returns bounding boxes + scores + class IDs.

- `yolov5pose_postprocess`  
  In addition to boxes/scores, extracts per‑person keypoints and their confidence.

- `deeplabv3_postprocess`  
  Converts segmentation logits into per‑pixel class labels or color maps.

- `*_ppu_postprocess`  
  Works with models compiled with **PPU** support, where some post‑processing runs on the NPU itself; these modules adapt and finalize the PPU outputs.

---

## 3. Relationship with C++ and Python Examples

These libraries are intended to be **shared between C++ and Python**:

- C++ examples (e.g. [`src/cpp_example/object_detection/yolov5/yolov5_sync.cpp`](../cpp_example/object_detection/yolov5/yolov5_sync.cpp) do:
  - Include the header: `#include "yolov5_postprocess.h"`
  - Create an instance: `post_processor =
        YOLOv5PostProcess(...);`
  - Call a method like `postprocessor.postprocess(outputs);`

- Python examples (e.g. [`src/python_example/object_detection/yolov5/yolov5_sync_cpp_postprocess.py`](../python_example/object_detection/yolov5/yolov5_sync_cpp_postprocess.py) use **pybind11** bindings that wrap the same C++ classes:
  - `from dx_postprocess import YOLOv5PostProcess`
  - Create and call the same logical post‑processor from Python.
  - This guarantees **identical decoding results** between C++ and Python.

---

## 4. Building the Post-processing Libraries

When you build the main project (see `dx_app/README.md`):

```bash
# From project root
./build.sh
```

the postprocess modules are built automatically, and the resulting shared libraries + headers are placed in locations used by the examples and bindings.

