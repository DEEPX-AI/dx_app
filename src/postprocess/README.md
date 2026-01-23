## C++ Post-processing Overview & Core Objectives 

This directory serves as the centralized repository for **optimized C++ post-processing modules**. These libraries provide the critical decoding logic required to transform raw NPU tensor outputs into actionable, human-readable data.  

- **Performance Optimization:** Computational bottlenecks like Non-Maximum Suppression (NMS), segmentation mask resizing, and keypoint extraction are implemented in high-performance C++.  
- **Logic Unification:** By sharing the exact same codebase between C++ examples and Python bindings (`dx_postprocess`), the SDK ensures identical inference results across all development environments.  
-	**Modularity:** Each model family (YOLO, DeepLab, etc.) is isolated into its own module, allowing for targeted updates and lightweight linking.  

---

## Functional Responsibilities

At a high level, these libraries manage the transition from **DX-RT Tensors** to **Structured Data**.  

**Core Processing Steps**  

- **Step 1.** Tensor Decoding: Converting raw logits and offsets into coordinates (bounding boxes), confidence scores, and class IDs.  
- **Step 2.** Filtering & NMS: Applying Non-Maximum Suppression to remove redundant overlapping detections and filtering results based on user-defined confidence thresholds.  
- **Step 3.** Specialized Extraction  
     : **Pose:** Extracting **x, y** coordinates and visibility scores for keypoint skeletons.  
     : **Segmentation:** Decoding segmentation logits into per-pixel masks or class color maps.  
- **Step 4.**	Hardware Adaptation: Finalizing and scaling coordinates for models that use the **PPU** to handle the initial stages of post-processing.  

**Implementation Examples**  

The principles above are applied across various model families as follows  

- `yolov5_postprocess`: Decodes YOLOv5 detection heads, runs NMS, and returns standard bounding boxes, scores, and class IDs.  
- `yolov5pose_postprocess`: In addition to boxes and scores, it extracts per-person keypoints and their corresponding confidence levels for skeletal mapping.  
- `deeplabv3_postprocess`: Converts segmentation logits into high-resolution per-pixel class labels or visual color maps.  
- `*_ppu_postprocess`: Designed for models compiled with **PPU support**. These modules receive data that has already been partially processed by the NPU, adapting and finalizing the outputs for the host CPU.  

---

## Directory Structure

The libraries are organized by model architecture. Modules appended with `_ppu` are specialized for models where partial post-processing is offloaded to the **DEEPX PPU (Post-Processing Unit)**.  

**Module Components**  

Each subdirectory follows a standardized pattern  

- `*_postprocess.h`: Defines the class interface (e.g., `YOLOv5PostProcess`) and the result structures (e.g., `YOLOv5Result`).  
- `*_postprocess.cpp`: Implementation of decoding algorithms and mathematical filters.  
- `CMakeLists.txt`: Build configuration to compile the module into a standalone shared library.  

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

---

## Cross-Language Integration

The SDK is designed so that C++ and Python developers utilize the same underlying engine, ensuring a seamless transition from prototyping to production.  

**C++ Usage**  

Include the header and instantiate the class directly within your inference loop.  
```bash
#include "yolov5_postprocess.h" 

auto post_processor = YOLOv5PostProcess(conf_threshold, nms_threshold); 
auto final_results = post_processor.postprocess(npu_output_tensors);
```

**Python Usage**  

Import the `pybind11` wrapper via the `dx_postprocess` module to access the same C++ performance.  
```bash
from dx_postprocess import YOLOv5PostProcess 

post_processor = YOLOv5PostProcess(conf_threshold, nms_threshold) 
final_results = post_processor.postprocess(npu_output_tensors)
```

---

## Build Configuration

Post-processing libraries are compiled automatically during the main SDK build process.  
```bash
# Execute from the dx_app/ root directory 
./build.sh
```

The resulting shared libraries and headers are staged in the global `build/` directory, where they are picked up by the examples and the Python library installer.  

---
