# DX-APP C++ Post-processing

## Overview

This directory serves as the centralized repository for **optimized C++ post-processing modules**. These libraries provide the critical decoding logic required to transform raw NPU tensor outputs into actionable, human-readable data.  

**Key Features**  

- **Performance Optimization:** Computational bottlenecks like Non-Maximum Suppression (NMS), segmentation mask resizing, and keypoint extraction are implemented in high-performance C++.  
- **Logic Unification:** By sharing the exact same codebase between C++ examples and Python bindings (`dx_postprocess`), the SDK ensures identical inference results across all development environments.  
- **Modularity:** Each model family (YOLO, DeepLab, etc.) is isolated into its own module, allowing for targeted updates and lightweight linking.  

---

## Architecture & Implementation

At a high level, these libraries manage the transition from **DX-RT Tensors** to **Structured Data**.  

### Core Processing Steps

- **Step 1.** Tensor Decoding: Converting raw logits and offsets into coordinates (bounding boxes), confidence scores, and class IDs.  
- **Step 2.** Filtering & NMS: Applying Non-Maximum Suppression to remove redundant overlapping detections and filtering results based on user-defined confidence thresholds.  
- **Step 3.** Specialized Extraction  
     : **Pose:** Extracting **x, y** coordinates and visibility scores for keypoint skeletons.  
     : **Segmentation:** Decoding segmentation logits into per-pixel masks or class color maps.  
- **Step 4.**	Hardware Adaptation: Finalizing and scaling coordinates for models that use the **PPU** to handle the initial stages of post-processing.  

### Implementation Examples

The principles above are applied across various model families as follows  

- `yolov5_postprocess`: Decodes YOLOv5 detection heads, runs NMS, and returns standard bounding boxes, scores, and class IDs.  
- `yolov5pose_postprocess`: In addition to boxes and scores, it extracts per-person keypoints and their corresponding confidence levels for skeletal mapping.  
- `deeplabv3_postprocess`: Converts segmentation logits into high-resolution per-pixel class labels or visual color maps.  
- `*_ppu_postprocess`: Designed for models compiled with **PPU support**. These modules receive data that has already been partially processed by the NPU, adapting and finalizing the outputs for the host CPU.  

### Directory Structure

The libraries are organized by model architecture. Modules appended with `_ppu` are specialized for models where partial post-processing is offloaded to the **DEEPX PPU (Post-Processing Unit)**.  

**Module Components**  

Each subdirectory follows a standardized pattern  

- `*_postprocess.h`: Defines the class interface (e.g., `YOLOv5PostProcess`) and the result structures (e.g., `YOLOv5Result`).  
- `*_postprocess.cpp`: Implementation of decoding algorithms and mathematical filters.  
- `CMakeLists.txt`: Build configuration to compile the module into a standalone shared library.  

```text
src/postprocess/
├── CMakeLists.txt
├── README.md
├── centerpose/
├── classification/
├── damoyolo/
├── deeplabv3/
├── depth/
├── dncnn/
├── efficientdet/
├── embedding/
├── espcn/
├── face3d/
├── hand_landmark/
├── nanodet/
├── obb/
├── retinaface/
├── scrfd/
├── scrfd_ppu/
├── semantic_seg/
├── ssd/
├── ulfgfd/
├── yolact/
├── yolov5/
├── yolov5_ppu/
├── yolov5face/
├── yolov5pose/
├── yolov5pose_ppu/
├── yolov5seg/
├── yolov7/
├── yolov7_ppu/
├── yolov8/
├── yolov8_ppu/
├── yolov8pose/
├── yolov8seg/
├── yolov9/
├── yolov10/
├── yolov11/
├── yolov12/
├── yolov26/
├── yolox/
├── yolox_ppu/
├── yolov3tiny_ppu/
└── zero_dce/
```

---

## Cross-Language Integration

### Usage Methods

The SDK is designed so that C++ and Python developers utilize the same underlying engine, ensuring a seamless transition from prototyping to production.  

**Method A. C++ Usage**   

Include the header and instantiate the class directly within your inference loop.  

```bash
#include "yolov5_postprocess.h" 

auto post_processor = YOLOv5PostProcess(conf_threshold, nms_threshold); 
auto final_results = post_processor.postprocess(npu_output_tensors);
```

**Method B. Python Usage**  

Import the `pybind11` wrapper via the `dx_postprocess` module to access the same C++ performance.  

```bash
from dx_postprocess import YOLOv5PostProcess 

post_processor = YOLOv5PostProcess(conf_threshold, nms_threshold) 
final_results = post_processor.postprocess(npu_output_tensors)
```

### Build Configuration

**Compilation Guide**  

Post-processing libraries are compiled automatically during the main SDK build process.  

```bash
# Execute from the dx_app/ root directory 
./build.sh
```

The resulting shared libraries and headers are staged in the project build output directories and are then used by the C++ examples and Python binding build flow.  

---

