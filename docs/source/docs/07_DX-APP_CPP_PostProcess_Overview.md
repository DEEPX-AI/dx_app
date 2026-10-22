# DX-APP C++ Post-processing

## Overview

This directory serves as the centralized repository for **optimized C++ post-processing modules**. These libraries provide the critical decoding logic required to transform raw NPU tensor outputs into actionable, human-readable data.  

**Key Features**  

- **Performance Optimization:** Computational bottlenecks like Non-Maximum Suppression (NMS), segmentation mask resizing, and keypoint extraction are implemented in high-performance C++.  
- **Python Bindings:** These libraries are consumed by the `pybind11` bindings (`dx_postprocess`), which enable the `*_cpp_postprocess.py` variants of the Python examples.  
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
- **Step 4.** Hardware Adaptation: Finalizing and scaling coordinates for models that use the **PPU** to handle the initial stages of post-processing.  

### Implementation Examples

The principles above are applied across various model families as follows  

- `yolov5_postprocess`: Decodes YOLOv5 detection heads, runs NMS, and returns standard bounding boxes, scores, and class IDs.  
- `yolov5pose_postprocess`: In addition to boxes and scores, it extracts per-person keypoints and their corresponding confidence levels for skeletal mapping.  
- `deeplabv3_postprocess`: Applies argmax to the segmentation logits and returns a per-pixel class-ID mask at the model's output resolution (resizing and coloring are left to the caller).  
- `*_ppu_postprocess`: Designed for models compiled with **PPU support**. These modules receive data that has already been partially processed by the NPU, adapting and finalizing the outputs for the host CPU.  

!!! note "Example-layer fast segmentation"

     The C++ example runtime also provides `FastSegmentationPostprocessor` in `src/cpp_example/common/processors/segmentation_postprocessor.hpp`. It is a generic low-resolution semantic-segmentation fast path that performs argmax before nearest-neighbor resize/crop. This class is used by generated examples through the `fast_segmentation` alias in `scripts/add_model_variant.py`; it is separate from the standalone `src/postprocess/` pybind library modules listed below.

### Directory Structure

The libraries are organized by model architecture. Modules appended with `_ppu` are specialized for models where partial post-processing is offloaded to the **DEEPX PPU (Post-Processing Unit)**.  

**Module Components**  

Each subdirectory follows a standardized pattern  

- `*_postprocess.h`: Defines the class interface (e.g., `YOLOv5PostProcess`) and the result structures (e.g., `YOLOv5Result`).  
- `*_postprocess.cpp`: Implementation of decoding algorithms and mathematical filters.  
- `CMakeLists.txt`: present in the modules listed in `src/postprocess/CMakeLists.txt`; `./build.sh --all --make_so` builds each of them as a standalone shared library (`lib/libdxapp_<module>_postprocess.so`). With the default `--minimal` build (or `--target`/`--category`), `--make_so` builds none of them.  

```text
src/postprocess/
├── CMakeLists.txt
├── README.md
├── anomaly/
├── attribute/
├── centerpose/
├── classification/
├── clip_text/
├── damoyolo/
├── deeplabv3/
├── depth/
├── dncnn/
├── dope/
├── efficientdet/
├── embedding/
├── espcn/
├── face3d/
├── hand_landmark/
├── mediapipe_hand/
├── nanodet/
├── obb/
├── picodet/
├── ppmatting/
├── preopt/
├── realesrgan/
├── retinaface/
├── rtdetr/
├── scrfd/
├── scrfd_ppu/
├── semantic_seg/
├── sfa3d/
├── ssd/
├── superpoint/
├── ulfgfd/
├── vitpose/
├── yolact/
├── yolopv2/
├── yolov4/
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

The SDK is designed so that C++ and Python developers utilize the same underlying engine, ensuring a seamless transition  from prototyping to production.  

**Method A. C++ Usage**   

Include the header and instantiate the class directly within your inference loop.  

```cpp
#include "yolov5_postprocess.h"

YOLOv5PostProcess post_processor(input_w, input_h, obj_threshold, score_threshold,
                                 nms_threshold, is_ort_configured);
auto final_results = post_processor.postprocess(npu_output_tensors);  // std::vector<YOLOv5Result>
```

**Method B. Python Usage**  

Import the `pybind11` wrapper via the `dx_postprocess` module to access the same C++ performance.  

```python
from dx_postprocess import YOLOv5PostProcess

post_processor = YOLOv5PostProcess(input_w, input_h, obj_threshold, score_threshold,
                                   nms_threshold, is_ort_configured)
final_results = post_processor.postprocess(ie_output)
```

### Build Configuration

**Compilation Guide**  

The sources in `src/postprocess/` are compiled into the `dx_postprocess` Python module when `./build.sh` installs it (default build) or when you run `python3 -m pip install .` in `src/bindings/python/dx_postprocess/`.  

```bash
# Execute from the dx_app/ root directory 
./build.sh
```

`./build.sh --all --make_so` additionally builds the modules listed in `src/postprocess/CMakeLists.txt` as standalone shared libraries in `lib/`. `--make_so` has no effect on the default `--minimal`, `--target` or `--category` builds. The C++ examples do not link these libraries. They use `src/cpp_example/common/processors/`, except for `sfa3d`, `superpoint`, `dope` and `yolopv2`, whose sources are compiled directly into the C++ examples.  

---

