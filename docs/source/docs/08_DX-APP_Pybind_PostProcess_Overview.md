# DX-APP Python Post-processing

## dx_postprocess Overview & Performance Benefits  

The `dx_postprocess` module provides high-performance Python bindings for the **DEEPX C++ post-processing library**. By utilizing `pybind11`, it allows Python applications to execute hardware-optimized decoding logic with near-native C++ performance.  

Integrating C++ post-processing into your Python pipeline offers two primary advantages  

- **Stage Acceleration:** The post-processing operation itself (e.g., NMS, coordinate scaling) is significantly faster than native Python implementations.  
- **Pipeline-Wide Improvement:** End-to-end FPS gains occur in specific scenarios.  
    - **Primary Bottleneck:** When the post-processing stage is the slowest part of the pipeline.  
    - **CPU Contention:** In CPU-constrained environments (e.g., embedded boards), reduced CPU usage in the post-processing stage helps other stages (Read, Preprocess) and NPU utilization function more efficiently.  

!!! note "NOTE"  
    While the C++ library is faster, overall pipeline improvement depends on the bottleneck location. If NPU inference or data reading is the limiting factor, end-to-end FPS gains may be minimal, though CPU power consumption will still decrease.  

For end-to-end Python example usage, refer to the DX-APP Python usage documentation in `docs/source/docs/05_DX-APP_Python_Example_Usage_Guide.md`.  

---

## Supported Models & Tasks

The library wraps **41 C++ post-processing classes** with pybind11 bindings, ensuring consistent results between C++ and Python implementations.  

- **Object Detection:** YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLOv26, YOLOX, NanoDet, DAMOYOLO, SSD, CenterPose, EfficientDet, YOLACT  
- **Face Detection:** SCRFD, YOLOv5Face, RetinaFace, ULFGFD, Face3D  
- **Pose Estimation:** YOLOv5Pose, YOLOv8Pose  
- **Semantic Segmentation:** DeepLabV3, SemanticSeg (BiSeNet/SegFormer)  
- **Instance Segmentation:** YOLOv5Seg, YOLOv8Seg  
- **OBB Detection:** YOLOv26-OBB  
- **Classification:** EfficientNet-family  
- **Depth Estimation:** DepthPostProcess (FastDepth, SCDepth)  
- **Image Denoising:** DnCNN  
- **Super Resolution:** ESPCN  
- **Image Enhancement:** Zero-DCE  
- **Embedding:** ArcFace  
- **Hand Landmark:** HandLandmark  
- **PPU Variants:** YOLOv5-PPU, YOLOv7-PPU, YOLOv8-PPU, SCRFD-PPU, YOLOv5Pose-PPU, YOLOX-PPU, YOLOv3Tiny-PPU  

---

### Installation

**Prerequisites:** Before building, ensure the following are installed on your system  

- **Python:** 3.8 or higher  
- **Compiler:** GCC 4.8+ or Clang 3.3+  
- **Build Tool:** CMake 3.16 or higher  

During the installation, `pybind11` is automatically cloned from GitHub to `extern/pybind11` if it is not already present.  

**Installation Methods:** The module is automatically built during the full SDK setup, but it can also be installed independently from the binding directory.  

| **Method** | **Command** | **Recommended For** | 
|----|----|----|
| **Full Build** | `./build.sh` | Initial environment setup and full SDK deployment | 
| **Standalone** | `cd ./src/bindings/python/dx_postprocess && python3 -m pip install .` | Focused updates to the post-processing logic | 

**Standalone Installation Notes:** The standalone installation uses the local `pyproject.toml` and the configured Python packaging backend. Ensure your active Python environment has access to the required build dependencies before running `python3 -m pip install .`.  

**Verification:** To ensure the module is correctly linked to your Python environment, run:
```bash
python3 -c "import dx_postprocess; print('dx_postprocess successfully installed!')"  
```

---

## Usage Guide & Running Examples

The `dx_postprocess` classes expect a list of NumPy arrays directly from the `InferenceEngine`.  

Basic Implementation Example  
```bash
from dx_postprocess import YOLOv9PostProcess 
import numpy as np 

# Initialize the optimized C++ post-processor 
postprocessor = YOLOv9PostProcess( 
input_w=640, 
input_h=640, 
score_threshold=0.25, 
nms_threshold=0.45, 
is_ort_configured=True 
) 

# Post-process inference output
# ie_output is a list of numpy arrays from InferenceEngine
detections = postprocessor.postprocess(ie_output)

# detections is a numpy array with shape (N, 6)
# Each row: [x1, y1, x2, y2, confidence, class_id]
```

Running Examples  
```bash
# Python examples in src/python_example/ utilize this library in their '_cpp_postprocess.py' variants.
# From dx_app/ directory

# Sync (Image Inference)
python src/python_example/object_detection/yolov9s/yolov9s_sync_cpp_postprocess.py --model assets/models/YoloV9S.dxnn --image sample/img/sample_kitchen.jpg

# Async (Stream Inference)
python src/python_example/object_detection/yolov9s/yolov9s_async_cpp_postprocess.py --model assets/models/YoloV9S.dxnn --video assets/videos/dance-group.mov
```

---

## Technical Details 

- **Implementation:** C++ post-processing classes wrapped with pybind11 for seamless Python integration.  
- **Build System:** Utilizes CMake with scikit-build-core for robust, cross-platform extension building.  
- **Source Code:**  
     : **Python Bindings:** `src/bindings/python/dx_postprocess/postprocess_pybinding.cpp`  
     : **C++ Implementations:** `src/postprocess/`  

---

