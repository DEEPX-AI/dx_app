# dx_postprocess

High-performance post-processing library for AI models with C++ implementation and Python bindings.

## Overview

`dx_postprocess` provides Python bindings to [C++-implemented post-processing classes](../../../postprocess/) for various AI models. It accelerates post-processing operations themselves, and depending on the computing environment and pipeline bottleneck, may also improve overall throughput.

### Supported Models

- **Object Detection**: YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOX, YOLOv5Face, SCRFD, YOLOv5Pose
- **Semantic Segmentation**: DeepLabv3

## Performance Benefits

C++ post-processing accelerates the post-processing stage itself, and may also improve overall throughput depending on your environment:

- **Post-processing acceleration**: Always faster than Python implementation for the post-processing operation
- **Pipeline-wide improvement**: May boost end-to-end FPS in specific scenarios:
  - When post-processing is the primary bottleneck
  - In CPU-constrained environments where reduced CPU usage helps other pipeline stages (Read, Preprocess) and NPU utilization

**Note**: While C++ post-processing itself is faster, overall pipeline improvement depends on where the bottleneck occurs. If inference or other stages are the limiting factor, end-to-end FPS gain may be minimal.

For detailed analysis, see the [Python Examples Guide](../../../python_example/README.md).

## Installation

### Prerequisites

You need to have these installed on your system before building:

- **Python 3.8 or higher**
- **C++ compiler** (gcc 4.8+ or clang 3.3+)
- **CMake 3.16 or higher**


During the installation process, `pybind11` is automatically cloned from GitHub to `extern/pybind11` if not present.

### Installation Methods

**Method 1: Install with full dx_app build**
```bash
# From dx_app/ directory
./build.sh
```

**Method 2: Standalone installation**
```bash
# From dx_app/ directory
./src/bindings/python/dx_postprocess/install.sh
```

Available options for [`install.sh`](../../../bindings/python/dx_postprocess/install.sh):
- `--python_exec PATH`: Specify Python executable (default: `python3`)
- `--type TYPE`: Build type - `Release`, `Debug`, or `RelWithDebInfo` (default: `Release`)
- `--help`: Show help message

### Verification

```bash
# Verify installation
python3 -c "import dx_postprocess; print('Successfully installed!')"
```

## Usage

### Basic Example

```python
from dx_postprocess import YOLOv9PostProcess
import numpy as np

# Create post-processor
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

## Running Examples

Python examples in [`src/python_example/`](../../../python_example/) use this library in their `'_cpp_postprocess.py'` variants:

```bash
# From dx_app/ directory

# Sync (Image Inference)
python src/python_example/object_detection/yolov9/yolov9_sync_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --image sample/img/1.jpg

# Async (Stream Inference)
python src/python_example/object_detection/yolov9/yolov9_async_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov
```

## Technical Details

- **Implementation**: C++ post-processing classes wrapped with `pybind11`
- **Build system**: CMake with scikit-build-core
- **Source code**: 
  - Python bindings: [`src/bindings/python/dx_postprocess/postprocess_pybinding.cpp`](postprocess_pybinding.cpp)
  - C++ implementations: [`src/postprocess/`](../../../postprocess/)
