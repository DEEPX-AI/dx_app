# DX Postprocess API Reference

> 37 pybind11 C++ postprocess bindings for dx_app, organized by AI task.

## Overview

`dx_postprocess` is a pybind11 module that exposes high-performance C++ postprocessing
functions to Python. These are 5-10x faster than equivalent Python implementations and
are used by the `*_cpp_postprocess` variants of dx_app examples.

**Installation:** Built and installed via `./build.sh` from dx_app root. The shared
library is installed into the Python environment as `dx_postprocess.so`.

**Import:**
```python
import dx_postprocess
# or import specific functions:
from dx_postprocess import YoloV8PostProcess, ClassificationPostProcess
```

---

## Detection (11 bindings)

### YoloV5PostProcess

```python
pp = dx_postprocess.YoloV5PostProcess(
    input_width: int,       # Model input width (e.g., 640)
    input_height: int,      # Model input height (e.g., 640)
    score_threshold: float, # Confidence threshold (e.g., 0.25)
    nms_threshold: float,   # NMS IoU threshold (e.g., 0.45)
    use_ort: bool           # True if using ONNX Runtime backend
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
# Detection: (x1, y1, x2, y2, score, class_id)
```

### YoloV5TinyPostProcess

```python
pp = dx_postprocess.YoloV5TinyPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### YoloV7PostProcess

```python
pp = dx_postprocess.YoloV7PostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### YoloV7TinyPostProcess

```python
pp = dx_postprocess.YoloV7TinyPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### YoloV8PostProcess

```python
pp = dx_postprocess.YoloV8PostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### YoloV10PostProcess

```python
pp = dx_postprocess.YoloV10PostProcess(
    input_width: int, input_height: int,
    score_threshold: float,  # NMS-free: only score_threshold
    use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

**Note:** YOLOv10 is NMS-free by design. The `nms_threshold` parameter is not used.

### YoloV11PostProcess

```python
pp = dx_postprocess.YoloV11PostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### Yolo26PostProcess

```python
pp = dx_postprocess.Yolo26PostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### SSDPostProcess

```python
pp = dx_postprocess.SSDPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### EfficientDetPostProcess

```python
pp = dx_postprocess.EfficientDetPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### NanodetPostProcess

```python
pp = dx_postprocess.NanodetPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float, use_ort: bool
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

### Detection Return Type

All detection postprocessors return `list[Detection]` where each Detection is:

```python
class Detection:
    x1: float       # Top-left x (normalized 0-1, relative to input_width)
    y1: float       # Top-left y (normalized 0-1, relative to input_height)
    x2: float       # Bottom-right x
    y2: float       # Bottom-right y
    score: float    # Confidence score (0-1)
    class_id: int   # Class index into labels file
```

---

## Classification (4 bindings)

### ClassificationPostProcess

```python
pp = dx_postprocess.ClassificationPostProcess(
    top_k: int = 5,         # Number of top predictions to return
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[ClassResult]
# ClassResult: (class_id, score)
```

### EfficientNetPostProcess

```python
pp = dx_postprocess.EfficientNetPostProcess(top_k: int, use_ort: bool)
results = pp.process(outputs: list[np.ndarray]) -> list[ClassResult]
```

### MobileNetV2PostProcess

```python
pp = dx_postprocess.MobileNetV2PostProcess(top_k: int, use_ort: bool)
results = pp.process(outputs: list[np.ndarray]) -> list[ClassResult]
```

### ResNetPostProcess

```python
pp = dx_postprocess.ResNetPostProcess(top_k: int, use_ort: bool)
results = pp.process(outputs: list[np.ndarray]) -> list[ClassResult]
```

### Classification Return Type

```python
class ClassResult:
    class_id: int    # Class index
    score: float     # Softmax probability (0-1)
```

---

## Segmentation (4 bindings)

### YoloV5SegPostProcess (Instance)

```python
pp = dx_postprocess.YoloV5SegPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    mask_threshold: float = 0.5,    # Mask binarization threshold
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[SegResult]
```

### YoloV8SegPostProcess (Instance)

```python
pp = dx_postprocess.YoloV8SegPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    mask_threshold: float = 0.5,
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[SegResult]
```

### BisenetV1PostProcess (Semantic)

```python
pp = dx_postprocess.BisenetV1PostProcess(
    input_width: int, input_height: int,
    num_classes: int = 19,    # Cityscapes default
    use_ort: bool = False
)
mask = pp.process(outputs: list[np.ndarray]) -> np.ndarray
# mask: (H, W) int32 array with class IDs per pixel
```

### DeepLabV3PostProcess (Semantic)

```python
pp = dx_postprocess.DeepLabV3PostProcess(
    input_width: int, input_height: int,
    num_classes: int = 21,    # VOC default
    use_ort: bool = False
)
mask = pp.process(outputs: list[np.ndarray]) -> np.ndarray
```

### Segmentation Return Types

Instance segmentation:
```python
class SegResult:
    x1: float        # Bounding box
    y1: float
    x2: float
    y2: float
    score: float     # Detection confidence
    class_id: int    # Class index
    mask: np.ndarray # (H, W) binary mask for this instance
```

Semantic segmentation returns `np.ndarray` of shape `(H, W)` with per-pixel class IDs.

---

## Pose Estimation (2 bindings)

### YoloV5PosePostProcess

```python
pp = dx_postprocess.YoloV5PosePostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    num_keypoints: int = 17,   # COCO keypoints
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[PoseResult]
```

### YoloV8PosePostProcess

```python
pp = dx_postprocess.YoloV8PosePostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    num_keypoints: int = 17,
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[PoseResult]
```

### Pose Return Type

```python
class PoseResult:
    x1: float              # Person bounding box
    y1: float
    x2: float
    y2: float
    score: float           # Person detection confidence
    keypoints: list[tuple] # [(x, y, confidence), ...] for each keypoint
```

COCO 17 keypoints: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle.

---

## Face Detection (3 bindings)

### SCRFDPostProcess

```python
pp = dx_postprocess.SCRFDPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[FaceResult]
```

### RetinaFacePostProcess

```python
pp = dx_postprocess.RetinaFacePostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[FaceResult]
```

### YoloV5FacePostProcess

```python
pp = dx_postprocess.YoloV5FacePostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[FaceResult]
```

### Face Return Type

```python
class FaceResult:
    x1: float               # Face bounding box
    y1: float
    x2: float
    y2: float
    score: float            # Detection confidence
    landmarks: list[tuple]  # 5 facial landmarks: [(x, y), ...]
    # Landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
```

---

## Depth Estimation (1 binding)

### FastDepthPostProcess

```python
pp = dx_postprocess.FastDepthPostProcess(
    input_width: int, input_height: int,
    use_ort: bool = False
)
depth_map = pp.process(outputs: list[np.ndarray]) -> np.ndarray
# depth_map: (H, W) float32 array with depth values in meters
```

---

## Image Denoising (1 binding)

### DnCNNPostProcess

```python
pp = dx_postprocess.DnCNNPostProcess(
    input_width: int, input_height: int,
    use_ort: bool = False
)
denoised = pp.process(outputs: list[np.ndarray]) -> np.ndarray
# denoised: (H, W, 3) uint8 BGR image
```

---

## Image Enhancement (1 binding)

### ZeroDCEPostProcess

```python
pp = dx_postprocess.ZeroDCEPostProcess(
    input_width: int, input_height: int,
    use_ort: bool = False
)
enhanced = pp.process(outputs: list[np.ndarray]) -> np.ndarray
# enhanced: (H, W, 3) uint8 BGR image with improved lighting
```

---

## Super Resolution (1 binding)

### ESPCNPostProcess

```python
pp = dx_postprocess.ESPCNPostProcess(
    input_width: int, input_height: int,
    scale_factor: int = 4,    # Upscale factor
    use_ort: bool = False
)
upscaled = pp.process(outputs: list[np.ndarray]) -> np.ndarray
# upscaled: (H*scale, W*scale, 3) uint8 BGR image
```

---

## Embedding (1 binding)

### ArcFacePostProcess

```python
pp = dx_postprocess.ArcFacePostProcess(
    use_ort: bool = False
)
embedding = pp.process(outputs: list[np.ndarray]) -> np.ndarray
# embedding: (128,) float32 normalized feature vector
```

---

## Hand Landmark (1 binding)

### HandLandmarkPostProcess

```python
pp = dx_postprocess.HandLandmarkPostProcess(
    input_width: int, input_height: int,
    score_threshold: float = 0.5,
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[HandResult]
```

### Hand Return Type

```python
class HandResult:
    score: float            # Hand detection confidence
    handedness: str         # "Left" or "Right"
    landmarks: list[tuple]  # 21 landmarks: [(x, y, z), ...]
    # MediaPipe hand landmark indices (0=wrist, 4=thumb_tip, 8=index_tip, ...)
```

---

## OBB Detection (1 binding)

### Yolo26OBBPostProcess

```python
pp = dx_postprocess.Yolo26OBBPostProcess(
    input_width: int, input_height: int,
    score_threshold: float,     # Only score_threshold — NO NMS for OBB
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[OBBResult]
```

**Important:** OBB (Oriented Bounding Box) detection uses `score_threshold` only.
There is no NMS threshold parameter for OBB models.

### OBB Return Type

```python
class OBBResult:
    cx: float       # Center x
    cy: float       # Center y
    width: float    # Box width
    height: float   # Box height
    angle: float    # Rotation angle in radians
    score: float    # Confidence
    class_id: int   # Class index
```

---

## PPU (1 binding)

### PPUPostProcess

```python
pp = dx_postprocess.PPUPostProcess(
    input_width: int, input_height: int,
    score_threshold: float, nms_threshold: float,
    use_ort: bool = False
)
results = pp.process(outputs: list[np.ndarray]) -> list[Detection]
```

**Important:** PPU (Pre/Post Processing Unit) models have dedicated postprocessors.
Do NOT use standard YOLO postprocessors with PPU models — output tensor format differs.

---

## Common Usage Pattern (cpp_postprocess variant)

```python
from dx_postprocess import YoloV8PostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections

def on_engine_init(runner):
    input_w = runner.input_width
    input_h = runner.input_height
    use_ort = InferenceOption().get_use_ort()
    runner._cpp_postprocessor = YoloV8PostProcess(
        input_w, input_h, 0.25, 0.45, use_ort
    )
    runner._cpp_convert_fn = convert_cpp_detections

runner = SyncRunner(factory, on_engine_init=on_engine_init)
```

## Build Requirements

```bash
# Build from dx_app root
./build.sh

# Verify installation
python -c "import dx_postprocess; print(dir(dx_postprocess))"
```

If `import dx_postprocess` raises `ImportError`, the pybind11 module has not been
built. Run `./build.sh` from the dx_app root directory.
