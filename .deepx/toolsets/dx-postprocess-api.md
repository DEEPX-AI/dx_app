# DX Postprocess API Reference

> Pybind11 C++ postprocess bindings exposed via `dx_postprocess` module.
> See `src/bindings/python/dx_postprocess/postprocess_pybinding.cpp` for the
> complete list of postprocessor classes. All classes live in a single pybind11
> module built from C++ for 5-10x speedup over equivalent Python implementations.

## ⚠️ Anti-Fabrication Notice

**Do NOT invent classes.** If a class is not listed in the table below, check the
pybind11 source file (`postprocess_pybinding.cpp`) before assuming it exists or
doesn't exist. When uncertain, check the source directly. Common mistakes:
- Using `process()` — the method is **`postprocess()`**
- Describing returns as Python dataclasses — returns are **always numpy arrays**
- Inventing architecture-specific wrappers (e.g., `MobileNetV2PostProcess`) —
  use the generic class instead (e.g., `ClassificationPostProcess`)

## Source File

```
dx_app/src/bindings/python/dx_postprocess/postprocess_pybinding.cpp  (1541 lines)
```

All constructor signatures, method names, and return types are defined in this
single file. When in doubt, read the source — it is the only authority.

## Key Rules

1. **Method is ALWAYS `postprocess(ie_output)`** — never `process()`, never `run()`
2. **`ie_output` is always `List[np.ndarray]`** — the raw output tensors from inference
3. **Returns are always numpy arrays** — never Python dataclasses, never lists of objects
4. **`is_ort_configured`** (not `use_ort`) — boolean flag for ONNX Runtime tensor layout
5. **Import:** `from dx_postprocess import YOLOv8PostProcess`

## Complete Class Reference Table

| # | Class | Category | Params | Return Shape |
|---|-------|----------|--------|-------------|
| 1 | `YOLOv5PostProcess` | Detection | 6 | `ndarray[N,6]` |
| 2 | `YOLOv7PostProcess` | Detection | 6 | `ndarray[N,6]` |
| 3 | `YOLOXPostProcess` | Detection | 6 | `ndarray[N,6]` |
| 4 | `YOLOv8PostProcess` | Detection | 5 | `ndarray[N,6]` |
| 5 | `YOLOv9PostProcess` | Detection | 5 | `ndarray[N,6]` |
| 6 | `YOLOv10PostProcess` | Detection | 5 | `ndarray[N,6]` |
| 7 | `YOLOv11PostProcess` | Detection | 5 | `ndarray[N,6]` |
| 8 | `YOLOv12PostProcess` | Detection | 5 | `ndarray[N,6]` |
| 9 | `YOLOv26PostProcess` | Detection | 5 | `ndarray[N,6]` |
| 10 | `SSDPostProcess` | Detection | 6 | `ndarray[N,6]` |
| 11 | `NanoDetPostProcess` | Detection | 6 | `ndarray[N,6]` |
| 12 | `DamoYOLOPostProcess` | Detection | 5 | `ndarray[N,6]` |
| 13 | `EfficientDetPostProcess` | Detection | 6 | `ndarray[N,6]` |
| 14 | `YOLOv5FacePostProcess` | Face | 6 | `ndarray[N,21]` |
| 15 | `SCRFDPostProcess` | Face | 5 | `ndarray[N,21]` |
| 16 | `RetinaFacePostProcess` | Face | 4 | `ndarray[N,16]` |
| 17 | `ULFGFDPostProcess` | Face | 4 | `ndarray[N,6]` |
| 18 | `Face3DPostProcess` | Face | 2 | `ndarray[P]` |
| 19 | `YOLOv5PosePostProcess` | Pose | 6 | `ndarray[N,57]` |
| 20 | `YOLOv8PosePostProcess` | Pose | 5 | `ndarray[N,57]` |
| 21 | `CenterPosePostProcess` | Pose | 5 | `ndarray[N,30]` |
| 22 | `HandLandmarkPostProcess` | Landmark | 3 | `ndarray` |
| 23 | `YOLOv5SegPostProcess` | Segmentation | 6 | `tuple(ndarray[N,6], ndarray[N,H,W])` |
| 24 | `YOLOv8SegPostProcess` | Segmentation | 5 | `tuple(ndarray[N,6], ndarray[N,H,W])` |
| 25 | `YOLACTPostProcess` | Instance Seg | 6 | `tuple(ndarray[N,6], ndarray[N,H,W])` |
| 26 | `DeepLabv3PostProcess` | Segmentation | 2 | `ndarray[H,W] int32` |
| 27 | `SemanticSegPostProcess` | Segmentation | 3 | `ndarray[H,W] int32` |
| 28 | `ClassificationPostProcess` | Classification | 1 | `ndarray[K,2] float32` |
| 29 | `YOLOv5PPUPostProcess` | PPU | 5 | `ndarray[N,6]` |
| 30 | `YOLOv7PPUPostProcess` | PPU | 5 | `ndarray[N,6]` |
| 31 | `YOLOv8PPUPostProcess` | PPU | 4 | `ndarray[N,6]` |
| 32 | `YOLOXPPUPostProcess` | PPU | 5 | `ndarray[N,6]` |
| 33 | `YOLOv3TinyPPUPostProcess` | PPU | 5 | `ndarray[N,6]` |
| 34 | `YOLOv5PosePPUPostProcess` | PPU | 4 | `ndarray[N,57]` |
| 35 | `SCRFDPPUPostProcess` | PPU | 4 | `ndarray[N,21]` |
| 36 | `DepthPostProcess` | Depth | 2 | `ndarray[H,W] uint8` |
| 37 | `DnCNNPostProcess` | Denoising | 2 | `ndarray[H,W] float32` |
| 38 | `EmbeddingPostProcess` | Embedding | 1 | `ndarray[D] float32` |
| 39 | `ESPCNPostProcess` | Super Resolution | 3 | `ndarray float32` |
| 40 | `ZeroDCEPostProcess` | Enhancement | 2 | `ndarray[C,H,W] float32` |
| 41 | `OBBPostProcess` | OBB Detection | 3 | `ndarray[N,7]` |

## Constructor Details by Category

### Detection — Anchor-Based (6 params, includes `obj_threshold`)

```python
# YOLOv5, YOLOv7, YOLOX — all share the same 6-param signature
pp = dx_postprocess.YOLOv5PostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold, is_ort_configured)
pp = dx_postprocess.YOLOv7PostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold, is_ort_configured)
pp = dx_postprocess.YOLOXPostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold, is_ort_configured)
```

### Detection — Anchor-Free (5 params, NO `obj_threshold`)

```python
# YOLOv8, v9, v10, v11, v12, v26 — all share the same 5-param signature
pp = dx_postprocess.YOLOv8PostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)
pp = dx_postprocess.YOLOv9PostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)
pp = dx_postprocess.YOLOv10PostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)
pp = dx_postprocess.YOLOv11PostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)
pp = dx_postprocess.YOLOv12PostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)
pp = dx_postprocess.YOLOv26PostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)
```

### Detection — Special Parameters

```python
pp = dx_postprocess.SSDPostProcess(input_w, input_h, score_threshold, nms_threshold, num_classes=20, has_background=True)
pp = dx_postprocess.NanoDetPostProcess(input_w, input_h, score_threshold, nms_threshold, num_classes=80, reg_max=10)
pp = dx_postprocess.DamoYOLOPostProcess(input_w, input_h, score_threshold, nms_threshold, num_classes=80)
pp = dx_postprocess.EfficientDetPostProcess(input_w, input_h, score_threshold=0.3, nms_threshold=0.45, num_classes=90, has_background=True)  # → ndarray[N,6]
```

### Face Detection

```python
pp = dx_postprocess.YOLOv5FacePostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold, is_ort_configured)  # → ndarray[N,21]
pp = dx_postprocess.SCRFDPostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)  # → ndarray[N,21]
pp = dx_postprocess.RetinaFacePostProcess(input_w, input_h, score_threshold=0.5, nms_threshold=0.4)       # → ndarray[N,16]
pp = dx_postprocess.ULFGFDPostProcess(input_w, input_h, score_threshold=0.7, nms_threshold=0.3)           # → ndarray[N,6]
pp = dx_postprocess.Face3DPostProcess(input_w, input_h)                                                    # → ndarray[P]
```

### Pose Estimation

```python
pp = dx_postprocess.YOLOv5PosePostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold, is_ort_configured)  # → ndarray[N,57]
pp = dx_postprocess.YOLOv8PosePostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured=False)           # → ndarray[N,57]
pp = dx_postprocess.CenterPosePostProcess(input_w, input_h, score_threshold=0.3, nms_threshold=0.5, num_keypoints=8)           # → ndarray[N,30]
```

### Landmark

```python
pp = dx_postprocess.HandLandmarkPostProcess(input_w, input_h, confidence_threshold=0.5)  # → ndarray
# Check pybinding source for exact return shape
```

### Segmentation

```python
# Instance segmentation — returns tuple of (detections, masks)
pp = dx_postprocess.YOLOv5SegPostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold, is_ort_configured=True)  # → tuple(ndarray[N,6], ndarray[N,H,W])
pp = dx_postprocess.YOLOv8SegPostProcess(input_w, input_h, score_threshold, nms_threshold, is_ort_configured)                      # → tuple(ndarray[N,6], ndarray[N,H,W])
pp = dx_postprocess.YOLACTPostProcess(input_w, input_h, score_threshold=0.3, nms_threshold=0.5, num_classes=80, has_background=True)  # → tuple(ndarray[N,6], ndarray[N,H,W])

# Semantic segmentation — returns per-pixel class map
pp = dx_postprocess.DeepLabv3PostProcess(input_w, input_h)                  # → ndarray[H,W] int32
pp = dx_postprocess.SemanticSegPostProcess(input_w, input_h, num_classes=0) # → ndarray[H,W] int32
```

### Classification

```python
pp = dx_postprocess.ClassificationPostProcess(top_k=5)  # → ndarray[K,2] float32
# Extra methods: get_top_k() — NOT get_input_width()/get_input_height()
```

### PPU Variants (no `is_ort_configured` parameter)

```python
pp = dx_postprocess.YOLOv5PPUPostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold)  # → ndarray[N,6]
pp = dx_postprocess.YOLOv7PPUPostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold)  # → ndarray[N,6]
pp = dx_postprocess.YOLOv8PPUPostProcess(input_w, input_h, score_threshold, nms_threshold)                 # → ndarray[N,6]
pp = dx_postprocess.YOLOXPPUPostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold)   # → ndarray[N,6]
pp = dx_postprocess.YOLOv3TinyPPUPostProcess(input_w, input_h, obj_threshold, score_threshold, nms_threshold)  # → ndarray[N,6]
pp = dx_postprocess.YOLOv5PosePPUPostProcess(input_w, input_h, score_threshold, nms_threshold)             # → ndarray[N,57]
pp = dx_postprocess.SCRFDPPUPostProcess(input_w, input_h, score_threshold, nms_threshold)                  # → ndarray[N,21]
```

### Depth, Denoising, Embedding, Super Resolution, Enhancement, OBB

```python
pp = dx_postprocess.DepthPostProcess(input_w, input_h)              # → ndarray[H,W] uint8
pp = dx_postprocess.DnCNNPostProcess(input_w, input_h)              # → ndarray[H,W] float32
pp = dx_postprocess.EmbeddingPostProcess(l2_normalize=True)         # → ndarray[D] float32
# Extra methods: get_l2_normalize() — NOT get_input_width()/get_input_height()
pp = dx_postprocess.ESPCNPostProcess(input_w, input_h, scale_factor=2)  # → ndarray float32
pp = dx_postprocess.ZeroDCEPostProcess(input_w, input_h)            # → ndarray[C,H,W] float32
pp = dx_postprocess.OBBPostProcess(input_w, input_h, score_threshold=0.3)  # → ndarray[N,7]
```

## Common Methods

Most classes expose these methods (exceptions noted in Constructor Details):

| Method | Description |
|--------|-------------|
| `postprocess(ie_output: List[np.ndarray])` | Run postprocessing on inference output |
| `get_input_width() -> int` | Returns configured input width |
| `get_input_height() -> int` | Returns configured input height |

**Exceptions:**
- `ClassificationPostProcess` has `get_top_k()` instead of width/height methods
- `EmbeddingPostProcess` has `get_l2_normalize()` instead of width/height methods

## Return Type Patterns

| Pattern | Shape | Column Layout |
|---------|-------|---------------|
| Detection | `ndarray[N,6]` | `[x1, y1, x2, y2, confidence, class_id]` |
| Face (5-pt) | `ndarray[N,21]` | `[x1, y1, x2, y2, conf, class_id, 5×(lm_x, lm_y, lm_conf)]` |
| Face (RetinaFace) | `ndarray[N,16]` | `[x1, y1, x2, y2, conf, class_id, 5×(lm_x, lm_y)]` |
| Face (ULFGFD) | `ndarray[N,6]` | `[x1, y1, x2, y2, confidence, class_id]` (no landmarks) |
| Pose (17-kp) | `ndarray[N,57]` | `[x1, y1, x2, y2, conf, class_id, 17×(kp_x, kp_y, kp_conf)]` |
| Pose (CenterPose) | `ndarray[N,30]` | `[x1, y1, x2, y2, conf, class_id, 8×(kp_x, kp_y, kp_conf)]` |
| Instance Seg | `tuple(ndarray[N,6], ndarray[N,H,W])` | Detections + per-instance binary masks |
| Semantic Seg | `ndarray[H,W] int32` | Per-pixel class ID map |
| Classification | `ndarray[K,2] float32` | `[class_id, confidence]` per row |
| OBB | `ndarray[N,7]` | `[cx, cy, w, h, confidence, class_id, angle]` |
| Depth | `ndarray[H,W] uint8` | Per-pixel depth values |
| Denoising | `ndarray[H,W] float32` | Denoised image |
| Embedding | `ndarray[D] float32` | Feature vector (L2-normalized if configured) |
| Enhancement | `ndarray[C,H,W] float32` | Enhanced image |
| Super Resolution | `ndarray float32` | Upscaled image |
| Face3D | `ndarray[P]` | 1D face parameter vector |

## ⚠️ Classes That Do NOT Exist

Do not fabricate these — they are common hallucinations:

| Fabricated Name | What To Use Instead |
|-----------------|---------------------|
| `BisenetV1PostProcess` | Use `SemanticSegPostProcess` |
| `DeepLabV3PostProcess` (capital V) | Use `DeepLabv3PostProcess` (lowercase v) |
| `MobileNetV2PostProcess` | Use `ClassificationPostProcess` |
| `ResNetPostProcess` | Use `ClassificationPostProcess` |
| `EfficientNetPostProcess` | Use `ClassificationPostProcess` |
| `ArcFacePostProcess` | Use `EmbeddingPostProcess` |
| `FastDepthPostProcess` | Use `DepthPostProcess` |
| `YoloV5TinyPostProcess` | Use `YOLOv5PostProcess` (handles all v5 variants) |
| `YoloV7TinyPostProcess` | Use `YOLOv7PostProcess` (handles all v7 variants) |
| `Yolo26OBBPostProcess` | Use `OBBPostProcess` |
| `PPUPostProcess` (generic) | Use specific PPU variants: `YOLOv5PPUPostProcess`, etc. |

## Usage Example

From actual skeleton code (`*_cpp_postprocess` examples):

```python
import numpy as np
from dx_postprocess import YOLOv8PostProcess

# Create postprocessor — matches model input dimensions
pp = YOLOv8PostProcess(
    640,     # input_w
    640,     # input_h
    0.25,    # score_threshold
    0.45,    # nms_threshold
    False    # is_ort_configured
)

# ie_output is List[np.ndarray] from inference engine
detections = pp.postprocess(ie_output)  # → ndarray[N,6]

# Each row: [x1, y1, x2, y2, confidence, class_id]
for det in detections:
    x1, y1, x2, y2, conf, cls = det
    print(f"Class {int(cls)}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) conf={conf:.2f}")
```

Instance segmentation example:

```python
from dx_postprocess import YOLOv8SegPostProcess

pp = YOLOv8SegPostProcess(640, 640, 0.25, 0.45, False)
detections, masks = pp.postprocess(ie_output)  # → tuple(ndarray[N,6], ndarray[N,H,W])

for i, det in enumerate(detections):
    x1, y1, x2, y2, conf, cls = det
    instance_mask = masks[i]  # ndarray[H,W] binary mask for this detection
```

## Build & Verify

```bash
# Build from dx_app root
./build.sh

# Verify installation
python -c "import dx_postprocess; print(dir(dx_postprocess))"
```

If `import dx_postprocess` raises `ImportError`, run `./build.sh` from `dx_app/`.
