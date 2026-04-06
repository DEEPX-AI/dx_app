# Model Zoo — dx_app

> Compact reference for the model zoo. Do NOT enumerate models here —
> query `config/model_registry.json` as the single source of truth for current
> model counts and task types.

---

## Anti-Fabrication Warning

**NEVER guess or fabricate model names.** The registry uses lowercase `model_name`
values (e.g., `yolov8n`, `scrfd_10g`), NOT PascalCase class names. If you are unsure
whether a model exists, query the registry. Fabricating names causes silent failures
because the framework looks up `.dxnn` files by exact `model_name`.

**NEVER guess postprocessor values.** The `postprocessor` field in the registry is a
lowercase string key (e.g., `yolov8`, `scrfd`), NOT a Python or C++ class name.
The framework maps these keys to implementation classes internally.

---

## Source of Truth

```
config/model_registry.json
```

This is a **JSON array** of model entries (not an object). Each entry has:

| Field | Required | Description |
|---|---|---|
| `model_name` | yes | Unique lowercase identifier (e.g., `yolov8n`) |
| `dxnn_file` | yes | Compiled model filename (e.g., `yolov8n.dxnn`) |
| `add_model_task` | yes | Task type (see Task Overview below) |
| `postprocessor` | yes | Registry key for postprocessor (see mapping below) |
| `supported` | yes | `true` if model is available for use |
| `original_name` | no | Upstream model name |
| `csv_task` | no | CSV-compatible task label |
| `input_width` | no | Model input width in pixels |
| `input_height` | no | Model input height in pixels |
| `config` | no | Path to model-specific config JSON |
| `source` | no | Model provenance |

---

## Quick Queries

List all supported models:
```bash
jq '[.[] | select(.supported == true)] | length' config/model_registry.json
```

List models for a specific task:
```bash
jq '[.[] | select(.add_model_task == "object_detection" and .supported == true)] | .[].model_name' config/model_registry.json
```

Get a model's postprocessor and input size:
```bash
jq '.[] | select(.model_name == "yolov8n") | {postprocessor, input_width, input_height}' config/model_registry.json
```

List all distinct postprocessor keys:
```bash
jq '[.[].postprocessor] | unique' config/model_registry.json
```

List all distinct task types:
```bash
jq '[.[].add_model_task] | unique' config/model_registry.json
```

---

## Postprocessor Mapping

The `postprocessor` field in the registry is a **lowercase string key**. The framework
maps it to a Python class (for Python runner) and a C++ binding (for C++ runner).

| Registry Key | Python Class | C++ `dx_postprocess` | Notes |
|---|---|---|---|
| `yolov5` | `YOLOv5Postprocessor` | `YOLOv5PostProcess` | |
| `yolov8` | `YOLOv8Postprocessor` | `YOLOv8PostProcess` | |
| `yolov26` | `YOLOv8Postprocessor` | `YOLOv26PostProcess` | Python reuses YOLOv8 |
| `yolov10` | `YOLOv8Postprocessor` | `YOLOv10PostProcess` | Python reuses YOLOv8 |
| `yolox` | `YOLOXPostprocessor` | `YOLOXPostProcess` | |
| `damoyolo` | `DamoYoloPostprocessor` | `DamoYOLOPostProcess` | |
| `nanodet` | `NanoDetPostprocessor` | `NanoDetPostProcess` | |
| `ssd` | `SSDPostprocessor` | `SSDPostProcess` | |
| `efficientnet` | `ClassificationPostprocessor` | `ClassificationPostProcess` | All classifiers |
| `yolov8pose` | `YOLOv8PosePostprocessor` | `YOLOv8PosePostProcess` | |
| `yolov5seg` | `YOLOv5InstanceSegPostprocessor` | `YOLOv5SegPostProcess` | |
| `yolov8seg` | `YOLOv8InstanceSegPostprocessor` | `YOLOv8SegPostProcess` | |
| `yolact` | `YOLACTPostprocessor` | — | No C++ binding |
| `bisenetv1` | `SemanticSegmentationPostprocessor` | `SemanticSegPostProcess` | |
| `bisenetv2` | `SemanticSegmentationPostprocessor` | `SemanticSegPostProcess` | Shares with bisenetv1 |
| `deeplabv3` | `SemanticSegmentationPostprocessor` | `DeepLabv3PostProcess` | |
| `segformer` | `SegFormerPostprocessor` | `SemanticSegPostProcess` | |
| `scrfd` | `SCRFDPostprocessor` | `SCRFDPostProcess` | |
| `yolov5face` | `YOLOv5FacePostprocessor` | `YOLOv5FacePostProcess` | |
| `yolov7face` | `YOLOv7FacePostprocessor` | — | No C++ binding |
| `retinaface` | `RetinaFacePostprocessor` | `RetinaFacePostProcess` | |
| `fastdepth` | `DepthEstimationPostprocessor` | `DepthPostProcess` | |
| `dncnn` | `DnCNNPostprocessor` | `DnCNNPostProcess` | |
| `espcn` | `ESPCNPostprocessor` | `ESPCNPostProcess` | |
| `zero_dce` | `ZeroDCEPostprocessor` | `ZeroDCEPostProcess` | |
| `arcface` | `ArcFacePostprocessor` | `EmbeddingPostProcess` | |
| `obb` | `OBBPostprocessor` | `OBBPostProcess` | Score threshold only, no NMS |
| `yolov5_ppu` | `YOLOv5PPUPostprocessor` | `YOLOv5PPUPostProcess` | |
| `yolov7_ppu` | `YOLOv7PPUPostprocessor` | `YOLOv7PPUPostProcess` | |
| `hand_landmark` | `HandLandmarkPostprocessor` | — | No C++ binding |

**Key insight:** `yolov10` and `yolov26` share `YOLOv8Postprocessor` in Python but
have distinct C++ postprocessors. PPU models have dedicated postprocessors — never
use standard YOLO postprocessors with PPU models.

---

## Task Overview

The following task types are known (query `config/model_registry.json` for the current full list):

| Task | Postprocessor Keys Used | Example Models (query registry for full list) |
|---|---|---|
| `object_detection` | `yolov5`, `yolov8`, `yolov10`, `yolov26`, `yolox`, `damoyolo`, `nanodet`, `ssd` | yolov8n, yolov5s, ssd_mobilenetv2 |
| `classification` | `efficientnet` | efficientnet_b0, resnet50 |
| `pose_estimation` | `yolov8pose` | yolov8n_pose, yolov8s_pose |
| `instance_segmentation` | `yolov5seg`, `yolov8seg`, `yolact` | yolov8n_seg, yolov5s_seg |
| `semantic_segmentation` | `bisenetv1`, `bisenetv2`, `deeplabv3`, `segformer` | bisenetv1, deeplabv3plus |
| `face_detection` | `scrfd`, `yolov5face`, `yolov7face`, `retinaface` | scrfd_10g, retinaface |
| `depth_estimation` | `fastdepth` | fastdepth |
| `image_denoising` | `dncnn` | dncnn_15, dncnn_25, dncnn_50 |
| `super_resolution` | `espcn` | espcn_x2, espcn_x3, espcn_x4 |
| `image_enhancement` | `zero_dce` | zero_dce |
| `embedding` | `arcface` | arcface_mobilefacenet |
| `obb_detection` | `obb` | yolo26n_obb |
| `ppu` | `yolov5_ppu`, `yolov7_ppu` | yolov5s_ppu, yolov7_ppu |
| `hand_landmark` | `hand_landmark` | hand_landmark_lite |
| `attribute_recognition` | — | Query registry for models |
| `reid` | — | Query registry for models |

**Do NOT hardcode model names from this table.** These are examples only. Always
query `config/model_registry.json` for the current list.

---

## Common Mistakes

1. **Fabricating model names** — e.g., `yolov11n` or `efficientdet_d0` may not exist
   in the registry. Always verify with `jq`.

2. **Using PascalCase for postprocessor** — the registry field is a lowercase string
   like `yolov8`, not `YoloV8PostProcess`. The mapping to classes is internal.

3. **Wrong postprocessor family** — `segformer` models use `segformer` postprocessor,
   not `efficientnet`. Classification postprocessor is only for `classification` task.

4. **Ignoring `supported` field** — some models have `"supported": false`. Always
   filter by `supported == true` when listing available models.

5. **Mixing PPU and standard postprocessors** — PPU models (`yolov5_ppu`, `yolov7_ppu`)
   require their dedicated postprocessors. Standard YOLO postprocessors will fail.

6. **Assuming input sizes** — not all YOLO models use 640x640. Check `input_width` and
   `input_height` in the registry. Some models (e.g., nanodet, ssd) use smaller inputs.

7. **Hardcoding model lists** — the registry is the live source. This document
   intentionally does not list every model to avoid going stale.
