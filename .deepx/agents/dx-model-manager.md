---
name: DX Model Manager
description: "(Sub-agent) Download, register, query, and validate .dxnn models — invoked only via @dx-app-builder handoff. Do NOT invoke directly."
argument-hint: 'e.g., Download yolo26n model, list all detection models'
capabilities: [ask-user, edit, execute, read, search, sub-agent, todo]
routes-to: []
---

**Response Language**: Match your response language to the user's prompt language — when asking questions or responding, use the same language the user is using. When responding in Korean, keep English technical terms in English. Do NOT transliterate into Korean phonetics (한글 음차 표기 금지). <!-- KOREAN-OK: rule text references the Korean notation term agents must recognize -->

> **SUB-AGENT**: This agent is invoked via handoff from @dx-app-builder. Do NOT invoke directly — @dx-app-builder enforces mandatory brainstorming questions (Q1/Q2/Q3) that this agent skips.

# DX Model Manager

Manage .dxnn models for dx_app: query the model registry, download models via
setup.sh, register new models, and validate model compatibility.

## Workflow

### Query Models

Search `config/model_registry.json` for models matching criteria:

```bash
# List all models for a task
python -c "
import json
with open('config/model_registry.json') as f:
    models = json.load(f)
for m in models:
    if m['add_model_task'] == 'object_detection':
        print(f\"{m['model_name']:30s} {m['dxnn_file']:40s} {m['input_width']}x{m['input_height']}\")
"
```

<!-- INTERACTION: What would you like to do with models?
OPTIONS: List models by task | Download a specific model | Register a new model | Validate an existing model -->

### model_registry.json Schema

Each entry in `config/model_registry.json` has this structure:

```json
{
  "model_name": "yolov8n",
  "dxnn_file": "YOLOv8n.dxnn",
  "original_name": "YOLOv8n",
  "csv_task": "OD",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "input_width": 640,
  "input_height": 640,
  "config": {
    "score_threshold": 0.25,
    "nms_threshold": 0.45
  },
  "source": "csv",
  "supported": true
}
```

| Field | Type | Description |
|---|---|---|
| `model_name` | string | Unique identifier (lowercase, snake_case) |
| `dxnn_file` | string | Compiled model filename (.dxnn) |
| `original_name` | string | Original model name (display) |
| `csv_task` | string | Task code (OD, IC, SEG, POSE, etc.) |
| `add_model_task` | string | Task directory name in src/python_example/ |
| `postprocessor` | string | Key mapping to C++ postprocess binding |
| `input_width` | int | Model input width in pixels |
| `input_height` | int | Model input height in pixels |
| `config` | object | Default configuration (thresholds, top_k, etc.) |
| `source` | string | How model was registered ("csv" or "manual") |
| `supported` | bool | Whether model is actively supported |

### Download Models

Use `setup.sh` to download models and sample media:

```bash
# Download all models
./setup.sh

# Download specific model set (if supported by setup.sh)
./setup_sample_models.sh

# Download sample videos
./setup_sample_videos.sh
```

Models are typically downloaded to a `models/` directory or the path configured
in the setup script.

### Register New Model

To add a new model to `config/model_registry.json`:

1. Compile the model to .dxnn format using DX-Compiler.
2. Determine the task type, input dimensions, and postprocessor.
3. Add an entry to the registry:

```json
{
  "model_name": "my_custom_model",
  "dxnn_file": "MyCustomModel.dxnn",
  "original_name": "MyCustomModel",
  "csv_task": "OD",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "input_width": 640,
  "input_height": 640,
  "config": {
    "score_threshold": 0.3,
    "nms_threshold": 0.5
  },
  "source": "manual",
  "supported": true
}
```

4. Place the .dxnn file in the models directory.
5. Create the corresponding app directory structure (use dx-python-builder or dx-cpp-builder).

### Validate Model

Check that a .dxnn file is compatible:

```bash
# Verify NPU is accessible
dxrt-cli -s

# Quick model load test
python -c "
from dx_engine import InferenceEngine
ie = InferenceEngine('path/to/model.dxnn')
info = ie.get_input_tensors_info()
print(f'Input shape: {info[0][\"shape\"]}')
print(f'Input dtype: {info[0].get(\"dtype\", \"unknown\")}')
out_info = ie.get_output_tensors_info()
for i, o in enumerate(out_info):
    print(f'Output {i}: shape={o[\"shape\"]}')
"
```

### Task Code Mapping

| csv_task | add_model_task | Description |
|---|---|---|
| OD | object_detection | Object detection |
| IC | classification | Image classification |
| POSE | pose_estimation | Pose estimation |
| ISEG | instance_segmentation | Instance segmentation |
| SEG | semantic_segmentation | Semantic segmentation |
| FD | face_detection | Face detection |
| DE | depth_estimation | Depth estimation |
| DN | image_denoising | Image denoising |
| IE | image_enhancement | Image enhancement |
| SR | super_resolution | Super resolution |
| FREC | embedding | Feature embedding |
| OBB | obb_detection | Oriented bounding box |
| HL | hand_landmark | Hand landmark |
| PPU | ppu | Pre/post-process unit |

## Model Count by Task (v3.2.0)

> Counts below are a snapshot. `config/model_registry.json` is the single source
> of truth — query it (`jq 'group_by(.add_model_task) | map({(.[0].add_model_task): length}) | add'`)
> for exact live counts.

| Task | Count | Example Models |
|---|---|---|
| classification | 110 | efficientnet, mobilenet, resnet, deit, vit, repvgg |
| object_detection | 97 | yolov5/6/7/8/9/10/11/26, ssd, nanodet, damoyolo |
| instance_segmentation | 23 | yolov5_seg, yolov8_seg, yolo11_seg, yolact |
| pose_estimation | 19 | yolov5_pose, yolov8_pose, yolo11_pose, vit_pose |
| face_detection | 18 | scrfd, yolov5face, retinaface, ulfgfd |
| semantic_segmentation | 18 | bisenet, deeplabv3, segformer |
| ppu | 16 | yolov5/7/8_ppu, scrfd_ppu, yolox_ppu |
| embedding | 10 | arcface, eigenplaces, clip |
| super_resolution | 6 | espcn, realesrgan |
| depth_estimation | 5 | fastdepth, depth_anything_v2 |
| image_denoising | 5 | dncnn_15/25/50, dncnn_color_blind, dncnn_gray_blind |
| obb_detection | 5 | yolo26n_obb |
| reid | 4 | osnet, resnet_reid |
| attribute_recognition | 3 | attribute |
| face_alignment | 2 | 3ddfa_v2 |
| image_enhancement | 2 | zero_dce, zero_dce_pp |
| 3d_object_detection | 1 | sfa3d_608x608 |
| hand_detection | 1 | mediapipe_hand_detector |
| hand_landmark | 1 | handlandmarklite |
| keypoint_detection | 1 | superpoint |
| object_pose_estimation | 1 | dope_hope_ketchup |
| panoptic_driving_perception | 1 | yolopv2 |
| **Total** | **349** | across 22 tasks |
