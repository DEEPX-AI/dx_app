# Model Registry Reference

> `config/model_registry.json` — the single source of truth for all 133 models
> across 15 AI tasks in dx_app.

## Overview

The model registry is a JSON file at `config/model_registry.json` that describes every
supported model. All dx_app tools and agents query this file to:
- Verify a model exists before creating an app
- Get model metadata (input size, thresholds, output format)
- Determine which postprocessor to use
- Check if a model is supported on the current platform

## Schema

Each entry in `model_registry.json` follows this schema:

```json
{
  "<model_name>": {
    "task": "<task_type>",
    "dxnn_file": "<model_name>.dxnn",
    "input_size": [<width>, <height>],
    "supported": true,
    "default_threshold": <float>,
    "nms_threshold": <float>,
    "labels_file": "<labels_filename>.txt",
    "output_format": "<format_description>",
    "family": "<model_family>",
    "variant": "<size_variant>",
    "num_classes": <int>,
    "source": "<origin_framework>"
  }
}
```

### Required Fields

| Field | Type | Description | Example |
|---|---|---|---|
| `task` | `string` | AI task category | `"object_detection"` |
| `dxnn_file` | `string` | Model filename (no path) | `"yolov8n.dxnn"` |
| `input_size` | `[int, int]` | `[width, height]` | `[640, 640]` |
| `supported` | `bool` | Whether model is available for download | `true` |
| `default_threshold` | `float` | Recommended score threshold | `0.25` |

### Optional Fields

| Field | Type | Description | Example |
|---|---|---|---|
| `nms_threshold` | `float` | NMS IoU threshold (detection only) | `0.45` |
| `labels_file` | `string` | Class labels filename | `"coco_labels.txt"` |
| `output_format` | `string` | Output tensor description | `"8400x84"` |
| `family` | `string` | Model architecture family | `"yolov8"` |
| `variant` | `string` | Size variant | `"nano"` |
| `num_classes` | `int` | Number of output classes | `80` |
| `source` | `string` | Original framework | `"ultralytics"` |

## Query Patterns

### Python: Load and Query

```python
import json
from pathlib import Path

# Load registry
registry_path = Path(__file__).parent.parent.parent.parent / "config" / "model_registry.json"
with open(registry_path) as f:
    registry = json.load(f)

# Check if model exists
def model_exists(name: str) -> bool:
    return name in registry

# Get model info
def get_model_info(name: str) -> dict:
    if name not in registry:
        raise KeyError(f"Model '{name}' not found in registry")
    return registry[name]

# List models by task
def list_models_by_task(task: str) -> list:
    return [
        name for name, info in registry.items()
        if info["task"] == task
    ]

# List supported models only
def list_supported_models() -> list:
    return [
        name for name, info in registry.items()
        if info.get("supported", False)
    ]
```

### Common Queries

```python
# Get input dimensions for a model
info = registry["yolov8n"]
width, height = info["input_size"]
# width=640, height=640

# Get recommended thresholds
score_thresh = info["default_threshold"]   # 0.25
nms_thresh = info.get("nms_threshold", 0.45)

# Get labels file path
labels = info.get("labels_file", "coco_labels.txt")
labels_path = Path("config/labels") / labels

# Filter by task
det_models = [k for k, v in registry.items() if v["task"] == "object_detection"]
# ['yolov5n', 'yolov5s', 'yolov5m', 'yolov7', 'yolov7_tiny', 'yolov8n', ...]
```

## 133 Models Across 15 Tasks

### Task Distribution

| Task | Count | Example Models |
|---|---|---|
| object_detection | ~45 | yolov5n/s/m, yolov7/tiny, yolov8n/s/m, yolov10n/s, yolov11n/s, yolo26n/s, ssd, efficientdet, nanodet |
| classification | ~20 | efficientnet_b0-b4, mobilenetv2, resnet18/34/50/101, inception_v3, squeezenet |
| pose_estimation | ~8 | yolov5s_pose, yolov5m_pose, yolov8n_pose, yolov8s_pose |
| instance_segmentation | ~10 | yolov5n_seg, yolov5s_seg, yolov8n_seg, yolov8s_seg |
| semantic_segmentation | ~10 | bisenetv1, deeplabv3plus_mobilenet, deeplabv3plus_resnet, segformer_b0/b1 |
| face_detection | ~8 | scrfd_10g, scrfd_2.5g, yolov5s_face, yolov5n_face, retinaface |
| depth_estimation | ~4 | fastdepth_1, fastdepth_2 |
| image_denoising | ~6 | dncnn_15, dncnn_25, dncnn_50 |
| image_enhancement | ~3 | zero_dce, zero_dce_pp |
| super_resolution | ~4 | espcn_x2, espcn_x3, espcn_x4 |
| embedding | ~3 | arcface_mobilefacenet, arcface_resnet50 |
| obb_detection | ~3 | yolo26n_obb, yolo26s_obb |
| hand_landmark | ~3 | handlandmarklite_1, handlandmarklite_2 |
| ppu | ~6 | yolov5s_ppu, yolov7_ppu, yolov8n_ppu |

**Note:** Exact counts vary as models are added or deprecated between releases.
Use `list_supported_models()` to get the current supported set.

## Model Naming Conventions

### Pattern

```
<family><variant>[_<suffix>]
```

| Component | Description | Examples |
|---|---|---|
| `family` | Architecture family | `yolov5`, `yolov8`, `efficientnet`, `resnet` |
| `variant` | Size variant (n/s/m/l/x) or parameter | `n`, `s`, `b0`, `50` |
| `suffix` | Task-specific suffix | `_pose`, `_seg`, `_face`, `_ppu`, `_obb` |

### Examples

| Name | Family | Variant | Suffix | Task |
|---|---|---|---|---|
| `yolov8n` | yolov8 | nano | — | object_detection |
| `yolov8n_seg` | yolov8 | nano | `_seg` | instance_segmentation |
| `yolov5s_pose` | yolov5 | small | `_pose` | pose_estimation |
| `yolov5s_face` | yolov5 | small | `_face` | face_detection |
| `efficientnet_b0` | efficientnet | b0 | — | classification |
| `resnet50` | resnet | 50 | — | classification |
| `yolo26n_obb` | yolo26 | nano | `_obb` | obb_detection |
| `yolov5s_ppu` | yolov5 | small | `_ppu` | ppu |

### Case Sensitivity

Model names in `model_registry.json` are **case-sensitive**. The convention is all
lowercase with underscores:
- Correct: `yolov8n`, `efficientnet_b0`
- Wrong: `YoloV8n`, `EfficientNet_B0`, `YOLOV8N`

See `memory/common_pitfalls.md` [UNIVERSAL] entry for the case mismatch pitfall.

## Querying from CLI

```bash
# Check if a model exists (using jq)
jq '.yolov8n' config/model_registry.json

# List all object_detection models
jq '[to_entries[] | select(.value.task == "object_detection") | .key]' config/model_registry.json

# Get input size for a model
jq '.yolov8n.input_size' config/model_registry.json

# Count models per task
jq '[to_entries[] | .value.task] | group_by(.) | map({task: .[0], count: length})' config/model_registry.json
```

## Registry Integrity

The registry must satisfy these constraints:
1. Every model's `dxnn_file` must be a valid filename (no path separators)
2. Every model's `task` must be one of the 15 supported tasks
3. Every model's `input_size` must be `[width, height]` with positive integers
4. If `supported: true`, the model must be downloadable via `setup.sh`
5. If `labels_file` is specified, the file must exist in `config/labels/`
6. Model names must be unique (enforced by JSON key uniqueness)

## Adding a New Model

```json
{
  "my_custom_model": {
    "task": "object_detection",
    "dxnn_file": "my_custom_model.dxnn",
    "input_size": [640, 640],
    "supported": true,
    "default_threshold": 0.25,
    "nms_threshold": 0.45,
    "labels_file": "coco_labels.txt",
    "family": "yolov8",
    "variant": "custom",
    "num_classes": 80,
    "source": "custom"
  }
}
```

After adding:
1. Place the `.dxnn` file in the models directory
2. Create the factory and app files under `src/python_example/<task>/<model>/`
3. Run `scripts/validate_app.py` to verify consistency
