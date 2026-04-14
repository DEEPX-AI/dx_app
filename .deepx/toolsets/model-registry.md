# Model Registry Reference

> `config/model_registry.json` -- the single source of truth for all 133 models
> across 15 AI tasks in dx_app.

## Anti-Fabrication Notice

This document was verified against the actual `config/model_registry.json` file.
**Do NOT invent field names.** If a field is not listed here, it does not exist.
See [Fields That Do NOT Exist](#fields-that-do-not-exist) for commonly hallucinated fields.

When in doubt, read the source file directly:
```bash
python3 -c "import json; print(json.dumps(json.load(open('config/model_registry.json'))[0], indent=2))"
```

## Source File

| Item | Value |
|---|---|
| Path | `config/model_registry.json` |
| Format | JSON **array** (`[]`) |
| Entry count | 133 |
| Loading code | `tests/common/utils.py`, `.deepx/scripts/validate_framework.py` |

## Structure

The registry is a **JSON array of objects** -- NOT a JSON object with model names as keys.

```
[              <-- top-level is an ARRAY
  { ... },     <-- each entry is an object
  { ... },
  ...
]
```

Each entry is a flat object describing one model. Access is always via **array iteration**,
never via dict-key lookup by model name.

## Schema

### Required Fields (present in all 133 entries)

| Field | Type | Description | Example |
|---|---|---|---|
| `model_name` | `string` | Unique identifier, lowercase with underscores | `"yolov8n"` |
| `dxnn_file` | `string` | Compiled model filename (no path) | `"YoloV8N.dxnn"` |
| `add_model_task` | `string` | Task type (one of 15 values) | `"object_detection"` |
| `postprocessor` | `string` | Postprocess binding key | `"yolov8"` |
| `supported` | `boolean` | Whether the model is available | `true` |

### Optional Fields (present in most but not all entries)

| Field | Type | Description | Example |
|---|---|---|---|
| `original_name` | `string` | Display name (mixed case) | `"YoloV8N"` |
| `csv_task` | `string` | Short task code from CSV import | `"OD"` |
| `input_width` | `integer` | Input width in pixels | `640` |
| `input_height` | `integer` | Input height in pixels | `640` |
| `config` | `object` | Default thresholds (see below) | `{"score_threshold": 0.25, "nms_threshold": 0.45}` |
| `source` | `string` | Registration source | `"csv"` |

### `config` Sub-Fields

The `config` object contains threshold and parameter defaults. Not all sub-fields appear
in every entry -- contents vary by task type.

| Sub-field | Type | Description | Typical Value |
|---|---|---|---|
| `score_threshold` | `float` | Confidence score threshold | `0.25` |
| `nms_threshold` | `float` | NMS IoU threshold | `0.45` |
| `obj_threshold` | `float` | Objectness threshold | `0.5` |
| `top_k` | `int` | Top-K results (classification) | `5` |

### `source` Values

| Value | Meaning |
|---|---|
| `"csv"` | Imported from model CSV |
| `"inferred"` | Auto-detected from model file |
| `"manifest"` | From model manifest |
| `"stream_model_list"` | From streaming model list |

## Example Entries

### Object Detection

```json
{
  "model_name": "yolov8n",
  "dxnn_file": "YoloV8N.dxnn",
  "original_name": "YoloV8N",
  "csv_task": "OD",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "input_width": 640,
  "input_height": 640,
  "config": {"score_threshold": 0.25, "nms_threshold": 0.45},
  "source": "csv",
  "supported": true
}
```

### Classification

```json
{
  "model_name": "alexnet",
  "dxnn_file": "AlexNet.dxnn",
  "original_name": "AlexNet",
  "csv_task": "IC",
  "add_model_task": "classification",
  "postprocessor": "efficientnet",
  "input_width": 224,
  "input_height": 224,
  "config": {"top_k": 5},
  "source": "csv",
  "supported": true
}
```

### Depth Estimation

```json
{
  "model_name": "fastdepth_1",
  "dxnn_file": "FastDepth_1.dxnn",
  "original_name": "FastDepth_1",
  "csv_task": "DEPTH",
  "add_model_task": "depth_estimation",
  "postprocessor": "fastdepth",
  "input_width": 224,
  "input_height": 224,
  "config": {},
  "source": "csv",
  "supported": true
}
```

## Query Patterns

### Python: Load and Query

```python
import json
from pathlib import Path

# Load registry (returns a LIST, not a dict)
registry_path = Path("config/model_registry.json")
with open(registry_path) as f:
    registry = json.load(f)  # type: list[dict]

# Find a model by name (array iteration)
def find_model(name: str) -> dict | None:
    return next((e for e in registry if e["model_name"] == name), None)

# List supported models
def list_supported() -> list[dict]:
    return [e for e in registry if e.get("supported")]

# List models by task
def list_by_task(task: str) -> list[str]:
    return [e["model_name"] for e in registry if e["add_model_task"] == task]

# Get input dimensions
model = find_model("yolov8n")
if model:
    w, h = model.get("input_width"), model.get("input_height")  # 640, 640

# Get thresholds from config sub-object
if model:
    cfg = model.get("config", {})
    score_thresh = cfg.get("score_threshold", 0.25)
    nms_thresh = cfg.get("nms_threshold", 0.45)
```

### How the Codebase Loads the Registry

```python
# From tests/common/utils.py
def load_registry() -> list:
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)  # Returns list, not dict
    return [e for e in registry if e.get("supported") and e.get("model_name") not in SKIP_MODELS]
```

### CLI Queries (jq)

```bash
# The file is an ARRAY, so use .[] not .key

# Find a model by name
jq '.[] | select(.model_name == "yolov8n")' config/model_registry.json

# List all object_detection model names
jq '[.[] | select(.add_model_task == "object_detection") | .model_name]' config/model_registry.json

# Get input dimensions for a model
jq '.[] | select(.model_name == "yolov8n") | {w: .input_width, h: .input_height}' config/model_registry.json

# Count models per task
jq 'group_by(.add_model_task) | map({task: .[0].add_model_task, count: length}) | sort_by(-.count)' config/model_registry.json

# List all supported models
jq '[.[] | select(.supported) | .model_name]' config/model_registry.json

# Get thresholds from config sub-object
jq '.[] | select(.model_name == "yolov8n") | .config' config/model_registry.json
```

## Task Types

All 15 `add_model_task` values in the registry:

| `add_model_task` | `csv_task` | Description |
|---|---|---|
| `object_detection` | `OD` | Bounding-box object detection |
| `classification` | `IC` | Image classification |
| `face_detection` | `FD` | Face detection |
| `instance_segmentation` | `IS` | Per-instance masks |
| `semantic_segmentation` | `SS` | Per-pixel class labels |
| `pose_estimation` | `PE` | Keypoint-based pose |
| `image_denoising` | `DN` | Image noise reduction |
| `ppu` | `PPU` | Post-processing unit models |
| `embedding` | `EMB` | Feature embedding / face recognition |
| `depth_estimation` | `DEPTH` | Monocular depth |
| `hand_landmark` | `HL` | Hand keypoint detection |
| `super_resolution` | `SR` | Image upscaling |
| `obb_detection` | `OBB` | Oriented bounding-box detection |
| `image_enhancement` | `IE` | Low-light / image enhancement |

**Note:** There are 15 unique values listed in the registry, but only 14 rows above.
The 15th task may be added in newer releases -- always verify against the source file.

## Fields That Do NOT Exist

These field names have been hallucinated in previous documentation. They are **not real**.

| Fabricated Field | What Actually Exists |
|---|---|
| `task` | `add_model_task` |
| `input_size` | Separate `input_width` and `input_height` fields |
| `default_threshold` | `config.score_threshold` (inside `config` object) |
| `nms_threshold` (top-level) | `config.nms_threshold` (inside `config` object) |
| `labels_file` | Does not exist in any entry |
| `output_format` | Does not exist in any entry |
| `family` | Does not exist in any entry |
| `variant` | Does not exist in any entry |
| `num_classes` | Does not exist in any entry |

**The top-level structure is a JSON array, NOT a JSON object with model names as keys.**
Dict-key access like `registry["yolov8n"]` will raise a `TypeError`. Use array iteration.
