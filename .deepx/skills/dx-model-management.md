# Skill: Model Management for dx_app

> Manage .dxnn models: query the registry, download via setup scripts,
> register new models, and validate compatibility.

## model_registry.json

Location: `config/model_registry.json`

### Schema

Each model entry:

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

### Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | string | Yes | Unique lowercase identifier |
| `dxnn_file` | string | Yes | Compiled model filename |
| `original_name` | string | Yes | Display name |
| `csv_task` | string | Yes | Task code (OD, IC, SEG, POSE, etc.) |
| `add_model_task` | string | Yes | Task directory under src/python_example/ |
| `postprocessor` | string | Yes | Maps to dx_postprocess binding |
| `input_width` | int | Yes | Model input width |
| `input_height` | int | Yes | Model input height |
| `config` | object | Yes | Default thresholds (can be empty `{}`) |
| `source` | string | Yes | "csv" (auto-generated) or "manual" |
| `supported` | bool | Yes | Whether actively maintained |

## Querying the Registry

### List all models

```bash
python -c "
import json
with open('config/model_registry.json') as f:
    models = json.load(f)
print(f'Total models: {len(models)}')
for m in sorted(models, key=lambda x: x['add_model_task']):
    print(f\"  {m['add_model_task']:25s} {m['model_name']:30s} {m['input_width']}x{m['input_height']}\")
"
```

### List models by task

```bash
python -c "
import json, sys
task = sys.argv[1] if len(sys.argv) > 1 else 'object_detection'
with open('config/model_registry.json') as f:
    models = json.load(f)
matches = [m for m in models if m['add_model_task'] == task]
print(f'{task}: {len(matches)} models')
for m in matches:
    print(f\"  {m['model_name']:30s} {m['dxnn_file']:40s} {m['input_width']}x{m['input_height']}\")
" object_detection
```

### Find a specific model

```bash
python -c "
import json, sys
name = sys.argv[1]
with open('config/model_registry.json') as f:
    models = json.load(f)
match = [m for m in models if m['model_name'] == name]
if match:
    print(json.dumps(match[0], indent=2))
else:
    print(f'Model \"{name}\" not found')
" yolov8n
```

### Count by task

```bash
python -c "
import json
from collections import Counter
with open('config/model_registry.json') as f:
    models = json.load(f)
counts = Counter(m['add_model_task'] for m in models)
for task, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {task:30s} {count}')
print(f'  {\"TOTAL\":30s} {len(models)}')
"
```

## Downloading Models

### setup.sh (All models)

```bash
./setup.sh
```

Downloads all supported models and test media to their default locations.

### setup_sample_models.sh (Sample subset)

```bash
./setup_sample_models.sh
```

Downloads a curated subset of models for quick testing.

### setup_sample_videos.sh (Test videos)

```bash
./setup_sample_videos.sh
```

Downloads sample video files for testing inference.

## Validating a .dxnn Model

### Basic load test

```python
from dx_engine import InferenceEngine

ie = InferenceEngine("path/to/model.dxnn")

# Input info
input_info = ie.get_input_tensors_info()
for i, info in enumerate(input_info):
    print(f"Input {i}: shape={info['shape']}")

# Output info
output_info = ie.get_output_tensors_info()
for i, info in enumerate(output_info):
    print(f"Output {i}: shape={info['shape']}")

# Model version
try:
    ver = ie.get_model_version()
    print(f"Model format version: {ver}")
except:
    print("Model version not available")
```

### NPU availability check

```bash
dxrt-cli -s
```

If this fails, the NPU is not accessible. Check:
- Hardware present
- DX-RT installed
- Kernel driver loaded (`lsmod | grep deepx`)

### Quick inference test

```python
import numpy as np
from dx_engine import InferenceEngine

ie = InferenceEngine("model.dxnn")
info = ie.get_input_tensors_info()
shape = info[0]["shape"]

# Create dummy input matching model shape
dummy = np.zeros(shape, dtype=np.float32)
outputs = ie.run([dummy])

print(f"Inference OK: {len(outputs)} output tensors")
for i, o in enumerate(outputs):
    print(f"  Output {i}: shape={np.array(o).shape}")
```

## Registering a New Model

### Step 1: Compile to .dxnn

Use DX-Compiler to convert ONNX/TFLite to .dxnn format.

### Step 2: Determine metadata

- Task type (object_detection, classification, etc.)
- Input dimensions
- Postprocessor family (which C++ binding to use)
- Default thresholds

### Step 3: Add registry entry

```python
import json

with open("config/model_registry.json") as f:
    models = json.load(f)

new_model = {
    "model_name": "my_custom_model",
    "dxnn_file": "MyCustomModel.dxnn",
    "original_name": "My Custom Model",
    "csv_task": "OD",
    "add_model_task": "object_detection",
    "postprocessor": "yolov8",
    "input_width": 640,
    "input_height": 640,
    "config": {
        "score_threshold": 0.25,
        "nms_threshold": 0.45
    },
    "source": "manual",
    "supported": true
}

models.append(new_model)

with open("config/model_registry.json", "w") as f:
    json.dump(models, f, indent=2)
```

### Step 4: Create application

Use `dx-build-python-app` or `dx-build-cpp-app` skill to generate the
application code.

## Task Code Reference

| csv_task | add_model_task | Count |
|---|---|---|
| OD | object_detection | ~50 |
| IC | classification | ~15 |
| ISEG | instance_segmentation | ~8 |
| POSE | pose_estimation | ~6 |
| FD | face_detection | ~8 |
| SEG | semantic_segmentation | ~5 |
| DE | depth_estimation | ~2 |
| DN | image_denoising | 3 |
| IE | image_enhancement | 1 |
| SR | super_resolution | 1 |
| FREC | embedding | 1 |
| OBB | obb_detection | 1 |
| HL | hand_landmark | 1 |
| PPU | ppu | ~2 |
