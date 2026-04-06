# Prompt: New Python Segmentation App

> Template for creating a Python segmentation application in dx_app.

## Variables

| Variable | Description | Example |
|---|---|---|
| `{model_name}` | Model name from model_registry.json | `yolo26n-seg`, `bisenetv1` |
| `{seg_type}` | Segmentation type | `instance` or `semantic` |
| `{variant}` | Python variant | `sync`, `async`, `sync_cpp_postprocess`, `async_cpp_postprocess` |
| `{input_source}` | Primary input for testing | `image`, `video`, `usb`, `rtsp` |

## Prompt

Build a Python segmentation application for `{model_name}` ({seg_type} segmentation)
using the `{variant}` runner with `{input_source}` as the primary input source.

### Step 1: Query Model Registry

Read `config/model_registry.json` and verify:
- `{model_name}` exists in the registry
- `task` is `"instance_segmentation"` or `"semantic_segmentation"`
- `supported` is `true`
- Extract `input_size`, `default_threshold`, `labels_file`

Determine segmentation type from the task field:
- `instance_segmentation` → use `IInstanceSegFactory`, outputs per-instance masks
- `semantic_segmentation` → use `ISegmentationFactory`, outputs per-pixel class map

### Step 2: Create Factory

**Instance Segmentation:**
```python
from common.base import IInstanceSegFactory
from common.processors import LetterboxPreprocessor, {ModelName}SegPostprocessor
from common.visualizers import InstanceSegVisualizer


class {ModelName}Factory(IInstanceSegFactory):
    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width, input_height):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width, input_height):
        return {ModelName}SegPostprocessor(
            input_width, input_height, self.config)

    def create_visualizer(self):
        return InstanceSegVisualizer()

    def get_model_name(self) -> str:
        return "{model_name}"

    def get_task_type(self) -> str:
        return "instance_segmentation"
```

**Semantic Segmentation:**
```python
from common.base import ISegmentationFactory
from common.processors import SegmentationPreprocessor, SemanticSegPostprocessor
from common.visualizers import SemanticSegVisualizer


class {ModelName}Factory(ISegmentationFactory):
    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width, input_height):
        return SegmentationPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width, input_height):
        return SemanticSegPostprocessor(
            input_width, input_height, self.config)

    def create_visualizer(self):
        return SemanticSegVisualizer()

    def get_model_name(self) -> str:
        return "{model_name}"

    def get_task_type(self) -> str:
        return "semantic_segmentation"
```

Create `factory/__init__.py`:
```python
from .{model_name}_factory import {ModelName}Factory
```

### Step 3: Create Application Script

Same pattern as detection (see `prompts/new-python-detection.md` Step 3).

### Step 4: Create config.json

**Instance Segmentation:**
```json
{
  "score_threshold": 0.25,
  "nms_threshold": 0.45,
  "mask_threshold": 0.5
}
```

**Semantic Segmentation:**
```json
{
  "num_classes": 19,
  "ignore_index": 255
}
```

### Step 5: Validate

1. Syntax check all .py files
2. Verify factory uses correct interface (IInstanceSegFactory vs ISegmentationFactory)
3. Verify config.json matches the segmentation type
4. Verify visualizer matches (InstanceSegVisualizer vs SemanticSegVisualizer)
5. Run `--help` test

### Key Differences from Detection

| Aspect | Detection | Instance Seg | Semantic Seg |
|---|---|---|---|
| Factory Interface | IDetectionFactory | IInstanceSegFactory | ISegmentationFactory |
| Preprocessor | LetterboxPreprocessor | LetterboxPreprocessor | SegmentationPreprocessor |
| Visualizer | DetectionVisualizer | InstanceSegVisualizer | SemanticSegVisualizer |
| Config extras | — | mask_threshold | num_classes |
| Output | Bounding boxes | Boxes + masks | Per-pixel labels |

### Checklist

- [ ] Model exists in model_registry.json with correct segmentation task
- [ ] Correct factory interface selected (IInstanceSegFactory vs ISegmentationFactory)
- [ ] Factory implements all 5 methods
- [ ] Correct preprocessor for segmentation type
- [ ] Correct visualizer for segmentation type
- [ ] config.json includes segmentation-specific fields
- [ ] factory/__init__.py exports the factory class
- [ ] Standard sys.path pattern in app scripts
- [ ] No hardcoded model paths
