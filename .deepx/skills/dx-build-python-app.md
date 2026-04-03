# Skill: Build Python Inference App for dx_app

> **This skill doc is sufficient.** Do NOT read source code files in `common/`
> unless this document explicitly tells you to. The templates and patterns here
> are the authoritative reference for building dx_app Python applications.

## Overview

Step-by-step guide to build a complete Python inference application for dx_app
v3.0.0, including factory, all 4 variants, config, and packaging.

## Output Isolation (MUST FOLLOW)

All AI-generated applications MUST be created under `dx-agentic-dev/`, NOT in the
production `src/` directory. This prevents accidental modification of existing code.

### Session Directory

```
dx-agentic-dev/<YYYYMMDD-HHMMSS>_<model>_<task>/
├── session.json          # Build metadata
├── README.md             # How to run this app
├── factory/
│   ├── __init__.py
│   └── <model>_factory.py
├── <model>_sync.py
├── <model>_async.py
├── <model>_sync_cpp_postprocess.py
├── <model>_async_cpp_postprocess.py
└── config.json
```

### session.json Template

```json
{
  "session_id": "<YYYYMMDD-HHMMSS>_<model>_<task>",
  "created_at": "<ISO 8601 timestamp>",
  "model": "<model_name>",
  "task": "<task_type>",
  "variants": ["sync", "async", "sync_cpp_postprocess", "async_cpp_postprocess"],
  "status": "complete",
  "notes": "<any relevant notes>"
}
```

### Import Boilerplate for dx-agentic-dev/

Since apps in `dx-agentic-dev/` are at a different directory depth than production apps,
use this dynamic root-finding pattern instead of the standard `_v3_dir` pattern:

```python
import sys
from pathlib import Path

# Find dx_app root dynamically
_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / 'src' / 'python_example' / 'common').exists():
        break
    _current = _current.parent
_v3_dir = _current / 'src' / 'python_example'
_module_dir = Path(__file__).parent

for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
```

### When to Use Production Path

Only create files in `src/python_example/<task>/<model>/` when the user EXPLICITLY says:
- "Add this to the production codebase"
- "Create this in src/"
- "Make this a permanent addition"

Default behavior: ALWAYS use `dx-agentic-dev/`.

## Prerequisites

- dx_app repository cloned
- Model exists in `config/model_registry.json`
- Know the task type and model name

## Step 1: Query Model Registry

```bash
python -c "
import json
with open('config/model_registry.json') as f:
    models = json.load(f)
match = [m for m in models if m['model_name'] == '<MODEL_NAME>']
if match:
    m = match[0]
    print(f'Model:  {m[\"model_name\"]}')
    print(f'Task:   {m[\"add_model_task\"]}')
    print(f'File:   {m[\"dxnn_file\"]}')
    print(f'Input:  {m[\"input_width\"]}x{m[\"input_height\"]}')
    print(f'Config: {json.dumps(m.get(\"config\", {}))}')
else:
    print('Model not found in registry')
"
```

## Step 2: Create Directory Structure

```bash
mkdir -p src/python_example/<TASK>/<MODEL>/factory
touch src/python_example/<TASK>/<MODEL>/__init__.py
touch src/python_example/<TASK>/<MODEL>/factory/__init__.py
```

## Step 3: Select IFactory Interface

| Task | Interface | Import |
|---|---|---|
| object_detection | `IDetectionFactory` | `from common.base import IDetectionFactory` |
| classification | `IClassificationFactory` | `from common.base import IClassificationFactory` |
| pose_estimation | `IPoseFactory` | `from common.base import IPoseFactory` |
| instance_segmentation | `IInstanceSegFactory` | `from common.base import IInstanceSegFactory` |
| semantic_segmentation | `ISegmentationFactory` | `from common.base import ISegmentationFactory` |
| face_detection | `IFaceFactory` | `from common.base import IFaceFactory` |
| depth_estimation | `IDepthEstimationFactory` | `from common.base import IDepthEstimationFactory` |
| image_denoising | `IRestorationFactory` | `from common.base import IRestorationFactory` |
| image_enhancement | `IRestorationFactory` | `from common.base import IRestorationFactory` |
| super_resolution | `IRestorationFactory` | `from common.base import IRestorationFactory` |
| embedding | `IEmbeddingFactory` | `from common.base import IEmbeddingFactory` |
| obb_detection | `IOBBFactory` | `from common.base import IOBBFactory` |
| hand_landmark | `IHandLandmarkFactory` | `from common.base import IHandLandmarkFactory` |

## Step 4: Select Components

### Preprocessors (from `common.processors`)

| Preprocessor | Used By |
|---|---|
| `LetterboxPreprocessor` | All YOLO variants, SSD, NanoDet, CenterPose |
| `ClassificationPreprocessor` | EfficientNet, MobileNet, ResNet |
| `SegmentationPreprocessor` | BiSeNet, DeepLabV3, SegFormer |
| `DepthPreprocessor` | FastDepth |
| `RestorationPreprocessor` | DnCNN, Zero-DCE |
| `SRPreprocessor` | ESPCN |

### Postprocessors (from `common.processors`)

| Family | Postprocessor Class | Models |
|---|---|---|
| YOLOv5 | `YOLOv5Postprocessor` | yolov5n/s/m/l, yolov3, yolox |
| YOLOv8 | `YOLOv8Postprocessor` | yolov8n/s/m/l/x, yolo26n/s/m/l/x |
| YOLOv10 | `YOLOv10Postprocessor` | yolov10n/s/m/b/l/x |
| YOLOv11 | `YOLOv11Postprocessor` | yolov11n/s/m/l/x |
| DamoYOLO | `DamoYoloPostprocessor` | damoyolo variants |
| NanoDet | `NanoDetPostprocessor` | nanodet_repvgg |
| SSD | `SSDPostprocessor` | ssdmv1, ssdmv2lite |
| Classification | `ClassificationPostprocessor` | all classification |
| Pose | `PosePostprocessor` | yolov5_pose, yolov8_pose |
| SegInstance | `InstanceSegPostprocessor` | yolov5_seg, yolov8_seg, yolact |
| SegSemantic | `SemanticSegPostprocessor` | bisenet, deeplabv3, segformer |
| Face | `FacePostprocessor` | scrfd, retinaface, yolov5face |
| Depth | `DepthPostprocessor` | fastdepth |
| Restoration | `RestorationPostprocessor` | dncnn |
| SR | `SRPostprocessor` | espcn |
| Embedding | `EmbeddingPostprocessor` | arcface |
| OBB | `OBBPostprocessor` | yolo26n_obb |

### Visualizers (from `common.visualizers`)

| Visualizer | Used By |
|---|---|
| `DetectionVisualizer` | All detection tasks |
| `ClassificationVisualizer` | Classification |
| `PoseVisualizer` | Pose estimation |
| `InstanceSegVisualizer` | Instance segmentation |
| `SemanticSegVisualizer` | Semantic segmentation |
| `FaceVisualizer` | Face detection |
| `DepthVisualizer` | Depth estimation |
| `RestorationVisualizer` | Denoising, enhancement |
| `SRVisualizer` | Super resolution |

## Step 5: Create Factory

### Template: `factory/<model>_factory.py`

```python
"""
<ModelDisplay> Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import <IFactoryInterface>
from common.processors import <Preprocessor>, <Postprocessor>
from common.visualizers import <Visualizer>


class <ModelClass>Factory(<IFactoryInterface>):
    """Factory for creating <ModelDisplay> components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return <Preprocessor>(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return <Postprocessor>(input_width, input_height, self.config)

    def create_visualizer(self):
        return <Visualizer>()

    def get_model_name(self) -> str:
        return "<model_name>"

    def get_task_type(self) -> str:
        return "<task_type>"
```

### Template: `factory/__init__.py`

```python
from .<model>_factory import <ModelClass>Factory
```

## Step 6: Create Sync Variant

### Template: `<model>_sync.py`

```python
#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
<ModelDisplay> Synchronous Inference Example - DX-APP v3.0.0

Usage:
    python <model>_sync.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import <ModelClass>Factory
from common.runner import SyncRunner, parse_common_args


def parse_args():
    return parse_common_args("<ModelDisplay> Sync Inference")


def main():
    args = parse_args()
    factory = <ModelClass>Factory()
    runner = SyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
```

## Step 7: Create Async Variant

### Template: `<model>_async.py`

```python
#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
<ModelDisplay> Asynchronous Inference Example - DX-APP v3.0.0

Usage:
    python <model>_async.py --model model.dxnn --video input.mp4
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import <ModelClass>Factory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("<ModelDisplay> Async Inference")


def main():
    args = parse_args()
    factory = <ModelClass>Factory()
    runner = AsyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
```

## Step 8: Create Sync C++ Postprocess Variant

### Template: `<model>_sync_cpp_postprocess.py`

```python
#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
<ModelDisplay> Synchronous Inference (C++ Postprocess) - DX-APP v3.0.0

Usage:
    python <model>_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import <CppPostProcess>
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import <ModelClass>Factory
from common.runner import SyncRunner, parse_common_args


def parse_args():
    return parse_common_args("<ModelDisplay> Sync Inference")


def main():
    args = parse_args()
    factory = <ModelClass>Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        runner._cpp_postprocessor = <CppPostProcess>(
            input_w, input_h, 0.3, 0.45, use_ort
        )
        runner._cpp_convert_fn = convert_cpp_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
```

**Note**: The `<CppPostProcess>` class name comes from `dx_postprocess` module.
Common bindings include: `YOLOv5PostProcess`, `YOLOv8PostProcess`,
`YOLOv10PostProcess`, `YOLOv11PostProcess`, `SSDPostProcess`, `NanoDetPostProcess`.
Check `src/postprocess/` for the full list of 37 bindings.

## Step 9: Create Async C++ Postprocess Variant

### Template: `<model>_async_cpp_postprocess.py`

```python
#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
<ModelDisplay> Asynchronous Inference (C++ Postprocess) - DX-APP v3.0.0

Usage:
    python <model>_async_cpp_postprocess.py --model model.dxnn --video input.mp4
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import <CppPostProcess>
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import <ModelClass>Factory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("<ModelDisplay> Async Inference")


def main():
    args = parse_args()
    factory = <ModelClass>Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        runner._cpp_postprocessor = <CppPostProcess>(
            input_w, input_h, 0.3, 0.45, use_ort
        )
        runner._cpp_convert_fn = convert_cpp_detections

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
```

## Step 10: Create config.json

### Detection models
```json
{
  "score_threshold": 0.25,
  "nms_threshold": 0.45
}
```

### Classification models
```json
{
  "top_k": 5
}
```

### Segmentation models
```json
{
  "score_threshold": 0.25,
  "mask_threshold": 0.5
}
```

### Restoration / Enhancement / SR
```json
{}
```

## Step 11: Validate

Run these checks in order:

```bash
# 1. Syntax check all Python files
for f in factory/<model>_factory.py <model>_sync.py <model>_async.py \
         <model>_sync_cpp_postprocess.py <model>_async_cpp_postprocess.py; do
    python -c "import py_compile; py_compile.compile('$f', doraise=True)" && echo "OK: $f"
done

# 2. JSON validation
python -c "import json; json.load(open('config.json')); print('OK: config.json')"

# 3. Factory import test (requires dx_engine environment)
PYTHONPATH=../../ python -c "from factory import <ModelClass>Factory; f = <ModelClass>Factory(); print(f'OK: {f.get_model_name()} / {f.get_task_type()}')"

# 4. Smoke test (requires NPU + model file)
python <model>_sync.py --model /path/to/<model>.dxnn --image test.jpg --no-display
```

## File Creation Checklist

Before declaring the app complete, verify all files exist:

- [ ] `src/python_example/<task>/<model>/__init__.py`
- [ ] `src/python_example/<task>/<model>/config.json`
- [ ] `src/python_example/<task>/<model>/factory/__init__.py`
- [ ] `src/python_example/<task>/<model>/factory/<model>_factory.py`
- [ ] `src/python_example/<task>/<model>/<model>_sync.py`
- [ ] `src/python_example/<task>/<model>/<model>_async.py`
- [ ] `src/python_example/<task>/<model>/<model>_sync_cpp_postprocess.py` (if applicable)
- [ ] `src/python_example/<task>/<model>/<model>_async_cpp_postprocess.py` (if applicable)

## Substitution Reference

When using templates, replace these placeholders:

| Placeholder | Example | Description |
|---|---|---|
| `<model>` | `yolov8n` | Lowercase model identifier |
| `<MODEL_NAME>` | `yolov8n` | Model name for registry lookup |
| `<ModelClass>` | `Yolov8` | PascalCase for class name |
| `<ModelDisplay>` | `YOLOv8n` | Display name for docstrings |
| `<TASK>` | `object_detection` | Task directory name |
| `<IFactoryInterface>` | `IDetectionFactory` | Factory interface class |
| `<Preprocessor>` | `LetterboxPreprocessor` | Preprocessor class |
| `<Postprocessor>` | `YOLOv8Postprocessor` | Postprocessor class |
| `<Visualizer>` | `DetectionVisualizer` | Visualizer class |
| `<CppPostProcess>` | `YOLOv8PostProcess` | C++ pybind11 class |
| `<task_type>` | `object_detection` | Task type string |

## Run Commands

```bash
# Sync with image
python <model>_sync.py --model /path/to/<model>.dxnn --image test.jpg

# Async with video
python <model>_async.py --model /path/to/<model>.dxnn --video test.mp4

# Async with camera
python <model>_async.py --model /path/to/<model>.dxnn --camera 0

# Sync C++ postprocess, no display, save output
python <model>_sync_cpp_postprocess.py --model /path/to/<model>.dxnn \
    --image test.jpg --no-display --save

# Benchmark: async, no display, 3 loops
python <model>_async_cpp_postprocess.py --model /path/to/<model>.dxnn \
    --video test.mp4 --no-display --loop 3 --verbose
```
