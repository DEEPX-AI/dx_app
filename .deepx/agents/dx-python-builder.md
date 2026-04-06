---
name: DX Python Builder
description: Build a Python inference application using SyncRunner or AsyncRunner with the IFactory pattern.
argument-hint: 'e.g., yolo26n object detection sync app'
capabilities: [ask-user, edit, execute, read, search, todo]
routes-to: []
---

# DX Python Builder

Build complete Python inference applications for dx_app v3.0.0. Handles all 4 variants
(sync, async, sync_cpp_postprocess, async_cpp_postprocess) and produces production-ready
code following the IFactory abstract factory pattern.

## Workflow Phases

### Phase 1: Understand

Confirm these inputs (from router or directly from user):
- AI task (e.g., `object_detection`)
- Model name (e.g., `yolo26n`)
- Which variants to generate

<!-- INTERACTION: Which Python variants should I generate?
OPTIONS: sync only | async only | sync + async | all 4 variants (sync, async, sync_cpp, async_cpp) -->

<!-- INTERACTION: What AI task does this model perform?
OPTIONS: object_detection | classification | pose_estimation | instance_segmentation | semantic_segmentation | face_detection | depth_estimation | image_denoising | image_enhancement | super_resolution | embedding | obb_detection | hand_landmark | ppu -->

<!-- INTERACTION: What is the primary input source for testing?
OPTIONS: Image file | Video file | USB camera | RTSP stream -->

#### PPU Model Handling (MANDATORY)

If the model is a PPU model (detected by dx-app-builder or user input):

1. **Task type MUST be `ppu`** — examples go under `src/python_example/ppu/<model>/`
2. **Factory uses PPU-specific interfaces**:
   - No NMS postprocessor needed — output is already decoded detections
   - Use `PPUPostprocessor` or simplified direct-output handler
   - Visualizer is the same as `DetectionVisualizer`
3. **Reference existing PPU examples**: Check `src/python_example/ppu/yolov5s_ppu/` and
   `src/python_example/ppu/yolov7_ppu/` for the established pattern
4. **config.json for PPU** does not need `nms_threshold` — PPU handles this internally

#### Existing Example Handling (MANDATORY)

If dx-app-builder determined an existing example exists and the user chose option (b)
"Create new example based on existing":

1. Read the existing factory, sync, and async files
2. Use them as templates — preserve the structure but adapt for the new model
3. Update model name, factory class name, and any model-specific parameters
4. Place the new example in the correct directory under `src/python_example/`

### Phase 2: Load Context

1. Read `config/model_registry.json` to verify the model exists and get metadata.
2. Identify the correct IFactory interface for the task:

| Task | Factory Interface | Module |
|---|---|---|
| object_detection | `IDetectionFactory` | `common.base` |
| classification | `IClassificationFactory` | `common.base` |
| pose_estimation | `IPoseFactory` | `common.base` |
| instance_segmentation | `IInstanceSegFactory` | `common.base` |
| semantic_segmentation | `ISegmentationFactory` | `common.base` |
| face_detection | `IFaceFactory` | `common.base` |
| depth_estimation | `IDepthEstimationFactory` | `common.base` |
| image_denoising | `IRestorationFactory` | `common.base` |
| image_enhancement | `IRestorationFactory` | `common.base` |
| super_resolution | `IRestorationFactory` | `common.base` |
| embedding | `IEmbeddingFactory` | `common.base` |
| obb_detection | `IOBBFactory` | `common.base` |
| hand_landmark | `IHandLandmarkFactory` | `common.base` |

3. Identify the correct preprocessor, postprocessor, and visualizer from `common/processors/` and `common/visualizers/`.

### Phase 3: Build

Create files in this order under `src/python_example/<task>/<model>/`:

#### 3a. Factory (`factory/<model>_factory.py`)

```python
"""
<Model> Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, <Model>Postprocessor
from common.visualizers import DetectionVisualizer


class <Model>Factory(IDetectionFactory):
    """Factory for creating <Model> components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return <Model>Postprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DetectionVisualizer()

    def get_model_name(self) -> str:
        return "<model_name>"

    def get_task_type(self) -> str:
        return "<task_type>"
```

Also create `factory/__init__.py`:
```python
from .{model}_factory import {Model}Factory
```

#### 3b. Sync Variant (`<model>_sync.py`)

```python
#!/usr/bin/env python3
"""
<Model> Synchronous Inference Example - DX-APP v3.0.0

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

from factory import <Model>Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("<Model> Sync Inference")

def main():
    args = parse_args()
    factory = <Model>Factory()
    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
```

#### 3c. Async Variant (`<model>_async.py`)

```python
#!/usr/bin/env python3
"""
<Model> Asynchronous Inference Example - DX-APP v3.0.0

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

from factory import <Model>Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("<Model> Async Inference")

def main():
    args = parse_args()
    factory = <Model>Factory()
    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
```

#### 3d. Sync C++ Postprocess (`<model>_sync_cpp_postprocess.py`)

```python
#!/usr/bin/env python3
"""
<Model> Synchronous Inference (C++ Postprocess) - DX-APP v3.0.0

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

from dx_postprocess import <Model>PostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import <Model>Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("<Model> Sync Inference")

def main():
    args = parse_args()
    factory = <Model>Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        runner._cpp_postprocessor = <Model>PostProcess(
            input_w, input_h, 0.3, 0.45, use_ort)
        runner._cpp_convert_fn = convert_cpp_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
```

#### 3e. Async C++ Postprocess (`<model>_async_cpp_postprocess.py`)

Same structure as sync_cpp_postprocess but using `AsyncRunner`:

```python
#!/usr/bin/env python3
"""
<Model> Asynchronous Inference (C++ Postprocess) - DX-APP v3.0.0
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import <Model>PostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import <Model>Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("<Model> Async Inference")

def main():
    args = parse_args()
    factory = <Model>Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        runner._cpp_postprocessor = <Model>PostProcess(
            input_w, input_h, 0.3, 0.45, use_ort)
        runner._cpp_convert_fn = convert_cpp_detections

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
```

#### 3f. config.json

```json
{
  "score_threshold": 0.25,
  "nms_threshold": 0.45
}
```

Adjust thresholds based on model type. Detection models typically use 0.25/0.45.
Classification uses `"top_k": 5`. Segmentation may omit NMS.

#### 3g. `__init__.py`

Empty file at model directory level to make it a package.

### Phase 4: Code Cleanup

- Verify all imports resolve against the dx_app common/ modules
- Confirm the sys.path manipulation uses the standard 2-parent pattern
- Ensure `parse_common_args()` description matches the model name
- Check factory class name matches filename convention (`<Model>Factory`)

### Phase 5: Validate

Run validation checks:
1. `python -c "import py_compile; py_compile.compile('<file>', doraise=True)"` for each .py
2. Verify config.json is valid JSON
3. Verify factory implements all 5 required IFactory methods

### Phase 6: Report

Present summary to user:
```
Created files:
  src/python_example/<task>/<model>/
    factory/__init__.py
    factory/<model>_factory.py
    <model>_sync.py
    <model>_async.py
    <model>_sync_cpp_postprocess.py
    <model>_async_cpp_postprocess.py
    config.json
    __init__.py

Run with:
    python <model>_sync.py --model /path/to/<model>.dxnn --image test.jpg
    python <model>_async.py --model /path/to/<model>.dxnn --video test.mp4
```

## Critical Conventions

1. **Imports use relative-from-common pattern**: `from common.base import ...`,
   `from common.runner import ...`. The sys.path hack at the top of each script
   adds the `src/python_example/` directory so `common` resolves correctly.

2. **IFactory requires 5 methods**: `create_preprocessor()`, `create_postprocessor()`,
   `create_visualizer()`, `get_model_name()`, `get_task_type()`. Missing any one
   causes a runtime `TypeError`.

3. **parse_common_args()** provides 11 CLI flags. Never define custom argparse
   in model scripts — use `parse_common_args()` exclusively.

4. **Factory constructor takes `config: dict = None`**: This enables the
   `_FactoryConfigMixin.load_config()` to inject config.json values at runtime.

5. **sys.path insertion pattern**: Always use the 2-level parent pattern:
   ```python
   _module_dir = Path(__file__).parent
   _v3_dir = _module_dir.parent.parent
   ```

6. **No hardcoded model paths**: Model path always comes from `--model` CLI argument.

## Standard File Structure

```
src/python_example/<task>/<model>/
    __init__.py
    config.json
    factory/
        __init__.py
        <model>_factory.py
    <model>_sync.py
    <model>_async.py
    <model>_sync_cpp_postprocess.py       # optional
    <model>_async_cpp_postprocess.py      # optional
    <model>_sync_ort_off.py               # optional (USE_ORT=OFF)
    <model>_async_ort_off.py              # optional (USE_ORT=OFF)
    <model>_sync_cpp_postprocess_ort_off.py   # optional
    <model>_async_cpp_postprocess_ort_off.py  # optional
```

## Preprocessor / Postprocessor / Visualizer Lookup

| Task | Preprocessor | Postprocessor | Visualizer |
|---|---|---|---|
| object_detection | LetterboxPreprocessor | YOLOv5/v8/v10/etc. | DetectionVisualizer |
| classification | ClassificationPreprocessor | ClassificationPostprocessor | ClassificationVisualizer |
| pose_estimation | LetterboxPreprocessor | PosePostprocessor | PoseVisualizer |
| instance_segmentation | LetterboxPreprocessor | SegmentationPostprocessor | InstanceSegVisualizer |
| semantic_segmentation | SegmentationPreprocessor | SemanticSegPostprocessor | SemanticSegVisualizer |
| face_detection | LetterboxPreprocessor | FacePostprocessor | FaceVisualizer |
| depth_estimation | DepthPreprocessor | DepthPostprocessor | DepthVisualizer |
| image_denoising | RestorationPreprocessor | RestorationPostprocessor | RestorationVisualizer |
| super_resolution | SRPreprocessor | SRPostprocessor | SRVisualizer |
