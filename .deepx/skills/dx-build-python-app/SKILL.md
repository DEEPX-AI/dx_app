---
name: dx-build-python-app
description: Build Python inference app for dx_app
---

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
├── setup.sh              # Environment setup (venv, pip dependencies)
├── run.sh                # One-command inference launcher
├── session.json          # Build metadata
├── session.log           # Actual command output (captured via tee)
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
  "created_at": "<ISO 8601 timestamp with timezone, e.g. 2026-04-30T04:23:25+09:00>",
  "model": "<model_name>",
  "task": "<task_type>",
  "variants": ["sync", "async", "sync_cpp_postprocess", "async_cpp_postprocess"],
  "compiler_session": "<relative path to compiler session, e.g. dx-compiler/dx-agentic-dev/20260430-032426_claude_yolo26n_compile>",
  "agent": "<tool identifier: copilot | cursor | claude | opencode>",
  "status": "complete",
  "notes": "<any relevant notes>"
}
```

> **REC-5 — session.json `agent` and `compiler_session` fields are REQUIRED (R47)**:
> - `agent`: The tool identifier — use `copilot`, `cursor`, `claude`, or `opencode`.
>   This field disambiguates sessions when multiple tools run concurrently.
> - `compiler_session`: The relative path (from suite root) to the companion
>   dx-compiler session dir. Required for cross-project (compile + app) sessions.
>   Omit only when no compilation was performed.
> - `created_at` MUST include a timezone offset (e.g. `+09:00`). Plain UTC timestamps
>   without offset are non-compliant.
> - ALL tools (copilot, cursor, opencode, claude_code) MUST produce `session.json`.
>   Do not skip this file for any tool.

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

> **NEVER reuse previous session artifacts.** Do NOT check, list, browse, or
> reference files from previous sessions in `dx-agentic-dev/`. Each build
> session MUST create a new session directory with a fresh timestamp. Even if
> a previous session built the same model, always start from scratch. Do NOT
> run `ls dx-agentic-dev/` or check for existing files from past runs.

## Phase 0: Prerequisites Check

Before starting the build workflow, verify:

1. **dx-runtime**: `bash ../../scripts/sanity_check.sh --dx_rt`
   - FAIL → `bash ../../install.sh --all --exclude-app --exclude-stream --skip-uninstall --venv-reuse`
   - Re-run sanity_check.sh — must PASS after install
   - **If still failing → STOP (unconditional).** User instructions to continue do NOT override this.
     If NPU hardware init failure ("Device initialization failed"): tell the user a cold boot /
     system reboot is required, then STOP. NEVER proceed with code generation while sanity check is failing.
     NEVER mark this check as "done" when it actually failed.
2. **dx_engine**: `python -c "import dx_engine"` — FAIL → `./install.sh && ./build.sh`
3. **dx_postprocess** (if cpp_postprocess variants): `python -c "import dx_postprocess"` — FAIL → `./build.sh`

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

## Step 2: Search for Existing Examples (MANDATORY)

**Before creating any new files**, check whether a working example already exists.
Existing examples are the best reference for correct postprocessor selection.

```bash
# Check standard location
ls src/python_example/<TASK>/<MODEL>/factory/ 2>/dev/null

# Check PPU location
ls src/python_example/ppu/<MODEL>/factory/ 2>/dev/null

# If found, inspect the factory to see which postprocessor is used
grep -n "Postprocessor\|PostProcess" src/python_example/<TASK>/<MODEL>/factory/*_factory.py
```

**If an existing example is found:**
1. Read the existing factory file — it has the correct postprocessor for this model
2. Use the same preprocessor/postprocessor/visualizer combination
3. Ask the user: (a) explain existing only, or (b) create new based on existing
4. **Never generate a factory with a different postprocessor than the existing working example**

**If no existing example is found:**
1. Use the Registry Key → Postprocessor mapping table (Step 4) to select the correct class
2. Cross-reference with `model_registry.json` postprocessor field
3. Proceed to Step 3

## Step 3: Create Directory Structure

```bash
mkdir -p src/python_example/<TASK>/<MODEL>/factory
touch src/python_example/<TASK>/<MODEL>/__init__.py
touch src/python_example/<TASK>/<MODEL>/factory/__init__.py
```

## Step 4: Select IFactory Interface

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

## Step 5: Select Components

### Preprocessors (from `common.processors`)

| Preprocessor | Used By | Source File |
|---|---|---|
| `LetterboxPreprocessor` | All YOLO detection/pose/face/seg, SSD, NanoDet, CenterPose, DamoYolo, OBB | `letterbox_preprocessor.py` |
| `SimpleResizePreprocessor` | Classification, Semantic Segmentation, SegFormer, Depth, Restoration, Enhancement, Super Resolution, Embedding | `simple_resize_preprocessor.py` |
| `GrayscaleResizePreprocessor` | DnCNN (grayscale denoising models) | `grayscale_preprocessor.py` |

> **WARNING**: Only 3 preprocessor classes exist. Do NOT fabricate task-specific preprocessors
> (e.g., `ClassificationPreprocessor`, `DepthPreprocessor` do not exist).

### Postprocessors (from `common.processors`)

> **CRITICAL**: The `postprocessor` field in `model_registry.json` is a **registry key**,
> NOT a Python class name. Always use this mapping table to find the correct Python class.

| Registry Key | Python Postprocessor Class | C++ Binding (`dx_postprocess`) |
|---|---|---|
| `yolov5` | `YOLOv5Postprocessor` | `YOLOv5PostProcess` |
| `yolov8` | `YOLOv8Postprocessor` | `YOLOv8PostProcess` |
| `yolov26` | `YOLOv8Postprocessor` | `YOLOv26PostProcess` |
| `yolov10` | `YOLOv8Postprocessor` | `YOLOv10PostProcess` |
| `yolox` | `YOLOXPostprocessor` | `YOLOXPostProcess` |
| `damoyolo` | `DamoYoloPostprocessor` | `DamoYOLOPostProcess` |
| `nanodet` | `NanoDetPostprocessor` | `NanoDetPostProcess` |
| `ssd` | `SSDPostprocessor` | `SSDPostProcess` |
| `efficientnet` | `ClassificationPostprocessor` | `ClassificationPostProcess` |
| `yolov8pose` | `YOLOv8PosePostprocessor` | `YOLOv8PosePostProcess` |
| `yolov5seg` | `YOLOv5InstanceSegPostprocessor` | `YOLOv5SegPostProcess` |
| `yolov8seg` | `YOLOv8InstanceSegPostprocessor` | `YOLOv8SegPostProcess` |
| `bisenetv1` / `bisenetv2` / `deeplabv3` | `SemanticSegmentationPostprocessor` | `SemanticSegPostProcess` / `DeepLabv3PostProcess` |
| `segformer` | `SegFormerPostprocessor` | `SemanticSegPostProcess` |
| `scrfd` | `SCRFDPostprocessor` | `SCRFDPostProcess` |
| `yolov5face` | `YOLOv5FacePostprocessor` | `YOLOv5FacePostProcess` |
| `retinaface` | `RetinaFacePostprocessor` | `RetinaFacePostProcess` |
| `fastdepth` | `DepthEstimationPostprocessor` | `DepthPostProcess` |
| `dncnn` | `DnCNNPostprocessor` | `DnCNNPostProcess` |
| `espcn` | `ESPCNPostprocessor` | `ESPCNPostProcess` |
| `zero_dce` | `ZeroDCEPostprocessor` | `ZeroDCEPostProcess` |
| `arcface` | `ArcFacePostprocessor` | `EmbeddingPostProcess` |
| `obb` | `OBBPostprocessor` | `OBBPostProcess` |
| `yolov5_ppu` | `YOLOv5PPUPostprocessor` | `YOLOv5PPUPostProcess` |
| `yolov7_ppu` | `YOLOv7PPUPostprocessor` | `YOLOv7PPUPostProcess` |
| `yolov8_ppu` | `YOLOv8PPUPostprocessor` | `YOLOv8PPUPostProcess` |
| `yolox_ppu` | `YOLOXPPUPostprocessor` | `YOLOXPPUPostProcess` |
| `yolov3tiny_ppu` | `YOLOv3TinyPPUPostprocessor` | `YOLOv3TinyPPUPostProcess` |
| `efficientdet` | `EfficientDetPostprocessor` | `EfficientDetPostProcess` |
| `yolact` | `YOLACTPostprocessor` | `YOLACTPostProcess` |
| `hand_landmark` | `HandLandmarkPostprocessor` | `HandLandmarkPostProcess` |

> **Note**: For the complete and authoritative list of C++ postprocessor bindings,
> see `src/bindings/python/dx_postprocess/postprocess_pybinding.cpp`.

> **WARNING — yolo26 trap**: `model_registry.json` uses registry key `"yolov26"`, but
> the correct Python class is `YOLOv8Postprocessor` (NOT `Yolo26Postprocessor` which
> does not exist). YOLO26 uses YOLOv8-compatible end-to-end output format `[1,300,6]`.
>
> **WARNING**: Generic names like `PosePostprocessor`, `FacePostprocessor` do NOT exist.
> Each model family has its own specific postprocessor class.

### Visualizers (from `common.visualizers`)

| Visualizer | Used By | Source File |
|---|---|---|
| `DetectionVisualizer` | All detection tasks | `detection_visualizer.py` |
| `ClassificationVisualizer` | Classification | `classification_visualizer.py` |
| `PoseVisualizer` | Pose estimation | `pose_visualizer.py` |
| `InstanceSegVisualizer` | Instance segmentation | `instance_seg_visualizer.py` |
| `SemanticSegmentationVisualizer` | Semantic segmentation | `segmentation_visualizer.py` |
| `FaceVisualizer` | Face detection | `face_visualizer.py` |
| `DepthVisualizer` | Depth estimation | `restoration_depth_visualizer.py` |
| `RestorationVisualizer` | Denoising | `restoration_depth_visualizer.py` |
| `SuperResolutionVisualizer` | Super resolution | `embedding_enhancement_visualizer.py` |
| `EnhancementVisualizer` | Image enhancement | `embedding_enhancement_visualizer.py` |
| `EmbeddingVisualizer` | Embedding | `embedding_enhancement_visualizer.py` |
| `OBBVisualizer` | OBB detection | `obb_visualizer.py` |
| `FaceAlignmentVisualizer` | Face alignment | `embedding_enhancement_visualizer.py` |
| `HandLandmarkVisualizer` | Hand landmark | `embedding_enhancement_visualizer.py` |

## Step 6: Create Factory

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

## Step 7: Create Sync Variant

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

## Step 8: Create Async Variant

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

## Step 9: Create Sync C++ Postprocess Variant

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
Check `src/bindings/python/dx_postprocess/postprocess_pybinding.cpp` for the full list of available bindings.

## Step 10: Create Async C++ Postprocess Variant

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

## Step 11: Create config.json

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

## Step 12: Validate

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

# 4. Postprocessor cross-check (CRITICAL — catches wrong postprocessor)
# Verify the factory uses the correct postprocessor for this model family.
# See the Registry Key → Python Postprocessor Class table in Step 5.
python -c "
import ast, json
# Read registry
with open('config/model_registry.json') as f:
    models = json.load(f)
match = [m for m in models if m['model_name'] == '<MODEL_NAME>']
if match:
    reg_key = match[0].get('postprocessor', '')
    print(f'Registry postprocessor key: {reg_key}')
# Read factory
tree = ast.parse(open('factory/<model>_factory.py').read())
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if 'Postprocessor' in (alias.asname or alias.name):
                print(f'Factory uses: {alias.asname or alias.name}')
"

# 5. Smoke test (requires NPU + model file)
python <model>_sync.py --model /path/to/<model>.dxnn --image <TASK_SAMPLE_IMAGE> --no-display

# 6. Output accuracy check (CRITICAL — catches zero-detection bugs)
# After smoke test, verify detection count > 0 on known-good sample image.
# If the app supports --save-json, check the JSON output:
python -c "
import subprocess, sys
result = subprocess.run(
    ['python', '<model>_sync.py', '--model', '/path/to/<model>.dxnn',
     '--image', '<TASK_SAMPLE_IMAGE>', '--no-display', '--verbose'],
    capture_output=True, text=True, timeout=60
)
if result.returncode != 0:
    print(f'FAIL: exit code {result.returncode}')
    sys.exit(1)
# Check stdout for detection count
output = result.stdout + result.stderr
if 'detections: 0' in output.lower() or 'no objects detected' in output.lower():
    print('FAIL: Zero detections on known-good sample image')
    print('Check: postprocessor class, score_threshold, model output format')
    sys.exit(1)
print('PASS: Inference completed with detections')
"
```

### 7. Cross-validation with reference model (if available)

If a precompiled DXNN for the same model exists in `assets/models/` or an existing
verified example exists in `src/python_example/<task>/<model>/`, run a differential
diagnosis to isolate app code vs compilation issues. See `dx-validate.md` Level 5.5.

```bash
# Check if precompiled reference model exists
MODEL_NAME="<MODEL_NAME>"
DX_APP_ROOT="$(cd ../.. && pwd)"
REF_MODEL="${DX_APP_ROOT}/assets/models/${MODEL_NAME}.dxnn"

if [ -f "$REF_MODEL" ]; then
    echo "=== Cross-validation: Generated app with precompiled model ==="
    python <model>_sync.py --model "$REF_MODEL" \
        --image <TASK_SAMPLE_IMAGE> --no-display --verbose
    REF_RESULT=$?

    echo "=== Cross-validation: Generated app with new model ==="
    python <model>_sync.py --model /path/to/<model>.dxnn \
        --image <TASK_SAMPLE_IMAGE> --no-display --verbose
    NEW_RESULT=$?

    if [ $REF_RESULT -eq 0 ] && [ $NEW_RESULT -ne 0 ]; then
        echo "DIAGNOSIS: Compilation problem — precompiled model works, new model fails"
    elif [ $REF_RESULT -ne 0 ] && [ $NEW_RESULT -ne 0 ]; then
        echo "DIAGNOSIS: Generated app code problem — both models fail"
    elif [ $REF_RESULT -eq 0 ] && [ $NEW_RESULT -eq 0 ]; then
        echo "PASS: Both models work with generated app"
    fi
fi

# Check if existing verified example exists
TASK="<TASK>"
EXISTING_APP="${DX_APP_ROOT}/src/python_example/${TASK}/${MODEL_NAME}/${MODEL_NAME}_sync.py"
if [ -f "$EXISTING_APP" ] && [ -f "$REF_MODEL" ]; then
    echo "=== Cross-validation: Existing app with new model ==="
    python "$EXISTING_APP" --model /path/to/<model>.dxnn \
        --image <TASK_SAMPLE_IMAGE> --no-display --verbose
    EXISTING_RESULT=$?

    if [ $EXISTING_RESULT -ne 0 ]; then
        echo "DIAGNOSIS: Compilation-level problem — existing verified app also fails"
    fi
fi
```

**Decision tree**: See `dx-validate.md` Level 5.5 Differential Diagnosis Decision Matrix.

### Task-Aware Sample Image for Smoke Test

Select the sample image based on the model's AI task:

| Task | Sample Image Path |
|---|---|
| object_detection | `../../sample/img/sample_dog.jpg` |
| face_detection | `../../sample/img/sample_face.jpg` |
| pose_estimation | `../../sample/img/sample_people.jpg` |
| hand_landmark | `../../sample/img/sample_hand.jpg` |
| obb_detection | `../../sample/dota8_test/P0177.png` |
| instance_segmentation, semantic_segmentation | `../../sample/img/sample_street.jpg` |
| classification | `../../sample/ILSVRC2012/0.jpeg` |
| super_resolution | `../../sample/img/sample_superresolution.png` |
| image_enhancement | `../../sample/img/sample_lowlight.jpg` |
| image_denoising | `../../sample/img/sample_denoising.jpg` |
| depth_estimation | `../../sample/img/sample_street.jpg` |
| embedding | `../../sample/img/sample_face.jpg` |

**MUST** use these task-matched images instead of generic `test.jpg` or `input.jpg`.

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
- [ ] `setup.sh` — environment setup script (**MUST** detect/activate venv — see setup.sh template below)
- [ ] `run.sh` — one-command inference launcher (**MUST** use real model + sample image paths — see run.sh template below)
- [ ] `session.log` — actual command output with structured blocks (see session.log template below)
- [ ] `session.json` — build metadata (**REQUIRED** — see session.json template at top of this skill doc)

  > **REC-T4 (iter-15) — session.json is MANDATORY for ALL tools including claude_code.**
  > The agent skipping session.json is the most common source of `test_session_json_exists` failures.
  > Write it BEFORE declaring DONE. Use this exact JSON structure:
  >
  > ⚠️ **`session.txt` is NOT `session.json`.** Both files may exist, but only `session.json` (a
  > JSON-formatted file) satisfies the `test_session_json_exists` requirement. Writing `session.txt`
  > instead of `session.json` is the most common mistake — the file MUST be named exactly `session.json`.
  >
  > ```json
  > {
  >   "session_id": "<YYYYMMDD-HHMMSS>_<agent>_<model>_inference",
  >   "created_at": "<ISO 8601 local timezone, e.g. 2026-04-30T08:20:54+09:00>",
  >   "model": "<model_name, e.g. yolo26n>",
  >   "task": "<task_type, e.g. object_detection>",
  >   "variants": ["sync"],
  >   "compiler_session": "dx-compiler/dx-agentic-dev/<compiler_session_id>",
  >   "agent": "<copilot | cursor | claude | opencode>",
  >   "status": "complete",
  >   "notes": ""
  > }
  > ```
  >
  > **STOP before DONE if session.json is absent — the test harness WILL catch this.**
  >
  > **REC-V2 (iter-17) — opencode session.json regression HARD GATE**:
  > opencode iter-17 did NOT produce `session.json` in the app session, causing
  > `test_session_json_exists` to FAIL. To guarantee session.json is written,
  > opencode agents MUST execute this Python snippet explicitly in the app session:
  >
  > ```python
  > import json
  > from datetime import datetime
  > from pathlib import Path
  >
  > # ⛔ BLOCKING GATE — write session.json BEFORE emitting [DX-AGENTIC-DEV: DONE]
  > session_json = {
  >     "session_id": SESSION_ID,   # e.g. "20260430-101324_opencode_yolo26n_inference"
  >     "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00"),
  >     "model": "yolo26n",
  >     "task": "object_detection",
  >     "variants": ["sync"],
  >     "compiler_session": "dx-compiler/dx-agentic-dev/<compiler_session_id>",
  >     "agent": "opencode",
  >     "status": "complete",
  >     "notes": "",
  > }
  > (Path(APP_SESSION_DIR) / "session.json").write_text(
  >     json.dumps(session_json, indent=2, ensure_ascii=False)
  > )
  > print("session.json written —", APP_SESSION_DIR)
  > ```
  >
  > **Do NOT proceed to DONE without verifying `session.json` exists in the app session dir.**
- [ ] `config.json` — inference thresholds / model config for the app session (**REQUIRED**)

## session.log Template (MANDATORY)

> **R23**: session.log must have a structured header and named phase blocks. Reference: opencode session `224919` (58 lines, all phases logged).

```bash
# ── session.log init (write before any other command) ──────────────────────
SESSION_ID="<YYYYMMDD-HHMMSS>_<agent>_<model>_<task>"
echo "===== SESSION LOG: ${SESSION_ID} =====" > session.log
echo "Date: $(date)" >> session.log
echo "" >> session.log

# After sanity check:
echo "--- sanity_check ---" >> session.log
echo "$ bash dx-runtime/scripts/sanity_check.sh --dx_rt" >> session.log
<paste actual output> >> session.log
echo "RESULT: PASS" >> session.log
echo "" >> session.log

# After inference run:
echo "--- inference ---" >> session.log
echo "$ python <model>_sync.py --input bus.jpg" >> session.log
<paste actual output (FPS, latency, detections)> >> session.log
echo "RESULT: PASS  (<N> FPS, <M> ms NPU)" >> session.log
echo "" >> session.log

# Artifacts listing:
echo "--- artifacts ---" >> session.log
ls -lh . >> session.log
echo "RESULT: PASS" >> session.log
```

**session.log MUST contain**:
- A `===== SESSION LOG: <session_id> =====` header
- Named blocks: `sanity_check`, `inference`, `artifacts`
- Each block ends with `RESULT: PASS` or `RESULT: FAIL`
- Actual command output (NOT hand-written summaries)

## setup.sh Template (MANDATORY)

> **CRITICAL**: `setup.sh` must be runnable standalone. A user must be able to `cd` into
> the session directory and run `./setup.sh` without manually activating any venv first.
> Failure to include venv detection causes NumPy/OpenCV version conflicts on the host system.

```bash
#!/bin/bash
# Environment setup for <ModelDisplay> <TaskType> app
# Generated by DX Agentic Dev

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Virtual environment detection & activation ---
# Search upward for the dx-runtime shared venv (preferred)
RUNTIME_VENV=""
_search="$SCRIPT_DIR"
for _i in 1 2 3 4 5; do
    _search="$(dirname "$_search")"
    if [ -d "$_search/venv-dx-runtime" ]; then
        RUNTIME_VENV="$_search/venv-dx-runtime"
        break
    fi
done

LOCAL_VENV="$SCRIPT_DIR/.venv"

if [ -n "$RUNTIME_VENV" ] && [ -d "$RUNTIME_VENV" ]; then
    echo "[INFO] Activating dx-runtime venv: $RUNTIME_VENV"
    source "$RUNTIME_VENV/bin/activate"
elif [ -d "$LOCAL_VENV" ]; then
    echo "[INFO] Activating local venv: $LOCAL_VENV"
    source "$LOCAL_VENV/bin/activate"
else
    echo "[INFO] Creating local venv at $LOCAL_VENV ..."
    python3 -m venv "$LOCAL_VENV"
    source "$LOCAL_VENV/bin/activate"
    pip install --upgrade pip
fi

# --- 2. Install dependencies ---
pip install opencv-python numpy

# --- 3. Verify dx_engine ---
python -c "import dx_engine; print('[OK] dx_engine available')" 2>/dev/null || {
    echo "[WARN] dx_engine not found in this venv."
    echo "       Run: cd $(cd "$SCRIPT_DIR/../.." && pwd) && ./install.sh && ./build.sh"
}

echo "[INFO] Setup complete. Run: bash run.sh"
```

**Customization rules:**
- Adjust `pip install` line to include model-specific dependencies if needed
- The 5-level upward search covers both `dx-agentic-dev/<session>/` (3 levels up)
  and `src/python_example/<task>/<model>/` (4 levels up) paths

## run.sh Template (MANDATORY)

> **CRITICAL**: `run.sh` must include **real, working paths** — never `/path/to/` placeholders.
> The model and image paths must be relative from the session directory.

```bash
#!/bin/bash
# One-command inference launcher for <ModelDisplay> <TaskType>
# Generated by DX Agentic Dev

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Activate venv ---
source "$SCRIPT_DIR/setup.sh" 2>/dev/null || true

# --- Default paths (relative from session dir) ---
# Model: use precompiled model or dx-compiler output
DEFAULT_MODEL="../../assets/models/<model>.dxnn"
# Sample image: task-appropriate (see Task-Aware Sample Image table)
DEFAULT_IMAGE="../../sample/img/<TASK_SAMPLE_IMAGE>"

MODEL="${1:-$DEFAULT_MODEL}"
IMAGE="${2:-$DEFAULT_IMAGE}"

if [ ! -f "$MODEL" ]; then
    echo "[ERROR] Model not found: $MODEL"
    echo "Usage: bash run.sh [model_path] [image_path]"
    echo ""
    echo "Model locations:"
    echo "  Precompiled: ../../assets/models/<model>.dxnn"
    echo "  dx-compiler: ../../../../dx-compiler/dx-agentic-dev/<session>/<model>.dxnn"
    exit 1
fi

echo "[INFO] Model: $MODEL"
echo "[INFO] Image: $IMAGE"
python "$SCRIPT_DIR/<model>_sync.py" --model "$MODEL" --image "$IMAGE" --no-display
```

**Customization rules:**
- Replace `<TASK_SAMPLE_IMAGE>` with the actual sample image from the Task-Aware table
  (e.g., `sample_dog.jpg` for object_detection)
- If the model exists in `assets/models/`, use that as `DEFAULT_MODEL`
- If using a dx-compiler output, compute the relative path from the session directory

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

> **RULE**: Never use `/path/to/<model>.dxnn` or generic `test.jpg` / `input.jpg` in
> generated run commands, README examples, or run.sh scripts. Always use **real relative
> paths** from the session directory.

### Model Path Resolution (MANDATORY)

When generating run commands, resolve the model path in this priority order:

1. **Precompiled model in assets**: `../../assets/models/<model>.dxnn` — check if it exists
2. **dx-compiler session output**: `../../../../dx-compiler/dx-agentic-dev/<session>/<model>.dxnn`
3. **User-provided path**: If the user specified a model location, use that

If the model file cannot be located, use the precompiled assets path as the default
and note in the README that the user may need to adjust the path.

> **R46 — Do NOT copy `.dxnn` to the app session directory**:
> The compiled `.dxnn` file belongs in `dx-compiler/dx-agentic-dev/<compiler_session>/`.
> Reference it via a relative path from `run.sh` / `yolo26n_sync.py` — do NOT copy it
> into `dx_app/dx-agentic-dev/<app_session>/`. Copying wastes 6–7 MB and blurs the
> dual-session boundary. Use path #2 above (`../../../../dx-compiler/…/<model>.dxnn`)
> or store the path in `config.json` and read it at runtime.

### Example Run Commands (with real paths)

```bash
# Sync with image (object_detection example)
python <model>_sync.py --model ../../assets/models/<model>.dxnn \
    --image ../../sample/img/sample_dog.jpg

# Async with video
python <model>_async.py --model ../../assets/models/<model>.dxnn \
    --video ../../assets/videos/dogs.mp4

# Async with camera
python <model>_async.py --model ../../assets/models/<model>.dxnn --camera 0

# Sync C++ postprocess, no display, save output
python <model>_sync_cpp_postprocess.py --model ../../assets/models/<model>.dxnn \
    --image ../../sample/img/sample_dog.jpg --no-display --save

# Benchmark: async, no display, 3 loops
python <model>_async_cpp_postprocess.py --model ../../assets/models/<model>.dxnn \
    --video ../../assets/videos/dogs.mp4 --no-display --loop 3 --verbose
```

**Note**: Replace `sample_dog.jpg` with the task-appropriate sample image from the
Task-Aware Sample Image table (Step 12).
