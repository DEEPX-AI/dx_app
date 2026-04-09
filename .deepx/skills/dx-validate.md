# Skill: Validation for dx_app

> Validate dx_app applications at every phase gate: static analysis,
> configuration checks, smoke tests, and full integration tests.

## 7-Level Validation Pyramid

```
Level 6:   Integration       (NPU + model + full pipeline)
Level 5.5: Cross-Validation  (NPU + differential diagnosis with precompiled reference model)
Level 5:   Output Accuracy   (NPU + verify detection output correctness)
Level 4:   Smoke             (NPU + model + quick single-frame inference)
Level 3:   Component         (preprocessor/postprocessor/visualizer individually)
Level 2:   Config            (JSON validity, schema compliance)
Level 1:   Static            (syntax, imports, factory interface)
```

Levels 1-3 can run without NPU hardware.
Levels 4-6 require NPU + model files.
Level 5.5 requires NPU + precompiled reference model in `assets/models/` (skip if unavailable).

## Level 1: Static Validation (11 Checks)

### Check 1: Python syntax

```bash
python -c "import py_compile; py_compile.compile('FILE.py', doraise=True)"
```

### Check 2: Factory has all 5 methods

```python
import ast, sys

tree = ast.parse(open(sys.argv[1]).read())
required = {'create_preprocessor', 'create_postprocessor',
            'create_visualizer', 'get_model_name', 'get_task_type'}
found = set()
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in required:
        found.add(node.name)
missing = required - found
if missing:
    print(f"FAIL: Missing methods: {missing}")
    sys.exit(1)
print("PASS: All 5 IFactory methods present")
```

### Check 3: sys.path pattern present

```python
import re, sys

content = open(sys.argv[1]).read()
if '_v3_dir = _module_dir.parent.parent' not in content:
    print("FAIL: Missing standard sys.path setup")
    sys.exit(1)
print("PASS: sys.path pattern found")
```

### Check 4: parse_common_args() used

```python
content = open(sys.argv[1]).read()
if 'parse_common_args' not in content:
    print("FAIL: Not using parse_common_args()")
    sys.exit(1)
print("PASS: Uses parse_common_args()")
```

### Check 5: No hardcoded model paths

```python
import re, sys

content = open(sys.argv[1]).read()
pattern = r'["\'][/\\].*\.dxnn["\']'
matches = re.findall(pattern, content)
if matches:
    print(f"FAIL: Hardcoded model paths: {matches}")
    sys.exit(1)
print("PASS: No hardcoded model paths")
```

### Check 6: File header present

```python
content = open(sys.argv[1]).read()
if not content.startswith('#!/usr/bin/env python3'):
    print("WARN: Missing shebang line")
```

### Check 7: __init__.py exists

```bash
test -f __init__.py && echo "PASS" || echo "FAIL: Missing __init__.py"
test -f factory/__init__.py && echo "PASS" || echo "FAIL: Missing factory/__init__.py"
```

### Check 8: Factory import in __init__.py

```python
content = open('factory/__init__.py').read()
if 'import' not in content:
    print("WARN: factory/__init__.py appears empty")
```

### Check 9: No relative imports

```python
import re, sys

content = open(sys.argv[1]).read()
if re.search(r'from\s+\.\.', content):
    print("FAIL: Contains relative imports (from ..)")
    sys.exit(1)
print("PASS: No relative imports")
```

### Check 10: Docstring present

```python
import ast, sys

tree = ast.parse(open(sys.argv[1]).read())
if not ast.get_docstring(tree):
    print("WARN: Missing module docstring")
```

### Check 11: No print() in factory

```python
import ast, sys

tree = ast.parse(open(sys.argv[1]).read())
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == 'print':
            print("WARN: Factory contains print() — use logging instead")
            break
```

## Level 2: Config Validation

### Check 1: config.json is valid JSON

```bash
python -c "import json; json.load(open('config.json')); print('PASS')"
```

### Check 2: config.json has expected keys

```python
import json, sys

config = json.load(open('config.json'))
task = sys.argv[1] if len(sys.argv) > 1 else 'object_detection'

expected = {
    'object_detection': ['score_threshold', 'nms_threshold'],
    'classification': ['top_k'],
    'pose_estimation': ['score_threshold', 'nms_threshold'],
    'instance_segmentation': ['score_threshold'],
    'semantic_segmentation': [],
    'face_detection': ['score_threshold', 'nms_threshold'],
    'depth_estimation': [],
    'image_denoising': [],
    'super_resolution': [],
}

for key in expected.get(task, []):
    if key not in config:
        print(f"WARN: Missing expected key '{key}' for {task}")
print("PASS: Config validation complete")
```

### Check 3: model_registry.json entry exists

```python
import json, sys

model_name = sys.argv[1]
with open('config/model_registry.json') as f:
    models = json.load(f)
match = [m for m in models if m['model_name'] == model_name]
if not match:
    print(f"WARN: {model_name} not found in model_registry.json")
else:
    print(f"PASS: Found in registry")
```

## Level 3: Component Validation (Smoke, No NPU)

### Check: Factory creates valid components

```python
import sys
sys.path.insert(0, '../..')  # Add python_example to path
from factory import <ModelClass>Factory

factory = <ModelClass>Factory()
assert factory.get_model_name() != ""
assert factory.get_task_type() != ""

pre = factory.create_preprocessor(640, 640)
assert pre is not None
assert hasattr(pre, 'process')

post = factory.create_postprocessor(640, 640)
assert post is not None
assert hasattr(post, 'process')

viz = factory.create_visualizer()
assert viz is not None
assert hasattr(viz, 'visualize')

print("PASS: All components created successfully")
```

## Level 4: Smoke Test (NPU Required)

Quick single-frame inference to verify the full pipeline works:

```bash
# Verify NPU first
dxrt-cli -s

# Single image, no display
python <model>_sync.py \
    --model /path/to/<model>.dxnn \
    --image test.jpg \
    --no-display

# Check exit code
echo "Exit code: $?"
```

Expected output contains `[INFO] Starting inference` and performance summary.

## Level 5: Output Accuracy Validation (NPU Required)

After smoke test passes (Level 4), verify the inference output is actually correct.
This catches the critical gap where exit code is 0 but detections are wrong or empty.

### Check 1: Detection Count > 0

```python
"""
Run after smoke test. Verifies that the model actually detects something
on a known-good sample image appropriate for the task.
"""
import subprocess, json, sys

model_script = sys.argv[1]    # e.g., yolo26n_sync.py
model_path = sys.argv[2]      # e.g., /path/to/yolo26n.dxnn
sample_image = sys.argv[3]    # Task-appropriate image (see table below)

result = subprocess.run(
    ["python", model_script, "--model", model_path,
     "--image", sample_image, "--no-display", "--save-json"],
    capture_output=True, text=True, timeout=60
)

if result.returncode != 0:
    print(f"FAIL: Script exited with code {result.returncode}")
    print(result.stderr[-500:])
    sys.exit(1)

# Parse saved JSON output (artifacts/<task>/<model>/detections.json)
try:
    with open("detections.json") as f:
        detections = json.load(f)
    count = len(detections.get("detections", []))
    if count == 0:
        print("FAIL: Zero detections on known-good sample image")
        print("  Possible causes:")
        print("  - Wrong postprocessor for this model family")
        print("  - score_threshold too high in config.json")
        print("  - Postprocessor/model output format mismatch")
        sys.exit(1)
    print(f"PASS: {count} detection(s) found")
except FileNotFoundError:
    # Fallback: check stdout for detection count
    if "detections: 0" in result.stdout.lower() or "no detections" in result.stdout.lower():
        print("FAIL: Zero detections reported in stdout")
        sys.exit(1)
    print("WARN: Could not verify detection count (no JSON output)")
```

### Check 2: Bounding Box Coordinate Validity

```python
"""
Verify that all bounding box coordinates are within the image dimensions.
Catches postprocessor bugs where coordinates are not properly rescaled.
"""
import json, sys

with open("detections.json") as f:
    data = json.load(f)

img_w = data.get("image_width", 640)
img_h = data.get("image_height", 480)
errors = []

for i, det in enumerate(data.get("detections", [])):
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    # Allow small overflow (5%) for letterbox rounding
    margin_w = img_w * 0.05
    margin_h = img_h * 0.05

    if x1 < -margin_w or y1 < -margin_h:
        errors.append(f"Det {i}: negative coords ({x1:.1f}, {y1:.1f})")
    if x2 > img_w + margin_w or y2 > img_h + margin_h:
        errors.append(f"Det {i}: coords exceed image ({x2:.1f}, {y2:.1f}) > ({img_w}, {img_h})")
    if x2 <= x1 or y2 <= y1:
        errors.append(f"Det {i}: inverted bbox ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

if errors:
    print(f"FAIL: {len(errors)} bbox coordinate error(s)")
    for e in errors[:5]:
        print(f"  {e}")
    sys.exit(1)
print("PASS: All bbox coordinates within image bounds")
```

### Check 3: Confidence Score Validity

```python
"""
Verify confidence scores are in the valid range [0.0, 1.0].
"""
import json, sys

with open("detections.json") as f:
    data = json.load(f)

errors = []
for i, det in enumerate(data.get("detections", [])):
    score = det.get("score", det.get("confidence", -1))
    if score < 0.0 or score > 1.0:
        errors.append(f"Det {i}: invalid score {score}")

if errors:
    print(f"FAIL: {len(errors)} invalid confidence score(s)")
    for e in errors[:5]:
        print(f"  {e}")
    sys.exit(1)
print("PASS: All confidence scores in [0.0, 1.0]")
```

### Check 4: Class ID Validity

```python
"""
Verify class IDs are non-negative integers within the expected range.
COCO: 0-79, ImageNet: 0-999, Custom: check model metadata.
"""
import json, sys

with open("detections.json") as f:
    data = json.load(f)

max_class_id = int(sys.argv[1]) if len(sys.argv) > 1 else 79  # Default: COCO 80 classes

errors = []
for i, det in enumerate(data.get("detections", [])):
    cls = det.get("class_id", -1)
    if not isinstance(cls, int) or cls < 0 or cls > max_class_id:
        errors.append(f"Det {i}: invalid class_id {cls} (expected 0-{max_class_id})")

if errors:
    print(f"FAIL: {len(errors)} invalid class ID(s)")
    for e in errors[:5]:
        print(f"  {e}")
    sys.exit(1)
print(f"PASS: All class IDs in valid range [0, {max_class_id}]")
```

### Check 5: Postprocessor-Model Family Cross-Check

```python
"""
Verify the factory's postprocessor matches the model family from model_registry.json.
This catches the critical bug where an agent uses the wrong postprocessor class.
"""
import json, ast, sys

model_name = sys.argv[1]
factory_path = sys.argv[2]  # e.g., factory/yolo26n_factory.py

# Registry key → expected Python postprocessor class
REGISTRY_TO_POSTPROCESSOR = {
    "yolov5": "YOLOv5Postprocessor",
    "yolov8": "YOLOv8Postprocessor",
    "yolov26": "YOLOv8Postprocessor",       # yolo26 reuses YOLOv8 (end-to-end)
    "yolov10": "YOLOv10Postprocessor",
    "yolov11": "YOLOv11Postprocessor",
    "ssd": "SSDPostprocessor",
    "nanodet": "NanoDetPostprocessor",
    "damoyolo": "DamoYoloPostprocessor",
    "classification": "ClassificationPostprocessor",
    "pose": "PosePostprocessor",
    "instance_seg": "InstanceSegPostprocessor",
    "semantic_seg": "SemanticSegPostprocessor",
    "face": "FacePostprocessor",
    "depth": "DepthPostprocessor",
    "restoration": "RestorationPostprocessor",
    "sr": "SRPostprocessor",
    "embedding": "EmbeddingPostprocessor",
    "obb": "OBBPostprocessor",
}

# Load registry
with open("config/model_registry.json") as f:
    registry = json.load(f)
match = [m for m in registry if m["model_name"] == model_name]
if not match:
    print(f"WARN: {model_name} not in registry — skipping cross-check")
    sys.exit(0)

registry_key = match[0].get("postprocessor", "")
expected_class = REGISTRY_TO_POSTPROCESSOR.get(registry_key)
if not expected_class:
    print(f"WARN: Unknown registry postprocessor key '{registry_key}'")
    sys.exit(0)

# Parse factory to find actual postprocessor class
tree = ast.parse(open(factory_path).read())
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            name = alias.asname or alias.name
            if "postprocessor" in name.lower() or "Postprocessor" in name:
                imports.append(name)

if not imports:
    print("FAIL: No postprocessor import found in factory")
    sys.exit(1)

actual_class = imports[0]
if actual_class != expected_class:
    print(f"FAIL: Postprocessor mismatch")
    print(f"  Registry key: '{registry_key}' → expected: {expected_class}")
    print(f"  Factory uses: {actual_class}")
    sys.exit(1)

print(f"PASS: Postprocessor matches — {registry_key} → {actual_class}")
```

### Task-Appropriate Sample Images for Output Validation

| Task | Sample Image | Expected Minimum Detections |
|---|---|---|
| object_detection | `sample/img/sample_dog.jpg` | >= 1 (dog, bicycle, etc.) |
| face_detection | `sample/img/sample_face.jpg` | >= 1 face |
| pose_estimation | `sample/img/sample_people.jpg` | >= 1 person |
| classification | `sample/ILSVRC2012/0.jpeg` | top_k >= 1 |
| instance_segmentation | `sample/img/sample_street.jpg` | >= 1 instance |
| semantic_segmentation | `sample/img/sample_street.jpg` | non-empty mask |
| obb_detection | `sample/dota8_test/P0177.png` | >= 1 oriented bbox |

## Level 5.5: Cross-Validation with Reference Model (NPU Required)

When a precompiled reference DXNN exists in `assets/models/` for the same model,
use it as a known-good baseline for **differential diagnosis** — isolating whether
a failure is in the generated app code or in the compiled model.

> **Skip condition**: If no precompiled DXNN for the same model exists in
> `assets/models/` AND no existing verified example exists in `src/python_example/`,
> skip this level entirely and proceed to Level 6.

### Prerequisite: Check for Reference Assets

```bash
MODEL_NAME="<model_name>"
DX_APP_ROOT="$(cd ../.. && pwd)"  # or appropriate path to dx_app root

# Check 1: Precompiled model in assets/models/
PRECOMPILED="${DX_APP_ROOT}/assets/models/${MODEL_NAME}.dxnn"
if [ -f "$PRECOMPILED" ]; then
    echo "REF MODEL FOUND: $PRECOMPILED"
else
    echo "SKIP: No precompiled reference model for ${MODEL_NAME}"
fi

# Check 2: Existing verified example in src/python_example/
TASK_TYPE="<task_type>"  # e.g., object_detection
EXISTING_APP="${DX_APP_ROOT}/src/python_example/${TASK_TYPE}/${MODEL_NAME}/${MODEL_NAME}_sync.py"
if [ -f "$EXISTING_APP" ]; then
    echo "REF APP FOUND: $EXISTING_APP"
else
    echo "SKIP: No existing verified example for ${MODEL_NAME}"
fi
```

### Test A: Run Generated App with Precompiled Model

If a precompiled DXNN exists for the same model family, run the generated app
with the known-good precompiled model to isolate app code vs compilation issues:

```bash
# Run generated app with the precompiled (known-good) model
python <generated_app>/<model>_sync.py \
    --model assets/models/<model>.dxnn \
    --image <TASK_SAMPLE_IMAGE> --no-display --verbose

# Then run the same generated app with the newly compiled model
python <generated_app>/<model>_sync.py \
    --model <new_model_path>/<model>.dxnn \
    --image <TASK_SAMPLE_IMAGE> --no-display --verbose
```

**Decision tree**:
- **PASS with precompiled, FAIL with new model** → **Compilation problem** (app code is correct; the newly compiled .dxnn is faulty)
- **FAIL with both** → **Generated app code problem** (or environment issue)
- **PASS with both, same results** → Both model and app are correct

### Test B: Compare Against Existing Verified Example

If an existing verified example for the same model exists in
`src/python_example/<task>/<model>/`, use it as a reference implementation:

```bash
# Step 1: Run existing verified example with --verbose (--show-log for C++)
EXISTING_APP="src/python_example/<task>/<model>/<model>_sync.py"
if [ -f "$EXISTING_APP" ]; then
    echo "=== Reference App Output ==="
    python "$EXISTING_APP" \
        --model assets/models/<model>.dxnn \
        --image <TASK_SAMPLE_IMAGE> --no-display --verbose 2>&1 | tee /tmp/ref_output.log

    # Step 2: Run generated app with the SAME precompiled model
    echo "=== Generated App Output ==="
    python <generated_app>/<model>_sync.py \
        --model assets/models/<model>.dxnn \
        --image <TASK_SAMPLE_IMAGE> --no-display --verbose 2>&1 | tee /tmp/gen_output.log

    # Step 3: Compare inference results
    echo "=== Comparing Outputs ==="
    diff <(grep -i "detect\|class\|score\|bbox\|confidence" /tmp/ref_output.log) \
         <(grep -i "detect\|class\|score\|bbox\|confidence" /tmp/gen_output.log) || true
fi
```

### Test C: Cross-Model Swap (Existing App + Generated Model)

Run the existing verified app with the newly compiled model to isolate
compilation-level problems from app-level problems:

```bash
# Existing verified app + generated (new) model
python src/python_example/<task>/<model>/<model>_sync.py \
    --model <new_model_path>/<model>.dxnn \
    --image <TASK_SAMPLE_IMAGE> --no-display --verbose 2>&1 | tee /tmp/existing_with_new.log

# Existing verified app + precompiled (reference) model
python src/python_example/<task>/<model>/<model>_sync.py \
    --model assets/models/<model>.dxnn \
    --image <TASK_SAMPLE_IMAGE> --no-display --verbose 2>&1 | tee /tmp/existing_with_ref.log
```

### Differential Diagnosis Decision Matrix

| Existing App + Precompiled | Existing App + New Model | Generated App + Precompiled | Generated App + New Model | Diagnosis |
|---|---|---|---|---|
| PASS | PASS | PASS | PASS | All correct |
| PASS | PASS | PASS | FAIL | **Generated app + new model interaction bug** |
| PASS | PASS | FAIL | FAIL | **Generated app code problem** |
| PASS | FAIL | PASS | FAIL | **Compilation problem** — new .dxnn is faulty |
| PASS | FAIL | FAIL | FAIL | **Generated app code problem + compilation problem** |
| FAIL | FAIL | FAIL | FAIL | **Environment problem** — NPU, dx_engine, or deps |

### Recovery Actions by Diagnosis

| Diagnosis | Action |
|---|---|
| Generated app code problem | Fix factory postprocessor, config.json thresholds, or preprocessing |
| Compilation problem | Re-check config.json, try different quantization, adjust PPU settings |
| Environment problem | Verify NPU with `dxrt-cli -s`, reinstall dx_engine, check dependencies |
| Interaction bug | Compare config.json between generated and existing app |

## Level 6: Integration Test (Full Pipeline)

```bash
# Video with save + loop
python <model>_async.py \
    --model /path/to/<model>.dxnn \
    --video test.mp4 \
    --no-display \
    --save \
    --loop 2 \
    --verbose

# Verify output was saved
ls -la artifacts/python_example/*/output.*
```

## validate_app.py (Automated)

Combined validation script that runs all Level 1-3 checks (no NPU required):

```bash
# Run from model directory
python -c "
import sys, os, json, ast, re

model_dir = '.'
errors = []
warnings = []

# Static checks
py_files = [f for f in os.listdir(model_dir)
            if f.endswith('.py') and f != '__init__.py']

for pyf in py_files:
    path = os.path.join(model_dir, pyf)
    # Syntax
    try:
        compile(open(path).read(), path, 'exec')
    except SyntaxError as e:
        errors.append(f'Syntax error in {pyf}: {e}')
    # No hardcoded paths
    content = open(path).read()
    if re.search(r'[\"\'][/\\\\].*\\.dxnn[\"\']', content):
        errors.append(f'Hardcoded model path in {pyf}')

# Factory checks
factory_dir = os.path.join(model_dir, 'factory')
if not os.path.isdir(factory_dir):
    errors.append('Missing factory/ directory')
else:
    factory_files = [f for f in os.listdir(factory_dir)
                     if f.endswith('.py') and f != '__init__.py']
    for ff in factory_files:
        tree = ast.parse(open(os.path.join(factory_dir, ff)).read())
        methods = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}
        required = {'create_preprocessor', 'create_postprocessor',
                    'create_visualizer', 'get_model_name', 'get_task_type'}
        missing = required - methods
        if missing:
            errors.append(f'Factory {ff} missing methods: {missing}')

# Config check
config_path = os.path.join(model_dir, 'config.json')
if os.path.isfile(config_path):
    try:
        json.load(open(config_path))
    except json.JSONDecodeError as e:
        errors.append(f'Invalid config.json: {e}')
else:
    warnings.append('No config.json found')

# __init__.py checks
if not os.path.isfile(os.path.join(model_dir, '__init__.py')):
    warnings.append('Missing __init__.py')
if factory_dir and not os.path.isfile(os.path.join(factory_dir, '__init__.py')):
    warnings.append('Missing factory/__init__.py')

# Report
if errors:
    print(f'FAIL: {len(errors)} error(s)')
    for e in errors:
        print(f'  ERROR: {e}')
    sys.exit(1)
if warnings:
    print(f'PASS with {len(warnings)} warning(s)')
    for w in warnings:
        print(f'  WARN: {w}')
else:
    print('PASS: All checks passed')
"
```

## Common Failures and Fixes

| Failure | Cause | Fix |
|---|---|---|
| `SyntaxError` in factory | Wrong Python version or typo | Check Python 3.8+ syntax |
| `ModuleNotFoundError: common` | sys.path not configured | Add 2-parent sys.path pattern |
| `ModuleNotFoundError: dx_engine` | DX-RT not installed | Install dx_engine package |
| `ModuleNotFoundError: dx_postprocess` | Bindings not built | Run `./build.sh` |
| `TypeError: Can't instantiate abstract class` | Missing factory method | Implement all 5 methods |
| `json.JSONDecodeError` | Malformed config.json | Fix JSON syntax |
| `FileNotFoundError: .dxnn` | Model not downloaded | Run `./setup.sh` |
| `RuntimeError: NPU not found` | No NPU hardware | Skip with `@requires_npu` |
| Hardcoded model path | Path embedded in source | Use `args.model` from CLI |
| Relative imports | `from ..common import` | Use `from common.base import` |
| Empty factory/__init__.py | Missing import line | Add `from .<model>_factory import` |
| Zero detections on known-good image | Wrong postprocessor or threshold too high | Verify registry key → postprocessor mapping (see Level 5 Check 5) |
| Bbox coordinates outside image | Postprocessor not rescaling letterbox | Verify letterbox inverse transform in postprocessor |
| Garbled detections with PPU model | Using standard YOLO postprocessor on PPU output | Use `PPUPostProcess` for PPU models |
| `yolov26` model, wrong postprocessor | Agent generates `Yolo26Postprocessor` (doesn't exist) | Use `YOLOv8Postprocessor` — yolo26 reuses YOLOv8 end-to-end format |
