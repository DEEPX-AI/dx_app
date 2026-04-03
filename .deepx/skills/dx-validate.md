# Skill: Validation for dx_app

> Validate dx_app applications at every phase gate: static analysis,
> configuration checks, smoke tests, and full integration tests.

## 5-Level Validation Pyramid

```
Level 5: Integration  (NPU + model + full pipeline)
Level 4: Smoke        (NPU + model + quick single-frame inference)
Level 3: Component    (preprocessor/postprocessor/visualizer individually)
Level 2: Config       (JSON validity, schema compliance)
Level 1: Static       (syntax, imports, factory interface)
```

Levels 1-3 can run without NPU hardware.
Levels 4-5 require NPU + model files.

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

## Level 5: Integration Test (Full Pipeline)

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

Combined validation script that runs all Level 1-2 checks:

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
