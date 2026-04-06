# Orchestration Guide for dx_app

How to coordinate multi-agent builds, manage sub-agent contracts, and avoid
common anti-patterns when building dx_app applications.

## 5-Phase Build Lifecycle

Every dx_app build follows this lifecycle, regardless of complexity:

```
Phase 1: DISCOVER    Understand task, model, variant requirements
    |
Phase 2: PLAN       Present file list, factory choice, config to user
    |
Phase 3: BUILD      Create files in dependency order
    |
Phase 4: VALIDATE   Run syntax, schema, and smoke tests
    |
Phase 5: REPORT     Summarize files created and run commands
```

### Phase 1: DISCOVER

Gather information:
- Read `config/model_registry.json` for model metadata
- Check if `src/python_example/<task>/<model>/` already exists
- Identify the IFactory interface (from `.deepx/instructions/factory-pattern.md`)
- Determine which preprocessor/postprocessor/visualizer to use

Exit criteria: Agent knows task, model, variant(s), and all component classes.

### Phase 2: PLAN

Present the plan:
```
Files to create:
  src/python_example/object_detection/yolov8n/
    factory/__init__.py
    factory/yolov8n_factory.py
    yolov8n_sync.py
    yolov8n_async.py
    config.json
    __init__.py

Factory: Yolov8Factory(IDetectionFactory)
Components: LetterboxPreprocessor, YOLOv8Postprocessor, DetectionVisualizer
Config: score_threshold=0.25, nms_threshold=0.45
```

Exit criteria: User confirms or modifies the plan.

### Phase 3: BUILD

Create files in dependency order:
1. `__init__.py` (model directory)
2. `factory/__init__.py`
3. `factory/<model>_factory.py` (factory must exist before variants)
4. `config.json`
5. `<model>_sync.py`
6. `<model>_async.py`
7. `<model>_sync_cpp_postprocess.py` (if requested)
8. `<model>_async_cpp_postprocess.py` (if requested)

Exit criteria: All files created.

### Phase 4: VALIDATE

Run validation checks:
1. `py_compile` on each .py file
2. JSON validation on config.json
3. Import test for the factory
4. Smoke test if NPU is available

Exit criteria: All checks pass.

### Phase 5: REPORT

Present results:
- List of created files with line counts
- Run commands for each variant
- Any warnings or notes

## Sub-Agent Contract Template

When the master router (dx-app-builder) delegates to a specialist:

```yaml
contract:
  task: "Build Python sync+async app"
  model: "yolov8n"
  task_type: "object_detection"
  variants: ["sync", "async"]
  target_dir: "src/python_example/object_detection/yolov8n/"
  factory_interface: "IDetectionFactory"
  config:
    score_threshold: 0.25
    nms_threshold: 0.45
  constraints:
    - "Do not modify files outside target_dir"
    - "Do not modify common/ framework"
    - "Follow coding-standards.md"
    - "Use parse_common_args() for CLI"
  deliverables:
    - "factory/yolov8n_factory.py"
    - "yolov8n_sync.py"
    - "yolov8n_async.py"
    - "config.json"
  validation:
    - "py_compile on all .py files"
    - "json.load on config.json"
```

## 6 Sub-Agent Types

### 1. Python Builder (`dx-python-builder`)
- Creates factory + sync/async variants
- Handles all 4 Python variant types
- Validates against IFactory interface

### 2. C++ Builder (`dx-cpp-builder`)
- Creates CMakeLists.txt + main.cpp + factory.hpp
- Handles sync and async C++ variants
- Validates CMake and compile checks

### 3. Benchmark Builder (`dx-benchmark-builder`)
- Profiles existing applications
- Compares variants (sync vs async, Python vs C++ postprocess)
- Recommends optimizations

### 4. Model Manager (`dx-model-manager`)
- Queries model_registry.json
- Manages model downloads via setup.sh
- Validates .dxnn compatibility

### 5. Validator (`dx-validate`)
- Runs static checks (imports, factory interface)
- Runs smoke tests (quick inference)
- Runs integration tests (full pipeline)

### 6. Config Manager
- Creates and validates config.json
- Updates model_registry.json entries
- Manages threshold tuning

## 7 Anti-Patterns

### 1. Framework Modification

**Anti-pattern**: Modifying `common/base/i_factory.py` to add a method for
one model's needs.

**Fix**: Create a model-specific mixin in the factory directory, not in the
shared framework.

### 2. Hardcoded Model Paths

**Anti-pattern**: Embedding `/home/user/models/yolov8n.dxnn` in source code.

**Fix**: Always use `parse_common_args()` which provides `args.model`.

### 3. Custom Argparse

**Anti-pattern**: Defining a new `argparse.ArgumentParser` in a model script.

**Fix**: Use `parse_common_args(description)`. If you need additional flags,
discuss adding them to the shared parser (Protocol 7 — no side effects).

### 4. Skipping Validation

**Anti-pattern**: Creating all files and declaring success without any checks.

**Fix**: Validate after each file (Protocol 4). Minimum: py_compile + JSON check.

### 5. Wrong Factory Interface

**Anti-pattern**: Using `IDetectionFactory` for a segmentation model because
"they both have the same methods."

**Fix**: Use the correct specialized interface. It matters for documentation,
type checking, and future interface evolution.

### 6. Monolithic Generation

**Anti-pattern**: Generating all files in one giant code block without
intermediate validation.

**Fix**: Follow the BUILD phase order. Create factory first, validate,
create variants, validate each.

### 7. Ignoring Existing Examples

**Anti-pattern**: Inventing a new file structure or naming convention.

**Fix**: Always look at existing examples for the same task type first
(Protocol 5). The codebase has established patterns — follow them.

## Inter-Agent Communication

When agents need to coordinate (e.g., Python builder needs model info
from Model Manager):

1. **Model Manager provides**: model_name, dxnn_file, input dimensions,
   task type, postprocessor key, default config.
2. **Python Builder receives**: uses this to select IFactory interface,
   preprocessor/postprocessor classes, and config.json defaults.
3. **Validator receives**: file list from builder, runs checks.

Data flows through the master router (dx-app-builder), not directly
between specialists.

## Parallel vs Sequential Tasks

### Can Run in Parallel
- Creating sync and async variants (both depend on factory, not each other)
- Running py_compile on independent files
- Querying model_registry.json and checking directory existence

### Must Run Sequentially
- Factory BEFORE variants (variants import from factory)
- `__init__.py` BEFORE factory module (package must exist)
- BUILD BEFORE VALIDATE (files must exist to validate)
- config.json BEFORE smoke test (runner loads config)
