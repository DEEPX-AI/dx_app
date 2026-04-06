# Prompt: Orchestrated dx_app Build

> Meta-template for multi-agent orchestrated builds. Coordinates multiple specialist
> agents across 5 phases to build a complete dx_app application.

## Overview

This template orchestrates a full dx_app build when the request spans multiple
components (e.g., Python + C++ variants, multiple models, or a model + custom
postprocessor). It coordinates the DX App Builder router with specialist agents.

## 5 Phases

### Phase 1: Discovery

**Agent:** DX App Builder (router)

1. Parse the user's request to identify:
   - Target AI task(s)
   - Model(s) to support
   - Language(s): Python, C++, or both
   - Variants: which of the 4 Python variants
   - Special requirements: custom postprocessor, RTSP input, batch mode
2. Query `config/model_registry.json` for each model
3. Check for existing implementations that might conflict

**Output:** Build specification document:
```yaml
build_spec:
  task: object_detection
  models:
    - name: yolo26n
      variants: [sync, async, sync_cpp_postprocess, async_cpp_postprocess]
      languages: [python, cpp]
    - name: yolo26s
      variants: [sync, async]
      languages: [python]
  input_source: usb
  custom_postprocess: false
  thresholds:
    score: 0.25
    nms: 0.45
```

### Phase 2: Parallel Build

**Agents:** DX Python Builder + DX C++ Builder (parallel)

Dispatch independent builds to specialist agents:

| Agent | Task | Dependencies |
|---|---|---|
| DX Python Builder | Build all Python variants for each model | model_registry.json |
| DX C++ Builder | Build C++ apps for each model | model_registry.json |

**Parallelism rules:**
- Different models can be built in parallel
- Python and C++ for the same model can be built in parallel
- Factory must be created before variant scripts (sequential within model)

### Phase 3: Integration

**Agent:** DX App Builder (router)

1. Verify all generated files exist and are syntactically correct
2. Check cross-references:
   - Factory imports resolve
   - config.json is shared correctly across variants
   - Model names match between registry, factory, and scripts
3. Create any shared infrastructure:
   - Labels file symlinks
   - Shared config directory entries

### Phase 4: Validation

**Agent:** DX App Builder (router)

Run validation suite:
```bash
# Static checks
python .deepx/scripts/validate_app.py src/python_example/<task>/<model>/

# Smoke tests (if NPU available)
python .deepx/scripts/validate_app.py src/python_example/<task>/<model>/ --smoke-test

# Framework integrity
python .deepx/scripts/validate_framework.py
```

Validation checklist:
- [ ] All .py files pass syntax check
- [ ] All config.json files are valid JSON
- [ ] All factories implement 5 required methods
- [ ] All variant scripts use parse_common_args()
- [ ] No hardcoded model paths
- [ ] No relative imports
- [ ] SIGINT handlers present in C++ loop examples
- [ ] CMakeLists.txt compiles without errors

### Phase 5: Report

**Agent:** DX App Builder (router)

Present final summary:

```
Build Complete
==============

Models built: 2 (yolo26n, yolo26s)
Python variants: 6 files
C++ apps: 2 files
Config files: 2 files

File tree:
  src/python_example/object_detection/
    yolo26n/
      factory/__init__.py
      factory/yolo26n_factory.py
      yolo26n_sync.py
      yolo26n_async.py
      yolo26n_sync_cpp_postprocess.py
      yolo26n_async_cpp_postprocess.py
      config.json
      __init__.py
    yolo26s/
      factory/__init__.py
      factory/yolo26s_factory.py
      yolo26s_sync.py
      yolo26s_async.py
      config.json
      __init__.py

  src/cpp_example/object_detection/
    yolo26n/
      CMakeLists.txt
      yolo26n_sync.cpp
      factory/yolo26n_factory.hpp
      config.json

Run:
  python src/python_example/object_detection/yolo26n/yolo26n_sync.py \
    --model models/yolo26n.dxnn --input test.jpg
```

## Error Recovery

| Phase | Failure | Recovery |
|---|---|---|
| Discovery | Model not in registry | Suggest closest match, ask user |
| Build | Import resolution failure | Check common/ module structure |
| Integration | Cross-reference mismatch | Regenerate the mismatched file |
| Validation | Syntax error | Fix and re-validate |
| Validation | Missing IFactory method | Add stub and warn user |

## When to Use This Template

- Building for 2+ models simultaneously
- Building both Python and C++ for the same model
- Building all 4 Python variants
- Any build that involves custom postprocessor creation
- When the user says "build everything for model X"
