# dx_app — Copilot Global Instructions

> DXNN Application framework for standalone inference on DEEPX NPU accelerators.

## Overview

dx_app provides 133 compiled `.dxnn` models across 15 AI tasks with Python (4 variants) and C++ examples.

## Context Routing Table

| If the task mentions... | Read these files |
|---|---|
| **Python app, detection, classification** | `.deepx/skills/dx-build-python-app.md`, `.deepx/toolsets/common-framework-api.md` |
| **C++ app, native** | `.deepx/skills/dx-build-cpp-app.md`, `.deepx/toolsets/dx-engine-api.md` |
| **Async, performance, throughput** | `.deepx/skills/dx-build-async-app.md`, `.deepx/memory/performance_patterns.md` |
| **Model, download, registry** | `.deepx/skills/dx-model-management.md`, `.deepx/toolsets/model-registry.md` |
| **Validation, testing** | `.deepx/skills/dx-validate.md`, `.deepx/instructions/testing-patterns.md` |
| **ALWAYS read (every task)** | `.deepx/memory/common_pitfalls.md`, `.deepx/instructions/coding-standards.md` |

## Skills

| Skill | Description |
|-------|-------------|
| dx-build-python-app | Build Python inference app (sync, async, cpp_postprocess, async_cpp_postprocess) |
| dx-build-cpp-app | Build C++ inference app with InferenceEngine |
| dx-build-async-app | Build async high-performance inference app |
| dx-model-management | Download, register, and configure .dxnn models |
| dx-validate | Run validation checks at every phase gate |
| dx-brainstorm-and-plan | Process: collaborative design session before code generation |
| dx-tdd | Process: test-driven development — validate each file immediately after creation |
| dx-verify-completion | Process: verify before claiming completion — evidence before assertions |

## Interactive Workflow (MUST FOLLOW)

**Always walk through key decisions with the user before building.** This is a HARD GATE.

Before ANY code generation:
1. Ask 2-3 clarifying questions (variant, task type, model)
2. Present a build plan and wait for user approval
3. After generation, validate each file

### Output Isolation
All AI-generated code goes to `dx-agentic-dev/<session_id>/` by default.
Only write to `src/` when explicitly requested by the user.

## Critical Conventions

1. **IFactory pattern**: All Python apps implement IFactory with 5 methods
2. **parse_common_args()**: All app scripts use it — never custom argparse
3. **No hardcoded paths**: Model path from `--model` CLI arg
4. **model_registry.json**: Always query before creating an app
5. **Logging**: `logging.getLogger(__name__)` — not `print()`
6. **4-variant naming**: `<model>_sync.py`, `<model>_async.py`, `<model>_sync_cpp_postprocess.py`, `<model>_async_cpp_postprocess.py`
7. **C++14**: C++ examples use C++14 standard only
8. **RAII**: C++ code uses `std::unique_ptr`, no raw `new`/`delete`

## Quick Reference

```bash
./install.sh && ./build.sh   # Build C++ and pybind11 bindings
./setup.sh                   # Download models and test media
dxrt-cli -s                  # Verify NPU availability
pytest tests/ -m "not npu_required"  # Run unit tests
```

## Python Import Pattern

```python
import sys
from pathlib import Path
_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from factory import ModelFactory
from common.runner import SyncRunner, parse_common_args
```

## Hardware

| Architecture | Value |
|---|---|
| DX-M1 | `dx_m1` |
| DX-M1A | `dx_m1a` |

## 15 Supported AI Tasks

object_detection, classification, pose_estimation, instance_segmentation,
semantic_segmentation, face_detection, depth_estimation, image_denoising,
image_enhancement, super_resolution, embedding, obb_detection, hand_landmark, ppu
