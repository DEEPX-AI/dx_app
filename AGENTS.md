# dx_app — Claude Code Entry Point

> Self-contained entry point for dx_app standalone inference development.

## Shared Knowledge

All skills, instructions, toolsets, and memory live in `.deepx/`.
Read `.deepx/README.md` for the complete index.

## Quick Reference

```bash
./install.sh && ./build.sh          # Build C++ and pybind11 bindings
./setup.sh                          # Download models and test media
dxrt-cli -s                         # Verify NPU availability
pytest tests/ -m "not npu_required" # Run unit tests (no NPU)
pytest tests/ -m npu_required       # Run NPU integration tests
```

## Skills

| Command | Description |
|---------|-------------|
| /dx-build-python-app | Build Python inference app (sync, async, cpp_postprocess, async_cpp_postprocess) |
| /dx-build-cpp-app | Build C++ inference app with InferenceEngine |
| /dx-build-async-app | Build async high-performance inference app |
| /dx-model-management | Download, register, and configure .dxnn models |
| /dx-validate | Run validation checks at every phase gate |
| /dx-brainstorm-and-plan | Brainstorm and plan before any code generation |
| /dx-tdd | Test-driven development — validate each file immediately after creation |
| /dx-verify-completion | Verify before claiming completion — evidence before assertions |

## Critical Conventions

1. **Absolute imports**: `from dx_app.src.python_example.common.xyz import ...`
2. **Model resolution**: Query `config/model_registry.json` — never hardcode .dxnn paths
3. **Factory pattern**: All apps implement IFactory with 5 methods (`create_preprocessor`, `create_postprocessor`, `create_visualizer`, `get_model_name`, `get_task_type`)
4. **CLI args**: Use `parse_common_args()` from `common/runner/args.py`
5. **NPU check**: `dxrt-cli -s` before any inference operation
6. **Logging**: `logging.getLogger(__name__)` — no bare `print()`
7. **Skill doc is sufficient**: Do NOT read source code unless skill is insufficient
8. **No relative imports**: Always use absolute imports from the package root
9. **No hardcoded model paths**: All model paths from CLI args or model_registry.json
10. **4 variants**: Python apps have sync, async, sync_cpp_postprocess, async_cpp_postprocess
11. **PPU model auto-detection**: Auto-detect PPU models by checking model name `_ppu` suffix, `model_registry.json` `csv_task: "PPU"`, or compiler session context. PPU models go under `src/python_example/ppu/` with simplified postprocessing.
12. **Existing example search**: Before generating code, search `src/python_example/<task>/<model>/` for existing examples. If found, ask user: (a) explain-only, or (b) create new based on existing. Never silently skip or overwrite.
13. **PPU example generation is MANDATORY**: If the compiled .dxnn model is PPU, the agent MUST generate a working example — never skip example generation for PPU models.

## Context Routing Table

| Task mentions... | Read these files |
|---|---|
| **Python app, detection, classification** | `.deepx/skills/dx-build-python-app.md`, `.deepx/toolsets/common-framework-api.md` |
| **C++ app, native** | `.deepx/skills/dx-build-cpp-app.md`, `.deepx/toolsets/dx-engine-api.md` |
| **Async, performance, throughput** | `.deepx/skills/dx-build-async-app.md`, `.deepx/memory/performance_patterns.md` |
| **Model, download, registry** | `.deepx/skills/dx-model-management.md`, `.deepx/toolsets/model-registry.md` |
| **Validation, testing** | `.deepx/skills/dx-validate.md`, `.deepx/instructions/testing-patterns.md` |
| **Validation, feedback, fix** | `.deepx/skills/dx-validate.md` (cross-project: `dx-runtime/.deepx/skills/dx-validate-and-fix.md`) |
| **ALWAYS read (every task)** | `.deepx/memory/common_pitfalls.md`, `.deepx/instructions/coding-standards.md` |

## Python Imports

```python
from dx_app.src.python_example.common.runner.args import parse_common_args
from dx_app.src.python_example.common.runner.factory_runner import FactoryRunner
from dx_app.src.python_example.common.utils.model_utils import load_model_config
import logging

logger = logging.getLogger(__name__)
```

## File Structure

```
src/python_example/{task}/{model}/
├── __init__.py
├── config.json
├── {model}_factory.py
├── {model}_sync.py
├── {model}_async.py
├── {model}_sync_cpp_postprocess.py
└── {model}_async_cpp_postprocess.py

src/cpp_example/{task}/{model}/
├── CMakeLists.txt
├── main.cpp
├── config.json
├── include/
└── src/
```

## Hardware

| Architecture | Value |
|---|---|
| DX-M1 | `dx_m1` |
| DX-M1A | `dx_m1a` |

## Memory

Persistent knowledge in `.deepx/memory/`. Read at task start, update when learning.
