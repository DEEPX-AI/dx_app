# dx_app — Copilot Global Instructions

> DXNN Application framework for standalone inference on DEEPX NPU accelerators.

## Response Language

Match your response language to the user's prompt language — when asking questions
or responding, use the same language the user is using.

**Technical term rule**: When responding in Korean, keep English technical terms in
their original English form. Do NOT transliterate English terms into Korean phonetics
(한글 음차 표기 금지). Established Korean loanwords (모델, 서버, 파일, 데이터) are acceptable.

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

### Mandatory Questions for App Building (HARD-GATE)

Before ANY code generation, the agent MUST ask and receive answers to:
1. **Language/variant**: Python (sync/async/cpp_postprocess/async_cpp_postprocess) or C++?
2. **AI task**: detection, classification, segmentation, pose, etc.
3. **Model**: Specific model name or auto-recommend

These questions MUST NOT be skipped even if the prompt seems clear.

### Agent Routing (MANDATORY)

**All app building requests MUST go through `@dx-app-builder`** (the master router).
Do NOT invoke `@dx-python-builder`, `@dx-cpp-builder`, or other specialist agents directly.
`@dx-app-builder` enforces mandatory brainstorming questions (Q1: language/variant,
Q2: AI task, Q3: model) that specialist agents skip.

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
9. **Cross-validation with reference model**: When a precompiled DXNN exists in `assets/models/` or an existing verified example exists in `src/python_example/`, run differential diagnosis (Level 5.5) to isolate app code vs compilation issues. Run generated app with precompiled model, compare against existing example with `--verbose`/`--show-log`. See `.deepx/skills/dx-validate.md` Level 5.5.

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

## Git Safety — Superpowers Artifacts

**NEVER `git add` or `git commit` files under `docs/superpowers/`.** These are temporary
planning artifacts generated by the superpowers skill system (specs, plans). They are
`.gitignore`d, but some tools may bypass `.gitignore` with `git add -f`. Creating the
files is fine — committing them is forbidden.

## Session Sentinels (MANDATORY for Automated Testing)

When processing a user prompt, output these exact markers for automated session
boundary detection by the test harness:

- **First line of your response**: `[DX-AGENTIC-DEV: START]`
- **Last line after ALL work is complete**: `[DX-AGENTIC-DEV: DONE (output-dir: <relative_path>)]`
  where `<relative_path>` is the session output directory (e.g., `dx-agentic-dev/20260409-143022_yolo26n_detection/`)

Rules:
1. **CRITICAL — Output `[DX-AGENTIC-DEV: START]` as the absolute first line of your
   first response.** This must appear before ANY other text, tool calls, or reasoning.
   Even if the user instructs you to "just proceed" or "use your own judgment",
   the START sentinel is non-negotiable — automated tests WILL fail without it.
2. Output `[DX-AGENTIC-DEV: DONE (output-dir: <path>)]` as the very last line after all work, validation,
   and file generation is complete
3. If you are a **sub-agent** invoked via handoff/routing from a higher-level agent,
   do NOT output these sentinels — only the top-level agent outputs them
4. If the user sends multiple prompts in a session, output START/DONE for each prompt
5. The `output-dir` in DONE must be the relative path from the project root to the
   session output directory. If no files were generated, omit the `(output-dir: ...)` part.
6. **NEVER output DONE after only producing planning artifacts** (specs, plans, design
   documents). DONE means all deliverables are produced — implementation code, scripts,
   configs, and validation results. If you completed a brainstorming or planning phase
   but have not yet implemented the actual code, do NOT output DONE. Instead, proceed
   to implementation or ask the user how to proceed.
7. **Pre-DONE mandatory deliverable check**: Before outputting DONE, verify that all
   mandatory deliverables exist in the session directory. If any mandatory file is
   missing, create it before outputting DONE. Each sub-project defines its own mandatory
   file list in its skill document (e.g., `dx-build-pipeline-app.md` File Creation Checklist).
8. **Session HTML export guidance**: Immediately before the DONE sentinel line, output:
   `To save this session as HTML, type: /share html` — this tells the user they can
   preserve the full conversation. The test harness (`test.sh`) will automatically
   detect and copy the exported HTML file to the session output directory.
