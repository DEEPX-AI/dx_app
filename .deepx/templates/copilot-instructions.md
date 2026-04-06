# dx_app — GitHub Copilot Instructions

> Auto-generated from `.deepx/`. Do not edit directly.
> Regenerate with: `python .deepx/scripts/generate_platforms.py --generate --platform copilot`

## Response Language

Match your response language to the user's prompt language — when asking questions
or responding, use the same language the user is using.

**Technical term rule**: When responding in Korean, keep English technical terms in
their original English form. Do NOT transliterate English terms into Korean phonetics
(한글 음차 표기 금지). Established Korean loanwords (모델, 서버, 파일, 데이터) are acceptable.

## Recommended Model (Pre-Flight Check)

DX Agentic Development requires strong instruction following and knowledge base
utilization. Before starting any task, check if you are running as one of the
recommended models:

- **Claude Sonnet 4.6** or later
- **Claude Opus 4.6** or later

If you are NOT one of these models (e.g., GPT-4.1, GPT-4o, Gemini, etc.),
output the following notice at the beginning of your first response:

```
══════════════════════════════════════════════════════════════
⚠  DX-AGENTIC-DEV: MODEL NOTICE

DX Agentic Development recommends Claude Sonnet 4.6+ or
Opus 4.6+. Your current model may produce lower quality
in instruction following, knowledge base utilization, and
API accuracy. Please switch to a recommended model.
══════════════════════════════════════════════════════════════
```

After the box notice, proceed with the task normally. This is an informational
warning, not a blocking gate.

### Model Notice Repeat (Non-Recommended Model Only)

If you output the MODEL NOTICE box above (i.e., you are NOT a recommended model),
you MUST also output this shortened reminder **immediately before the actual question
line** that the user needs to answer — NOT at the beginning of the brainstorming flow.

**Timing**: After all file reading, context analysis, and preamble text, right before
the line that contains the `?` (the actual question), insert this reminder:

```
---
⚠ **Non-recommended model** — output quality may be degraded. Recommended: Claude Sonnet 4.6+ / Opus 4.6+
---
```

**Example — WRONG** (repeat scrolls past with the box):
```
[DX-AGENTIC-DEV: START]
══ MODEL NOTICE ══
---  ⚠ Non-recommended model ---     ← TOO EARLY, scrolls past
... (reads files, analyzes context) ...
First question: ...?
```

**Example — CORRECT** (repeat appears right before the question):
```
[DX-AGENTIC-DEV: START]
══ MODEL NOTICE ══
... (reads files, analyzes context) ...
---  ⚠ Non-recommended model ---     ← RIGHT BEFORE the question
First question: ...?
```

Only output this reminder ONCE (before the first question), not before every question.

## Overview

dx_app is a DEEPX standalone inference application framework. It provides 133 compiled
`.dxnn` models across 15 AI tasks with Python (4 variants) and C++ examples.

## Context Routing Table

{ROUTING_TABLE}

## Skills

{SKILLS_TABLE}

## Hardware

{HARDWARE_TABLE}

## Quick Reference

```bash
./build.sh      # Build C++ and pybind11 postprocess bindings
./setup.sh      # Download models and test media
```

## Critical Conventions

1. **IFactory pattern**: All Python model apps must implement IFactory with 5 methods:
   `create_preprocessor()`, `create_postprocessor()`, `create_visualizer()`,
   `get_model_name()`, `get_task_type()`.

2. **parse_common_args()**: All app scripts use `parse_common_args()` from
   `common.runner`. Never define custom `argparse.ArgumentParser`.

3. **No hardcoded paths**: Model path comes from `--model` CLI argument.
   Never embed `.dxnn` file paths in source code.

4. **model_registry.json**: Always query `config/model_registry.json` to verify
   a model exists before creating an application.

5. **Logging**: Use `logging.getLogger(__name__)`, not `print()`.

6. **sys.path pattern**: Python scripts use the 2-level parent path insertion:
   ```python
   _module_dir = Path(__file__).parent
   _v3_dir = _module_dir.parent.parent
   ```

7. **4-variant naming**: `<model>_sync.py`, `<model>_async.py`,
   `<model>_sync_cpp_postprocess.py`, `<model>_async_cpp_postprocess.py`.

8. **C++14**: C++ examples use C++14 standard only. No C++17 features.

9. **SIGINT handler**: All C++ loop-based examples must install a SIGINT handler.

10. **RAII**: C++ code uses `std::unique_ptr`, no raw `new`/`delete`.

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

## Directory Structure

```
src/python_example/<task>/<model>/
    __init__.py
    config.json
    factory/
        __init__.py
        <model>_factory.py
    <model>_sync.py
    <model>_async.py
```

## 15 Supported AI Tasks

object_detection, classification, pose_estimation, instance_segmentation,
semantic_segmentation, face_detection, depth_estimation, image_denoising,
image_enhancement, super_resolution, embedding, obb_detection, hand_landmark, ppu
