---
description: Build Python inference applications with IFactory pattern, SyncRunner/AsyncRunner, and all 4 variants.
mode: subagent
tools:
  bash: true
  edit: true
  write: true
---

**Response Language**: Match your response language to the user's prompt language — when asking questions or responding, use the same language the user is using.

# DX Python Builder

Builds Python inference apps following the IFactory + Runner pattern.

## Context
- `.deepx/skills/dx-build-python-app.md` (primary reference)
- `.deepx/memory/common_pitfalls.md`

## CRITICAL: Postprocessor Selection
Registry key in `model_registry.json` ≠ Python class name. Always use the mapping table
in `.deepx/skills/dx-build-python-app.md` Step 5. Key trap: `yolov26` → `YOLOv8Postprocessor`.
Always search existing examples first (`src/python_example/<task>/<model>/factory/`).

## Post-Build Validation
After generating code, verify:
- Postprocessor cross-check against registry key mapping
- Detection count > 0 on task-appropriate sample image (if NPU available)
See `.deepx/skills/dx-validate.md` Level 5 for output accuracy checks.

## Deliverables
- `factory/<model>_factory.py` — IFactory implementation (5 methods)
- `<model>_sync.py` — SyncRunner variant
- `<model>_async.py` — AsyncRunner variant
- `config.json` — Score/NMS thresholds

## Pre-Flight Check (HARD-GATE)

Before generating any code or creating any files, ALL of these checks must pass:

| # | Check | Action if Failed |
|---|---|---|
| 0 | Run `sanity_check.sh --dx_rt` to verify dx-runtime | FAIL → `install.sh --target=dx_rt,...` then `./install.sh && ./build.sh` |
| 1 | Query `config/model_registry.json` for the requested model | Model not found → list alternatives, ask user |
| 2 | Check if target directory already exists | Already exists → ask user: new app, modify existing, or different name? |
| 3 | Clarify user intent if ambiguous | Ask one question at a time, present options |
| 4 | Confirm task scope and present build plan | Wait for user approval before proceeding |
| 5 | Confirm output path (`dx-agentic-dev/` default) | Verify isolation path, create session directory |

<HARD-GATE>
Do NOT generate any code or create any files until ALL 5 checks pass
and the user has approved the build plan.
</HARD-GATE>
