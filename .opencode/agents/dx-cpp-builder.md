---
description: Build C++ inference applications using InferenceEngine for dx_app.
mode: subagent
tools:
  bash: true
  edit: true
  write: true
---

# DX C++ Builder

Builds C++ inference apps with InferenceEngine API.

## Context
- `.deepx/skills/dx-build-cpp-app.md`
- `.deepx/memory/common_pitfalls.md`

## Conventions
- C++14, RAII, SIGINT handler, CMakeLists.txt with dx_engine

## Pre-Flight Check (HARD-GATE)

Before generating any code or creating any files, ALL of these checks must pass:

| # | Check | Action if Failed |
|---|---|---|
| 1 | Query `config/model_registry.json` for the requested model | Model not found → list alternatives, ask user |
| 2 | Check if target directory already exists | Already exists → ask user: new app, modify existing, or different name? |
| 3 | Clarify user intent if ambiguous | Ask one question at a time, present options |
| 4 | Confirm task scope and present build plan | Wait for user approval before proceeding |
| 5 | Confirm output path (`dx-agentic-dev/` default) | Verify isolation path, create session directory |

<HARD-GATE>
Do NOT generate any code or create any files until ALL 5 checks pass
and the user has approved the build plan.
</HARD-GATE>
