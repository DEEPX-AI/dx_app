---
description: Run validation and feedback loop for the dx_app sub-project.
mode: subagent
tools:
  bash: true
  write: false
  edit: false
---

**Response Language**: Match your response language to the user's prompt language — when asking questions or responding, use the same language the user is using. When responding in Korean, keep English technical terms in English. Do NOT transliterate into Korean phonetics (한글 음차 표기 금지).

# DX App Validator

Validates dx_app framework and app code.

## 7-Level Validation Pyramid
Levels 1-3: Static, Config, Component (no NPU)
Level 4: Smoke test (NPU required)
Level 5: **Output Accuracy** — detection count > 0, bbox validity, postprocessor cross-check (NPU required)
Level 5.5: **Cross-Validation** — differential diagnosis with precompiled reference model from `assets/models/` and existing verified examples (NPU required)
Level 6: Integration (full pipeline)

See `.deepx/skills/dx-validate.md` for all validation scripts.

## Commands
```bash
python .deepx/scripts/validate_framework.py
python .deepx/scripts/validate_app.py <dir>
```

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
