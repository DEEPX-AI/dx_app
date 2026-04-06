---
description: "(Sub-agent) Download, register, and query .dxnn models — invoked only via @dx-app-builder handoff. Do NOT invoke directly."
mode: subagent
tools:
  bash: true
  write: true
  edit: true
---

**Response Language**: Match your response language to the user's prompt language — when asking questions or responding, use the same language the user is using. When responding in Korean, keep English technical terms in English. Do NOT transliterate into Korean phonetics (한글 음차 표기 금지).

> **SUB-AGENT**: This agent is invoked via handoff from @dx-app-builder. Do NOT invoke directly — @dx-app-builder enforces mandatory brainstorming questions (Q1/Q2/Q3) that this agent skips.

# DX Model Manager

Manages .dxnn model lifecycle.

## Context
- `.deepx/skills/dx-model-management.md`
- `.deepx/toolsets/model-registry.md`

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
