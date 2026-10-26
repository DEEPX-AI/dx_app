# DX-APP Agent-Driven Development Guide

## Overview

DX-APP supports DEEPX agent-driven development (dx-agent-dev) for building standalone
inference applications on DEEPX NPU accelerators. Instead of manually writing
boilerplate, you describe what you want in natural language and a network of
specialized agents generates production-ready inference code, validates it, and
reports results.

This guide covers the agent architecture, available skills, the validation
framework, and troubleshooting for dx_app standalone inference development.

---

## Agent Architecture

Six agents collaborate to build, validate, and manage dx_app inference applications.

| Agent | Description | Routes To |
|---|---|---|
| `dx-app-builder` | Master router — classifies request type<br>and dispatches to specialist agent | `dx-python-builder`<br>`dx-cpp-builder`<br>`dx-benchmark-builder`<br>`dx-model-manager` |
| `dx-python-builder` | Builds Python inference apps in 4 variants:<br>`sync`, `async`, `sync_cpp_postprocess`,<br>`async_cpp_postprocess`<br>(Sub-agent — invoked by dx-app-builder) | — |
| `dx-cpp-builder` | Builds C++ inference apps using the `InferenceEngine` API (Sub-agent — invoked by dx-app-builder) | — |
| `dx-benchmark-builder` | Benchmarks and profiles inference performance on target hardware (Sub-agent — invoked by dx-app-builder) | `dx-python-builder`<br>(rebuild as a faster variant) |
| `dx-model-manager` | Downloads, registers, and manages `.dxnn` compiled models (Sub-agent — invoked by dx-app-builder) | — |
| `dx-validator` | Validates generated app code and `.deepx/` framework integrity | — |

### Routing Flow

```
User Request
    │
    ▼
dx-app-builder  (classifies intent)
    │
    ├──► dx-python-builder    (Python inference app)
    ├──► dx-cpp-builder       (C++ inference app)
    ├──► dx-benchmark-builder (performance profiling)
    └──► dx-model-manager     (model operations)
            │
            ▼
      validate_app.py  (run by the builder after generation)
```

---

## Skills

Skills encapsulate reusable workflows that agents invoke during code generation.

**General SWE Process**  

| Skill | Description |
|-------|-------------|
| `dx-agent-brainstorm` | Brainstorm and plan before any code generation |
| `dx-agent-tdd` | Test-driven development — validate each file immediately after creation |
| `dx-agent-verify` | Verify before claiming completion — evidence before assertions |
| `dx-swe-writing-plans` | Write structured implementation plans before touching code |
| `dx-swe-executing-plans` | Execute a written implementation plan with review checkpoints |
| `dx-swe-debugging` | Systematic debugging — diagnose before proposing fixes |
| `dx-swe-parallel-agents` | Dispatch 2+ independent tasks to parallel agents |
| `dx-swe-subagent-dev` | Execute implementation plans with independent sub-agents |
| `dx-swe-receiving-review` | Receive and process code review feedback with technical rigor |
| `dx-swe-requesting-review` | Request code review to verify work meets requirements |
| `dx-skill-router` | Route tasks to the appropriate skill automatically |

**DEEPX Build**  

| Skill | Description |
|-------|-------------|
| `dx-agent-app-build-python` | Build a Python inference app in any of the 4 variants using the IFactory pattern |
| `dx-agent-app-build-cpp` | Build a C++ inference app with the `InferenceEngine` runtime API |
| `dx-agent-app-build-async` | Build an async high-performance app with pipelined pre/infer/post stages |
| `dx-agent-app-model-management` | Download `.dxnn` models from the registry and configure model paths |
| `dx-agent-app-validate` | Run the validation levels (static, config, component, NPU smoke, output accuracy, cross-validation, integration) against generated code |

**Harness**  

| Skill | Description |
|-------|-------------|
| `dx-harness-writing-skills` | Create and edit skill files |
| `dx-harness-validate` | Validate this repo's `.deepx/` integrity |

---

## Supported AI Tools

dx_app agent-driven development works with five AI coding tools. Each auto-loads
the knowledge base through its own configuration.

| Tool | Config Files | Agents Available |
|---|---|---|
| **Claude Code** | `CLAUDE.md`<br>6 agents in `.claude/agents/`<br>18 skills in `.claude/skills/` | `@dx-app-builder`, `@dx-validator`<br>(+ 4 sub-agents via handoff) |
| **GitHub Copilot** | `.github/copilot-instructions.md`<br>6 agents in `.github/agents/`<br>18 skills in `.github/skills/`<br>4 instructions in `.github/instructions/` | `@dx-app-builder`, `@dx-validator`<br>(`dx-python-builder`, `dx-cpp-builder`,<br>`dx-benchmark-builder`, `dx-model-manager`<br>are sub-agents reached through `@dx-app-builder`) |
| **Cursor** | `.cursor/rules/dx-app.mdc` (always)<br>6 agent rules + 18 skill rules<br>`python-example.mdc`, `cpp-example.mdc`,<br>`tests.mdc` (28 total) | Free-form with<br>auto-applied rules |
| **OpenCode** | `AGENTS.md`, `opencode.json`<br>6 agents in `.opencode/agents/`<br>18 skills in `.deepx/skills/` | `@dx-app-builder`<br>or `/dx-agent-app-build-python` |
| **Codex CLI** | `AGENTS.md`<br>`.codex/skills/dx-codex-identity/SKILL.md` | Free-form with<br>auto-loaded instructions |

### Copilot File-Specific Instructions

When editing files matching these globs, Copilot automatically injects
context-specific instructions:

| Glob Pattern | Injected Instruction | Content |
|---|---|---|
| `src/python_example/**` | `python-example.instructions.md` | IFactory pattern, `parse_common_args()`, 4-variant naming, no relative imports |
| `src/cpp_example/**` | `cpp-example.instructions.md` | C++14 standard, RAII patterns, InferenceEngine API |
| `src/postprocess/**` | `postprocess.instructions.md` | Postprocessing conventions, pybind bindings |
| `tests/**` | `tests.instructions.md` | pytest markers (`slow`, `smoke`), validation references |

### OpenCode Skills (Slash Commands)

**General SWE Process**

| Slash Command | Description |
|---|---|
| `/dx-agent-brainstorm` | Brainstorm and plan before code generation |
| `/dx-agent-tdd` | Test-driven development with incremental validation |
| `/dx-agent-verify` | Verify completion with evidence before assertions |
| `/dx-swe-writing-plans` | Write structured implementation plans |
| `/dx-swe-executing-plans` | Execute implementation plans with review checkpoints |
| `/dx-swe-debugging` | Systematic debugging — diagnose before fixing |
| `/dx-swe-parallel-agents` | Dispatch independent tasks to parallel agents |
| `/dx-swe-subagent-dev` | Execute plans with independent sub-agents |
| `/dx-swe-receiving-review` | Receive code review feedback with technical rigor |
| `/dx-swe-requesting-review` | Request code review to verify work |
| `/dx-skill-router` | Route tasks to the appropriate skill |

**DEEPX Build**

| Slash Command | Description |
|---|---|
| `/dx-agent-app-build-python` | Step-by-step Python app generation with IFactory |
| `/dx-agent-app-build-cpp` | C++ app with InferenceEngine |
| `/dx-agent-app-build-async` | Async high-performance app |
| `/dx-agent-app-model-management` | Model download and registry |
| `/dx-agent-app-validate` | Run the validation levels (static, config, component, NPU smoke, output accuracy, cross-validation, integration) |

**Harness**

| Slash Command | Description |
|---|---|
| `/dx-harness-writing-skills` | Create and edit skill files |
| `/dx-harness-validate` | Validate this repo's `.deepx/` integrity |

### Platform File Loading Reference

Each AI coding agent auto-loads different configuration files at the dx_app level.

**Auto-Loaded Files**  

| File | Auto-loaded by | Loading |
|------|----------------|---------|
| `.github/copilot-instructions.md` | Copilot Chat/CLI | Auto |
| `CLAUDE.md` | Claude Code | Auto |
| `AGENTS.md` + `opencode.json` | OpenCode | Auto |
| `AGENTS.md` + `.codex/skills/dx-codex-identity/SKILL.md` | Codex CLI | Auto |
| `.cursor/rules/dx-app.mdc` | Cursor | Auto |
| `.cursor/rules/python-example.mdc` | Cursor | Auto (glob) |
| `.cursor/rules/cpp-example.mdc` | Cursor | Auto (glob) |
| `.cursor/rules/tests.mdc` | Cursor | Auto (glob) |

**Agent Files (Manual @mention)**  

| Agent | Copilot (`@mention`) | OpenCode (`@mention`) | Claude Code (`.claude/agents/`) |
|-------|------|---------|---------|
| `dx-app-builder` | `.github/agents/`<br>`dx-app-builder.agent.md` | `.opencode/agents/`<br>`dx-app-builder.md` | `.claude/agents/`<br>`dx-app-builder.md` |
| `dx-python-builder` | `.github/agents/`<br>`dx-python-builder.agent.md` | `.opencode/agents/`<br>`dx-python-builder.md` | `.claude/agents/`<br>`dx-python-builder.md` |
| `dx-cpp-builder` | `.github/agents/`<br>`dx-cpp-builder.agent.md` | `.opencode/agents/`<br>`dx-cpp-builder.md` | `.claude/agents/`<br>`dx-cpp-builder.md` |
| `dx-benchmark-builder` | `.github/agents/`<br>`dx-benchmark-builder.agent.md` | `.opencode/agents/`<br>`dx-benchmark-builder.md` | `.claude/agents/`<br>`dx-benchmark-builder.md` |
| `dx-model-manager` | `.github/agents/`<br>`dx-model-manager.agent.md` | `.opencode/agents/`<br>`dx-model-manager.md` | `.claude/agents/`<br>`dx-model-manager.md` |
| `dx-validator` | `.github/agents/`<br>`dx-validator.agent.md` | `.opencode/agents/`<br>`dx-validator.md` | `.claude/agents/`<br>`dx-validator.md` |

**Copilot File-Specific Instructions**  

| File | Scope |
|------|-------|
| `.github/instructions/python-example.instructions.md` | Python example files |
| `.github/instructions/cpp-example.instructions.md` | C++ example files |
| `.github/instructions/postprocess.instructions.md` | Postprocess files |
| `.github/instructions/tests.instructions.md` | Test files |

**Skill Files (All Platforms — `/slash-command`)**  

| Skill | File |
|-------|------|
| `/dx-agent-brainstorm` | `.deepx/skills/dx-agent-brainstorm/SKILL.md` |
| `/dx-agent-app-build-async` | `.deepx/skills/dx-agent-app-build-async/SKILL.md` |
| `/dx-agent-app-build-cpp` | `.deepx/skills/dx-agent-app-build-cpp/SKILL.md` |
| `/dx-agent-app-build-python` | `.deepx/skills/dx-agent-app-build-python/SKILL.md` |
| `/dx-swe-parallel-agents` | `.deepx/skills/dx-swe-parallel-agents/SKILL.md` |
| `/dx-swe-executing-plans` | `.deepx/skills/dx-swe-executing-plans/SKILL.md` |
| `/dx-agent-app-model-management` | `.deepx/skills/dx-agent-app-model-management/SKILL.md` |
| `/dx-swe-receiving-review` | `.deepx/skills/dx-swe-receiving-review/SKILL.md` |
| `/dx-swe-requesting-review` | `.deepx/skills/dx-swe-requesting-review/SKILL.md` |
| `/dx-skill-router` | `.deepx/skills/dx-skill-router/SKILL.md` |
| `/dx-swe-subagent-dev` | `.deepx/skills/dx-swe-subagent-dev/SKILL.md` |
| `/dx-swe-debugging` | `.deepx/skills/dx-swe-debugging/SKILL.md` |
| `/dx-agent-tdd` | `.deepx/skills/dx-agent-tdd/SKILL.md` |
| `/dx-agent-app-validate` | `.deepx/skills/dx-agent-app-validate/SKILL.md` |
| `/dx-agent-verify` | `.deepx/skills/dx-agent-verify/SKILL.md` |
| `/dx-swe-writing-plans` | `.deepx/skills/dx-swe-writing-plans/SKILL.md` |
| `/dx-harness-writing-skills` | `.deepx/skills/dx-harness-writing-skills/SKILL.md` |
| `/dx-harness-validate` | `.deepx/skills/dx-harness-validate/SKILL.md` |

**Shared Knowledge Base (`.deepx/`)**  

The `.deepx/` directory is the canonical source of truth for all agent knowledge.
Platform-specific files (`.github/agents|skills|instructions/`, `.github/copilot-instructions.md`,
`.claude/`, `.cursor/rules/`, `.opencode/agents/`, `CLAUDE.md`, `AGENTS.md` and their
`-KO` twins) are generated from `.deepx/` by `dx-agent-gen generate`, which ships with
dx-all-suite. It is not auto-loaded as a whole (OpenCode's `opencode.json` loads only
`.deepx/memory/common_pitfalls.md` and `.deepx/instructions/coding-standards.md`);
agents and skills read specific files as needed during task execution.

| Directory | Files | Description |
|-----------|-------|-------------|
| `.deepx/agents/` | 6 files (`dx-app-builder.md`, `dx-python-builder.md`, `dx-cpp-builder.md`, `dx-benchmark-builder.md`, `dx-model-manager.md`, `dx-validator.md`) | Authoritative agent definitions |
| `.deepx/skills/` | 18 directories | Detailed skill workflows (one `SKILL.md` per directory) |
| `.deepx/templates/` | 6 template files | Instruction templates for the platform files (processed by `dx-agent-gen`) |
| `.deepx/knowledge/` | 1 file | Domain-specific reference material |
| `.deepx/toolsets/` | 6 files | API references and schema documentation |
| `.deepx/instructions/` | 6 files | Coding standards and workflow rules |
| `.deepx/memory/` | 5 files | Persistent knowledge — pitfalls and session memory |
| `.deepx/contextual-rules/` | 4 files | Scope-aware coding rules (C++ examples, Python examples, postprocess, tests) |
| `.deepx/prompts/` | 4 files | Reusable prompt templates |
| `.deepx/scripts/` | 3 files | Validation and harness scripts |
| `.deepx/docs/` | 2 files | CI sub-repo gate guide (EN/KO) |

---

## User Scenarios

### Scenario 1: Build a Python Detection App

**Prompt:**

```
"Build a yolo26n person detection app using Python"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. `CLAUDE.md` runs `/dx-skill-router` first and hands app builds to the `dx-app-builder` agent (`.claude/agents/`), which asks the 3 mandatory questions (language/variant, AI task, model), loads the `dx-agent-app-build-python` skill, generates files in `dx-agent-dev/<session_id>/` (or `src/...` if explicitly requested), and validates. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder`, generates the requested variants and runs `validate_app.py`. |
| **Cursor** | Type the prompt directly. `dx-app.mdc` (always loaded) provides context. `python-example.mdc` activates for `src/python_example/` files. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-agent-app-build-python` skill directly. |

### Scenario 2: Build a C++ App

**Prompt:**

```
"Build a C++ inference app for yolo26n using InferenceEngine"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. `CLAUDE.md` hands the build to the `dx-app-builder` agent, which loads the `dx-agent-app-build-cpp` skill. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-cpp-builder`. |
| **Cursor** | Type the prompt directly. `cpp-example.mdc` activates for `src/cpp_example/` files, injecting C++14 and RAII conventions. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-agent-app-build-cpp` skill directly. |

### Scenario 3: Download and Register a Model

**Prompt:**

```
"Download yolo26n model for DX-M1"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | `@dx-app-builder` followed by the prompt. Routes to `dx-model-manager`. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-model-manager`. |
| **Cursor** | Type the prompt directly. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-agent-app-model-management` skill. |

### Scenario 4: Validate Generated Code

**Prompt:**

```
"Validate the detection app I just created"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | `@dx-validator` followed by the prompt. |
| **GitHub Copilot** | `@dx-validator` followed by the prompt. |
| **Cursor** | Type the prompt directly. |
| **OpenCode** | `@dx-validator` followed by the prompt, or run manually: `python .deepx/scripts/validate_app.py dx-agent-dev/<session_id>/` |

### Scenario 5: Build a Pose Estimation App

**Prompt:**

```
"Build a pose estimation app with yolo26n-pose"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. `CLAUDE.md` hands the build to the `dx-app-builder` agent, which loads the `dx-agent-app-build-python` skill with the `pose_estimation` task type. Generates keypoint visualization and skeleton drawing logic. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with pose-specific postprocessing. |
| **Cursor** | Type the prompt directly. `dx-app.mdc` (always loaded) provides context; `python-example.mdc` also applies when you place the app under `src/python_example/pose_estimation/`. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-agent-app-build-python` skill directly. |

### Scenario 6: Build an Instance Segmentation App

**Prompt:**

```
"Build an instance segmentation app with yolo26n-seg"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. `CLAUDE.md` hands the build to the `dx-app-builder` agent, which loads the `dx-agent-app-build-python` skill with the `instance_segmentation` task type. Generates mask overlay visualization. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with segmentation-specific postprocessing. |
| **Cursor** | Type the prompt directly. `dx-app.mdc` (always loaded) provides context; `python-example.mdc` also applies when you place the app under `src/python_example/instance_segmentation/`. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-agent-app-build-python` skill directly. |

### Scenario 7: Build a Classification App

**Prompt:**

```
"Build an image classification app with EfficientNet-Lite0"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. `CLAUDE.md` hands the build to the `dx-app-builder` agent, which loads the `dx-agent-app-build-python` skill with the `image_classification` task. Generates top-K label prediction logic. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with classification postprocessing (softmax + top-K). |
| **Cursor** | Type the prompt directly. `dx-app.mdc` (always loaded) provides context; `python-example.mdc` also applies when you place the app under `src/python_example/image_classification/`. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-agent-app-build-python` skill directly. |

### Scenario 8: Build an Async High-Performance App

**Prompt:**

```
"Build an async high-performance detection app with yolo26n"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. `CLAUDE.md` hands the build to the `dx-app-builder` agent, which loads the `dx-agent-app-build-async` skill. Generates pipelined pre/infer/post stages with queue-based parallelism. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with async variant focus. |
| **Cursor** | Type the prompt directly. `dx-app.mdc` (always loaded) provides context; `python-example.mdc` also applies when you place the app under `src/python_example/`. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-agent-app-build-async` skill directly. |

---

## Quick Start

Request a person detection app in natural language:

```
@dx-app-builder "Build a Python person detection app using yolo26n"
```

The agent will:

1. **Ask the 3 mandatory questions** — language/variant (Python `sync` / `async` / `cpp_postprocess` or C++), AI task (`detection`, `classification`, `segmentation`, etc.), model
2. **Present a build plan** — list of files to generate, model to download, config to write
3. **Route to `dx-python-builder`** — the specialist agent takes over
4. **Generate files** in `dx-agent-dev/<session_id>/` (or `src/` if explicitly requested)
5. **Validate and report** — the builder runs `validate_app.py`, runs the sync example once if an NPU is present, and prints a completion report

### Mandatory Questions (HARD-GATE)

When using `@dx-app-builder`, the agent enforces 3 mandatory questions before
generating any code:

1. **Language/variant**: Python (sync / async / cpp_postprocess / async_cpp_postprocess) or C++?
2. **AI task**: detection, classification, segmentation, pose, etc.
3. **Model**: Specific model name (e.g., `yolo26n`) or auto-recommend

These questions are **non-skippable** — even if your prompt provides enough context,
the agent will confirm each decision explicitly before proceeding.

---

## What Gets Created

By default, agent-generated code is placed in the `dx-agent-dev/` isolation directory
to prevent conflicts with existing source code.

### Default Output (dx-agent-dev/)

```
dx-agent-dev/<session_id>/
├── README.md              # Session metadata and run instructions
├── session.json           # Machine-readable session config
├── setup.sh               # Environment setup script (mandatory)
├── run.sh                 # App launch script (mandatory)
├── session.log            # Agent session log (mandatory)
├── __init__.py
├── config.json
├── factory/
│   ├── __init__.py
│   └── {model}_factory.py
├── {model}_sync.py
├── {model}_async.py                  # if requested
├── {model}_sync_cpp_postprocess.py   # if requested
└── {model}_async_cpp_postprocess.py  # if requested
```

`{model}_sync.py` is always generated; the other three variants are generated when you ask for them.

Session ID format: `YYYYMMDD-HHMMSS_<agent>_<coding_model>_<target_model>_<task>` where `<agent>` is `claude`, `codex`, `copilot`, `cursor`, or `opencode` (e.g., `20260403-143022_claude_sonnet46_yolo26n_detection`). The timestamp is the system local time.

### Production Output (src/)

When you explicitly request production placement, files are written directly to
`src/python_example/{task}/{family}/{variant}/` — the standard source tree
(`{variant}` is the `.dxnn` stem).

### File Descriptions

| File | Purpose |
|---|---|
| `config.json` | Inference thresholds read by the factory (e.g. `score_threshold`, `nms_threshold`, `top_k`); the model path comes from `--model` or `model_registry.json` |
| `{model}_factory.py` | Implements `IFactory`: `create_preprocessor`, `create_postprocessor`, `create_visualizer`, `get_model_name`, `get_task_type` |
| `{model}_sync.py` | Synchronous single-threaded inference entry point |
| `{model}_async.py` | Asynchronous pipelined inference entry point |
| `{model}_sync_cpp_postprocess.py` | Sync inference with C++ post-processing via pybind |
| `{model}_async_cpp_postprocess.py` | Async inference with C++ post-processing via pybind |

!!! note "NOTE"

    `setup.sh`, `run.sh`, and `session.log` are mandatory artifacts in every session output directory

---

## Validation Levels

The `dx-agent-app-validate` skill defines these levels. Levels 1–3 run without an NPU; Levels 4–6 need the NPU and the model file.

| Level | What It Checks | Requires Hardware |
|---|---|---|
| 1 — Static | Syntax, imports, IFactory methods, `parse_common_args()`, no hard-coded paths | No |
| 2 — Config | `config.json` is valid JSON and has the task's keys; the model has a `model_registry.json` entry | No |
| 3 — Component | Factory creates the preprocessor, postprocessor and visualizer; OpenCV has a GUI backend; the app runs when copied outside the suite | No |
| 4 — Smoke | Single-frame inference on the NPU | Yes |
| 5 — Output Accuracy | Detections > 0, valid boxes, scores and class IDs, postprocessor matches the model family | Yes |
| 5.5 — Cross-Validation | Comparison with a precompiled reference model or an existing verified example (skipped if neither exists) | Yes |
| 6 — Integration | Full pipeline on video with `--save` | Yes |

---

## Validation Commands

```bash
# Static checks (11 checks: required files, syntax, relative imports, logging, hard-coded paths, IFactory, parse_common_args, unused imports, README, app.yaml, unreachable code)
python .deepx/scripts/validate_app.py dx-agent-dev/<session_id>/

# Add the 3 smoke tests (--help, factory import, NPU status) — run after the session's setup.sh
python .deepx/scripts/validate_app.py dx-agent-dev/<session_id>/ --smoke-test

# Framework integrity — verify .deepx/ directory structure
python .deepx/scripts/validate_framework.py
```

!!! note "NOTE"

    `validate_app.py` checks a session output directory; source-tree variant directories have no `__init__.py` and fail the required-files check. Other options: `--strict`, `--json`, `--verbose`.

---

## Knowledge Base Structure

Agent knowledge lives in the `.deepx/` directory at the dx_app project root.

| Directory | Count | Contents |
|---|---|---|
| `agents/` | 6 | Agent definitions and routing rules |
| `skills/` | 18 | Skill workflow directories (one `SKILL.md` each) |
| `templates/` | 6 | Instruction templates for the platform files (processed by `dx-agent-gen`) |
| `knowledge/` | 1 | Domain-specific reference material |
| `toolsets/` | 6 | API references (InferenceEngine, common framework / IFactory, `.dxnn` model format, model registry, pybind postprocess, OCR app) |
| `memory/` | 5 | Persistent knowledge (index, common pitfalls, model zoo, performance patterns, platform API) |
| `contextual-rules/` | 4 | Scope-aware rules for C++ examples, Python examples, postprocess, and tests |
| `prompts/` | 4 | Reusable prompt templates (new C++ app, Python detection/segmentation, orchestrated build) |
| `scripts/` | 3 | `validate_app.py`, `validate_framework.py`, `harness_bootstrap.sh` |
| `docs/` | 2 | CI sub-repo gate guide (EN/KO) |

!!! note "NOTE"

    Platform files (`.github/agents|skills|instructions/`, `.github/copilot-instructions.md`, `.claude/`, `.cursor/rules/`, `.opencode/agents/`, `CLAUDE.md`, `AGENTS.md` and their `-KO` twins) are generated by `dx-agent-gen generate`, which ships with dx-all-suite. 
    Do not edit them directly. Edit `.deepx/` source files and re-run the generator. In a standalone dx_app checkout, run `bash .deepx/scripts/harness_bootstrap.sh --check` before editing `.deepx/`.

Agents read from these directories at task start. Memory files are updated when
new patterns or fixes are discovered during development.

---

## Session Sentinels

Agents output fixed markers at the start and end of each task for automated testing:

| Marker | When |
|---|---|
| `[DX-AGENT-DEV: START]` | First line of the agent's response |
| `[DX-AGENT-DEV: DONE (output-dir: <relative_path>)]` | Last line after all work is complete. `<relative_path>` is the session output directory relative to the project root. If no files were generated, omit the `(output-dir: ...)` part. |

Sub-agents invoked via handoff do not output sentinels — only the top-level agent does.

Rules:  

1. **Required** — Output `[DX-AGENT-DEV: START]` as the absolute first line of the first response, before any other text, tool call, or reasoning. It cannot be skipped even if the user says "just proceed" — automated tests fail without it.  
2. Output `[DX-AGENT-DEV: DONE (output-dir: <path>)]` as the very last line after all work, validation, and file generation is complete.  
3. If you are a sub-agent invoked via handoff/routing, do NOT output sentinels — only the top-level agent outputs them.  
4. If the user sends multiple prompts in a session, output START/DONE for each prompt.  
5. The `output-dir` in DONE must be the relative path from the project root to the session output directory.  
6. **Never output DONE after only producing planning artifacts** (specs, plans, design documents). DONE means all deliverables are produced — implementation code, scripts, configs, and validation results.  
7. Right after START and right before DONE the agent prints the DEEPX banner; the banner never moves the sentinel lines.  
8. Before DONE the agent checks that every mandatory file exists in the session directory.  

---

## Troubleshooting

- **Problem:** Agent writes relative imports (`from .factory import ...`)  
    **Cause:** Default LLM behavior  
    **Solution:** Use absolute imports only.  
    Example: `from common.processors import YOLOv8Postprocessor`

- **Problem:** Factory class missing methods  
    **Cause:** Incomplete `IFactory` implementation  
    **Solution:** Implement all 5 required methods:  
    `create_preprocessor`, `create_postprocessor`, `create_visualizer`, `get_model_name`, `get_task_type`

- **Problem:** Model not found at runtime  
    **Cause:** `.dxnn` file path not registered  
    **Solution:** Query `model_registry.json` via `dx-model-manager` to download and register the model.

- **Problem:** NPU not available / device error  
    **Cause:** Accelerator not detected by driver  
    **Solution:** Run `dxcli -s` to check device status and verify the DEEPX kernel module is loaded (`dxrt-cli -s` is the same tool).

- **Problem:** `validate_app.py` fails immediately  
    **Cause:** The target directory has no `__init__.py`, `config.json` or `factory/`, or (with `--smoke-test`) `common` cannot be imported  
    **Solution:** Run it from the dx_app root on a session directory (`dx-agent-dev/<session_id>/`), after the session's `setup.sh` has copied `./common` into it.

---

## Further Reading

- [DX-APP Architecture and Usage Reference](09_DX-APP_Project_Overview.md)
- [DX-APP Python Example Usage Guide](05_DX-APP_Python_Example_Usage_Guide.md)
- [DX-APP C++ Example Usage Guide](03_DX-APP_CPP_Example_Usage_Guide.md)
- [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)
