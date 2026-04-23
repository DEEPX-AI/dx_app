# DX-APP Agentic Development Guide

## Overview

DX-APP supports AI-powered agentic development for building standalone inference
applications on DEEPX NPU accelerators. Instead of manually writing boilerplate,
you describe what you want in natural language and a network of specialized agents
generates production-ready inference code, validates it, and reports results.

This guide covers the agent architecture, available skills, the validation
framework, and troubleshooting for dx_app standalone inference development.

---

## Agent Architecture

Six agents collaborate to build, validate, and manage dx_app inference applications.

| Agent | Description | Routes To |
|---|---|---|
| `dx-app-builder` | Master router — classifies the request type and dispatches to the appropriate specialist agent | `dx-python-builder`, `dx-cpp-builder`, `dx-benchmark-builder`, `dx-model-manager` |
| `dx-python-builder` | Builds Python inference apps in 4 variants: `sync`, `async`, `sync_cpp_postprocess`, `async_cpp_postprocess` (Sub-agent — invoked by dx-app-builder) | — |
| `dx-cpp-builder` | Builds C++ inference apps using the `InferenceEngine` API (Sub-agent — invoked by dx-app-builder) | — |
| `dx-benchmark-builder` | Benchmarks and profiles inference performance on target hardware (Sub-agent — invoked by dx-app-builder) | — |
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
      dx-validator  (called automatically after generation)
```

---

## Skills

Skills encapsulate reusable workflows that agents invoke during code generation.

| Skill | Description |
|---|---|
| `dx-brainstorm-and-plan` | Brainstorm and plan before any code generation (process skill) |
| `dx-build-async-app` | Build an async high-performance app with pipelined pre/infer/post stages |
| `dx-build-cpp-app` | Build a C++ inference app with the `InferenceEngine` runtime API |
| `dx-build-python-app` | Build a Python inference app in any of the 4 variants using the IFactory pattern |
| `dx-dispatching-parallel-agents` | Dispatch 2+ independent tasks to parallel agents |
| `dx-executing-plans` | Execute a written implementation plan with review checkpoints |
| `dx-model-management` | Download `.dxnn` models from the registry and configure model paths |
| `dx-receiving-code-review` | Receive and process code review feedback with technical rigor |
| `dx-requesting-code-review` | Request code review to verify work meets requirements |
| `dx-skill-router` | Route tasks to the appropriate skill automatically |
| `dx-subagent-driven-development` | Execute implementation plans with independent sub-agents |
| `dx-systematic-debugging` | Systematic debugging — diagnose before proposing fixes |
| `dx-tdd` | Test-driven development — validate each file immediately after creation (process skill) |
| `dx-validate` | Run the 5-level validation pyramid against generated code |
| `dx-verify-completion` | Verify before claiming completion — evidence before assertions (process skill) |
| `dx-writing-plans` | Write structured implementation plans before touching code |
| `dx-writing-skills` | Create or edit skill definitions |

---

## Supported AI Tools

dx_app agentic development works with four AI coding tools. Each auto-loads
the knowledge base through its own configuration.

| Tool | Config Files | Agents Available |
|---|---|---|
| **Claude Code** | `CLAUDE.md` | All 6 agents via context routing |
| **GitHub Copilot** | `.github/copilot-instructions.md`, 6 agents in `.github/agents/`, 17 skills in `.github/skills/`, 4 instructions in `.github/instructions/` | `@dx-app-builder`, `@dx-python-builder`, `@dx-cpp-builder`, `@dx-benchmark-builder`, `@dx-model-manager`, `@dx-validator` |
| **Cursor** | `.cursor/rules/dx-app.mdc` (always), 6 agent rules, 17 skill rules, `python-example.mdc`, `cpp-example.mdc`, `tests.mdc` (27 total) | Free-form with auto-applied rules |
| **OpenCode** | `AGENTS.md`, `opencode.json`, 6 agents in `.opencode/agents/`, 17 skills in `.deepx/skills/` | `@dx-app-builder` or `/dx-build-python-app` |

### Copilot File-Specific Instructions

When editing files matching these globs, Copilot automatically injects
context-specific instructions:

| Glob Pattern | Injected Instruction | Content |
|---|---|---|
| `src/python_example/**` | `python-example.instructions.md` | IFactory pattern, SyncRunner/AsyncRunner usage, 4-variant naming |
| `src/cpp_example/**` | `cpp-example.instructions.md` | C++14 standard, RAII patterns, InferenceEngine API |
| `src/postprocess/**` | `postprocess.instructions.md` | Postprocessing conventions, pybind bindings |
| `tests/**` | `tests.instructions.md` | pytest patterns, fixtures, NPU markers |

### OpenCode Skills (Slash Commands)

| Slash Command | Description |
|---|---|
| `/dx-build-python-app` | Step-by-step Python app generation with IFactory |
| `/dx-build-cpp-app` | C++ app with InferenceEngine |
| `/dx-build-async-app` | Async high-performance app |
| `/dx-model-management` | Model download and registry |
| `/dx-validate` | Run the 5-level validation pyramid |
| `/dx-brainstorm-and-plan` | Brainstorm and plan before code generation |
| `/dx-dispatching-parallel-agents` | Dispatch independent tasks to parallel agents |
| `/dx-executing-plans` | Execute implementation plans with review checkpoints |
| `/dx-receiving-code-review` | Receive code review feedback with technical rigor |
| `/dx-requesting-code-review` | Request code review to verify work |
| `/dx-skill-router` | Route tasks to the appropriate skill |
| `/dx-subagent-driven-development` | Execute plans with independent sub-agents |
| `/dx-systematic-debugging` | Systematic debugging — diagnose before fixing |
| `/dx-tdd` | Test-driven development with incremental validation |
| `/dx-verify-completion` | Verify completion with evidence before assertions |
| `/dx-writing-plans` | Write structured implementation plans |
| `/dx-writing-skills` | Create or edit skill definitions |

### Platform File Loading Reference

Each AI coding agent auto-loads different configuration files at the dx_app level.

#### Auto-Loaded Files

| File | Copilot Chat/CLI | OpenCode | Claude Code | Cursor | Loading |
|------|:---:|:---:|:---:|:---:|---------|
| `.github/copilot-instructions.md` | ✅ | — | — | — | Auto |
| `CLAUDE.md` | — | — | ✅ | — | Auto |
| `AGENTS.md` + `opencode.json` | — | ✅ | — | — | Auto |
| `.cursor/rules/dx-app.mdc` | — | — | — | ✅ | Auto |
| `.cursor/rules/python-example.mdc` | — | — | — | ✅ | Auto (glob) |
| `.cursor/rules/cpp-example.mdc` | — | — | — | ✅ | Auto (glob) |
| `.cursor/rules/tests.mdc` | — | — | — | ✅ | Auto (glob) |

#### Agent Files (Manual @mention)

| Agent | Copilot (`@mention`) | OpenCode (`@mention`) | Claude Code (`.claude/agents/`) |
|-------|------|---------|---------|
| `dx-app-builder` | `.github/agents/dx-app-builder.agent.md` | `.opencode/agents/dx-app-builder.md` | `.claude/agents/dx-app-builder.md` |
| `dx-python-builder` | `.github/agents/dx-python-builder.agent.md` | `.opencode/agents/dx-python-builder.md` | `.claude/agents/dx-python-builder.md` |
| `dx-cpp-builder` | `.github/agents/dx-cpp-builder.agent.md` | `.opencode/agents/dx-cpp-builder.md` | `.claude/agents/dx-cpp-builder.md` |
| `dx-benchmark-builder` | `.github/agents/dx-benchmark-builder.agent.md` | `.opencode/agents/dx-benchmark-builder.md` | `.claude/agents/dx-benchmark-builder.md` |
| `dx-model-manager` | `.github/agents/dx-model-manager.agent.md` | `.opencode/agents/dx-model-manager.md` | `.claude/agents/dx-model-manager.md` |
| `dx-validator` | `.github/agents/dx-validator.agent.md` | `.opencode/agents/dx-validator.md` | `.claude/agents/dx-validator.md` |

#### Copilot File-Specific Instructions

| File | Scope |
|------|-------|
| `.github/instructions/python-example.instructions.md` | Python example files |
| `.github/instructions/cpp-example.instructions.md` | C++ example files |
| `.github/instructions/postprocess.instructions.md` | Postprocess files |
| `.github/instructions/tests.instructions.md` | Test files |

#### Skill Files (All Platforms — `/slash-command`)

| Skill | File |
|-------|------|
| `/dx-brainstorm-and-plan` | `.deepx/skills/dx-brainstorm-and-plan/SKILL.md` |
| `/dx-build-async-app` | `.deepx/skills/dx-build-async-app/SKILL.md` |
| `/dx-build-cpp-app` | `.deepx/skills/dx-build-cpp-app/SKILL.md` |
| `/dx-build-python-app` | `.deepx/skills/dx-build-python-app/SKILL.md` |
| `/dx-dispatching-parallel-agents` | `.deepx/skills/dx-dispatching-parallel-agents/SKILL.md` |
| `/dx-executing-plans` | `.deepx/skills/dx-executing-plans/SKILL.md` |
| `/dx-model-management` | `.deepx/skills/dx-model-management/SKILL.md` |
| `/dx-receiving-code-review` | `.deepx/skills/dx-receiving-code-review/SKILL.md` |
| `/dx-requesting-code-review` | `.deepx/skills/dx-requesting-code-review/SKILL.md` |
| `/dx-skill-router` | `.deepx/skills/dx-skill-router/SKILL.md` |
| `/dx-subagent-driven-development` | `.deepx/skills/dx-subagent-driven-development/SKILL.md` |
| `/dx-systematic-debugging` | `.deepx/skills/dx-systematic-debugging/SKILL.md` |
| `/dx-tdd` | `.deepx/skills/dx-tdd/SKILL.md` |
| `/dx-validate` | `.deepx/skills/dx-validate/SKILL.md` |
| `/dx-verify-completion` | `.deepx/skills/dx-verify-completion/SKILL.md` |
| `/dx-writing-plans` | `.deepx/skills/dx-writing-plans/SKILL.md` |
| `/dx-writing-skills` | `.deepx/skills/dx-writing-skills/SKILL.md` |

#### Shared Knowledge Base (`.deepx/`)

The `.deepx/` directory is the canonical source of truth for all agent knowledge.
Platform-specific files (`.github/`, `.cursor/`, `.opencode/`) are generated from
`.deepx/` by `dx-agentic-gen generate --repo dx-runtime/dx_app`. It is NOT
auto-loaded — agents and skills reference specific files as needed during task
execution.

| Directory | Files | Description |
|-----------|-------|-------------|
| `.deepx/agents/` | 6 files (`dx-app-builder.md`, `dx-python-builder.md`, `dx-cpp-builder.md`, `dx-benchmark-builder.md`, `dx-model-manager.md`, `dx-validator.md`) | Authoritative agent definitions |
| `.deepx/skills/` | 17 directories | Detailed skill workflows (one `SKILL.md` per directory) |
| `.deepx/templates/` | Template files | Code generation templates |
| `.deepx/knowledge/` | Knowledge files | Domain-specific reference material |
| `.deepx/toolsets/` | 5 files | API references and schema documentation |
| `.deepx/instructions/` | 6 files | Coding standards and workflow rules |
| `.deepx/memory/` | 5 files | Persistent knowledge — pitfalls and session memory |
| `.deepx/contextual-rules/` | 4 files | Context-dependent rules activated by task type |

---

## User Scenarios

### Scenario 1: Build a Python Detection App

**Prompt:**

```
"Build a yolo26n person detection app using Python"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. `CLAUDE.md` routes to `dx-build-python-app` skill. Asks 2-3 questions (variant, task type, model), generates files in `dx-agentic-dev/<session_id>/` (or `src/...` if explicitly requested), and validates. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder`, generates all 4 variants, runs `dx-validator`. |
| **Cursor** | Type the prompt directly. `dx-app.mdc` (always loaded) provides context. `python-example.mdc` activates for `src/python_example/` files. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-build-python-app` skill directly. |

### Scenario 2: Build a C++ App

**Prompt:**

```
"Build a C++ inference app for yolo26n using InferenceEngine"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. Routes to `dx-build-cpp-app` skill. |
| **GitHub Copilot** | `@dx-cpp-builder` followed by the prompt. |
| **Cursor** | Type the prompt directly. `cpp-example.mdc` activates for `src/cpp_example/` files, injecting C++14 and RAII conventions. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-build-cpp-app` skill directly. |

### Scenario 3: Download and Register a Model

**Prompt:**

```
"Download yolo26n model for DX-M1"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | `@dx-model-manager` followed by the prompt. |
| **GitHub Copilot** | `@dx-model-manager` followed by the prompt. |
| **Cursor** | Type the prompt directly. |
| **OpenCode** | `@dx-model-manager` followed by the prompt, or `/dx-model-management` skill. |

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
| **OpenCode** | `@dx-validator` followed by the prompt, or run manually: `python .deepx/scripts/validate_app.py src/python_example/object_detection/yolo26n/` |

### Scenario 5: Build a Pose Estimation App

**Prompt:**

```
"Build a pose estimation app with yolo26n-pose"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. Routes to `dx-build-python-app` skill with `pose_estimation` task type. Generates keypoint visualization and skeleton drawing logic. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with pose-specific postprocessing. |
| **Cursor** | Type the prompt directly. `python-example.mdc` activates for generated files in `src/python_example/pose_estimation/`. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-build-python-app` skill directly. |

### Scenario 6: Build an Instance Segmentation App

**Prompt:**

```
"Build an instance segmentation app with yolo26n-seg"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. Routes to `dx-build-python-app` skill with `instance_segmentation` task type. Generates mask overlay visualization. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with segmentation-specific postprocessing. |
| **Cursor** | Type the prompt directly. `python-example.mdc` activates for generated files in `src/python_example/instance_segmentation/`. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-build-python-app` skill directly. |

### Scenario 7: Build a Classification App

**Prompt:**

```
"Build an image classification app with EfficientNet-B0"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. Routes to `dx-build-python-app` skill with `classification` task type. Generates top-K label prediction logic. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with classification postprocessing (softmax + top-K). |
| **Cursor** | Type the prompt directly. `python-example.mdc` activates for generated files in `src/python_example/classification/`. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-build-python-app` skill directly. |

### Scenario 8: Build an Async High-Performance App

**Prompt:**

```
"Build an async high-performance detection app with yolo26n"
```

| Tool | How to Use |
|---|---|
| **Claude Code** | Type the prompt directly. Routes to `dx-build-async-app` skill. Generates pipelined pre/infer/post stages with queue-based parallelism. |
| **GitHub Copilot** | `@dx-app-builder` followed by the prompt. Routes to `dx-python-builder` with async variant focus. |
| **Cursor** | Type the prompt directly. `python-example.mdc` activates for generated async files. |
| **OpenCode** | `@dx-app-builder` followed by the prompt, or `/dx-build-async-app` skill directly. |

---

## Quick Start

Request a person detection app in natural language:

```
@dx-app-builder "yolo26n으로 사람 감지하는 Python 앱 만들어줘"
```

The agent will:

1. **Ask clarifying questions** — variant (`sync` / `async`), model precision, task type (`detection`, `classification`, `segmentation`, etc.)
2. **Present a build plan** — list of files to generate, model to download, config to write
3. **Route to `dx-python-builder`** — the specialist agent takes over
4. **Generate files** in `dx-agentic-dev/<session_id>/` (or `src/` if explicitly requested)
5. **Validate and report** — `dx-validator` runs checks and prints a summary

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

By default, AI-generated code is placed in the `dx-agentic-dev/` isolation directory
to prevent conflicts with existing source code.

### Default Output (dx-agentic-dev/)

```
dx-agentic-dev/<session_id>/
├── README.md              # Session metadata and run instructions
├── session.json           # Machine-readable session config
├── setup.sh               # Environment setup script (mandatory)
├── run.sh                 # App launch script (mandatory)
├── session.log            # Agent session log (mandatory)
└── src/python_example/{task}/{model}/
    ├── __init__.py
    ├── config.json
    ├── {model}_factory.py
    ├── {model}_sync.py
    ├── {model}_async.py
    ├── {model}_sync_cpp_postprocess.py
    └── {model}_async_cpp_postprocess.py
```

Session ID format: `YYYYMMDD-HHMMSS_model_task` (e.g., `20260403-143022_yolo26n_detection`).

### Production Output (src/)

When you explicitly request production placement, files are written directly to
`src/python_example/{task}/{model}/` — the standard source tree.

### File Descriptions

| File | Purpose |
|---|---|
| `config.json` | Model path, task type, input dimensions, label map |
| `{model}_factory.py` | Implements `IFactory` — the 5-method interface for pre/post processing |
| `{model}_sync.py` | Synchronous single-threaded inference entry point |
| `{model}_async.py` | Asynchronous pipelined inference entry point |
| `{model}_sync_cpp_postprocess.py` | Sync inference with C++ post-processing via pybind |
| `{model}_async_cpp_postprocess.py` | Async inference with C++ post-processing via pybind |

> **Note:** `setup.sh`, `run.sh`, and `session.log` are mandatory artifacts in every session output directory.

---

## 5-Level Validation Pyramid

`dx-validator` applies checks in ascending order of cost. Each level gates the next.

```
        ▲
       /5\       Performance benchmarks (FPS targets)
      /───\
     / 4   \     NPU integration tests (requires hardware)
    /───────\
   /   3     \   Smoke tests (--help, module import)
  /───────────\
 /     2       \ Config validation (model paths, task types)
/───────────────\
       1         Static checks (imports, naming, structure)
```

| Level | What It Checks | Requires Hardware |
|---|---|---|
| 1 — Static | Absolute imports, naming conventions, file structure, IFactory methods | No |
| 2 — Config | `config.json` schema, `.dxnn` model path resolution, valid task types | No |
| 3 — Smoke | `--help` flag runs without error, modules import cleanly | No |
| 4 — NPU Integration | End-to-end inference on a sample image with NPU present | Yes |
| 5 — Performance | FPS meets target thresholds for the model and accelerator | Yes |

---

## Validation Commands

```bash
# Static checks (11 checks across Level 1 and Level 2)
python .deepx/scripts/validate_app.py src/python_example/{task}/{model}/

# Include smoke tests (Levels 1–3)
python .deepx/scripts/validate_app.py src/python_example/{task}/{model}/ --smoke-test

# Framework integrity — verify .deepx/ directory structure
python .deepx/scripts/validate_framework.py
```

---

## Knowledge Base Structure

Agent knowledge lives in the `.deepx/` directory at the dx_app project root.

| Directory | Count | Contents |
|---|---|---|
| `agents/` | 6 | Agent definitions and routing rules |
| `skills/` | 17 | Skill workflow directories (one `SKILL.md` each) |
| `templates/` | — | Code generation templates |
| `knowledge/` | — | Domain-specific reference material |
| `toolsets/` | 5 | API references (InferenceEngine, IFactory, dxrt-cli, model registry, pybind helpers) |
| `memory/` | 5 | Persistent knowledge (common pitfalls, platform API notes, optimization patterns, camera/display notes, model config cache) |
| `contextual-rules/` | 4 | Coding standards, import rules, naming conventions, directory layout rules |
| `prompts/` | 4 | System prompts for each specialist agent |
| `scripts/` | 2 | `validate_app.py`, `validate_framework.py` |

> **Note:** Platform files (`.github/`, `.cursor/`, `.opencode/`, `CLAUDE.md`, `AGENTS.md`) are
> generated by `dx-agentic-gen generate --repo dx-runtime/dx_app`. Do not edit them directly —
> edit `.deepx/` source files and re-run the generator. A pre-commit hook runs `dx-agentic-gen`
> automatically.

Agents read from these directories at task start. Memory files are updated when
new patterns or fixes are discovered during development.

---

## Session Sentinels

Agents output fixed markers at the start and end of each task for automated testing:

| Marker | When |
|---|---|
| `[DX-AGENTIC-DEV: START]` | First line of the agent's response |
| `[DX-AGENTIC-DEV: DONE (output-dir: <relative_path>)]` | Last line after all work is complete. `<relative_path>` is the session output directory relative to the project root. If no files were generated, omit the `(output-dir: ...)` part. |

Sub-agents invoked via handoff do not output sentinels — only the top-level agent does.

Rules:
1. **CRITICAL** — Output `[DX-AGENTIC-DEV: START]` as the absolute first line of the first response, before any other text. This is non-negotiable even if the user says to proceed autonomously.
2. Output `[DX-AGENTIC-DEV: DONE (output-dir: <path>)]` as the very last line after all work, validation, and file generation is complete.
3. If you are a sub-agent invoked via handoff/routing, do NOT output sentinels — only the top-level agent outputs them.
4. If the user sends multiple prompts in a session, output START/DONE for each prompt.
5. The `output-dir` in DONE must be the relative path from the project root to the session output directory.
6. **Never output DONE after only producing planning artifacts** (specs, plans, design documents). DONE means all deliverables are produced — implementation code, scripts, configs, and validation results.

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| Agent writes relative imports (`from .factory import ...`) | Default LLM behavior | All imports must be absolute: `from dx_app.python_example.detection.yolo26n.yolo26n_factory import ...` |
| Factory class missing methods | Incomplete `IFactory` implementation | Implement all 5 required methods: `create_preprocessor`, `create_postprocessor`, `create_label_map`, `create_input_config`, `create_visualizer` |
| Model not found at runtime | `.dxnn` file path not registered | Query `model_registry.json` via `dx-model-manager` to download and register the model |
| NPU not available / device error | Accelerator not detected by driver | Run `dxrt-cli -s` to check device status; verify the DEEPX kernel module is loaded |
| `validate_app.py` fails immediately | Python path or venv not configured | Activate the dx_app virtual environment and ensure `PYTHONPATH` includes the project root |

---

## Further Reading

- [DX-APP Project Overview](09_DX-APP_Project_Overview.md)
- [DX-APP Python Example Usage Guide](05_DX-APP_Python_Example_Usage_Guide.md)
- [DX-APP C++ Example Usage Guide](03_DX-APP_CPP_Example_Usage_Guide.md)
- [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)
