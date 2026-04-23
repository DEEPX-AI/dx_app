---
name: dx-brainstorm-and-plan
description: Brainstorm and plan before implementation
---

<!-- AUTO-GENERATED from .deepx/ — DO NOT EDIT DIRECTLY -->
<!-- Source: .deepx/skills/dx-brainstorm-and-plan/SKILL.md -->
<!-- Run: dx-agentic-gen generate -->

# Skill: Brainstorm and Plan for dx_app

> **Type: RIGID — HARD-GATE.** No code generation without an approved plan.

## When to Invoke

- User asks to build, create, or add any dx_app application
- Before invoking `/dx-build-python-app`, `/dx-build-cpp-app`, or `/dx-build-async-app`

## HARD-GATE Rule

```
IF plan not approved by user → DO NOT generate any code
IF plan approved → proceed to build skill
```

No exceptions. "Just build it" means use defaults — it does NOT mean skip planning.

## Step 1: Context Check

Before asking the user anything, gather facts silently:

```bash
# 1. Query model registry
python -c "import json; models=json.load(open('config/model_registry.json')); print(f'Registry: {len(models)} models')"

# 2. Check for existing examples matching the model
ls src/python_example/<task>/<model>/ 2>/dev/null
```

If an existing example is found, ask: **(a)** explain existing, or **(b)** create new based on it.
Never silently overwrite existing apps.

### Skeleton-First Constraint (MANDATORY — applies to ALL plans)

> **All demo scripts MUST be built using skeleton-first development.** Do NOT
> propose, design, or plan alternative implementation approaches (e.g., "single
> image processing", "direct `engine.run()` calls", "`wait()` approach",
> "`register_callback()` approach"). The ONLY valid implementation method is:
>
> 1. Copy the closest existing example from `src/python_example/<task>/`
> 2. Modify ONLY model-specific parts (factory class, model name, pre/postprocessor)
> 3. Use SyncRunner (sync) or AsyncRunner (async) — these are the ONLY two runners
>
> This constraint is NON-NEGOTIABLE. The brainstorming phase decides WHAT to build
> (task, model, variant, input source), not HOW to implement it. The implementation
> method is always skeleton-copy + minimal modification.
>
> See `memory/common_pitfalls.md` Pitfall #20 for the task→skeleton mapping table.

## Step 2: Key Decisions (Ask the User)

### Decision 1: App Type

| Option | When to suggest |
|--------|----------------|
| **Python app** (default) | Most models, rapid prototyping |
| **C++ app** | Performance-critical, production deployment |

### Decision 2: Variant Selection (Python only)

| Variant | Description |
|---------|-------------|
| `sync` | Single-threaded, image or short video |
| `async` | Multi-threaded, camera or long video |
| `sync_cpp_postprocess` | Sync with C++ pybind11 postprocessor |
| `async_cpp_postprocess` | Async with C++ pybind11 postprocessor |
| **All 4** (default) | Generate all variants |

### Decision 3: Input Source & Special Requirements

- Image (`--image`), video (`--video`), or camera (`--camera 0`)?
- Custom preprocessing? PPU model? Specific thresholds?

## Step 3: Build Plan

Present this plan template for user approval:

```
## Build Plan: <ModelDisplay> <TaskType> App

**App type**: Python / C++
**Model**: <model_name>  |  **Task**: <task_type>
**Variants**: sync, async, sync_cpp_postprocess, async_cpp_postprocess
**Skeleton source**: src/python_example/<task>/<closest_model>/
**Output dir**: dx-agentic-dev/<YYYYMMDD-HHMMSS>_<model>_<task>/

### Files to Create
1. factory/<model>_factory.py    8. __init__.py
2. factory/__init__.py           9. session.json
3. config.json                  10. README.md
4. <model>_sync.py              11. setup.sh
5. <model>_async.py             12. run.sh
6. <model>_sync_cpp_postprocess.py  13. session.log
7. <model>_async_cpp_postprocess.py

### Components
- Preprocessor: <Preprocessor>  |  Postprocessor: <Postprocessor>
- Visualizer: <Visualizer>      |  Factory: <IFactoryInterface>
```

For C++ apps: `CMakeLists.txt`, `main.cpp`, `config.json`, `include/`, `src/`.

## Step 4: Pre-flight Check

- [ ] Model exists in `config/model_registry.json`
- [ ] No naming collision with existing apps
- [ ] Components (preprocessor, postprocessor, visualizer) are known
- [ ] For C++ apps: `dx_engine` headers available after `./build.sh`

## Step 5: Route to Build Skill

| App type | Invoke |
|----------|--------|
| Python (any variant) | `/dx-build-python-app` |
| C++ | `/dx-build-cpp-app` |
| Async-focused | `/dx-build-async-app` |

## Defaults (for "just build it")

- App type: Python, Variants: All 4, Output: `dx-agentic-dev/`
- Input: image (sync), video (async), Config: task-appropriate defaults

## Anti-Pattern: Proposing Alternative Implementation Approaches

Do NOT present choices like:
- "Option A: Single image processing with direct engine calls"
- "Option B: Batch processing with wait() pattern"
- "Option C: Streaming with register_callback()"

These are FABRICATED approaches that do not exist in the dx_app framework.
The dx_app framework has exactly 2 runners: `SyncRunner` (for sync variants)
and `AsyncRunner` (for async variants). Both are used via skeleton-copy from
existing examples. There is NO other valid approach.

If the user asks about implementation alternatives, explain that dx_app uses
the IFactory + Runner pattern exclusively, and the brainstorming phase only
decides: app type, variant selection, model, and input source.
