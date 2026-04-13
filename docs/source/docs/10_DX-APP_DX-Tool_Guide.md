# DX-APP DX Tool Guide

This document is intended for contributors and maintainers working on the DX-APP example repository.

`scripts/dx_tool.sh` is the unified developer entry point for repetitive example maintenance tasks such as model onboarding, package extraction, example discovery, validation, execution, and benchmarking.

---

## Overview

The tool provides both:

- **Interactive mode** for guided workflows
- **Command mode** for repeatable developer operations and automation

Primary script:

- `scripts/dx_tool.sh`

Related helper scripts:

- `scripts/add_model.sh`
- `scripts/extract_model_package.sh`
- `scripts/validate_models.sh`
- `scripts/verify_inference_output.py`
- `scripts/run_examples.sh`
- `scripts/bench_models.sh`

Key data files:

- `config/model_registry.json` — model registry, single source of truth
- `scripts/inference_verify_rules.json` — numerical verification thresholds per task

---

## When to Use `dx_tool.sh`

Use `dx_tool.sh` when you need to:

- add a new example model to the repository structure
- inspect existing models by task or keyword
- extract a standalone package into the current example layout
- validate example assets and generated code structure
- run a filtered subset of examples
- benchmark examples across C++ and Python variants

For end-user installation, setup, and basic inference execution, refer to the installation and usage documents instead of this guide.

---

## Execution Modes

### Interactive mode

Run the tool with no subcommand to enter the guided menu.

```bash
./scripts/dx_tool.sh
```

This mode is useful when:

- you are exploring the repository for the first time
- you do not remember the exact subcommand
- you want menu-based task selection

> **Note:** `dx_tool.sh run` with no arguments delegates to `scripts/run_examples.sh` interactive mode,
> which provides a 6-stage guided menu (language, category, model filter, sync/async, input type, display/save options)
> with a configuration summary before execution. Each test also displays its performance table.

### Command mode

Run a subcommand directly when you already know the intended task.

```bash
./scripts/dx_tool.sh list
./scripts/dx_tool.sh search yolov9
./scripts/dx_tool.sh validate
```

This mode is better for:

- repeatable contributor workflows
- shell history reuse
- scripting and CI-friendly operations

---

## Command Summary

| Command | Purpose |
|---|---|
| `add` | Create a new model/example skeleton |
| `extract` | Extract a standalone package into repository layout |
| `list` | List registered models |
| `search` | Search models by keyword |
| `info` | Show model details |
| `delete` | Remove a model from the repository |
| `new-task` | Create a new task directory skeleton |
| `validate` | Validate current model/example layout |
| `run` | Execute examples with filters |
| `bench` | Benchmark examples with filters |
| `help` | Show usage help |

---

## Model Registry

`config/model_registry.json` is a JSON array that serves as the single source of truth for all model metadata. Each entry contains:

| Field | Purpose |
|-------|---------|
| `model_name` | Unique identifier (e.g., `yolov9s`) |
| `dxnn_file` | Compiled model filename (e.g., `YOLOV9S.dxnn`) |
| `add_model_task` | Task category (e.g., `object_detection`) |
| `postprocessor` | Which shared processor to use (e.g., `yolov8`) |
| `input_width`, `input_height` | Model input dimensions |
| `config` | Extra parameters (thresholds, num_classes, etc.) |
| `supported` | Whether the model is part of the standard validation flow |

The `add` command reads this registry to auto-generate factory files, `config.json`, and all entry-point scripts.

---

## Model Validation and Numerical Verification

### Basic validation

```bash
./scripts/dx_tool.sh validate
# or directly:
./scripts/validate_models.sh --lang py
```

This runs code generation + NPU inference for all supported models.

### Numerical verification

```bash
./scripts/validate_models.sh --numerical --lang py
```

This additionally verifies that inference outputs are numerically correct:

1. **Inference** — runs each model through NPU
2. **Serialization** — `common/runner/verify_serialize.py` converts results to JSON
3. **Validation** — `scripts/verify_inference_output.py` checks results against `scripts/inference_verify_rules.json`

Verification covers 17 task types: bounding boxes, confidence ranges, class IDs, keypoints, segmentation masks, depth maps, embeddings, attributes, re-identification, face alignment, etc.

### `validate_models.sh` options

| Option | Purpose |
|--------|---------|
| `--lang cpp\|py\|both` | Language filter |
| `--numerical` | Enable output verification |
| `--skip-verify` | Code generation only (no inference) |
| `--no-video` | Image-only mode |
| `--list` | Print commands without executing |
| `--clean` | Remove all generated packages |
| `--start-from <model>` | Resume from a specific model |
| `<task_filter>` | Filter by task (e.g., `object_detection`) |

---

## Frequently Used Command Patterns

The exact workflow depends on the contributor task, but the following command patterns are the most commonly used.

### Inspect the current repository state

```bash
./scripts/dx_tool.sh list
./scripts/dx_tool.sh search yolov8
./scripts/dx_tool.sh info yolov9
```

### Validate after adding or refactoring examples

```bash
./scripts/dx_tool.sh validate
```

### Run language-specific example subsets

```bash
# Interactive — 6-stage guided menu
./scripts/dx_tool.sh run
scripts/run_examples.sh

# Non-interactive — pass options directly
./scripts/dx_tool.sh run --lang cpp
./scripts/dx_tool.sh run --lang py
./scripts/dx_tool.sh run --lang both
```

### Benchmark language-specific example subsets

```bash
./scripts/dx_tool.sh bench --lang cpp
./scripts/dx_tool.sh bench --lang py
```

---

## Common Workflows

### 1. Discover existing examples

```bash
./scripts/dx_tool.sh list
./scripts/dx_tool.sh search yolov8
./scripts/dx_tool.sh info yolov9
```

Use this first to avoid creating duplicate or inconsistent example names.

### 2. Add a new model example

```bash
./scripts/dx_tool.sh add
```

The add flow is intended for contributors creating a new example under the current task/model layout. Typical information includes:

- target language (`cpp`, `py`, or both)
- task category
- model name
- post-processing selection
- sync-only or multi-variant generation choice

You can also invoke `add_model.sh` directly for non-interactive usage:

```bash
# Create from a postprocessor template
./scripts/add_model.sh yolov30 detection --postprocessor yolov8

# Copy from an existing model directory (useful for same-family variants)
./scripts/add_model.sh yolov7_w6 detection --base-model yolov7 --postprocessor yolov7

# Generate, verify, and push in one step
./scripts/add_model.sh yolov30 detection --postprocessor yolov8 --verify --model assets/models/YoloV30.dxnn --git-push
```

Key `add_model.sh` options:

| Option | Description |
|--------|-------------|
| `--postprocessor <type>` | Select the post-processing family to use as template |
| `--base-model <name>` | Copy from a specific existing model directory instead of the default reference |
| `--lang <cpp\|py\|both>` | Target language (default: `both`) |
| `--verify` | Build and run inference verification after generation |
| `--model <path>` | `.dxnn` model file for `--verify` |
| `--no-video` | Skip video verification (image only) |
| `--git-push` | After successful `--verify`, commit and push |
| `--auto-add` | Batch-generate source packages for all unregistered models |

After generation, review the resulting files under `src/cpp_example/` and/or `src/python_example/`.

### 3. Extract a standalone package

```bash
./scripts/dx_tool.sh extract
```

Use this when you need to convert an external model package into the repository layout used by DX-APP.

### 4. Validate repository consistency

```bash
./scripts/dx_tool.sh validate
```

Run validation after adding or restructuring examples. This helps catch mismatched files, missing variants, or incomplete model onboarding.

### 5. Run examples selectively

```bash
# Interactive — guided category/model selection with performance output
./scripts/dx_tool.sh run
scripts/run_examples.sh

# Non-interactive
./scripts/dx_tool.sh run --lang cpp
./scripts/dx_tool.sh run --lang py
./scripts/dx_tool.sh run --lang both
```

Use `run` to execute filtered example sets without manually locating every command.
In interactive mode, selecting a category shows all available models in that category.

### 6. Benchmark examples

```bash
./scripts/dx_tool.sh bench --lang cpp
./scripts/dx_tool.sh bench --lang py
```

Use `bench` when you want comparable runtime results across example variants.

---

## First 5 Commands for a Contributor

If you are new to the repository, this is a practical starting sequence.

```bash
./setup.sh
./build.sh --clean
./scripts/dx_tool.sh list
./scripts/dx_tool.sh validate
./scripts/validate_models.sh --numerical --lang py
./run_tc.sh --cpp --cli
```

This sequence prepares assets, rebuilds the repository, inspects available models, validates layout consistency, and performs a fast example-oriented test pass.

---

## When to Use `add` vs `new-task`

- Use `add` when you are onboarding a new model into an existing task category.
- Use `new-task` when you are introducing a new task-level grouping that does not yet exist in the current repository layout.

In both cases, validate the result and then update the related test registration if the example should be covered by automated tests.

---

## Relationship to Other Top-Level Scripts

`dx_tool.sh` is a **developer workflow tool**, not a replacement for every top-level script.

- `setup.sh`: prepares shared assets such as models and videos
- `build.sh`: builds C++ binaries and Python bindings
- `run_tc.sh`: runs the repository test suites
- `scripts/dx_tool.sh`: helps contributors add, inspect, validate, run, and benchmark example content

A typical contributor workflow is:

1. prepare assets with `./setup.sh`
2. build binaries with `./build.sh`
3. inspect or add examples with `./scripts/dx_tool.sh`
4. run validations and tests with `./scripts/dx_tool.sh validate` and `./run_tc.sh`

---

## Recommended Contributor Workflow

### Add or refactor an example

1. inspect the existing repository layout
2. add or update the example structure
3. prepare required models/assets
4. build the repository
5. validate layout and run examples
6. execute relevant tests

Example sequence:

```bash
./setup.sh
./build.sh --clean
./scripts/dx_tool.sh validate
./scripts/dx_tool.sh run --lang cpp
./run_tc.sh --cpp --cli
```

If the change also affects Python examples, extend the flow with:

```bash
./scripts/dx_tool.sh run --lang py
./run_tc.sh --python
```

---

## Notes for Automation

For CI or scripted usage:

- prefer direct subcommands over interactive mode
- keep `dx_tool.sh` for contributor automation, not end-user setup
- use `setup.sh`, `build.sh`, and `run_tc.sh` for deterministic pipeline steps

---

## See Also

- `scripts/dx_tool.sh`
- `scripts/add_model.sh`
- `scripts/validate_models.sh`
- `scripts/verify_inference_output.py`
- `scripts/inference_verify_rules.json`
- `scripts/run_examples.sh`
- `scripts/bench_models.sh`
- `config/model_registry.json`
- `docs/11_DX-APP_Example_Source_Structure.md`
