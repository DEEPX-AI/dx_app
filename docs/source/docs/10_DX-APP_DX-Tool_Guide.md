# DX-APP DX Tool Guide

This guide is intended for contributors and maintainers working on the DX-APP example repository.  

`./scripts/dx_tool.sh` is the unified developer entry point for repetitive example maintenance tasks such as model onboarding, package extraction, example discovery, validation, execution, and benchmarking.  

---

## Overview

The tool provides both:  

- **Interactive mode** for guided workflows  
- **Command mode** for repeatable developer operations and automation  

Primary script:  

- `./scripts/dx_tool.sh`  

Related helper scripts:  

- `./scripts/add_model.sh`  
- `./scripts/extract_model_package.sh`  
- `./scripts/extract_sln_package.bat`  
- `./scripts/validate_models.sh`  
- `./scripts/run_examples.sh`  
- `./scripts/bench_models.sh`  

Key data files:  

- `./config/model_registry.json` — model registry, single source of truth  

**When to Use `dx_tool.sh`**

Use `dx_tool.sh` when you need to:

- add a new example model to the repository structure
- inspect existing models by task or keyword
- extract a standalone package into the current example layout
- validate example assets and generated code structure
- run a filtered subset of examples
- benchmark examples across C++ and Python variants

For end-user installation, setup, and basic inference execution, refer to the installation and usage documents instead of this guide.

---

## Quick Start for Contributors

### Step 1. Full Developer Sequence

If you are adding or refactoring an example, follow this standard sequence to ensure everything is built and tested correctly  

```bash
# 1. Prepare assets & Build
./setup.sh
./build.sh --clean

# 2. Inspect & Validate repository state
./scripts/dx_tool.sh list
./scripts/dx_tool.sh validate

# 3. Run and Verify (C++ and/or Python)
./scripts/dx_tool.sh run --lang both
./run_tc.sh --cpp --cli
./run_tc.sh --python
```

This sequence prepares assets, rebuilds the repository, inspects available models, validates layout consistency, and performs a fast example-oriented test pass.  

### Step 2. Execution Modes

**Interactive mode** 

Run the tool with no subcommand to enter the guided menu.  

```bash
./scripts/dx_tool.sh
```

This mode is useful when:

- you are exploring the repository for the first time
- you do not remember the exact subcommand
- you want menu-based task selection

!!! note "NOTE" 

    `dx_tool.sh run` with no arguments delegates to `scripts/run_examples.sh` interactive mode, which provides a 6-stage guided menu (language, category, model filter, sync/async, input type, display/save options) with a configuration summary before execution. Each test also displays its performance table.

**Command mode**   

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

### Step 3. Command Summary

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

## Core Capabilities & Reference

### Model Management

**Model Registry**  

`config/model_registry.json` is a JSON array that serves as the single source of truth for all model metadata. Each entry contains:  

| Field | Purpose |
|-------|---------|
| `model_name` | Unique identifier (e.g., `yolov9s`) |
| `dxnn_file` | Compiled model filename (e.g., `yolov9-s_640x640.dxnn`) |
| `add_model_task` | Task category (e.g., `object_detection`) |
| `postprocessor` | Which shared processor to use (e.g., `yolov8`) |
| `input_width`, `input_height` | Model input dimensions |
| `config` | Extra parameters (thresholds, num_classes, etc.) |
| `supported` | Whether the model is part of the standard validation flow |

The `add` command reads this registry to auto-generate factory files, `config.json`, and all entry-point scripts.  

**Adding Examples**  

- Use `add` when you are onboarding a new model into an existing task category.  
- Use `new-task` when you are introducing a new task-level grouping that does not yet exist in the current repository layout.  

In both cases, validate the result and then update the related test registration if the example should be covered by automated tests.  

### Validation & Verification 

**Basic validation**  

```bash
./scripts/dx_tool.sh validate
# or directly:
./scripts/validate_models.sh --lang py
```

This runs code generation + NPU inference for all supported models.  

**`validate_models.sh` options**

| Option | Purpose |
|--------|---------|
| `--lang cpp\|py\|both` | Language filter |
| `--skip-verify` | Code generation only (no inference) |
| `--no-video` | Image-only mode |
| `--list` | Print commands without executing |
| `--clean` | Remove all generated packages |
| `--start-from <model>` | Resume from a specific model |
| `<task_filter>` | Filter by task (e.g., `object_detection`) |

### Execution & Benchmarking

- **Selective Run**: The `run` command abstracts the complexity of finding specific binaries. It uses filters to execute subsets of the 130+ models instantly.  

- **Performance Benchmarking**: The `bench` command runs models in a loop, calculating average latency and throughput (FPS) while minimizing system noise.  

---

## Common Workflows

### Step 1. New Model/Task Integration

**Step 1-1. Discover existing examples**  

```bash
./scripts/dx_tool.sh list
./scripts/dx_tool.sh search yolov8
./scripts/dx_tool.sh info yolov9
```

Use this first to avoid creating duplicate or inconsistent example names.

**Step 1-2. Add a new model example**  

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

For YOLO-family model onboarding and postprocessor selection details, refer to [DX-APP YOLO Customizing Guide](12_DX-APP_YOLO_Customizing_Guide.md).

```bash
# Create from a postprocessor template
./scripts/add_model.sh yolo_custom detection --postprocessor yolov8

# Copy from an existing model directory (useful for same-family variants)
./scripts/add_model.sh yolov7_w6 detection --base-model yolov7 --postprocessor yolov7

# Generate, verify, and push in one step
./scripts/add_model.sh yolo_custom detection --postprocessor yolov8 --verify --model assets/models/yolo_custom.dxnn --git-push
```

Common `--postprocessor` values include:

| Task | Common values |
|------|---------------|
| Object detection | `yolov5`, `yolov7`, `yolov8`, `yolov9`, `yolov10`, `yolov11`, `yolov12`, `yolov26`, `yolox`, `damoyolo`, `nanodet`, `ssd` |
| Semantic segmentation | `deeplabv3`, `bisenetv1`, `bisenetv2`, `segformer`, `fast_segmentation` |
| Instance segmentation | `yolov5seg`, `yolov8seg`, `yolov26seg` |
| Pose estimation | `yolov5pose`, `yolov8pose`, `yolov26pose` |
| Face detection | `scrfd`, `yolov5face`, `yolov7face` |

Use `fast_segmentation` for generic semantic-segmentation models that output low-resolution logits or class maps and should use the `FastSegmentationPostprocessor` fast path.

Key `add_model.sh` options

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

**Step 1-3. Verify the integration**  

After generation, verify that the new example is correctly integrated and runs without issues.  

```bash
# Check if all required files and registry entries are consistent
./scripts/dx_tool.sh validate

# Test the new example (e.g., for C++)
./scripts/dx_tool.sh run --lang cpp --model <your_model_name>
```

!!! note "NOTE"  

    If you used the `--verify` flag with `add_model.sh` in the previous step, this manual verification might be redundant but is still recommended for visual confirmation.

### Step 2. Packaging & Distribution

**Step 2-1. Extract a standalone package**

```bash
./scripts/dx_tool.sh extract
```

Use this when you need to convert an external model package into the repository layout used by DX-APP.  

**Step 2-2. Extract a Visual Studio solution package on Windows**

Use `extract_sln_package.bat` when you want to extract a single C++ example into a Visual Studio/CMake package that can be opened or built outside the full DX-APP solution.

```powershell
.\scripts\extract_sln_package.bat classification/resnet50 --output-dir out_resnet50
```

The output is created under:

```text
out_resnet50\sln\classification\resnet50\
```

If CMake and the Visual Studio 2022 generator are available, the extractor also configures the package immediately and generates a solution file:

```text
out_resnet50\sln\classification\resnet50\build\dxapp_resnet50_sln_package.sln
```

The package includes the selected model sources, shared C++ example helpers, `CMakeLists.txt`, `build.bat`, and generated dependency defaults. To build it directly:

```powershell
cd .\out_resnet50\sln\classification\resnet50
.\build.bat
```

OpenCV and DXRT paths are configured through CMake. The extractor writes the dependency defaults it can detect at extraction time to:

```text
cmake\dxapp_package_deps.cmake
cmake\dxapp_package_deps.bat
```

In the usual local developer environment, users should not need to edit Visual Studio property pages manually. If the package is moved to another PC or dependency paths change, set one of these variables before running `build.bat`, or edit `cmake\dxapp_package_deps.cmake`:

| Variable | Purpose |
|---|---|
| `DEEPX_SDK_DIR` | DEEPX SDK (DXRT) root |
| `DXRT_INSTALLED_DIR` | DXRT install root used for `include`, `lib`, and `bin` |
| `OpenCV_DIR` | OpenCV CMake package directory |
| `VCPKG_INSTALLED_DIR` | vcpkg installed tree used for runtime DLL lookup |

To create only the package skeleton without configuring CMake or generating `.sln` files:

```powershell
.\scripts\extract_sln_package.bat classification/resnet50 --output-dir out_resnet50 --no-generate-sln
```


### Step 3. Repository Maintenance

**Step 3-1. Validate repository consistency**  

```bash
./scripts/dx_tool.sh validate
```

Run validation after adding or restructuring examples. This helps catch mismatched files, missing variants, or incomplete model onboarding.  

**Step 3-2. Run examples selectively**  

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

**Step 3-3. Benchmark examples**  

```bash
./scripts/dx_tool.sh bench --lang cpp
./scripts/dx_tool.sh bench --lang py
```

Use `bench` when you want comparable runtime results across example variants.  

**Notes for Automation**  

For CI or scripted usage:

- prefer direct subcommands over interactive mode  
- keep `dx_tool.sh` for contributor automation, not end-user setup  
- use `setup.sh`, `build.sh`, and `run_tc.sh` for deterministic pipeline steps  

---

## Supplementary Information

### Script Relationships

`dx_tool.sh` is a **developer workflow tool**, not a replacement for every top-level script.  

- `setup.sh`: prepares shared assets such as models and videos  
- `build.sh`: builds C++ binaries and Python bindings  
- `run_tc.sh`: runs the repository test suites  
- `scripts/dx_tool.sh`: helps contributors add, inspect, validate, run, and benchmark example   content

A typical contributor workflow is:  

- (1) prepare assets with `./setup.sh`
- (2) build binaries with `./build.sh`  
- (3) inspect or add examples with `./scripts/dx_tool.sh`  
- (4) run validations and tests with `./scripts/dx_tool.sh validate` and `./run_tc.sh`  

### FAQ & Troubleshooting

The exact workflow depends on the contributor task, but the following command patterns are the most commonly used.  

**Inspect the current repository state**    

```bash
./scripts/dx_tool.sh list
./scripts/dx_tool.sh search yolov8
./scripts/dx_tool.sh info yolov9
```

**Validate after adding or refactoring examples**  

```bash
./scripts/dx_tool.sh validate
```

**Run language-specific example subsets**  

```bash
# Interactive — 6-stage guided menu
./scripts/dx_tool.sh run
scripts/run_examples.sh

# Non-interactive — pass options directly
./scripts/dx_tool.sh run --lang cpp
./scripts/dx_tool.sh run --lang py
./scripts/dx_tool.sh run --lang both
```

**Benchmark language-specific example subsets**  

```bash
./scripts/dx_tool.sh bench --lang cpp
./scripts/dx_tool.sh bench --lang py
```

### See Also

- `scripts/dx_tool.sh`
- `scripts/add_model.sh`
- `scripts/validate_models.sh`
- `scripts/run_examples.sh`
- `scripts/bench_models.sh`
- `config/model_registry.json`
- `docs/11_DX-APP_Example_Source_Structure.md`

---
