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
- export an example as a standalone package for deployment
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
./build.sh --clean --all

# 2. Inspect & Validate repository state
./scripts/dx_tool.sh list
./scripts/dx_tool.sh validate

# 3. Run and Verify (C++ and/or Python)
./scripts/dx_tool.sh run --lang both
./run_tc.sh --cpp --cli
./run_tc.sh --python
```

This sequence prepares assets, rebuilds the repository, inspects available models, runs registry-driven validation, and performs an example-oriented test pass.  

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

`validate` prompts for the language and scope; for scripts, call `./scripts/validate_models.sh` directly.  

This mode is better for:  

- repeatable contributor workflows  
- shell history reuse  
- scripting and CI-friendly operations  

### Step 3. Command Summary

| Command | Purpose |
|---|---|
| `add` | Create a new model/example skeleton |
| `extract` | Export an example as a standalone package |
| `list` | List example models by task (C++/Python) |
| `search` | Search models by keyword |
| `info` | Show model details |
| `delete` | Delete example directories (`<task>/<family>/<variant>`, or a bare family name = the whole family); registry and `test_models.conf` rows are left in place |
| `new-task` | Generate skeleton `common/` files (postprocessor, visualizer, factory interface, sync/async runner) for a new task type |
| `validate` | Run registry-driven validation via `validate_models.sh`: build and run (NPU inference) every supported registry model, generating any that has no example yet |
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
| `model_name` | Registry key used by `./setup.sh --models` (e.g., `yolov9s`) |
| `variant` | Example directory and `.dxnn` stem (e.g., `yolov9-s_640x640`) |
| `family` | Family directory (e.g., `yolov9`) |
| `task` | Task directory (e.g., `object_detection`) |
| `dxnn_file` | Compiled model filename (e.g., `yolov9-s_640x640.dxnn`) |
| `add_model_task` | Legacy task name passed to `add_model.sh` by `validate_models.sh` (e.g., `object_detection`, `classification`, `ppu`) |
| `postprocessor` | Which shared processor to use (e.g., `yolov8`) |
| `input_width`, `input_height` | Model input dimensions |
| `config` | Extra parameters (thresholds, num_classes, etc.) |
| `supported` | Whether the model is part of the standard validation flow |
| `alias_of`, `published`, `image_only`, ... | Alias, publication and test metadata |

The `add` command copies a reference example picked from this registry and rewrites its factory files, `config.json`, and entry-point scripts for the new variant.  

**Adding Examples**  

- Use `add` when you are onboarding a new model into an existing task category.  
- Use `new-task` when you are introducing a new task-level grouping that does not yet exist in the current repository layout.  

In both cases, validate the result. `add_model.sh` registers the new variant in `model_registry.json` and `test_models.conf` automatically (unless `--no-register` is given). For a new task, check whether the test helpers need the new task as well.  

### Validation & Verification 

**Basic validation**  

```bash
./scripts/dx_tool.sh validate
# or directly:
./scripts/validate_models.sh --lang py
```

This builds and runs (NPU inference) the example of every supported registry row whose `.dxnn` is present; a row without an example is generated first. With `--skip-verify` only missing examples are generated.  

**`validate_models.sh` options**

| Option | Purpose |
|--------|---------|
| `--lang <lang>` | Language filter: `cpp`, `py` or `both` |
| `--skip-verify` | Code generation only (no inference) |
| `--no-video` | Image-only mode |
| `--list` | Print commands without executing |
| `--clean` | Legacy cleanup for the pre-v3.3.0 flat layout. **Do not run it on the current tree**: it deletes every `<task>/<family>/` directory whose name is not a registry `model_name`, including committed examples |
| `--start-from <model_name>` | Resume from a registry `model_name` |
| `<task_filter>` | Filter by registry `add_model_task` (e.g., `object_detection`, `classification`, `pose`, `obb`, `ppu`); task directory names such as `image_classification` are rejected |

### Execution & Benchmarking

- **Selective Run**: The `run` command abstracts the complexity of finding specific binaries. It uses filters to execute subsets of the 497 registered models instantly.  

- **Performance Benchmarking**: The `bench` command runs models in a loop, calculating average latency and throughput (FPS) while minimizing system noise.  

---

## Common Workflows

### Step 1. New Model/Task Integration

**Step 1-1. Discover existing examples**  

```bash
./scripts/dx_tool.sh list
./scripts/dx_tool.sh search yolov8
./scripts/dx_tool.sh info yolov9-s_640x640
```

Use this first to avoid creating duplicate or inconsistent example names.

**Step 1-2. Add a new model example**  

```bash
./scripts/dx_tool.sh add
```

The add flow is intended for contributors creating a new example under the `<task>/<family>/<variant>/` layout. Typical information includes:  

- target language (C++, Python, or both)  
- task type  
- postprocessor (reference)  
- variant name (the `.dxnn` stem)  
- family directory (empty = the reference's family)  
- generation mode (sync + async, or sync only)  

For YOLO-family model onboarding and postprocessor selection details, refer to [DX-APP YOLO Customizing Guide](12_DX-APP_YOLO_Customizing_Guide.md).

You can also invoke `add_model.sh` directly for non-interactive usage:  

```bash
# Create from a postprocessor family (the reference is the alphabetically first registered variant that uses it, preferring the same task)
./scripts/add_model.sh yolo-custom_640x640 object_detection --postprocessor yolov8

# Copy from an existing variant (useful for same-family variants)
./scripts/add_model.sh yolov7-custom_1280x1280 object_detection --base-model yolov7_640x640 --input-size 1280x1280

# Open a new family for the variant
./scripts/add_model.sh my-det_640x640 object_detection --family my_det --postprocessor yolov5

# Generate and verify in one step
./scripts/add_model.sh yolo-custom_640x640 object_detection --postprocessor yolov8 --verify --model assets/models/yolo-custom_640x640.dxnn
```

The first argument is the variant: the `.dxnn` stem, which is also the directory name. The example lands in `src/{cpp,python}_example/<task>/<family>/<variant>/` with the C++ factory scoped in `dxapp::v_<variant>`, `config.json` names the variant, and `config/model_registry.json` and `config/test_models.conf` gain the variant unless `--no-register` is given (a variant that is already registered takes its family, model file and input size from its row). The generator is `scripts/add_model_variant.py`; `add_model.sh` adds `--verify` (with `--model`, `--image`, `--video`, `--no-video`) and `--auto-add` around it.

Common `--postprocessor` values include:

| Task | Common values |
|------|---------------|
| Object detection | `yolov5`, `yolov7`, `yolov8`, `yolov9`, `yolov10`, `yolov11`, `yolov12`, `yolov13`, `yolov26`, `yolox`, `damoyolo`, `nanodet`, `ssd` |
| Semantic segmentation | `deeplabv3`, `bisenetv1`, `bisenetv2`, `segformer`, `fast_segmentation` |
| Instance segmentation | `yolov5seg`, `yolov8seg`, `yolov26seg` |
| Pose estimation | `yolov5pose`, `yolov8pose`, `yolov26pose` |
| Face detection | `scrfd`, `yolov5face`, `yolov7face` |

Use `fast_segmentation` for generic semantic-segmentation models that output low-resolution logits or class maps and should use the `FastSegmentationPostprocessor` fast path.

Key `add_model.sh` options

| Option | Description |
|--------|-------------|
| `--postprocessor <type>` | Select the post-processing family to use as template |
| `--base-model <variant>` | Copy from a specific existing variant instead of the default reference |
| `--family <name>` | Family directory (default: the registry row's family, else the reference's) |
| `--input-size <WxH>` | Model input size (default: the registry row's, else the reference's) |
| `--dxnn <file>` | Model file name (default: `<variant>.dxnn`) |
| `--no-register` | Do not add the variant to `model_registry.json` / `test_models.conf` |
| `--lang <lang>` | Target language: `cpp`, `py` or `both` (default: `both`) |
| `--sync-only` | Generate only the sync entries |
| `--verify` | Build and run inference verification after generation |
| `--model <path>` | `.dxnn` model file for `--verify` |
| `--no-video` | Skip video verification (image only) |
| `--auto-add` | Generate the example of every registry row that has none |

After generation, review the resulting files under `src/cpp_example/` and/or `src/python_example/`.

**Step 1-3. Verify the integration**  

After generation, verify that the new example is correctly integrated and runs without issues.  

```bash
# Build and run every supported registry model (long; interactive)
./scripts/dx_tool.sh validate

# Test the new example (e.g., for C++)
./scripts/dx_tool.sh run --lang cpp --filter <variant>
```

!!! note "NOTE"  

    If you used the `--verify` flag with `add_model.sh` in the previous step, this manual verification might be redundant but is still recommended for visual confirmation.

### Step 2. Packaging & Distribution

**Step 2-1. Extract a standalone package**

```bash
./scripts/dx_tool.sh extract
```

Use this when you need a standalone copy of one example (its model directory plus the `common/` files it depends on) for deployment outside the repository.  

The extracted package carries only the `common/` files the model actually depends on: only a fraction of the shared files, instead of
every postprocessor, runner, and visualizer for all 29 tasks.

To get the complete shared framework instead, answer `n` at the `Prune unused common/ files?
[Y/n]` prompt, or pass `--no-prune`:

```bash
./scripts/extract_model_package.sh object_detection/yolov7 --output-dir out_yolov7 --no-prune
```

Pruned and full packages are equivalent: they compile identically and produce byte-identical
inference output. Note that the Python visualizers use an unseeded color palette, so rendered colors
differ between runs either way.

**Step 2-2. Extract a Visual Studio solution package on Windows**

Use `extract_sln_package.bat` when you want to extract a single C++ example into a Visual Studio/CMake package that can be opened or built outside the full DX-APP solution.

```powershell
.\scripts\extract_sln_package.bat image_classification/resnet/resnet50_224x224 --output-dir out_resnet50
```

The output is created under:

```text
out_resnet50\sln\image_classification\resnet50_224x224\
```

If CMake and the Visual Studio 2022 generator are available, the extractor also configures the package immediately and generates a solution file:

```text
out_resnet50\sln\image_classification\resnet50_224x224\build\dxapp_resnet50_224x224_sln_package.sln
```

The package includes the selected model sources, shared C++ example helpers, `CMakeLists.txt`, `build.bat`, and generated dependency defaults. To build it directly:

```powershell
cd .\out_resnet50\sln\image_classification\resnet50_224x224
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
.\scripts\extract_sln_package.bat image_classification/resnet/resnet50_224x224 --output-dir out_resnet50 --no-generate-sln
```


### Step 3. Repository Maintenance

**Step 3-1. Validate repository consistency**  

```bash
./scripts/dx_tool.sh validate
```

Run validation after adding or restructuring examples. It builds and runs every supported registry model, so it catches examples that no longer build or run, and it generates any registered variant that has no example yet.  

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
./scripts/dx_tool.sh info yolov9-s_640x640
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
- [DX-APP Example Source Structure](11_DX-APP_Example_Source_Structure.md)

---
