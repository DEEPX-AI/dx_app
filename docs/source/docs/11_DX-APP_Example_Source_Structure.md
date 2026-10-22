# DX-APP Example Source Structure

This guide is intended for contributors and maintainers who need to understand or extend the refactored DX-APP example layout.

---

## Overview 

The `src/` directory is the implementation root for DX-APP examples and shared runtime logic.

Its structure is designed to:

- separate C++ and Python example implementations clearly  
- organize examples by **AI task** and then by **model family**  
- share post-processing logic across languages  
- make onboarding of new models predictable  
- support testing, validation, and benchmarking workflows  

---

## Directory Hierarchy & Organization

### Top-Level Layout Strategy

```text
src/
├── cpp_example/          # C++ examples organized by task/model
│   └── common/           # ← C++ shared runtime layer
├── python_example/       # Python examples organized by task/model
│   └── common/           # ← Python shared runtime layer
├── postprocess/          # C++ post-processing (consumed by pybind11 bindings)
├── bindings/             # pybind11 bridge exposing src/postprocess/ to Python
└── utility/              # Shared utility code used by build flow
```

- **`src/cpp_example/`**: end-to-end C++ example applications — each model directory contains thin entry-point files that delegate to the shared `common/` layer  
- **`src/cpp_example/common/`**: shared C++ runtime layer — base interfaces (4 hpp), processors, runners, visualizers, input sources (5 hpp), config, and utilities  
- **`src/python_example/`**: end-to-end Python example applications — same factory-based delegation pattern as C++  
- **`src/python_example/common/`**: shared Python runtime layer — base interfaces (4 py), processors, runners, visualizers, input sources (5 py), config, and utilities  
- **`src/postprocess/`**: C++ post-processing libraries consumed by the **pybind11 bindings** (`src/bindings/`). This enables `*_cpp_postprocess` variants to use C++ decode logic from Python  
- **`src/bindings/`**: pybind11 bridge exposing `src/postprocess/` to Python as the `dx_postprocess` package  
- **`src/utility/`**: common support code shared across the build flow  

Both `cpp_example/common/` and `python_example/common/` share the same core modules: `base/`, `config/`, `processors/`, `runner/`, `inputs/`, `visualizers/`, `utility/`. They are language-specific implementations of the same architectural pattern.

**Task-First Organization**  

Both `cpp_example/` and `python_example/` follow the same task-first structure.  

Representative task directories include:  

- `image_classification/`  
- `object_detection/` — PPU models are a family here, for example `yolo_ppu/`  
- `face_detection/`  
- `pose_estimation/`  
- `semantic_segmentation/`  
- `instance_segmentation/`  
- `depth_estimation/`  
- `face_recognition/`  
- `image_denoising/`  
- `low_light_enhancement/`  
- `super_resolution/`  
- `oriented_object_detection/`  
- `hand_landmark/`  
- `person_attribute/`  
- `person_reid/`  
- `face_landmark/`  

This makes it easy to answer both questions below:  

- what task does this example belong to?  
- where should a new model example be added?  

**Family and Variant Subdirectories**  

Within each task directory, examples are split by model family, then by variant. The variant directory name is the `.dxnn` stem.

Examples:  

```text
src/cpp_example/object_detection/yolov9/yolov9-s_640x640/
src/cpp_example/object_detection/yolov8/yolov8-n_640x640/
src/python_example/object_detection/yolov9/yolov9-s_640x640/
src/python_example/object_detection/ssd/ssd-mobilenetv2-lite_300x300/
```

This gives each variant its own workspace for:  

- config files  
- source files  
- factory helpers  
- model-specific wiring  

---

## Language-Specific Structure & Variants

### Common Runtime Layer (common/)

**Shared Runtime Layer** (`common/`)

Both `src/cpp_example/common/` and `src/python_example/common/` implement the same shared runtime architecture. Each provides base interfaces, configuration loading, input sources, processors, runners, visualizers, and utilities — so individual model directories only need thin entry points and a factory.

C++ Common Structure

```text
src/cpp_example/common/
├── base/                          # Abstract interfaces (.hpp)
│   ├── i_factory.hpp              #   IFactory pattern: task-specific I<Task>Factory interfaces (e.g., IDetectionFactory) that create the preprocessor, postprocessor and visualizer
│   ├── i_processor.hpp            #   IPreprocessor / IPostprocessor — image → input tensor, model outputs → results
│   ├── i_visualizer.hpp           #   IVisualizer — draw(frame, results, ctx) → frame
│   └── i_input_source.hpp         #   IInputSource — image/video/camera/RTSP abstraction
├── config/
│   └── model_config.hpp           # ModelConfig — loads config.json
├── processors/                    # Shared processors
│   ├── yolo_detection_postprocessor.hpp     # YOLOv5/v7/v8 and the later YOLO detectors
│   ├── scrfd_face_postprocessor.hpp
│   ├── nanodet_postprocessor.hpp
│   ├── damoyolo_postprocessor.hpp
│   ├── ssd_postprocessor.hpp
│   ├── segmentation_postprocessor.hpp       # DeepLabV3 plus FastSegmentationPostprocessor
│   ├── anchorless_instance_seg_postprocessor.hpp
│   ├── depth_postprocessor.hpp
│   └── ...
├── runner/                        # Task-specific runner pairs
│   ├── sync_detection_runner.hpp  #   Sequential detection loop
│   ├── async_detection_runner.hpp #   Pipelined detection loop
│   ├── sync_classification_runner.hpp
│   ├── async_classification_runner.hpp
│   ├── sync_segmentation_runner.hpp
│   ├── async_segmentation_runner.hpp
│   └── ...                        #   one sync + one async runner per task type
├── inputs/                        # 5 input source headers
│   ├── camera_source.hpp
│   ├── image_source.hpp
│   ├── video_source.hpp
│   ├── rtsp_source.hpp
│   └── input_factory.hpp
├── visualizers/                   # Task-specific visualizers
│   ├── detection_visualizer.hpp
│   ├── classification_visualizer.hpp
│   ├── segmentation_visualizer.hpp
│   ├── face_visualizer.hpp
│   ├── pose_visualizer.hpp
│   └── ...
├── utility/                       # Utility headers
│   ├── common_util.hpp
│   ├── labels.hpp
│   ├── preprocessing.hpp
│   ├── profiling.hpp
│   ├── run_dir.hpp
│   ├── safe_queue.hpp
│   ├── verify_serialize.hpp
│   ├── visualization.hpp
│   └── ...                        #   colorspace, frame_reorder, kitti_calib,
│                                  #   lidar_util, sfa3d_geometry, sr_tiling
├── graph/                         # Multi-model graph engine
├── registry/                      # Concrete model registry used by the graph engine
├── trackers/                      # IoU tracker (stable instance colors across frames)
├── test/                          # Unit test for common/
└── third_party/                   # Third-party header-only libraries
    └── nlohmann_json.hpp          #   JSON for Modern C++
```

C++ runners are **task-specific**: each task type has a dedicated sync/async runner pair (e.g., `sync_detection_runner`, `async_detection_runner`). This provides optimal performance for each task's specific data flow.

Python Common Structure

```text
src/python_example/common/
├── base/                  # Abstract interfaces (.py)
│   ├── i_factory.py       #   task-specific I<Task>Factory interfaces (e.g., IDetectionFactory) — create the preprocessor, postprocessor and visualizer
│   ├── i_processor.py     #   IPreprocessor / IPostprocessor — image → input tensor, model outputs → results
│   ├── i_visualizer.py    #   IVisualizer — visualize(frame, results) → frame
│   └── i_input_source.py  #   IInputSource — image/video/camera/RTSP abstraction
├── config/
│   ├── config_schema.py   # Config schema validation for config.json values
│   └── model_config.py    # ModelConfig — loads config.json (input size, labels, thresholds)
├── processors/            # Shared processors
│   ├── yolo_postprocessor.py           # YOLOv5/v7/v8/v9/v10/v11/v12/YOLOX
│   ├── fast_yolo_postprocessor.py      # opt-in exact fast path (YOLOv5/v7, byte-identical)
│   ├── face_postprocessor.py           # SCRFD, YOLOv5Face, YOLOv7Face
│   ├── segmentation_postprocessor.py   # BiSeNet, DeepLabV3+, SegFormer
│   ├── fast_segmentation_postprocessor.py # opt-in fast path (SegFormer / low-res seg)
│   ├── instance_seg_postprocessor.py   # YOLOv5Seg, YOLOv8Seg, YOLOv26Seg
│   ├── fast_instance_seg_postprocessor.py # opt-in fast mask path (YOLOv8-seg)
│   ├── yolact_postprocessor.py         # YOLACT instance segmentation
│   ├── fast_yolact_postprocessor.py    # opt-in fast mask path (YOLACT)
│   ├── obb_postprocessor.py            # YOLOv26OBB
│   ├── pose_postprocessor.py           # YOLOv5-Pose, YOLOv8-Pose
│   ├── ppu_postprocessor.py            # PPU variants (YOLOv3/v4/v5/v7/v8/v9/v10/v11/v12, YOLOX)
│   ├── classification_postprocessor.py # EfficientNet, AlexNet, etc.
│   ├── efficientdet_postprocessor.py   # EfficientDet
│   ├── fast_efficientdet_postprocessor.py # opt-in exact fast path (EfficientDet, byte-identical)
│   ├── depth_postprocessor.py          # FastDepth, SCDepth
│   ├── nanodet_postprocessor.py        # NanoDet
│   ├── ssd_postprocessor.py            # SSD MobileNet
│   ├── damoyolo_postprocessor.py       # DAMOYOLO
│   ├── embedding_postprocessor.py      # GenericEmbeddingPostprocessor (base of CLIP / ArcFace)
│   ├── restoration_postprocessor.py    # DnCNN, Real-ESRGAN
│   ├── nms_utils.py                    # Shared NMS / box utilities
│   ├── letterbox_preprocessor.py       # Shared letterbox preprocessing
│   └── ...
├── runner/                # Generic runner files
│   ├── sync_runner.py     # SyncRunner — sequential Pre→Infer→Post→Display loop
│   ├── async_runner.py    # AsyncRunner — pipelined multi-thread runner
│   ├── args.py            # Unified CLI argument parser (--model, --image, --video, etc.)
│   ├── entry.py           # run_entry() used by every entry script
│   ├── interrupts.py      # Graceful stop on SIGINT/SIGTERM
│   ├── run_dir.py         # `--save` run-directory and run_info helper
│   ├── sr_tiling.py       # Tiled super-resolution helper
│   └── verify_serialize.py # Serialize results to JSON for numerical verification
├── inputs/                # 5 input source files
│   ├── image_source.py
│   ├── video_source.py
│   ├── camera_source.py
│   ├── rtsp_source.py
│   └── input_factory.py
├── visualizers/           # Task-specific visualizers
│   ├── detection_visualizer.py
│   ├── classification_visualizer.py
│   ├── segmentation_visualizer.py
│   ├── face_visualizer.py
│   ├── pose_visualizer.py
│   ├── instance_seg_visualizer.py
│   ├── obb_visualizer.py
│   └── ...
├── trackers/              # IoU tracker (stable instance colors across frames)
├── variant_config.py      # Builds processors from the variant config.json
├── variants.py            # Variant lookup in config/model_registry.json
└── utility/                       # Utility files
    ├── common_util.py     # General utilities
    ├── labels.py          # COCO / ImageNet label constants
    ├── preprocessing.py   # Shared resize/normalize/letterbox
    ├── profiling.py       # Stage-wise latency profiler
    ├── safe_queue.py      # Thread-safe queue for async pipeline
    ├── skeleton.py        # Pose skeleton definitions
    ├── visualization.py   # Drawing helpers (boxes, text, masks)
    └── ...                # colorspace, kitti_calib, lidar_input, sfa3d_geometry, video_io
```

Python runners are **generic**: `SyncRunner` and `AsyncRunner` work for all task types via the factory pattern. This provides simplicity and uniform usage across all models.

**Key Difference: C++ vs Python Runners**

| Aspect | C++ (`cpp_example/common/runner/`) | Python (`python_example/common/runner/`) |
|--------|-----------------------------------|-----------------------------------------|
| Runner set | One sync + one async runner per task type | `SyncRunner` + `AsyncRunner` plus shared helpers |
| Dispatch | Task-specific runner per category | Generic runner for all tasks |
| Example | `sync_detection_runner.hpp` | `sync_runner.py` |


### How Model Directories Connect to `common/`

**C++ variant directory**

The factory and the sync/async entries live in `<task>/<family>/<variant>/`. The directory name is the `.dxnn` stem.

```text
src/cpp_example/object_detection/yolov8/yolov8-n_640x640/
├── factory/
│   └── yolov8-n_640x640_factory.hpp
├── yolov8-n_640x640_sync.cpp        # Entry → sync_detection_runner
├── yolov8-n_640x640_async.cpp       # Entry → async_detection_runner
└── config.json
```

**Python variant directory**

The same folder holds the factory, `config.json`, and one thin script per execution mode. Each script fixes that variant.

```text
src/python_example/object_detection/yolov8/yolov8-n_640x640/
├── factory/
│   └── yolov8-n_640x640_factory.py
├── config.json
├── yolov8-n_640x640_sync.py
├── yolov8-n_640x640_async.py
├── yolov8-n_640x640_sync_cpp_postprocess.py
└── yolov8-n_640x640_async_cpp_postprocess.py
```

The factory imports shared components from `common/`:

```cpp
// C++ factory example
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
```

```python
# Python factory example (config-driven: processors come from the variant's config.json)
from common.base import IDetectionFactory
from common.variant_config import build_processor, default_variant, load_variant_config
```

The entry-point script/program delegates to the runner:

```python
# Python
from common.runner import SyncRunner
from common.runner.entry import run_entry

run_entry(_script, factory_cls=Yolov9Factory, runner_cls=SyncRunner,
          description="yolov9-s_640x640 sync inference",
          fixed_variant="yolov9-s_640x640")
```

Adding a variant means a new `<task>/<family>/<variant>/` directory with `config.json`, a factory, and that variant's entry files.

### Entry Naming

**C++**

```text
src/cpp_example/<task>/<family>/<variant>/
├── factory/<variant>_factory.hpp
├── <variant>_sync.cpp
├── <variant>_async.cpp
└── config.json
```

- `<variant>_sync.cpp`: sequential execution for that variant
- `<variant>_async.cpp`: pipelined execution for that variant
- `bin/<variant>_sync` and `bin/<variant>_async`: built executables

**Python**

```text
src/python_example/<task>/<family>/<variant>/
├── factory/<variant>_factory.py
├── config.json
├── <variant>_sync.py
├── <variant>_async.py
├── <variant>_sync_cpp_postprocess.py
└── <variant>_async_cpp_postprocess.py
```

- `*_sync.py`: synchronous path
- `*_async.py`: asynchronous path
- `*_sync_cpp_postprocess.py`: synchronous path using shared C++ post-processing bindings
- `*_async_cpp_postprocess.py`: asynchronous path using shared C++ post-processing bindings

---

## Cross-Language Integration & Frameworks

### Shared Post-Processing Layer

**C++ and Python Shared Runtime** (`common/`)

Both `cpp_example/` and `python_example/` contain a `common/` directory with the same core modules:

| Module | Contents | Role |
|--------|----------|------|
| `common/base/` | 4 interface headers (.hpp) | `IFactory`, `IPreprocessor` / `IPostprocessor`, `IVisualizer`, `IInputSource` |
| `common/config/` | `model_config.hpp` | Loads `config.json` (input size, labels, thresholds) |
| `common/processors/` | Postprocessor / preprocessor headers | Shared decode logic for all model families |
| `common/runner/` | Runner headers | A dedicated sync/async runner pair per task type |
| `common/inputs/` | 5 source headers | Image, Video, Camera, RTSP input abstraction |
| `common/visualizers/` | Visualizer headers | Task-specific result rendering |
| `common/utility/` | Utility headers | Labels, preprocessing, profiling, run_dir, colorspace, verify_serialize |

This is the core architectural pattern of DX-APP: model directories are thin wrappers (factory + entry points) that delegate all heavy logic to their respective `common/` layer.  

**Pybind11 Post-Processing Bridge** (`src/postprocess/` + `src/bindings/`)  

`src/postprocess/` contains C++ post-processing implementations that are **not** linked by the C++ examples, except for four modules (`sfa3d`, `superpoint`, `dope`, `yolopv2`) whose sources are compiled into them. They are consumed by the pybind11 bindings under `src/bindings/python/dx_postprocess/`.  

This bridge enables the `*_cpp_postprocess.py` Python variants to use C++ decode logic for higher performance, while the pure Python variants (`*_sync.py`, `*_async.py`) use `python_example/common/processors/` natively.  

| Variant | Post-processing source |
|---------|----------------------|
| `*_sync.cpp` / `*_async.cpp` | `cpp_example/common/processors/` |
| `*_sync.py` / `*_async.py` | `python_example/common/processors/` |
| `*_sync_cpp_postprocess.py` / `*_async_cpp_postprocess.py` | `src/postprocess/` via pybind11 |

### Numerical Verification (`DXAPP_VERIFY`)

DX-APP can serialize post-processing results to JSON for inspection and debugging:  

- **`DXAPP_VERIFY=1`**: serializes post-processing results to `logs/verify/{model}.json` (last frame) and `{model}.frames.jsonl` (every frame, in input order)  
- **`common/runner/verify_serialize.py`**: serializes postprocess results to JSON for comparison  

This helps catch regressions such as broken post-processing, incorrect model configurations, or NPU output changes.  

---

## Management & Extension

### Model Registry & Automation

**`config/model_registry.json`**

The model registry is a JSON array that serves as the single source of truth for model metadata:

```json
{
  "model_name": "yolov9s",
  "dxnn_file": "yolov9-s_640x640.dxnn",
  "variant": "yolov9-s_640x640",
  "family": "yolov9",
  "task": "object_detection",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "input_width": 640,
  "input_height": 640,
  "config": { "score_threshold": 0.25, "nms_threshold": 0.45 },
  "supported": true
}
```

`model_name` is the registry key used by `./setup.sh --models`. The example directory is `<task>/<family>/<variant>`, here `object_detection/yolov9/yolov9-s_640x640`.

**`scripts/add_model.sh`**

The `add_model.sh` script copies a reference example picked from the registry and rewrites, for the new variant:

- `config.json` (variant, model file, input size, registry thresholds; processor classes from the reference)
- Python `factory/<variant>_factory.py` and `factory/__init__.py` (plus `custom_ops.py` when the reference has one)
- the entry-point scripts the reference has (sync/async × python/cpp_postprocess)
- C++ `factory/<variant>_factory.hpp` (namespace `dxapp::v_<variant>`), `<variant>_sync.cpp` and `<variant>_async.cpp` under `src/cpp_example/<task>/<family>/<variant>/`
- a `config/model_registry.json` row and a `config/test_models.conf` line (unless `--no-register`)

This enables onboarding a new model with zero manual code in most cases; see the [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md) for the options.

### Contributor Rules

**Adding New Examples**  

When adding a new example, keep the following rules:  

- (1) choose the correct task directory first  
- (2) create `<task>/<family>/<variant>/` for that model  
- (3) follow existing variant naming conventions  
- (4) place shared decode logic in `src/cpp_example/common/processors/` (C++) or `src/python_example/common/processors/` (Python) when appropriate  
- (5) add pybind11 bindings in `src/postprocess/` + `src/bindings/` only if the new flow needs `*_cpp_postprocess.py` variants  
- (6) register the model in `config/model_registry.json` if it should be part of the standard validation flow  
- (7) update validation/test registration where required  

**Contributor checklist**  

When a new example is intended to become part of the maintained repository flow, confirm all of the following:  

- source files are placed under the correct `<task>/<family>/<variant>/` directory  
- naming follows the current variant convention  
- required shared post-processing logic exists or is added under `src/cpp_example/common/processors/` (C++) or `src/python_example/common/processors/` (Python)  
- model assets can be prepared through the standard setup flow  
- validation passes  
- test registration is updated if automated coverage is required  
- user-facing or contributor-facing docs are updated when the structure meaningfully changes  

---

## Supplementary Information

### Connection to Testing

The source tree and the test tree are related, but not identical.

**Shared Test Infrastructure** (`tests/test_helpers/`)  

`tests/test_helpers/` provides shared constants and utilities used by both C++ and Python test suites:  

- `constants.py` — paths, sample-image maps, timeout values  
- `utils.py` — executable/script and model discovery, environment setup (full module list: see [DX-APP C++ Example Tests](04_DX-APP_CPP_Example_Test.md))  

**C++ tests**  

C++ tests are organized as follows. For the full list, see [DX-APP C++ Example Tests](04_DX-APP_CPP_Example_Test.md).

| Category | File | Description |
|----------|------|-------------|
| CLI Help | `test_cli_help.py` | Validates `--help` output for all executables |
| CLI Basic | `test_cli_basic.py` | Validates invalid-argument and no-argument handling |
| E2E | `test_e2e.py` | End-to-end inference with image/video inputs |
| Visualization | `test_visualization.py` | Validates output image generation with `--save` |
| Feature: Save Mode | `test_save_mode.py` | Tests `--save` / `--save-dir` behavior |
| Feature: Dump Tensors | `test_dump_tensors.py` | Tests `--dump-tensors` output |
| Feature: Verify | `test_verify.py` | Tests `DXAPP_VERIFY` environment variable |
| Feature: Multi-Loop | `test_multi_loop.py` | Tests `--loop` repeated execution |
| Feature: Signal | `test_signal_handling.py` | Tests SIGINT graceful shutdown |

The tests discover the executables in `bin/`.

**Python tests**  

- Python example tests discover variant scripts through `tests/test_helpers/utils.py` and `config/model_registry.json`  
- `tests/python_example/test_visualization.py` validates output image generation across task types  
- `tests/python_example/unit/` holds processor and layout checks  
- adding a new source directory alone does **not** automatically guarantee full test coverage  

Relevant files include:  

- `tests/test_helpers/utils.py`  
- `tests/test_helpers/constants.py`  
- `tests/python_example/test_visualization.py`  
- `tests/python_example/unit/`  
- `config/test_models.conf`  

**Model validation**  

- `scripts/validate_models.sh` runs registry-driven validation across all supported models  
- `config/model_registry.json` is the primary reference for which models are validated  

This means source layout, test coverage, and registry entries must be updated together when onboarding new examples.  

### Developer Resources

**Recommended Onboarding**  

When you add or refactor a model example:  

- (1) create or update the source layout under `src/`  
- (2) prepare the required model assets  
- (3) build the repository  
- (4) validate the example structure  
- (5) run relevant tests  

Typical related commands:  

```bash
./setup.sh
./build.sh --clean --all
./scripts/dx_tool.sh validate
./scripts/validate_models.sh --lang py
./run_tc.sh --cpp --cli
./run_tc.sh --python
```

If the example is intended to be part of the standard repository workflow, also review the relevant documentation pages so that user and contributor guidance stays aligned with the current structure.

### See Also

- `src/cpp_example/`
- `src/cpp_example/common/`
- `src/python_example/`
- `src/python_example/common/`
- `src/postprocess/` (pybind11 source)
- `src/bindings/python/dx_postprocess/`
- `config/model_registry.json`
- `scripts/validate_models.sh`
- `tests/cpp_example/`
- `tests/python_example/`
- [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md)

---
