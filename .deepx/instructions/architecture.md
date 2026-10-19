# dx_app v3.0.0 Architecture

This document describes the architecture of **dx_app only** — the standalone
inference application framework. It does not cover dx_stream (GStreamer pipelines)
or any streaming components.

## Overview

dx_app provides a three-layer architecture for building AI inference applications
on DEEPX NPU hardware. Applications are organized by AI task (28 categories) and
model (497 variants), with a shared framework providing common runners, factories,
preprocessors, postprocessors, and visualizers.

## Three-Layer Architecture

```
+===================================================================+
|  Layer 3: Application Layer                                        |
|  src/python_example/<task>/<family>/<variant>/                     |
|      config.json + factory/ + 4 entry scripts                      |
|  src/cpp_example/<task>/<family>/<variant>/                        |
|      config.json + factory/ + sync/async entries                   |
+===================================================================+
        |                                       |
        | uses                                  | uses
        v                                       v
+============================+   +============================+
|  Layer 2: Framework Layer  |   |  Layer 2: C++ Framework    |
|  common/                   |   |  common/                   |
|    base/     (IFactory)    |   |    runner/                 |
|    runner/   (Sync/Async)  |   |    base/                   |
|    processors/             |   |    processors/             |
|    visualizers/            |   |    visualizers/            |
|    inputs/                 |   +============================+
|    config/                 |
|    utility/                |
+============================+
        |
        | imports
        v
+===================================================================+
|  Layer 1: C++ Core                                                 |
|  dx_engine       InferenceEngine, InferenceOption, Configuration   |
|  dx_postprocess  37 pybind11 postprocess bindings                  |
+===================================================================+
        |
        | talks to
        v
+===================================================================+
|  DEEPX NPU Hardware (DX-M1 / DX-M1A (discontinued))               |
+===================================================================+
```

## Layer 1: C++ Core

### dx_engine

The NPU runtime library. Provides:

- **InferenceEngine**: Load a `.dxnn` model file and run inference.
  - `InferenceEngine(model_path)` — create with default options
  - `InferenceEngine(model_path, option)` — create with custom options
  - `run(inputs)` — synchronous inference
  - `run_async(inputs)` — submit async inference, returns request ID
  - `wait(req_id)` — wait for async result
  - `get_input_tensors_info()` — query model input shape/dtype
  - `get_output_tensors_info()` — query model output shape/dtype
  - `get_model_version()` — query .dxnn format version

- **InferenceOption**: Configure runtime behavior.
  - `set_use_ort(bool)` — enable/disable ONNX Runtime fallback
  - `get_use_ort()` — query current setting

- **Configuration**: Runtime metadata.
  - `get_version()` — DX-RT version string

### dx_postprocess

37 C++ postprocessor bindings exposed via pybind11. Each binding implements
high-performance NMS, decode, or transform logic for a specific model family.

Example usage from Python:
```python
from dx_postprocess import YOLOv8PostProcess
pp = YOLOv8PostProcess(input_w, input_h, score_thresh, nms_thresh, use_ort)
results = pp.postprocess(output_tensors)
```

## Layer 2: Python Framework

Located in `src/python_example/common/`:

### base/ — Abstract Interfaces

- **IFactory**: Abstract factory pattern. 11 specialized interfaces:
  `IDetectionFactory`, `IClassificationFactory`, `IPoseFactory`,
  `IInstanceSegFactory`, `ISegmentationFactory`, `IFaceFactory`,
  `IDepthEstimationFactory`, `IRestorationFactory`, `IOBBFactory`,
  `IEmbeddingFactory`, `IFaceAlignmentFactory`, `IHandLandmarkFactory`

- **_FactoryConfigMixin**: Mixin that adds `load_config(dict)` to all factories.
  Handles alias mapping (e.g., `score_threshold` -> `conf_threshold`).

- **IPreprocessor**: `process(image) -> (tensor, context)`
- **IPostprocessor**: `process(outputs, context) -> results`
- **IVisualizer**: `visualize(image, results) -> image`

### runner/ — Execution Engines

- **SyncRunner**: Sequential execution. Preprocess -> Infer -> Postprocess ->
  Visualize, one frame at a time. Supports image, video, camera, RTSP.
  Features: multi-loop, tensor dump, run directory, 7-field metrics.

- **AsyncRunner**: Pipelined execution using 5 worker threads:
  `read -> preprocess+run_async -> wait -> postprocess -> render`
  Display runs on the main thread (cv2.imshow GUI constraint).
  Features: inflight tracking, queue-based handoff, graceful SENTINEL shutdown.

- **parse_common_args()**: Unified CLI parser with 11 flags.

### processors/ — Pre/Post Processors

Per-model-family implementations of IPreprocessor and IPostprocessor:
- `LetterboxPreprocessor` — aspect-ratio-preserving resize with padding
- `YOLOv5Postprocessor`, `YOLOv8Postprocessor`, etc.
- `ClassificationPostprocessor`, `PosePostprocessor`, etc.
- `PPMattingPostprocessor` — a continuous alpha matte, not a class map
- `GalleryRetrievalPostprocessor` — a descriptor ranked against a committed gallery
  (see "Reference galleries" below)

### visualizers/ — Drawing Utilities

Per-task visualization:
- `DetectionVisualizer` — bounding boxes + labels
- `PoseVisualizer` — skeleton overlay
- `SegmentationVisualizer` — mask overlay
- `DepthVisualizer` — depth colormap
- `MattingVisualizer` — original / alpha matte / checkerboard composite
- `RetrievalVisualizer` — query beside its top-k gallery hits

### inputs/ — Input Sources

- **InputFactory**: Creates iterators for different input sources
  (video file, camera device, RTSP stream, image directory).

### config/ — Configuration

- `load_config(path, verbose)` — load and parse config.json

### utility/ — Helpers

- Performance summary printers (sync and async)
- Coordinate scaling (`scale_to_original`)
- SafeQueue (thread-safe queue wrapper)
- C++ detection result converters

## Layer 3: Application Layer

### Python Applications

Located in `src/python_example/<task>/<family>/<variant>/`, one directory per
variant. Each variant has its own factory and entry scripts; its settings live in
`config.json`, loaded by `common.variant_config.load_variant_config`.
The variant directory name is the `.dxnn` stem.

```
<family>/
    __init__.py
    <variant>/                           # e.g. yolov8-n_640x640
        config.json                      # Thresholds and processor spec
        factory/
            __init__.py
            <variant>_factory.py         # IFactory for this variant
        <variant>_sync.py                # Entry; variant is fixed
        <variant>_async.py
        <variant>_sync_cpp_postprocess.py
        <variant>_async_cpp_postprocess.py
```

A single-model extract may keep one `config.json` beside the entry script
instead of a nested variant directory.

### C++ Applications

Located in `src/cpp_example/<task>/<family>/<variant>/`, one directory per
variant. The variant is the `.dxnn` stem, so the model path alone identifies it.

```
<family>/
    <variant>/
        config.json
        factory/
            <variant>_factory.hpp
        <variant>_sync.cpp
        <variant>_async.cpp
```

## Sync vs Async Execution

### Synchronous (SyncRunner)

```
Frame N:  [Read] -> [Preprocess] -> [Infer] -> [Postprocess] -> [Render] -> [Display]
Frame N+1:                                                                    [Read] -> ...
```

- Simple, deterministic
- Good for: single images, debugging, low-throughput scenarios
- Bottleneck: total latency = sum of all phases

### Asynchronous (AsyncRunner)

```
Thread 1 (read):       [Read N] [Read N+1] [Read N+2]
Thread 2 (preprocess): [Pre N]  [Pre N+1]  [Pre N+2]
Thread 3 (wait):       [Infer N] [Infer N+1]
Thread 4 (postprocess):         [Post N]   [Post N+1]
Thread 5 (render):                         [Render N]
Main (display):                                      [Display N]
```

- Overlapped execution: preprocess(N+1) runs during infer(N)
- Good for: video, camera, RTSP — any continuous stream
- Typical speedup: 1.3-2.0x over sync for inference-bound models

## 4 Python Variants

| Variant | Runner | Postprocess | When to Use |
|---|---|---|---|
| `_sync.py` | SyncRunner | Python | Development, single images, debugging |
| `_async.py` | AsyncRunner | Python | Video/camera, real-time demo |
| `_sync_cpp_postprocess.py` | SyncRunner | C++ (dx_postprocess) | NMS-heavy models, benchmarking |
| `_async_cpp_postprocess.py` | AsyncRunner | C++ (dx_postprocess) | Maximum throughput |

## Module Dependency Graph

```
Application Script
    -> factory/<model>_factory.py
        -> common.base.IFactory (IDetectionFactory, etc.)
        -> common.processors (preprocessors + postprocessors)
        -> common.visualizers (task-specific visualizers)
    -> common.runner (SyncRunner or AsyncRunner)
        -> common.runner.args (parse_common_args)
        -> dx_engine (InferenceEngine, InferenceOption)
        -> common.config (load_config)
        -> common.utility (performance summaries)
        -> common.inputs (InputFactory) [async only]
    -> dx_postprocess [cpp_postprocess variants only]
```

## Configuration Flow

```
CLI args (--model, --config, etc.)
    |
    v
parse_common_args() -> argparse.Namespace
    |
    v
SyncRunner.run(args) / AsyncRunner.run(args)
    |
    +-> _resolve_config_path(args) -> finds config.json
    |       |
    |       v
    |   load_config(path) -> dict
    |       |
    |       v
    +-> factory.load_config(dict)  # _FactoryConfigMixin
    |
    +-> factory.create_preprocessor(w, h)
    +-> factory.create_postprocessor(w, h)
    +-> factory.create_visualizer()
    |
    +-> _dispatch_input(args) -> image/video/camera/rtsp
```


## Reference galleries

Three of the task categories DX Model Zoo added in release 2_5_0 -- `image_retrieval`,
`visual_place_recognition`, `person_reid` -- cannot be answered by a model alone. An
embedding model returns a descriptor; "which place is this?" needs a *set of known
descriptors* to rank it against. So those examples carry a committed gallery.

```
scripts/fetch_vpr_toy_dataset.py     third-party sample images -> sample/vpr/
scripts/build_reid_crops.py          person detector -> sample/reid/  (ReID needs crops)
scripts/build_gallery_database.py    encode a folder ON THE NPU -> sample/gallery/*.bin
                                        |
                        config.json "gallery": "sample/gallery/<purpose>_<variant>.bin"
                                        |
        GalleryRetrievalPostprocessor (py)      loadRetrievalGallery (cpp)
                cosine top-k -> RetrievalResult / EmbeddingResult::matches
                                        |
                              RetrievalVisualizer
```

Three properties are load-bearing:

1. **One file, one format.** `common/processors/gallery_format.py` defines it and
   `common/processors/gallery_retrieval_postprocessor.hpp` reads the same bytes. A
   first attempt shipped a `.npz` for Python and a `.bin` for C++ and the two copies
   disagreed immediately. A gallery that belongs to another encoder does not error --
   it ranks plausibly and wrongly -- so the file records the producing `.dxnn` stem and
   its width, and both readers refuse a mismatch.
2. **Built through the variant's own factory.** `build_gallery_database.py` drives
   `config.json`, so the gallery cannot be encoded with different preprocessing than
   the query path. A later preprocessing fix is inherited by rebuilding.
3. **The query is held out of its gallery.** A query that is also a gallery member
   scores a meaningless 1.0000 self-match, and the repo ships the same photo under two
   names, so `--exclude` drops by content hash rather than by path.

Rebuild after changing a variant's preprocessing:

```bash
python3 scripts/build_gallery_database.py --variant <dxnn-stem> \
    --images <folder> --out sample/gallery/<purpose>_<dxnn-stem>.bin
```
