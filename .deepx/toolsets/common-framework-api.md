# Common Framework API Reference

> SyncRunner, AsyncRunner, IFactory hierarchy, IInputSource, and parse_common_args() —
> the application framework for dx_app.

## ⚠️ Anti-Fabrication Notice

This document is a **reference-level overview**. For exact signatures, default values,
and edge cases, always read the source files listed below. If this document and source
code disagree, **source code wins**. Do not invent CLI flags, method names, or
constructor parameters — see the [Common Fabrications](#-common-fabrications-to-avoid)
section at the end.

---

## Source Files

| Component | Path |
|-----------|------|
| CLI parser | `src/python_example/common/runner/args.py` |
| SyncRunner | `src/python_example/common/runner/sync_runner.py` |
| AsyncRunner | `src/python_example/common/runner/async_runner.py` |
| IFactory interfaces | `src/python_example/common/base/i_factory.py` |
| IPreprocessor / IPostprocessor | `src/python_example/common/base/i_processor.py` |
| IVisualizer | `src/python_example/common/base/i_visualizer.py` |
| IInputSource | `src/python_example/common/base/i_input_source.py` |
| Concrete input sources | `src/python_example/common/inputs/` |

---

## parse_common_args()

**Location:** `src/python_example/common/runner/args.py`

```python
from common.runner import parse_common_args

args = parse_common_args("YOLOv8n Object Detection")
# or with output flag:
args = parse_common_args("My App", include_output=True)
```

Signature: `parse_common_args(description="DX-APP Inference", *, include_output=False)`

Parser uses `allow_abbrev=False`.

### Input Sources — Mutually Exclusive Group (required=True)

Input source must be exactly one of:

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--image` | `-i` | `str` | Input image path or directory |
| `--video` | `-v` | `str` | Input video path |
| `--camera` | `-c` | `int` | Camera device ID |
| `--rtsp` | `-r` | `str` | RTSP stream URL |

These are **mutually exclusive** — you cannot combine them.

### Other Arguments

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | `str` | **Required** | Path to `.dxnn` model file |
| `--display` | — | `store_true` | `True` | Show output window |
| `--no-display` | — | `store_false` → `display` | — | Disable display |
| `--save` | `-s` | `store_true` | `False` | Save output frames |
| `--save-dir` | — | `str` | — | Output save directory |
| `--loop` | `-l` | `int` | `1` (bare `--loop` = `2`) | Inference loops |
| `--dump-tensors` | — | `store_true` | `False` | Dump raw tensors |
| `--config` | — | `str` | — | Path to config.json |
| `--verbose` | — | `store_true` | `False` | Detailed per-frame logs |
| `--output` | `-o` | `str` | — | Only present when `include_output=True` |

### Usage Examples

```bash
# Image inference
python yolov8n_sync.py -m yolov8n.dxnn -i test.jpg

# Video inference with save
python yolov8n_sync.py -m yolov8n.dxnn -v video.mp4 --save --save-dir ./out

# USB camera, headless mode
python yolov8n_async.py -m yolov8n.dxnn -c 0 --no-display

# RTSP stream
python yolov8n_async.py -m yolov8n.dxnn -r rtsp://192.168.1.100:554/stream
```

---

## SyncRunner

Single-threaded inference runner. Everything runs on the main thread in a sequential
loop: read → preprocess → infer → postprocess → visualize.

**Location:** `src/python_example/common/runner/sync_runner.py`

### Constructor

```python
from common.runner import SyncRunner

runner = SyncRunner(
    factory,                        # IFactory implementation (duck-typed)
    use_ort=None,                   # None=auto, True=force ORT, False=disable
    cpp_postprocessor=None,         # Optional C++ postprocessor
    cpp_convert_fn=None,            # Optional C++ result converter
    cpp_visualize_fn=None,          # Optional custom viz function
    on_engine_init=None,            # Callback after engine init
    display_size=None,              # Default (960, 640)
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `factory` | IFactory subclass | Yes | Abstract factory for model components |
| `use_ort` | `bool` or `None` | No | `None`=auto, `True`=force ONNX Runtime, `False`=disable |
| `cpp_postprocessor` | callable | No | C++ postprocessor binding |
| `cpp_convert_fn` | callable | No | C++ result converter |
| `cpp_visualize_fn` | callable | No | Custom C++ visualization function |
| `on_engine_init` | callable | No | Hook called after InferenceEngine is created |
| `display_size` | tuple | No | Display window size, default `(960, 640)` |

### run()

```python
runner.run(args: argparse.Namespace)
```

Takes the parsed CLI arguments from `parse_common_args()`.

### Threading Model

**NO threads.** Everything executes sequentially on the main thread.

Public pipeline methods: `preprocess()`, `infer()`, `postprocess()`, `visualize()`

### Shutdown

No signal handler — uses `except KeyboardInterrupt` and window close detection.

### Performance Metrics

SyncRunner tracks 7 cumulative timing sums:

| Field | Description |
|-------|-------------|
| `sum_read` | Total frame reading time |
| `sum_preprocess` | Total preprocessing time |
| `sum_inference` | Total NPU inference time |
| `sum_postprocess` | Total postprocessing time |
| `sum_render` | Total visualization/rendering time |
| `sum_save` | Total frame saving time |
| `sum_display` | Total display time |

> **Note:** These are cumulative sums, not per-frame averages or named attributes
> like `fps` or `inference_time`. See source for exact access patterns.

---

## AsyncRunner

Multi-threaded inference runner. Overlaps all pipeline stages across 6 threads for
maximum throughput.

**Location:** `src/python_example/common/runner/async_runner.py`

### Constructor

```python
from common.runner import AsyncRunner

runner = AsyncRunner(
    factory,                        # IFactory implementation (duck-typed)
    use_ort=None,                   # None=auto, True=force ORT, False=disable
    on_engine_init=None,            # Callback after engine init
    display_size=(960, 640),        # Display window size
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `factory` | IFactory subclass | Yes | Abstract factory for model components |
| `use_ort` | `bool` or `None` | No | `None`=auto, `True`=force ORT, `False`=disable |
| `on_engine_init` | callable | No | Hook called after InferenceEngine is created |
| `display_size` | tuple | No | Display window size, default `(960, 640)` |

### 6-Thread Architecture (5 Workers + Main)

```
read_worker ──► preprocess_worker ──► wait_worker ──► postprocess_worker ──► render_worker
                                                                                   │
                                                                                   ▼
                                                                          Main Thread (display)
```

| Thread | Target Method | Purpose |
|--------|---------------|---------|
| `read_worker` | `_read_worker` | Reads frames from input source |
| `preprocess_worker` | `_preprocess_worker` | Preprocess + `ie.run_async()` submit |
| `wait_worker` | `_wait_worker` | `ie.wait(job_id)` to collect results |
| `postprocess_worker` | `_postprocess_worker` | Run postprocessing |
| `render_worker` | `_render_worker` | Visualize + save frames |
| **Main thread** | `_run_display_loop` | `cv2.imshow()` (must be main for GUI) |

### Queue Pipeline

```
read_queue → reqid_queue → output_queue → render_queue → display_queue
```

All queues are `SafeQueue(maxsize=4)`.

### Threading Details

- All 5 worker threads are `daemon=True`
- Threads joined with `timeout=5.0`
- Uses `_stop_event = threading.Event()` for graceful shutdown
- Worker exceptions are captured and re-raised on the main thread

### Performance Metrics

AsyncRunner tracks **17 metrics fields** (including inflight tracking).
See source for exact field names.

---

## Factory Interfaces (12)

**Location:** `src/python_example/common/base/i_factory.py`

12 abstract factory interfaces. All extend the base `IFactory` interface.

### Common Methods (all 12 interfaces)

Every factory must implement these 5 methods:

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_preprocessor` | `(input_width, input_height)` | IPreprocessor |
| `create_postprocessor` | `(input_width, input_height)` | IPostprocessor |
| `create_visualizer` | `()` | IVisualizer |
| `get_model_name` | `()` | `str` |
| `get_task_type` | `()` | `str` |

### 12 Task-Specific Interfaces

| # | Interface | Extra Methods |
|---|-----------|---------------|
| 1 | `IDetectionFactory` | — |
| 2 | `ISegmentationFactory` | — |
| 3 | `IClassificationFactory` | — |
| 4 | `IPoseFactory` | `get_num_keypoints()` |
| 5 | `IInstanceSegFactory` | — |
| 6 | `IFaceFactory` | `get_num_keypoints()` |
| 7 | `IDepthEstimationFactory` | — |
| 8 | `IRestorationFactory` | — |
| 9 | `IOBBFactory` | — |
| 10 | `IEmbeddingFactory` | — |
| 11 | `IFaceAlignmentFactory` | — |
| 12 | `IHandLandmarkFactory` | — |

**Note:** `IRestorationFactory` serves multiple tasks (denoising, enhancement,
super resolution) since they share the same input/output pattern.

### Example Implementation

```python
class YoloV8nFactory(IDetectionFactory):
    def create_preprocessor(self, input_width, input_height):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width, input_height):
        return YoloV8Postprocessor(input_width, input_height)

    def create_visualizer(self):
        return DetectionVisualizer()

    def get_model_name(self) -> str:
        return "yolov8n"

    def get_task_type(self) -> str:
        return "object_detection"
```

---

## IInputSource

**Location:** `src/python_example/common/base/i_input_source.py`

### 10 Abstract Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `get_frame` | `()` | `Tuple[bool, Optional[np.ndarray]]` |
| `is_opened` | `()` | `bool` |
| `release` | `()` | `None` |
| `get_type` | `()` | `InputType` |
| `get_width` | `()` | `int` |
| `get_height` | `()` | `int` |
| `get_fps` | `()` | `float` |
| `get_total_frames` | `()` | `int` |
| `get_description` | `()` | `str` |
| `is_live_source` | `()` | `bool` |

Also supports context manager protocol (`__enter__` / `__exit__`).

### InputType Enum

`IMAGE`, `VIDEO`, `CAMERA`, `RTSP`, `UNKNOWN`

### 4 Concrete Sources

| Class | InputType | Location |
|-------|-----------|----------|
| `ImageSource` | `IMAGE` | `common/inputs/` |
| `VideoSource` | `VIDEO` | `common/inputs/` |
| `CameraSource` | `CAMERA` | `common/inputs/` |
| `RTSPSource` | `RTSP` | `common/inputs/` |

---

## Processor Interfaces

### IPreprocessor

**Location:** `src/python_example/common/base/i_processor.py`

```python
def process(self, image) -> Tuple[tensor, PreprocessContext]:
    ...
```

### IPostprocessor

**Location:** `src/python_example/common/base/i_processor.py`

```python
def process(self, outputs, ctx) -> results:
    ...
```

### IVisualizer

**Location:** `src/python_example/common/base/i_visualizer.py`

```python
def visualize(self, frame, results) -> frame:
    ...
```

> **Note:** Python framework uses `process()` for preprocessor/postprocessor and
> `visualize()` for visualizer. The C++ `dx_postprocess` bindings use `postprocess()`.
> These are different layers — do not confuse them.

---

## ⚠️ Common Fabrications to Avoid

| Fabrication | Reality |
|-------------|---------|
| `--input` / `-i` single input arg | **Does not exist.** Input is a mutually exclusive group: `--image`/`-i`, `--video`/`-v`, `--camera`/`-c`, `--rtsp`/`-r` |
| `--score-threshold` / `-s` | **Does not exist.** `-s` is `--save`. Thresholds come from `config.json` |
| `--nms-threshold` / `-n` | **Does not exist.** NMS config is in `config.json` |
| `--output` / `-o` always available | Only present when `include_output=True` is passed to `parse_common_args()` |
| `--save-video` flag | **Does not exist.** Use `--save` + `--save-dir` |
| `--max-frames` flag | **Does not exist.** Use `--loop` for iteration control |
| `--use-ort` CLI flag | **Does not exist.** ORT is configured via `use_ort` constructor param or `InferenceOption` in code |
| `runner.metrics.fps` / `.inference_time` | **Do not exist.** Metrics are cumulative timing sums (`sum_read`, `sum_inference`, etc.) |
| AsyncRunner is 2-thread | **Wrong.** It is 6 threads: 5 daemon workers + main thread |
| IFactory has 11 interfaces | **Wrong.** There are 12 interfaces |
| `is_open()` method | **Wrong.** The method is `is_opened()` |
| `read()` method on IInputSource | **Wrong.** The method is `get_frame()` → `Tuple[bool, Optional[np.ndarray]]` |
| `get_frame_size()` method | **Wrong.** Use `get_width()` and `get_height()` separately |
| `draw()` method on IVisualizer | **Wrong.** The method is `visualize()` |
| `SyncRunner(factory, on_engine_init, config)` | **Wrong.** Constructor has 7 params: `factory`, `use_ort`, `cpp_postprocessor`, `cpp_convert_fn`, `cpp_visualize_fn`, `on_engine_init`, `display_size` |
