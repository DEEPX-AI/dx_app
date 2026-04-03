# Common Framework API Reference

> SyncRunner, AsyncRunner, IFactory hierarchy, IInputSource, and parse_common_args() —
> the application framework for dx_app v3.0.0.

## Overview

The common framework (`src/python_example/common/`) provides:
- **Runners** — SyncRunner (1104 lines) and AsyncRunner manage the inference lifecycle
- **IFactory** — 11 abstract factory interfaces, each requiring 5 methods
- **IInputSource** — Pluggable input sources (camera, video, image, RTSP)
- **parse_common_args()** — Standard CLI argument parser with 11 flags

---

## SyncRunner

Single-threaded inference runner. Reads frames, preprocesses, infers on NPU, postprocesses,
and visualizes in a sequential loop.

**Location:** `src/python_example/common/runner/sync_runner.py` (1104 lines)

### Constructor

```python
from common.runner import SyncRunner

runner = SyncRunner(
    factory: IFactory,                # Required: factory implementing 5 methods
    on_engine_init: callable = None,  # Optional: callback after engine initialization
    config: dict = None               # Optional: override config.json values
)
```

**Parameters:**
| Parameter | Type | Required | Description |
|---|---|---|---|
| `factory` | `IFactory` subclass | Yes | Abstract factory for model components |
| `on_engine_init` | `callable(runner)` | No | Hook called after InferenceEngine is created |
| `config` | `dict` | No | Config overrides (merged with config.json) |

### run()

```python
runner.run(args: argparse.Namespace)
```

Executes the full inference pipeline:
1. Loads config.json from the model directory
2. Creates InferenceEngine with InferenceOption
3. Calls `on_engine_init(self)` if provided
4. Creates preprocessor, postprocessor, visualizer via factory
5. Opens input source (camera/video/image)
6. Runs inference loop with signal handling
7. Prints performance metrics on exit

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `args` | `argparse.Namespace` | Parsed CLI arguments from `parse_common_args()` |

### Performance Metrics

SyncRunner tracks 7 metrics automatically:

| Metric | Description | Unit |
|---|---|---|
| `preprocess_time` | Frame preprocessing duration | ms |
| `inference_time` | NPU inference duration | ms |
| `postprocess_time` | Result postprocessing duration | ms |
| `visualize_time` | Drawing/rendering duration | ms |
| `total_time` | End-to-end frame time | ms |
| `fps` | Frames per second | fps |
| `frame_count` | Total frames processed | count |

Access metrics after `run()` completes:
```python
runner.run(args)
print(f"Average FPS: {runner.metrics.fps:.1f}")
print(f"Inference time: {runner.metrics.inference_time:.1f}ms")
print(f"Frames processed: {runner.metrics.frame_count}")
```

### Signal Handling

SyncRunner installs a `SIGINT` handler automatically. Pressing Ctrl+C triggers a
graceful shutdown:
1. Sets internal `_running = False`
2. Completes current frame processing
3. Releases input source
4. Prints final metrics summary
5. Exits with code 0

### Key Properties

| Property | Type | Description |
|---|---|---|
| `input_width` | `int` | Model input width (available after engine init) |
| `input_height` | `int` | Model input height (available after engine init) |
| `engine` | `InferenceEngine` | The underlying inference engine |
| `metrics` | `Metrics` | Performance metrics object |
| `_running` | `bool` | Loop control flag |

---

## AsyncRunner

Multi-threaded inference runner. Overlaps preprocessing of frame N+1 with NPU inference
of frame N for 2-3x throughput improvement.

**Location:** `src/python_example/common/runner/async_runner.py`

### Constructor

```python
from common.runner import AsyncRunner

runner = AsyncRunner(
    factory: IFactory,
    on_engine_init: callable = None,
    config: dict = None
)
```

Parameters are identical to SyncRunner.

### run()

```python
runner.run(args: argparse.Namespace)
```

### Thread Architecture

```
Main Thread                 Inference Thread
===========                 ================
read frame N        -->     [idle]
preprocess frame N  -->     [idle]
submit frame N      -->     infer frame N (NPU)
read frame N+1      <--     [inferring...]
preprocess frame N+1 <--    [inferring...]
submit frame N+1    -->     return results N
                            infer frame N+1 (NPU)
postprocess N       <--     [inferring...]
visualize N         <--     [inferring...]
```

**Key difference from SyncRunner:** AsyncRunner uses `engine.run_async()` and
`engine.wait()` to pipeline CPU work with NPU inference.

### Frame Order

AsyncRunner maintains strict frame ordering. Results are always returned in
submission order. However, if postprocessing is significantly slower than inference,
a frame order inversion can occur. See `memory/common_pitfalls.md` [DX_APP] entry.

---

## IFactory Hierarchy

11 abstract factory interfaces, each requiring 5 methods. All factories extend
the base `IFactory` interface.

### Base Interface: IFactory

```python
from abc import ABC, abstractmethod

class IFactory(ABC):
    @abstractmethod
    def create_preprocessor(self, input_width: int, input_height: int):
        """Return a preprocessor instance for the model."""
        ...

    @abstractmethod
    def create_postprocessor(self, input_width: int, input_height: int):
        """Return a postprocessor instance for the model."""
        ...

    @abstractmethod
    def create_visualizer(self):
        """Return a visualizer instance for displaying results."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name (e.g., 'yolov8n')."""
        ...

    @abstractmethod
    def get_task_type(self) -> str:
        """Return the task type (e.g., 'object_detection')."""
        ...
```

### 11 Task-Specific Interfaces

| Interface | Task | Module |
|---|---|---|
| `IDetectionFactory` | object_detection | `common.base` |
| `IClassificationFactory` | classification | `common.base` |
| `IPoseFactory` | pose_estimation | `common.base` |
| `IInstanceSegFactory` | instance_segmentation | `common.base` |
| `ISegmentationFactory` | semantic_segmentation | `common.base` |
| `IFaceFactory` | face_detection | `common.base` |
| `IDepthEstimationFactory` | depth_estimation | `common.base` |
| `IRestorationFactory` | image_denoising, image_enhancement, super_resolution | `common.base` |
| `IEmbeddingFactory` | embedding | `common.base` |
| `IOBBFactory` | obb_detection | `common.base` |
| `IHandLandmarkFactory` | hand_landmark | `common.base` |

**Note:** `IRestorationFactory` serves 3 tasks (denoising, enhancement, super resolution)
since they share the same input/output pattern.

### Required Methods (ALL 5 must be implemented)

```python
class YoloV8nFactory(IDetectionFactory):
    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width, input_height):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width, input_height):
        return YoloV8Postprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DetectionVisualizer()

    def get_model_name(self) -> str:
        return "yolov8n"

    def get_task_type(self) -> str:
        return "object_detection"
```

**Critical:** Omitting any method raises `TypeError` at instantiation. The constructor
must accept `config: dict = None` to support `_FactoryConfigMixin.load_config()`.

---

## IInputSource

Pluggable input sources for the runners.

### Available Sources

| Source | Class | CLI Argument |
|---|---|---|
| USB Camera | `CameraSource` | `--input usb` or `--input /dev/video2` |
| Image File | `ImageSource` | `--input image.jpg` |
| Video File | `VideoSource` | `--input video.mp4` |
| RTSP Stream | `RTSPSource` | `--input rtsp://...` |
| Image Directory | `DirectorySource` | `--input /path/to/images/` |

### InputFactory

Automatically selects the correct source based on the `--input` argument:

```python
from common.inputs import InputFactory

source = InputFactory.create(args.input)
# Returns CameraSource, ImageSource, VideoSource, etc.

while source.is_open():
    frame = source.read()
    if frame is None:
        break
    # ... process frame ...
source.release()
```

### Custom Input Source

```python
from common.inputs import IInputSource

class MyCustomSource(IInputSource):
    def open(self) -> bool:
        """Initialize the source. Return True on success."""
        ...

    def read(self) -> np.ndarray:
        """Read next frame as BGR numpy array. Return None at end."""
        ...

    def is_open(self) -> bool:
        """Return True if source is active."""
        ...

    def release(self):
        """Release resources."""
        ...

    def get_frame_size(self) -> tuple:
        """Return (width, height) of frames."""
        ...
```

---

## parse_common_args()

Standard CLI argument parser. All dx_app examples use this exclusively — never define
custom argparse.

```python
from common.runner import parse_common_args

args = parse_common_args("YOLOv8n Object Detection")
```

### 11 CLI Arguments

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | `str` | Required | Path to `.dxnn` model file |
| `--input` | `-i` | `str` | `"usb"` | Input source (file, camera, RTSP URL) |
| `--output` | `-o` | `str` | `None` | Output file path (save results) |
| `--score-threshold` | `-s` | `float` | `0.25` | Detection confidence threshold |
| `--nms-threshold` | `-n` | `float` | `0.45` | NMS IoU threshold |
| `--no-display` | | `flag` | `False` | Run headless (no cv2.imshow) |
| `--save-video` | | `str` | `None` | Save output to video file |
| `--max-frames` | | `int` | `None` | Stop after N frames |
| `--use-ort` | | `flag` | `False` | Use ONNX Runtime CPU backend |
| `--config` | `-c` | `str` | `"config.json"` | Path to config file |
| `--verbose` | `-v` | `flag` | `False` | Enable debug logging |

### Usage Examples

```bash
# Image inference with custom threshold
python yolov8n_sync.py --model yolov8n.dxnn --input test.jpg --score-threshold 0.3

# Video inference, save output
python yolov8n_sync.py --model yolov8n.dxnn --input video.mp4 --save-video output.mp4

# USB camera, headless mode
python yolov8n_async.py --model yolov8n.dxnn --input usb --no-display

# RTSP stream, max 1000 frames
python yolov8n_async.py --model yolov8n.dxnn --input rtsp://192.168.1.100:554/stream --max-frames 1000

# CPU-only mode (no NPU required)
python yolov8n_sync.py --model yolov8n.dxnn --input test.jpg --use-ort
```

### Argument Precedence

1. CLI arguments (highest priority)
2. config.json values
3. Default values (lowest priority)

**Note on `--score-threshold` alias:** In `config.json`, the key is `score_threshold`
(underscore). On the CLI, it is `--score-threshold` (hyphen). Both map to the same
value. See `memory/common_pitfalls.md` for the alias confusion pitfall.
