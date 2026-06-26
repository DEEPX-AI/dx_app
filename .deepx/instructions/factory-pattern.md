# IFactory Pattern Guide

The IFactory pattern is the core design pattern in dx_app v3.0.0. Every model
application is built around a concrete factory that creates matched sets of
preprocessor, postprocessor, and visualizer components.

## Why IFactory?

The Abstract Factory pattern solves the component compatibility problem: a YOLOv8
postprocessor only works with YOLOv8-format outputs, and its visualizer expects
detection result objects. The factory guarantees that all three components are
created as a compatible set.

## IFactory Interface

All factory interfaces share the same 5 abstract methods:

```python
class IFactory(ABC):
    @abstractmethod
    def create_preprocessor(self, input_width: int, input_height: int) -> IPreprocessor:
        """Create a preprocessor that transforms raw images to model input format."""
        pass

    @abstractmethod
    def create_postprocessor(self, input_width: int, input_height: int) -> IPostprocessor:
        """Create a postprocessor that decodes model outputs to structured results."""
        pass

    @abstractmethod
    def create_visualizer(self) -> IVisualizer:
        """Create a visualizer that draws results on images."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier (e.g., 'yolov8n')."""
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        """Return the task type (e.g., 'object_detection')."""
        pass
```

## _FactoryConfigMixin

All 11 factory interfaces inherit from `_FactoryConfigMixin`, which provides
a `load_config(dict)` method:

```python
class _FactoryConfigMixin:
    """Mixin providing load_config() for all factory interfaces."""

    _CONFIG_ALIASES = {
        "score_threshold": "conf_threshold",
    }

    def load_config(self, config: dict) -> None:
        """Merge external config into self.config with alias translation."""
        if hasattr(self, "config") and isinstance(self.config, dict):
            for key, value in config.items():
                alias = self._CONFIG_ALIASES.get(key)
                if alias and alias not in config:
                    self.config[alias] = value
                self.config[key] = value
```

This means concrete factories should define `self.config` in `__init__`:

```python
class MyFactory(IDetectionFactory):
    def __init__(self, config: dict = None):
        self.config = config or {}
```

The runner calls `factory.load_config(config)` after loading `config.json`,
which merges the JSON values into `self.config`.

## 11 Specialized Factory Interfaces

Each task type has its own factory interface. All inherit from
`_FactoryConfigMixin` and `ABC`, and define the same 5 abstract methods:

| Interface | Task | Extra Methods |
|---|---|---|
| `IDetectionFactory` | object_detection | — |
| `IClassificationFactory` | classification | — |
| `IPoseFactory` | pose_estimation | `get_num_keypoints() -> int` |
| `IInstanceSegFactory` | instance_segmentation | — |
| `ISegmentationFactory` | semantic_segmentation | — |
| `IFaceFactory` | face_detection | `get_num_keypoints() -> int` |
| `IDepthEstimationFactory` | depth_estimation | — |
| `IRestorationFactory` | image_denoising, image_enhancement, super_resolution | — |
| `IOBBFactory` | obb_detection | — |
| `IEmbeddingFactory` | embedding | — |
| `IFaceAlignmentFactory` | face_alignment (3D) | — |
| `IHandLandmarkFactory` | hand_landmark | — |

Note: `IPoseFactory` and `IFaceFactory` have an additional `get_num_keypoints()`
method because keypoint count varies by model (17 for COCO pose, 5 for face).

## Complete Implementation Example

Here is a full concrete factory for YOLOv8n object detection:

### factory/yolov8n_factory.py

```python
"""
YOLOv8 Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, YOLOv8Postprocessor
from common.visualizers import DetectionVisualizer


class Yolov8Factory(IDetectionFactory):
    """Factory for creating YOLOv8n components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return YOLOv8Postprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DetectionVisualizer()

    def get_model_name(self) -> str:
        return "yolov8n"

    def get_task_type(self) -> str:
        return "object_detection"
```

### factory/__init__.py

```python
from .yolov8n_factory import Yolov8Factory
```

## How the Runner Uses the Factory

The SyncRunner and AsyncRunner follow identical initialization:

```python
# Inside SyncRunner._init_engine():
self.preprocessor = self.factory.create_preprocessor(self.input_width, self.input_height)
self.postprocessor = self.factory.create_postprocessor(self.input_width, self.input_height)
self.visualizer = self.factory.create_visualizer()
```

The runner calls these three methods exactly once during initialization, after
loading the model and determining input dimensions from `InferenceEngine.get_input_tensors_info()`.

## config.json Schema

Each model directory contains a `config.json` with model-specific parameters:

### Detection models
```json
{
  "score_threshold": 0.25,
  "nms_threshold": 0.45
}
```

### Classification models
```json
{
  "top_k": 5
}
```

### Pose estimation models
```json
{
  "score_threshold": 0.3,
  "nms_threshold": 0.45,
  "kpt_threshold": 0.5
}
```

### Segmentation models
```json
{
  "score_threshold": 0.25,
  "mask_threshold": 0.5
}
```

### Restoration/SR/Denoising models
```json
{}
```

The config is loaded by `load_config()` in `common/config/` and passed to
the factory via `factory.load_config(config)`. The factory's concrete
postprocessor reads values from `self.config` during initialization.

## Config Flow

```
config.json  --load_config()--> dict
                                  |
                                  v
              factory.load_config(dict)  # _FactoryConfigMixin
                                  |
                                  v
              self.config updated with merged values
                                  |
                                  v
              factory.create_postprocessor(w, h)
                                  |
                                  v
              Postprocessor reads self.config["score_threshold"] etc.
```

## Common Mistakes

1. **Missing `self.config` in __init__**: The `load_config()` mixin checks
   `hasattr(self, "config")`. If you forget to initialize it, config values
   are silently dropped.

2. **Wrong factory interface**: Using `IDetectionFactory` for a segmentation
   model compiles fine but produces confusing errors when the visualizer
   receives wrong result types.

3. **Missing `__init__.py` in factory/**: The model script imports
   `from factory import <Model>Factory` — this requires `factory/__init__.py`.

4. **Hardcoding thresholds in factory**: Put thresholds in `config.json`, not
   in the factory constructor. This allows runtime tuning via `--config`.

5. **Not implementing all 5 methods**: Omitting any abstract method causes
   `TypeError: Can't instantiate abstract class` at runtime.
