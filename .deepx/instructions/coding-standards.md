# dx_app Coding Standards

Standards for all code within the dx_app repository. Applies to Python applications
under `src/python_example/`, C++ applications under `src/cpp_example/`, and any
new code contributed to the common framework.

## Python Standards

### Imports

**Always use the relative-from-common pattern.** Every model script adds
`src/python_example/` to `sys.path` so that `common` is importable:

```python
# CORRECT - relative from common
from common.base import IDetectionFactory
from common.runner import SyncRunner, parse_common_args
from common.processors import LetterboxPreprocessor

# WRONG - absolute from package root
from dx_app.src.python_example.common.base import IDetectionFactory

# WRONG - relative imports
from ...common.base import IDetectionFactory
```

The sys.path setup must appear at the top of every model script:
```python
import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
```

### Logging

Use Python `logging` module, not `print()`, for informational output in
framework code. Application scripts may use `print()` for user-facing messages.

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Model loaded: %s", model_path)
logger.warning("Config file not found, using defaults")
logger.error("Failed to open video source")
```

### CLI Arguments

**Always use `parse_common_args()`.** Never define custom `argparse.ArgumentParser`
in model scripts.

```python
from common.runner import parse_common_args

# Simple usage
args = parse_common_args("YOLOv8n Sync Inference")

# With --output flag (for SR/depth/denoising)
args = parse_common_args("ESPCN Sync Inference", include_output=True)
```

The 11 standard flags:
| Flag | Short | Type | Description |
|---|---|---|---|
| `--model` | `-m` | str | Model path (.dxnn), required |
| `--image` | `-i` | str | Input image or directory |
| `--video` | `-v` | str | Input video path |
| `--camera` | `-c` | int | Camera device ID |
| `--rtsp` | `-r` | str | RTSP stream URL |
| `--no-display` | | flag | Disable display |
| `--save` | `-s` | flag | Save output |
| `--save-dir` | | str | Output directory |
| `--loop` | `-l` | int? | Loop count (bare=2, default=1) |
| `--dump-tensors` | | flag | Dump raw tensors |
| `--config` | | str | Path to config.json |
| `--verbose` | | flag | Detailed per-frame logs |

### IFactory Pattern

Every model MUST implement a concrete factory. The factory is the single source
of truth for creating matching preprocessor/postprocessor/visualizer sets.

Required methods (5):
1. `create_preprocessor(input_width, input_height) -> IPreprocessor`
2. `create_postprocessor(input_width, input_height) -> IPostprocessor`
3. `create_visualizer() -> IVisualizer`
4. `get_model_name() -> str`
5. `get_task_type() -> str`

### Type Hints

Use type hints for all public method signatures:

```python
def create_preprocessor(self, input_width: int, input_height: int) -> IPreprocessor:
    ...
```

### Model Paths

**Never hardcode model paths.** Model paths always come from:
1. `--model` CLI argument (primary)
2. `config/model_registry.json` lookup (for validation)

```python
# CORRECT
args = parse_common_args("Example")
engine = InferenceEngine(args.model)

# WRONG
engine = InferenceEngine("/path/to/models/yolov8n.dxnn")
```

### Docstrings

All modules, classes, and public methods must have docstrings:

```python
"""
YOLOv8 Factory - DX-APP v3.0.0 Abstract Factory Pattern
"""

class Yolov8Factory(IDetectionFactory):
    """Factory for creating YOLOv8n detection components."""

    def create_preprocessor(self, input_width: int, input_height: int):
        """Create letterbox preprocessor for YOLOv8 input format."""
        ...
```

### File Headers

Every Python file must start with the shebang and copyright:

```python
#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
```

## C++ Standards

### Language Version

C++14. Do not use C++17 or later features.

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Files | snake_case | `yolov8n_sync.cpp` |
| Classes | PascalCase | `YOLOv8Factory` |
| Functions/methods | snake_case | `create_preprocessor()` |
| Variables | snake_case | `input_width` |
| Constants | UPPER_SNAKE | `MAX_DETECTIONS` |
| Namespaces | snake_case | `dxapp` |
| Member variables | trailing_ | `score_threshold_` |

### Error Handling

1. Check return values from `InferenceEngine` operations.
2. Use exceptions for unrecoverable errors (model not found, NPU unavailable).
3. Use return codes for expected failures (empty frame, end of video).
4. Always wrap `InferenceEngine` creation in try/catch.

### SIGINT Handler

All examples that process continuous streams (video, camera, RTSP) MUST install
a SIGINT handler for graceful shutdown:

```cpp
#include <csignal>

static volatile bool g_running = true;
void sigint_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    std::signal(SIGINT, sigint_handler);
    // ... main loop checks g_running ...
}
```

### RAII

Use `std::unique_ptr` for dynamically allocated objects. Never use raw
`new`/`delete`:

```cpp
// CORRECT
auto factory = std::make_unique<dxapp::YOLOv8Factory>();

// WRONG
dxapp::YOLOv8Factory* factory = new dxapp::YOLOv8Factory();
```

### CMake

Each example directory has its own `CMakeLists.txt`. The parent CMakeLists.txt
adds each subdirectory.

Required targets: `<model>_sync`, `<model>_async`.
Required links: `dx_engine::dx_engine`, `dx_postprocess`, `${OpenCV_LIBS}`.

## Convention Checklist

Before submitting any new model application, verify:

- [ ] **sys.path**: Uses the standard 2-parent pattern
- [ ] **Factory**: Implements all 5 IFactory methods
- [ ] **parse_common_args**: Used exclusively (no custom argparse)
- [ ] **config.json**: Present in model directory with valid JSON
- [ ] **No hardcoded paths**: Model path from `--model` only
- [ ] **Docstrings**: Module, class, and public methods documented
- [ ] **File header**: Shebang + copyright present
- [ ] **__init__.py**: Present in model dir and factory/ subdir

## Config File Standards

### config.json

```json
{
  "score_threshold": 0.25,
  "nms_threshold": 0.45
}
```

- Use `score_threshold` (not `conf_threshold`) — the `_FactoryConfigMixin`
  handles the alias mapping automatically.
- Keep config minimal — only thresholds that differ from defaults.
- Classification models use `"top_k": 5`.
- Segmentation models may omit `nms_threshold`.

### model_registry.json

Entries in `config/model_registry.json` must include all required fields.
See `.deepx/agents/dx-model-manager.md` for the full schema.
