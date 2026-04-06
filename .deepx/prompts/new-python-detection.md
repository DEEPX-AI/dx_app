# Prompt: New Python Detection App

> Template for creating a Python object detection application in dx_app.

## Variables

| Variable | Description | Example |
|---|---|---|
| `{model_name}` | Model name from model_registry.json | `yolo26n` |
| `{variant}` | Python variant to generate | `sync`, `async`, `sync_cpp_postprocess`, `async_cpp_postprocess` |
| `{input_source}` | Primary input for testing | `image`, `video`, `usb`, `rtsp` |

## Prompt

Build a Python object detection application for `{model_name}` using the `{variant}` runner
with `{input_source}` as the primary input source.

### Step 1: Query Model Registry

Read `config/model_registry.json` and verify:
- `{model_name}` exists in the registry
- `task` is `"object_detection"`
- `supported` is `true`
- Extract `input_size`, `default_threshold`, `nms_threshold`, `labels_file`

If the model is not found, list available object_detection models and ask the user to choose.

### Step 2: Create Factory

Create `src/python_example/object_detection/{model_name}/factory/{model_name}_factory.py`:

```python
from common.base import IDetectionFactory
from common.processors import LetterboxPreprocessor, {ModelName}Postprocessor
from common.visualizers import DetectionVisualizer


class {ModelName}Factory(IDetectionFactory):
    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width, input_height):
        return LetterboxPreprocessor(input_width, input_height)

    def create_postprocessor(self, input_width, input_height):
        return {ModelName}Postprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return DetectionVisualizer()

    def get_model_name(self) -> str:
        return "{model_name}"

    def get_task_type(self) -> str:
        return "object_detection"
```

Create `factory/__init__.py`:
```python
from .{model_name}_factory import {ModelName}Factory
```

### Step 3: Create Application Script

Based on `{variant}`, create the appropriate runner script:

**For sync:**
```python
#!/usr/bin/env python3
import sys
from pathlib import Path
_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import {ModelName}Factory
from common.runner import SyncRunner, parse_common_args

def main():
    args = parse_common_args("{ModelName} Sync Detection")
    factory = {ModelName}Factory()
    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
```

**For async:** Replace `SyncRunner` with `AsyncRunner`.

**For cpp_postprocess variants:** Add the `on_engine_init` callback with the appropriate
C++ postprocessor from `dx_postprocess`.

### Step 4: Create config.json

```json
{
  "score_threshold": <default_threshold from registry>,
  "nms_threshold": <nms_threshold from registry>
}
```

### Step 5: Validate

1. Syntax check: `python -c "import py_compile; py_compile.compile('<file>', doraise=True)"`
2. Verify factory implements all 5 IFactory methods
3. Verify config.json is valid JSON
4. Verify model name matches between factory, filename, and registry
5. Run `--help` to confirm parse_common_args works:
   ```bash
   python {model_name}_{variant}.py --help
   ```

### Checklist

- [ ] Model exists in model_registry.json with task=object_detection
- [ ] Factory implements all 5 methods (create_preprocessor, create_postprocessor, create_visualizer, get_model_name, get_task_type)
- [ ] Factory constructor accepts config: dict = None
- [ ] factory/__init__.py exports the factory class
- [ ] App script uses parse_common_args() (no custom argparse)
- [ ] App script has standard sys.path insertion pattern
- [ ] config.json has score_threshold and nms_threshold
- [ ] __init__.py exists at model directory level
- [ ] No hardcoded model paths
- [ ] No relative imports (all use sys.path pattern)
