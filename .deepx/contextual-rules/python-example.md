---
glob: "src/python_example/**"
description: Rules for Python example applications in dx_app.
---

# Python Example Rules

## IFactory Is Mandatory

Every Python model application MUST implement an IFactory subclass in `factory/<model>_factory.py`.
The factory MUST implement all 5 abstract methods:

1. `create_preprocessor(self, input_width, input_height)`
2. `create_postprocessor(self, input_width, input_height)`
3. `create_visualizer(self)`
4. `get_model_name(self) -> str`
5. `get_task_type(self) -> str`

The factory constructor MUST accept `config: dict = None`.

## parse_common_args() Only

All app scripts MUST use `parse_common_args()` from `common.runner`. Never define
custom `argparse.ArgumentParser` in model-specific scripts. This ensures all 11
standard CLI flags work consistently.

```python
from common.runner import parse_common_args
args = parse_common_args("Model Description")
```

## 4-Variant Naming Convention

| Variant | Filename Pattern |
|---|---|
| Sync | `<model>_sync.py` |
| Async | `<model>_async.py` |
| Sync C++ Postprocess | `<model>_sync_cpp_postprocess.py` |
| Async C++ Postprocess | `<model>_async_cpp_postprocess.py` |

File names MUST match exactly. The model name portion must be lowercase with
underscores, matching the key in `config/model_registry.json`.

## Absolute Imports via sys.path

All imports use the sys.path insertion pattern — never use relative imports:

```python
import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Now these resolve:
from factory import ModelFactory        # from _module_dir
from common.runner import SyncRunner    # from _v3_dir
```

## Per-variant config.json

Each `<family>/<variant>/` directory MUST contain a `config.json`, next to the
variant's `factory/` and entry scripts. A single-model extract may place one
`config.json` beside the entry script.

Minimum keys:

```json
{
  "score_threshold": 0.25
}
```

Detection models also include `"nms_threshold"`. Each variant's factory loads
its settings through `common.variant_config.load_variant_config`.

## Directory Structure

```
src/python_example/<task>/<family>/
    __init__.py                          # Required (empty)
    <variant>/                           # Directory name is the .dxnn stem
        config.json                      # Required
        factory/
            __init__.py                  # Exports the factory
            <variant>_factory.py         # IFactory for this variant
        <variant>_sync.py                # Entry; variant is fixed
        <variant>_async.py
        <variant>_sync_cpp_postprocess.py
        <variant>_async_cpp_postprocess.py
```

## Prohibited Patterns

- No `import argparse` in model scripts (use parse_common_args)
- No `from . import` or `from .. import` (no relative imports)
- No hardcoded `.dxnn` paths (model path comes from `--model` CLI arg)
- No `cv2.imshow()` without checking `args.no_display`
- No bare `except:` clauses (always specify exception type)
