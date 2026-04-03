---
applyTo: "src/python_example/**"
---

# Python Example — Contextual Instructions

Working on dx_app Python inference examples.

## Required Context
- `.deepx/skills/dx-build-python-app.md`
- `.deepx/memory/common_pitfalls.md`
- `.deepx/instructions/coding-standards.md`

## Rules
- IFactory is mandatory — implement all 5 methods
- Use `parse_common_args()` only — no custom argparse
- 4-variant naming: `_sync.py`, `_async.py`, `_sync_cpp_postprocess.py`, `_async_cpp_postprocess.py`
- sys.path insertion pattern for imports — no relative imports
- Every model directory needs `config.json` and `__init__.py`
