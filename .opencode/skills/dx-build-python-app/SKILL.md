---
name: dx-build-python-app
description: Build Python inference application for dx_app with IFactory pattern, SyncRunner/AsyncRunner, and all 4 variants (sync, async, sync_cpp_postprocess, async_cpp_postprocess)
---

# Build Python Inference App

Read the full skill document at `.deepx/skills/dx-build-python-app.md` for complete patterns, templates, and step-by-step instructions.

## Quick Reference

1. Query `config/model_registry.json` for the model
2. **Search existing examples** in `src/python_example/<task>/<model>/` — if found, use same postprocessor
3. Create directory `src/python_example/<task>/<model>/`
4. Implement IFactory in `factory/<model>_factory.py`
5. Create `<model>_sync.py` with SyncRunner
6. Create `<model>_async.py` with AsyncRunner
7. Add `config.json` with thresholds
8. **Postprocessor cross-check** — verify registry key → Python class mapping (e.g., `yolov26` → `YOLOv8Postprocessor`)
9. **Output accuracy** — detection count > 0 on task-appropriate sample image
10. Validate with `python .deepx/scripts/validate_app.py <dir>`
