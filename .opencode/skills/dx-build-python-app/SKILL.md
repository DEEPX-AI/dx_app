---
name: dx-build-python-app
description: Build Python inference application for dx_app with IFactory pattern, SyncRunner/AsyncRunner, and all 4 variants (sync, async, sync_cpp_postprocess, async_cpp_postprocess)
---

# Build Python Inference App

Read the full skill document at `.deepx/skills/dx-build-python-app.md` for complete patterns, templates, and step-by-step instructions.

## Quick Reference

1. Query `config/model_registry.json` for the model
2. Create directory `src/python_example/<task>/<model>/`
3. Implement IFactory in `factory/<model>_factory.py`
4. Create `<model>_sync.py` with SyncRunner
5. Create `<model>_async.py` with AsyncRunner
6. Add `config.json` with thresholds
7. Validate with `python .deepx/scripts/validate_app.py <dir>`
