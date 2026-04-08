---
name: dx-validate
description: Run validation checks at every phase gate for dx_app framework and application code
---

# Validation

Read `.deepx/skills/dx-validate.md` for full validation patterns.

## 6-Level Validation Pyramid
- Levels 1-3: Static, Config, Component (no NPU)
- Level 4: Smoke test (NPU required)
- Level 5: **Output Accuracy** — detection count > 0, bbox validity, confidence range, class ID range, postprocessor cross-check
- Level 6: Integration (full pipeline)

## Quick Reference
```bash
python .deepx/scripts/validate_framework.py   # Framework structure
python .deepx/scripts/validate_app.py <dir>    # App code patterns
pytest tests/ -m "not npu_required"            # Unit tests
```
