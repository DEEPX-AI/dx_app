---
name: dx-validate
description: Run validation checks at every phase gate for dx_app framework and application code
---

# Validation

Read `.deepx/skills/dx-validate.md` for full validation patterns.

## Quick Reference
```bash
python .deepx/scripts/validate_framework.py   # Framework structure
python .deepx/scripts/validate_app.py <dir>    # App code patterns
pytest tests/ -m "not npu_required"            # Unit tests
```
