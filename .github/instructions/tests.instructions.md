---
applyTo: "tests/**"
---

# Tests — Contextual Instructions

Working on dx_app test files.

## Required Context
- `.deepx/skills/dx-validate.md`
- `.deepx/instructions/testing-patterns.md`

## Rules
- Use pytest markers: `@pytest.mark.npu_required`, `@pytest.mark.slow`, `@pytest.mark.smoke`
- Tests not requiring NPU: `pytest tests/ -m "not npu_required"`
- NPU integration tests: `pytest tests/ -m npu_required`
