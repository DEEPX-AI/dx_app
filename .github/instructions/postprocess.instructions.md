---
applyTo: "src/postprocess/**"
---

# Postprocess — Contextual Instructions

Working on dx_app C++ postprocess libraries and pybind11 bindings.

## Required Context
- `.deepx/toolsets/dx-engine-api.md`
- `.deepx/memory/common_pitfalls.md`

## Rules
- pybind11 bindings must match Python postprocessor interface
- Build via `./build.sh` — never compile individually
- Shared libraries go to `lib/` directory
