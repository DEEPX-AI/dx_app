---
applyTo: "src/cpp_example/**"
---

# C++ Example — Contextual Instructions

Working on dx_app C++ inference examples.

## Required Context
- `.deepx/skills/dx-build-cpp-app.md`
- `.deepx/toolsets/dx-engine-api.md`
- `.deepx/memory/common_pitfalls.md`

## Rules
- C++14 standard only
- RAII with std::unique_ptr — no raw new/delete
- SIGINT handler for loop-based apps
- CMakeLists.txt must link dx_engine
