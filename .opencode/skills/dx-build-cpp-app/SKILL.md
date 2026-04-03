---
name: dx-build-cpp-app
description: Build C++ inference application using InferenceEngine for dx_app with C++14, RAII, and CMake
---

# Build C++ Inference App

Read `.deepx/skills/dx-build-cpp-app.md` for full patterns and templates.

## Quick Reference
1. Create directory `src/cpp_example/<task>/<model>/`
2. Write `main.cpp` with InferenceEngine
3. Create `CMakeLists.txt` linking dx_engine
4. Build with `./build.sh`
