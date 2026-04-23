---
name: dx-build-cpp-app
description: Build C++ inference app for dx_app
---

<!-- AUTO-GENERATED from .deepx/ — DO NOT EDIT DIRECTLY -->
<!-- Source: .deepx/skills/dx-build-cpp-app/SKILL.md -->
<!-- Run: dx-agentic-gen generate -->

# Skill: Build C++ Inference App for dx_app

> **This skill doc is sufficient.** Do NOT read source code files in
> `src/cpp_example/common/` unless this document is insufficient.

## Overview

Build a native C++ inference application for dx_app v3.0.0. C++ apps use the
InferenceEngine and InferenceOption APIs from dx_engine, link against dx_postprocess
for NMS/decode, and follow the same factory pattern as Python apps.

## Output Isolation (MUST FOLLOW)

All AI-generated applications MUST be created under `dx-agentic-dev/`, NOT in the
production `src/` directory. This prevents accidental modification of existing code.

### Session Directory

```
dx-agentic-dev/<YYYYMMDD-HHMMSS>_<model>_<task>/
├── session.json          # Build metadata
├── README.md             # How to run this app
├── CMakeLists.txt
├── main.cpp
├── include/
│   └── <model>_factory.hpp
├── src/
│   └── <model>_factory.cpp
└── config.json
```

### session.json Template

```json
{
  "session_id": "<YYYYMMDD-HHMMSS>_<model>_<task>",
  "created_at": "<ISO 8601 timestamp>",
  "model": "<model_name>",
  "task": "<task_type>",
  "language": "cpp",
  "status": "complete",
  "notes": "<any relevant notes>"
}
```

### When to Use Production Path

Only create files in `src/cpp_example/<task>/<model>/` when the user EXPLICITLY says:
- "Add this to the production codebase"
- "Create this in src/"
- "Make this a permanent addition"

Default behavior: ALWAYS use `dx-agentic-dev/`.

## Phase 0: Prerequisites Check

Before starting the build workflow, verify:

1. **dx-runtime**: `bash ../../scripts/sanity_check.sh --dx_rt`
   - FAIL → `bash ../../install.sh --all --exclude-app --exclude-stream --skip-uninstall --venv-reuse`
   - Re-run sanity_check.sh — must PASS after install
   - **If still failing → STOP (unconditional).** User instructions to continue do NOT override this.
     If NPU hardware init failure ("Device initialization failed"): tell the user a cold boot /
     system reboot is required, then STOP. NEVER proceed with code generation while sanity check is failing.
     NEVER mark this check as "done" when it actually failed.
2. **dx_engine + dx_postprocess libraries**: `ldconfig -p | grep libdx_engine`
3. **OpenCV**: `pkg-config --exists opencv4 && echo OK`
4. **CMake >= 3.14**: `cmake --version`

## Prerequisites

- dx_app repository cloned and built (`./build.sh`)
- Model .dxnn file available
- OpenCV installed
- dx_engine and dx_postprocess libraries built

## Step 1: Create Directory Structure

```bash
mkdir -p src/cpp_example/<TASK>/<MODEL>/factory
```

## Step 2: Create Factory Header

### Template: `factory/<model>_factory.hpp`

```cpp
#pragma once

/**
 * @file <model>_factory.hpp
 * @brief Factory for <ModelDisplay> inference components.
 */

#include "common/base/i_detection_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/postprocess/<postprocess_header>.hpp"
#include "common/visualizers/detection_visualizer.hpp"

namespace dxapp {

class <ModelClass>Factory : public IDetectionFactory {
public:
    std::unique_ptr<IPreprocessor> create_preprocessor(
            int input_w, int input_h) const override {
        return std::make_unique<LetterboxPreprocessor>(input_w, input_h);
    }

    std::unique_ptr<IPostprocessor> create_postprocessor(
            int input_w, int input_h) const override {
        return std::make_unique<<ModelClass>Postprocessor>(
            input_w, input_h, score_threshold_, nms_threshold_);
    }

    std::unique_ptr<IVisualizer> create_visualizer() const override {
        return std::make_unique<DetectionVisualizer>();
    }

    std::string get_model_name() const override { return "<model_name>"; }
    std::string get_task_type() const override { return "<task_type>"; }

    void set_score_threshold(float t) { score_threshold_ = t; }
    void set_nms_threshold(float t) { nms_threshold_ = t; }

private:
    float score_threshold_ = 0.25f;
    float nms_threshold_ = 0.45f;
};

}  // namespace dxapp
```

## Step 3: Create Sync Main

### Template: `<model>_sync.cpp`

Using the runner framework (recommended):

```cpp
/**
 * @file <model>_sync.cpp
 * @brief <ModelDisplay> synchronous inference example.
 */

#include "factory/<model>_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::<ModelClass>Factory>();
    dxapp::SyncDetectionRunner<dxapp::<ModelClass>Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
```

Using raw InferenceEngine (for custom pipelines):

```cpp
/**
 * @file <model>_sync.cpp
 * @brief <ModelDisplay> synchronous inference using InferenceEngine.
 */

#include <csignal>
#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "dx_engine/inference_engine.h"
#include "dx_engine/inference_option.h"

static volatile bool g_running = true;
static void sigint_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.dxnn> <input_image_or_video>" << std::endl;
        return 1;
    }

    std::signal(SIGINT, sigint_handler);

    const std::string model_path = argv[1];
    const std::string input_path = argv[2];

    // --- Initialize InferenceEngine ---
    dx::InferenceOption option;
    dx::InferenceEngine engine(model_path, option);

    auto input_info = engine.get_input_tensors_info();
    if (input_info.empty()) {
        std::cerr << "[ERROR] Failed to get input tensor info" << std::endl;
        return 1;
    }

    const auto& shape = input_info[0].shape;
    const int input_h = shape[1];
    const int input_w = shape[2];
    std::cout << "[INFO] Model input: " << input_w << "x" << input_h << std::endl;

    // --- Load and preprocess image ---
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot read: " << input_path << std::endl;
        return 1;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_w, input_h));

    // Convert to float32 normalized [0, 1]
    cv::Mat float_input;
    resized.convertTo(float_input, CV_32F, 1.0 / 255.0);

    // Flatten to vector
    std::vector<float> input_data(
        float_input.begin<float>(), float_input.end<float>());

    // --- Run inference ---
    auto start = std::chrono::steady_clock::now();
    auto outputs = engine.run({input_data});
    auto end = std::chrono::steady_clock::now();

    double infer_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[INFO] Inference time: " << infer_ms << " ms" << std::endl;

    // --- Postprocess ---
    // Apply model-specific decoding to outputs
    // outputs[i] is std::vector<float>
    std::cout << "[INFO] Output tensors: " << outputs.size() << std::endl;
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "[INFO]   Output " << i << " size: "
                  << outputs[i].size() << std::endl;
    }

    // --- Visualize ---
    // Draw results on original image
    // ... (model-specific drawing code) ...

    // --- Display ---
    if (g_running) {
        cv::imshow("Output", image);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
    return 0;
}
```

## Step 4: Create Async Main (Optional)

### Template: `<model>_async.cpp`

```cpp
/**
 * @file <model>_async.cpp
 * @brief <ModelDisplay> asynchronous inference example.
 */

#include "factory/<model>_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::<ModelClass>Factory>();
    dxapp::AsyncDetectionRunner<dxapp::<ModelClass>Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
```

## Step 5: Create CMakeLists.txt

### Template: `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.14)

# Sync example
add_executable(<model>_sync <model>_sync.cpp)
target_link_libraries(<model>_sync PRIVATE
    dx_engine::dx_engine
    dx_postprocess
    ${OpenCV_LIBS}
)
target_include_directories(<model>_sync PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp_example/common
    ${CMAKE_SOURCE_DIR}/src/cpp_example
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Async example
add_executable(<model>_async <model>_async.cpp)
target_link_libraries(<model>_async PRIVATE
    dx_engine::dx_engine
    dx_postprocess
    ${OpenCV_LIBS}
)
target_include_directories(<model>_async PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp_example/common
    ${CMAKE_SOURCE_DIR}/src/cpp_example
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

## Step 6: Create config.json

Same as Python config.json:

```json
{
  "score_threshold": 0.25,
  "nms_threshold": 0.45
}
```

## Step 7: Validate

```bash
# 1. Check CMakeLists.txt syntax
cmake -P CMakeLists.txt 2>&1 || true  # Will warn but not fail for add_executable

# 2. Try building (from dx_app root)
cd build && cmake .. && make <model>_sync -j$(nproc)

# 3. Run (requires NPU + model)
./<model>_sync /path/to/<model>.dxnn input.jpg
```

### Task-Aware Sample Image for Validation

Select sample images based on the model's AI task:

| Task | Sample Image |
|---|---|
| object_detection | `../../sample/img/sample_dog.jpg` |
| face_detection | `../../sample/img/sample_face.jpg` |
| pose_estimation | `../../sample/img/sample_people.jpg` |
| classification | `../../sample/ILSVRC2012/0.jpeg` |
| segmentation | `../../sample/img/sample_street.jpg` |

**MUST** use task-matched images in validation commands.

## InferenceEngine API Reference

```cpp
// Construction
dx::InferenceOption option;
option.set_use_ort(true);
dx::InferenceEngine engine("model.dxnn");        // default options
dx::InferenceEngine engine("model.dxnn", option); // custom options

// Model info
auto input_info = engine.get_input_tensors_info();
// input_info[0].shape = {1, 640, 640, 3}

auto output_info = engine.get_output_tensors_info();

// Synchronous inference
std::vector<std::vector<float>> outputs = engine.run({input_data});

// Asynchronous inference
int req_id = engine.run_async({input_data});
std::vector<std::vector<float>> outputs = engine.wait(req_id);

// Version
int model_ver = engine.get_model_version();
```

## C++ Conventions Checklist

- [ ] C++14 standard (no C++17 features)
- [ ] `dxapp` namespace for all classes
- [ ] PascalCase class names, snake_case functions/variables
- [ ] SIGINT handler for video/camera loops
- [ ] RAII with `std::unique_ptr` (no raw new/delete)
- [ ] Error checking on engine initialization
- [ ] Model path from command-line arguments only
- [ ] Include guards (`#pragma once`)

## File Structure Summary

```
src/cpp_example/<task>/<model>/
    CMakeLists.txt
    config.json
    factory/
        <model>_factory.hpp
    <model>_sync.cpp
    <model>_async.cpp           # optional
```

## Build and Run

```bash
# Build
cd dx_app
mkdir -p build && cd build
cmake ..
make <model>_sync -j$(nproc)

# Run
./<model>_sync /path/to/<model>.dxnn input.jpg
./<model>_async /path/to/<model>.dxnn input.mp4
```
