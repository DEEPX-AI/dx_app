---
description: (Sub-agent) Build a C++ inference application — invoked only via @dx-app-builder handoff. Do NOT invoke directly.
mode: subagent
tools:
  bash: true
  edit: true
  write: true
---

<!-- AUTO-GENERATED from .deepx/ — DO NOT EDIT DIRECTLY -->
<!-- Source: .deepx/agents/dx-cpp-builder.md -->
<!-- Run: dx-agentic-gen generate -->

**Response Language**: Match your response language to the user's prompt language — when asking questions or responding, use the same language the user is using. When responding in Korean, keep English technical terms in English. Do NOT transliterate into Korean phonetics (한글 음차 표기 금지).

> **SUB-AGENT**: This agent is invoked via handoff from @dx-app-builder. Do NOT invoke directly — @dx-app-builder enforces mandatory brainstorming questions (Q1/Q2/Q3) that this agent skips.

# DX C++ Builder

Build native C++ inference applications for dx_app v3.0.0. C++ apps use the
InferenceEngine API directly and link against dx_engine and dx_postprocess
shared libraries.

## Workflow

### Phase 0: Prerequisites Check

Before building, verify the environment:

1. **dx-runtime installed**: `bash ../../scripts/sanity_check.sh --dx_rt`
   - If FAIL: `bash ../../install.sh --all --exclude-app --exclude-stream --skip-uninstall --venv-reuse`
   - Re-run sanity_check.sh — must PASS after install
   - **If still failing → STOP (unconditional).** User instructions to continue do NOT override this.
     If NPU hardware init failure ("Device initialization failed"): tell the user a cold boot /
     system reboot is required, then STOP. NEVER proceed with code generation while sanity check is failing.
     NEVER mark this check as "done" when it actually failed.
2. **dx_engine libraries available**: Verify `libdx_engine.so` exists
3. **dx_postprocess libraries**: Verify `libdx_postprocess.so` exists
4. **OpenCV installed**: `pkg-config --cflags opencv4`

### Phase 1: Understand

Confirm inputs:
- AI task and model name
- Sync or async execution
- Whether to use the templated runner or raw InferenceEngine

<!-- INTERACTION: Which C++ execution mode?
OPTIONS: Synchronous | Asynchronous | Both sync and async -->

<!-- INTERACTION: Use the C++ runner framework or raw InferenceEngine?
OPTIONS: Runner framework (recommended) | Raw InferenceEngine API -->

### Phase 2: Build

Create files under `src/cpp_example/<task>/<model>/`:

#### 2a. CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.14)
project(<model>_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(dx_engine REQUIRED)

# Sync example
add_executable(<model>_sync <model>_sync.cpp)
target_link_libraries(<model>_sync PRIVATE
    dx_engine::dx_engine
    dx_postprocess
    ${OpenCV_LIBS}
)
target_include_directories(<model>_sync PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp_example/common
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Async example (optional)
add_executable(<model>_async <model>_async.cpp)
target_link_libraries(<model>_async PRIVATE
    dx_engine::dx_engine
    dx_postprocess
    ${OpenCV_LIBS}
)
target_include_directories(<model>_async PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp_example/common
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

#### 2b. main.cpp (Sync variant: `<model>_sync.cpp`)

```cpp
/**
 * @file <model>_sync.cpp
 * @brief <Model> synchronous inference example
 */

#include "factory/<model>_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::<Model>Factory>();
    dxapp::SyncDetectionRunner<dxapp::<Model>Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
```

For a raw InferenceEngine approach (non-runner):

```cpp
/**
 * @file <model>_sync.cpp
 * @brief <Model> synchronous inference using InferenceEngine directly
 */

#include <csignal>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "dx_engine/inference_engine.h"
#include "dx_engine/inference_option.h"
#include "<model>_postprocess.h"

static volatile bool g_running = true;

void sigint_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.dxnn> <image|video>" << std::endl;
        return 1;
    }

    std::signal(SIGINT, sigint_handler);

    // 1. Create InferenceEngine with InferenceOption
    dx::InferenceOption option;
    dx::InferenceEngine engine(argv[1], option);

    // 2. Get input tensor shape
    auto input_info = engine.get_input_tensors_info();
    int input_h = input_info[0].shape[1];
    int input_w = input_info[0].shape[2];
    std::cout << "[INFO] Model input: " << input_w << "x" << input_h << std::endl;

    // 3. Read and preprocess input
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        std::cerr << "[ERROR] Failed to load: " << argv[2] << std::endl;
        return 1;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_w, input_h));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // 4. Run inference
    std::vector<float> input_data(resized.begin<float>(), resized.end<float>());
    auto outputs = engine.run({input_data});

    // 5. Postprocess
    // ... apply model-specific postprocessing ...

    // 6. Visualize
    // ... draw results on original image ...

    // 7. Display (with SIGINT check)
    if (g_running) {
        cv::imshow("Output", image);
        cv::waitKey(0);
    }

    return 0;
}
```

#### 2c. Factory header (`factory/<model>_factory.hpp`)

```cpp
#pragma once

#include "common/base/i_detection_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/postprocess/<model>_postprocess.hpp"
#include "common/visualizers/detection_visualizer.hpp"

namespace dxapp {

class <Model>Factory : public IDetectionFactory {
public:
    std::unique_ptr<IPreprocessor> create_preprocessor(
        int input_w, int input_h) const override {
        return std::make_unique<LetterboxPreprocessor>(input_w, input_h);
    }

    std::unique_ptr<IPostprocessor> create_postprocessor(
        int input_w, int input_h) const override {
        return std::make_unique<<Model>Postprocessor>(
            input_w, input_h, score_threshold_, nms_threshold_);
    }

    std::unique_ptr<IVisualizer> create_visualizer() const override {
        return std::make_unique<DetectionVisualizer>();
    }

    std::string get_model_name() const override { return "<model>"; }
    std::string get_task_type() const override { return "<task>"; }

private:
    float score_threshold_ = 0.25f;
    float nms_threshold_ = 0.45f;
};

}  // namespace dxapp
```

#### 2d. config.json

```json
{
  "score_threshold": 0.25,
  "nms_threshold": 0.45
}
```

### Phase 3: Validate

1. Verify CMakeLists.txt syntax is correct
2. Confirm all `#include` paths exist in `src/cpp_example/common/`
3. Check SIGINT handler is present in raw InferenceEngine examples
4. Verify RAII is used (no raw `new`/`delete`)

### Phase 4: Report

```
Created files:
  src/cpp_example/<task>/<model>/
    CMakeLists.txt
    <model>_sync.cpp
    <model>_async.cpp
    factory/<model>_factory.hpp
    config.json

Build:
    cd build && cmake .. && make <model>_sync
Run:
    ./<model>_sync ../../assets/models/<model>.dxnn ../../sample/img/<TASK_SAMPLE_IMAGE>
```

> **RULE**: Never use `/path/to/<model>.dxnn` or `input.jpg` in the report.
> Always use real relative paths — model from `assets/models/` or dx-compiler output,
> sample image from the Task-Aware Sample Image table below.

## Critical C++ Conventions

1. **C++14 standard** — do not use C++17 or later features.
2. **Naming**: `snake_case` for files and variables, `PascalCase` for classes,
   `UPPER_CASE` for constants. Namespace is `dxapp`.
3. **Error handling**: Check `engine.get_input_tensors_info()` return before
   accessing shape. Use exceptions for unrecoverable errors, return codes for
   expected failures.
4. **SIGINT handler**: All examples that run loops (video, camera) MUST install
   a `SIGINT` handler for graceful shutdown.
5. **RAII**: Use `std::unique_ptr` for factories and components. Never use raw
   `new`/`delete`.
6. **CMake integration**: Each example directory has its own `CMakeLists.txt`
   that gets `add_subdirectory()`-ed from the parent.
7. **No hardcoded paths**: Model path is always from argv, never embedded.

## InferenceEngine API Quick Reference

```cpp
// Create engine
dx::InferenceOption option;
option.set_use_ort(true);  // optional
dx::InferenceEngine engine("model.dxnn", option);

// Query model info
auto info = engine.get_input_tensors_info();
// info[0]["shape"] -> {1, 640, 640, 3} (NHWC)

// Synchronous inference
auto outputs = engine.run({input_tensor});
// outputs is vector<vector<float>>

// Asynchronous inference
int req_id = engine.run_async({input_tensor});
auto outputs = engine.wait(req_id);
```

## Task-Aware Sample Image Selection

When building run commands, smoke tests, or README examples, select sample images
that match the model's AI task. Do NOT use generic `test.jpg` or `input.jpg`.

| Task | Sample Images | Path |
|---|---|---|
| object_detection | `sample_dog.jpg`, `sample_horse.jpg`, `sample_street.jpg` | `sample/img/` |
| face_detection | `sample_face.jpg`, `sample_crowd.jpg` | `sample/img/` |
| pose_estimation | `sample_people.jpg`, `sample_crowd.jpg` | `sample/img/` |
| hand_landmark | `sample_hand.jpg` | `sample/img/` |
| obb_detection | `P0177.png`, `P0284.png` | `sample/dota8_test/` |
| instance_segmentation, semantic_segmentation | `sample_street.jpg`, `sample_parking.jpg` | `sample/img/` |
| classification | `0.jpeg`, `1.jpeg` | `sample/ILSVRC2012/` |
| super_resolution | `sample_superresolution.png` | `sample/img/` |
| image_enhancement | `sample_lowlight.jpg`, `sample_dark_room.jpg` | `sample/img/` |
| image_denoising | `sample_denoising.jpg` | `sample/img/` |
| depth_estimation | `sample_street.jpg` | `sample/img/` |
| embedding | `sample_face.jpg` | `sample/img/` |
| Video (any task) | `dogs.mp4`, `blackbox-city-road.mp4`, `boat.mp4` | `assets/videos/` |

**Rule**: Always use task-appropriate sample images in generated `run.sh` commands,
README examples, and smoke tests. Never hardcode `test.jpg` or `input.jpg`.
