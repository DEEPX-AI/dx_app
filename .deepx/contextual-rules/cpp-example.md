---
glob: "src/cpp_example/**"
description: Rules for C++ example applications in dx_app.
---

# C++ Example Rules

## CMakeLists.txt Required

Every C++ example directory MUST contain a `CMakeLists.txt` that:
- Sets `CMAKE_CXX_STANDARD 14` (no C++17 features)
- Links against `dx_engine::dx_engine`, `dx_postprocess`, and `${OpenCV_LIBS}`
- Sets include directories for `common/` and the current source directory
- Defines at least one `add_executable()` target

```cmake
cmake_minimum_required(VERSION 3.14)
project(<model>_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(dx_engine REQUIRED)

add_executable(<model>_sync <model>_sync.cpp)
target_link_libraries(<model>_sync PRIVATE
    dx_engine::dx_engine
    dx_postprocess
    ${OpenCV_LIBS}
)
```

## SIGINT Handler Mandatory

All C++ examples that process video or camera input (any loop) MUST install a
SIGINT handler for graceful shutdown:

```cpp
#include <csignal>

static volatile bool g_running = true;
void sigint_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    std::signal(SIGINT, sigint_handler);
    // ...
    while (g_running) {
        // process frames
    }
    return 0;
}
```

Without this, Ctrl+C leaves the NPU in a locked state requiring `dxrt-cli --reset`.

## InferenceEngine RAII

Use RAII for all resource management:

```cpp
// CORRECT: stack allocation or unique_ptr
dx::InferenceEngine engine("model.dxnn", option);
auto factory = std::make_unique<dxapp::ModelFactory>();

// WRONG: raw new/delete
dx::InferenceEngine* engine = new dx::InferenceEngine("model.dxnn", option);
delete engine;
```

## Error Code Checks

Always check return values and handle errors:

```cpp
auto input_info = engine.get_input_tensors_info();
if (input_info.empty()) {
    std::cerr << "[ERROR] Failed to get input tensor info" << std::endl;
    return 1;
}

cv::Mat image = cv::imread(argv[2]);
if (image.empty()) {
    std::cerr << "[ERROR] Cannot load: " << argv[2] << std::endl;
    return 1;
}
```

## Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Files | `snake_case` | `yolov8n_sync.cpp` |
| Classes | `PascalCase` | `YoloV8nFactory` |
| Variables | `snake_case` | `input_width` |
| Constants | `UPPER_CASE` | `MAX_DETECTIONS` |
| Namespace | `dxapp` | `namespace dxapp { ... }` |
| Private members | trailing underscore | `score_threshold_` |
| Header guards | `#pragma once` | — |

## Prohibited Patterns

- No C++17 features (`std::optional`, `std::filesystem`, structured bindings, `if constexpr`)
- No raw `new`/`delete` (use smart pointers)
- No global mutable state (except SIGINT flag)
- No hardcoded model paths (always from argv)
- No `using namespace std;` in headers
