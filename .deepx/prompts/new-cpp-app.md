# Prompt: New C++ Application

> Template for creating a C++ inference application in dx_app.

## Variables

| Variable | Description | Example |
|---|---|---|
| `{model_name}` | Model name from model_registry.json | `yolo26n` |
| `{task}` | AI task | `object_detection` |
| `{mode}` | Execution mode | `sync`, `async`, `both` |

## Prompt

Build a C++ inference application for `{model_name}` (`{task}`) with `{mode}` execution
using the InferenceEngine API directly.

### Step 1: Verify Model

Query `config/model_registry.json`:
- Confirm `{model_name}` exists and `supported: true`
- Get `input_size`, `task`, `default_threshold`, `nms_threshold`

### Step 2: Create CMakeLists.txt

Create `src/cpp_example/{task}/{model_name}/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project({model_name}_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(dx_engine REQUIRED)

add_executable({model_name}_sync {model_name}_sync.cpp)
target_link_libraries({model_name}_sync PRIVATE
    dx_engine::dx_engine
    dx_postprocess
    ${OpenCV_LIBS}
)
target_include_directories({model_name}_sync PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp_example/common
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

### Step 3: Create main.cpp

Create `src/cpp_example/{task}/{model_name}/{model_name}_sync.cpp`:

```cpp
#include <csignal>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "dx_engine/inference_engine.h"
#include "dx_engine/inference_option.h"

static volatile bool g_running = true;
void sigint_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.dxnn> <image|video>" << std::endl;
        return 1;
    }

    std::signal(SIGINT, sigint_handler);

    // Create engine
    dx::InferenceOption option;
    dx::InferenceEngine engine(argv[1], option);

    auto input_info = engine.get_input_tensors_info();
    int input_h = input_info[0].shape[1];
    int input_w = input_info[0].shape[2];
    std::cout << "[INFO] Input: " << input_w << "x" << input_h << std::endl;

    // Read image
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot load: " << argv[2] << std::endl;
        return 1;
    }

    // Preprocess
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_w, input_h));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // Infer
    std::vector<float> input_data(
        resized.begin<float>(), resized.end<float>());
    auto outputs = engine.run({input_data});

    // Postprocess (model-specific)
    std::cout << "[INFO] Got " << outputs.size() << " output tensors" << std::endl;

    // Display
    if (g_running) {
        cv::imshow("Result", image);
        cv::waitKey(0);
    }

    return 0;
}
```

### Step 4: Create Factory Header (Optional Runner Pattern)

Create `src/cpp_example/{task}/{model_name}/factory/{model_name}_factory.hpp`:

```cpp
#pragma once
#include "common/base/i_{task_interface}_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/visualizers/{task}_visualizer.hpp"

namespace dxapp {

class {ModelName}Factory : public I{TaskInterface}Factory {
public:
    std::unique_ptr<IPreprocessor> create_preprocessor(
        int input_w, int input_h) const override;
    std::unique_ptr<IPostprocessor> create_postprocessor(
        int input_w, int input_h) const override;
    std::unique_ptr<IVisualizer> create_visualizer() const override;
    std::string get_model_name() const override;
    std::string get_task_type() const override;

private:
    float score_threshold_ = {default_threshold};
    float nms_threshold_ = {nms_threshold};
};

}  // namespace dxapp
```

### Step 5: Create config.json

```json
{
  "score_threshold": {default_threshold},
  "nms_threshold": {nms_threshold}
}
```

### Step 6: Build

```bash
cd build && cmake .. && make {model_name}_sync
```

### Step 7: Validate

1. CMakeLists.txt compiles without errors
2. SIGINT handler is present for all loop-based examples
3. InferenceEngine uses RAII (no raw `new`/`delete`)
4. Error code checks after `engine.get_input_tensors_info()`
5. No hardcoded model paths (argv only)
6. C++14 compliance (no C++17 features)

### Checklist

- [ ] CMakeLists.txt with correct targets and link libraries
- [ ] SIGINT handler installed before any loop
- [ ] RAII: std::unique_ptr for factories and components
- [ ] Error handling: check engine creation, file loading, tensor info
- [ ] C++14 standard (no std::optional, structured bindings, etc.)
- [ ] Namespace: all classes in `dxapp` namespace
- [ ] config.json with model-specific thresholds
- [ ] No hardcoded paths
- [ ] snake_case files, PascalCase classes, UPPER_CASE constants
