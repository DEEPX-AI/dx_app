---
glob: "src/postprocess/**"
description: Rules for pybind11 postprocess bindings in dx_app.
---

# Postprocess Module Rules

## pybind11 Binding Pattern

Each postprocessor is a C++ class exposed to Python via pybind11. The binding
follows a consistent pattern:

```cpp
// src/postprocess/bindings/<model>_postprocess_binding.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "<model>_postprocess.h"

namespace py = pybind11;

void bind_<model>_postprocess(py::module& m) {
    py::class_<dx::<Model>PostProcess>(m, "<Model>PostProcess")
        .def(py::init<int, int, float, float, bool>(),
             py::arg("input_width"),
             py::arg("input_height"),
             py::arg("score_threshold"),
             py::arg("nms_threshold"),
             py::arg("use_ort"))
        .def("process", &dx::<Model>PostProcess::process,
             py::arg("outputs"));
}
```

## Module Naming

The pybind11 module is named `dx_postprocess`. All bindings are registered
in a single module:

```cpp
// src/postprocess/bindings/module.cpp

#include <pybind11/pybind11.h>

// Forward declarations
void bind_yolov5_postprocess(py::module& m);
void bind_yolov8_postprocess(py::module& m);
// ... all 37 bindings ...

PYBIND11_MODULE(dx_postprocess, m) {
    m.doc() = "DEEPX postprocess bindings for dx_app";
    bind_yolov5_postprocess(m);
    bind_yolov8_postprocess(m);
    // ... register all bindings ...
}
```

## build.sh Integration

The postprocess module is built by `build.sh` at the dx_app root:

```bash
#!/bin/bash
# Build postprocess pybind11 module
mkdir -p build && cd build
cmake .. -DBUILD_POSTPROCESS=ON
make dx_postprocess -j$(nproc)
# Install into Python environment
cp dx_postprocess*.so $(python -c "import site; print(site.getsitepackages()[0])")/
```

## CMakeLists Registration

New postprocessors must be registered in the CMakeLists.txt:

```cmake
# src/postprocess/CMakeLists.txt

pybind11_add_module(dx_postprocess
    bindings/module.cpp
    bindings/yolov5_postprocess_binding.cpp
    bindings/yolov8_postprocess_binding.cpp
    # ... add new binding files here ...
    bindings/<new_model>_postprocess_binding.cpp
)

target_link_libraries(dx_postprocess PRIVATE
    dx_engine::dx_engine
    ${OpenCV_LIBS}
)
```

## Adding a New Postprocessor

1. Create the C++ implementation: `src/postprocess/<model>_postprocess.{h,cpp}`
2. Create the binding: `src/postprocess/bindings/<model>_postprocess_binding.cpp`
3. Register in `module.cpp`: add forward declaration and `bind_<model>_postprocess(m);`
4. Add to `CMakeLists.txt`: add the binding .cpp file
5. Rebuild: `./build.sh`
6. Test: `python -c "from dx_postprocess import <Model>PostProcess; print('OK')"`

## Conventions

- Binding class names use `PascalCase` matching the C++ class: `YoloV8PostProcess`
- Python-side names are identical to C++ class names (no renaming)
- All constructors take `input_width, input_height` as first two parameters
- All process() methods accept `list[np.ndarray]` and return typed results
- Use `py::arg()` for all parameters to enable keyword arguments in Python
