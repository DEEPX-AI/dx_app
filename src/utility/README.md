
# src/cpp_example/utility

Small collection of shared utility helpers used by the example applications (e.g. camera, image, EfficientNet samples). These utilities provide simple filesystem helpers and lightweight debug output functions.

## Files
- `common_util_inline.hpp` - Small inline utility functions (extension/filename extraction, vector printing, etc.).
- `common_util.hpp` / `common_util.cpp` - Additional shared utilities with separated implementation.

Note: In this repository the utilities may also appear under `lib/utils`; the include paths are managed by the project's CMake configuration.

## Public API (high level)
- `dxapp::common::getExtension(const std::string& path)`
  - Returns the file extension (without the dot) for the given path. Example: `"image.jpg" -> "jpg"`.
- `dxapp::common::getFileName(const std::string& path)`
  - Returns the filename component from a full path. Example: `"/foo/bar/image.jpg" -> "image.jpg"`.
- `dxapp::common::show(std::vector<T> vec)`
  - Debug helper that prints the contents of a vector to stdout (templated).

## Example
```cpp
#include "common_util_inline.hpp"
#include <iostream>

int main() {
    std::string path = "/home/user/data/image.jpg";
    std::cout << dxapp::common::getFileName(path) << "\n";    // -> image.jpg
    std::cout << dxapp::common::getExtension(path) << "\n";   // -> jpg

    std::vector<int> v = {1,2,3};
    dxapp::common::show(v); // prints vector contents
    return 0;
}
```

## Build
These utility headers/sources are normally included by the top-level CMake targets. To test locally, ensure the include directory containing these headers is added to your compile flags or CMake target.

## Contributing
This module is intentionally small and focused. Improvements are welcome (for example: better cross-platform path handling, empty input handling, or optimizing the debug output). Please open a pull request with a brief description of the change.
