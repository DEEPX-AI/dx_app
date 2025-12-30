#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace dxapp {
namespace common {
// Small utility functions that benefit from inlining

/**
 * Get file extension from path
 * @param path File path
 * @return File extension without dot
 */
inline std::string getExtension(const std::string &path) {
    size_t pos = path.find_last_of(".");
    if (pos == std::string::npos) return "";
    return path.substr(pos + 1);
}

/**
 * Get filename from full path
 * @param path Full file path
 * @return Filename only
 */
inline std::string getFileName(const std::string &path) {
    return path.substr(path.find_last_of("/\\") + 1);
}

/**
 * Display vector contents to console
 * @param vec Vector to display
 */
template <typename T>
inline void show(std::vector<T> vec) {
    std::cout << "\n[ ";
    for (auto &v : vec) {
        std::cout << std::dec << v << ", ";
    }
    std::cout << " ]" << std::endl;
}

}  // namespace common
}  // namespace dxapp
