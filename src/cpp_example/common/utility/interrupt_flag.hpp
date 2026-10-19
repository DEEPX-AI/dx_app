/**
 * @file interrupt_flag.hpp
 * @brief Single definition of the shared graceful-shutdown interrupt flag.
 *
 * `g_interrupted()` is used both by common_util.hpp (window-close / 'q' key
 * handling) and run_dir.hpp (SIGINT/SIGTERM handling). Previously it was
 * declared inline in common_util.hpp and defined (also inline) in
 * run_dir.hpp; any translation unit that included common_util.hpp without
 * also including run_dir.hpp (e.g. the generated graph_registry_*.cpp files)
 * referenced a function template/inline entity that was declared but never
 * defined in that TU — ill-formed, no diagnostic required (NDR), and GCC
 * warns "used but never defined" for it. This header gives it one inline
 * definition that both headers include, so every TU sees a real definition.
 */

#ifndef DXAPP_INTERRUPT_FLAG_HPP
#define DXAPP_INTERRUPT_FLAG_HPP

#include <atomic>

namespace dxapp {

/**
 * @brief Shared flag set by SIGINT/SIGTERM or window-close/'q'-key handling.
 * All runner loops should check this flag in their iteration condition.
 */
inline std::atomic<bool>& g_interrupted() {
    static std::atomic<bool> flag{false};
    return flag;
}

}  // namespace dxapp

#endif  // DXAPP_INTERRUPT_FLAG_HPP
