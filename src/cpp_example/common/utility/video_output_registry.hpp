/**
 * @file video_output_registry.hpp
 * @brief Output videos opened by initVideoWriter, so a stream run that read no
 *        frame does not leave an empty, unplayable file behind.
 *
 * Every runner opens its VideoWriter before the first read. A live source
 * stopped before it delivers a frame (an RTSP server still waiting for its
 * first keyframe) then releases a writer that was never written: a 0-byte
 * output.mp4. initVideoWriter (run_dir.hpp) records each file it opens, and
 * emptyStreamStatus (common_util.hpp) - which every runner calls after it has
 * released the writer - deletes them when no frame was read. Both headers
 * include this one, so every translation unit sees one definition.
 */

#ifndef DXAPP_VIDEO_OUTPUT_REGISTRY_HPP
#define DXAPP_VIDEO_OUTPUT_REGISTRY_HPP

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace dxapp {

namespace detail {

struct VideoOutputRegistry {
    std::mutex mutex;
    std::vector<std::string> paths;
};

inline VideoOutputRegistry& videoOutputRegistry() {
    static VideoOutputRegistry registry;
    return registry;
}

}  // namespace detail

/// Records an output video file that a writer has just opened.
inline void registerVideoOutput(const std::string& path) {
    if (path.empty()) return;
    detail::VideoOutputRegistry& registry = detail::videoOutputRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.paths.push_back(path);
}

/**
 * @brief Forgets every recorded output video; with @p discard, deletes them.
 *
 * Call only after the writers have been released.
 * @return The number of files deleted.
 */
inline int takeVideoOutputs(bool discard) {
    std::vector<std::string> paths;
    {
        detail::VideoOutputRegistry& registry = detail::videoOutputRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        paths.swap(registry.paths);
    }
    int removed = 0;
    if (discard) {
        for (const std::string& path : paths) {
            if (std::remove(path.c_str()) == 0) ++removed;
        }
    }
    return removed;
}

}  // namespace dxapp

#endif  // DXAPP_VIDEO_OUTPUT_REGISTRY_HPP
