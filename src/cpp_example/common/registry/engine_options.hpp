/**
 * @file engine_options.hpp
 * @brief The DX-RT options a graph stage's engine opens with.
 *
 * A node's "engine": {"buffers": N} has to reach the place a stage maker
 * opens its dxrt::InferenceEngine, and the makers are one generated table of
 * function pointers (StageMaker) shared by every model. Rather than change
 * that signature in every generated row, StaticModelRegistry::createStageWith
 * sets the count for THIS thread for the length of one maker call
 * (EngineBufferScope), and every engine is opened through OpenEngine, which
 * reads it. createStage is serialized by its caller (i_registry.hpp,
 * registry THREADING), so a thread-local scope is exact; a stage opened
 * outside any scope gets DX-RT's default.
 *
 * Lives on the zoo side of the i_registry.hpp boundary, next to
 * typed_stage.hpp: the engine never includes it.
 */
#ifndef DXAPP_REGISTRY_ENGINE_OPTIONS_HPP
#define DXAPP_REGISTRY_ENGINE_OPTIONS_HPP

#include <dxrt/dxrt_api.h>

#include <memory>
#include <string>

namespace dxapp {
namespace graph {
namespace detail {

/// The I/O buffer count for an engine opened on this thread now; 0 = DX-RT's
inline int& EngineBuffersSlot() {
    static thread_local int buffers = 0;
    return buffers;
}

/// Sets EngineBuffersSlot() for one scope and restores it on every exit.
class EngineBufferScope {
 public:
    explicit EngineBufferScope(int buffers) : saved_(EngineBuffersSlot()) {
        EngineBuffersSlot() = buffers;
    }
    ~EngineBufferScope() { EngineBuffersSlot() = saved_; }
    EngineBufferScope(const EngineBufferScope&) = delete;
    EngineBufferScope& operator=(const EngineBufferScope&) = delete;

 private:
    int saved_;
};

/// DX-RT's defaults, with this thread's buffer count.
inline dxrt::InferenceOption EngineOptionForThisThread() {
    dxrt::InferenceOption option;
    option.bufferCount = EngineBuffersSlot();
    return option;
}

/// Open the engine of `loadable_path` (already through LoadableModelPath).
inline std::unique_ptr<dxrt::InferenceEngine> OpenEngine(const std::string& loadable_path) {
    dxrt::InferenceOption option = EngineOptionForThisThread();
    return std::unique_ptr<dxrt::InferenceEngine>(
        new dxrt::InferenceEngine(loadable_path, option));
}

}  // namespace detail
}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_REGISTRY_ENGINE_OPTIONS_HPP
