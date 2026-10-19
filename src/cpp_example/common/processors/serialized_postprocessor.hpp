/**
 * @file serialized_postprocessor.hpp
 * @brief Decorator that lets one IPostprocessor be shared by concurrent
 *        completion callbacks, by admitting one process() call at a time.
 *
 * WHY. Every async runner (common/runner/async_*_runner.hpp) builds ONE
 * postprocessor and captures it in the lambda it hands to
 * InferenceEngine::RegisterCallback. dxrt runs those callbacks on its output
 * worker pool (4 threads on x86), and up to ASYNC_BUFFER_SIZE (40) jobs are in
 * flight, so two frames' callbacks can be inside the same process() at once.
 * process() is not re-entrant in general: NanoDet resizes dfl_weights_,
 * YOLACT rebuilds priors_ lazily, ULFG-face rewrites box_format_ per frame,
 * and the YOLO decoders write num_classes_ - each is a data race when two
 * frames overlap, and some can decode one frame with another frame's state.
 *
 * WHAT. The same trade common/registry/typed_stage.hpp makes in
 * TypedStage::Decode() with its postprocess_mutex_ (see the doc comment
 * there): one mutex per shared postprocessor, held for exactly the inner
 * process() call. Decoding is CPU work done while the NPU is busy with the
 * next jobs, and the sync runners already decode one frame at a time, so
 * serializing it costs the async path nothing the sync path does not pay.
 *
 * WHAT IT MUST NOT DO. The lock covers process() only. It is never held
 * across a display-queue push, metrics, or a user callback - those happen in
 * the runner after process() returns - so it cannot deadlock against the
 * runner's own locks or widen into a pipeline-wide bottleneck. Keep it that
 * way: the only way to add work under this lock is to edit process() below.
 *
 * Every virtual of IPostprocessor<ResultT> forwards to the wrapped object;
 * only process() takes the lock (getModelName() is const and reads nothing
 * process() writes).
 */
#ifndef DXAPP_SERIALIZED_POSTPROCESSOR_HPP
#define DXAPP_SERIALIZED_POSTPROCESSOR_HPP

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/base/i_processor.hpp"

namespace dxapp {

template <typename ResultT>
class SerializedPostprocessor : public IPostprocessor<ResultT> {
public:
    explicit SerializedPostprocessor(std::unique_ptr<IPostprocessor<ResultT>> inner)
        : inner_(std::move(inner)) {
        if (!inner_) {
            throw std::invalid_argument("SerializedPostprocessor: null postprocessor");
        }
    }

    SerializedPostprocessor(const SerializedPostprocessor&) = delete;
    SerializedPostprocessor& operator=(const SerializedPostprocessor&) = delete;

    std::vector<ResultT> process(const dxrt::TensorPtrs& outputs,
                                 const PreprocessContext& ctx) override {
        std::lock_guard<std::mutex> lock(mutex_);
        return inner_->process(outputs, ctx);
    }

    std::string getModelName() const override { return inner_->getModelName(); }

private:
    std::unique_ptr<IPostprocessor<ResultT>> inner_;
    std::mutex mutex_;
};

/// Wraps `inner` so it can be captured by every completion callback of an
/// async runner. Throws std::invalid_argument on a null `inner`.
template <typename ResultT>
std::shared_ptr<IPostprocessor<ResultT>> Serialize(
    std::unique_ptr<IPostprocessor<ResultT>> inner) {
    return std::make_shared<SerializedPostprocessor<ResultT>>(std::move(inner));
}

}  // namespace dxapp

#endif  // DXAPP_SERIALIZED_POSTPROCESSOR_HPP
