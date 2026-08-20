/**
 * @file frame_reorder.hpp
 * @brief Emit async display items in submit order
 */

#ifndef DXAPP_FRAME_REORDER_HPP
#define DXAPP_FRAME_REORDER_HPP

#include <cstdint>
#include <map>
#include <utility>

namespace dxapp {

/**
 * @brief Reorders async display items back into submission order.
 *
 * The async runners push each completed frame into their display queue from
 * inside the dxrt completion callback, and the display thread renders / writes
 * (VideoWriter) in pop order. RunAsync completions do NOT preserve submission
 * order — postprocess runs in the callback thread, so a heavier frame can
 * finish after a lighter successor — so without this buffer a saved video comes
 * out locally shuffled inside the in-flight window, which reads as jitter on
 * playback. Frame *count* stays correct, so it is a silent defect.
 *
 * Usage: assign a monotonic index at submit time (``AsyncUserData::frame_index``,
 * carried through to the display args), then feed every popped item through
 * push() and call drain() once the queue is finished.
 *
 * @tparam T display-args type with a ``uint64_t frame_index`` member.
 */
template <typename T>
class FrameReorderBuffer {
public:
    /**
     * @param cap Force-flush threshold. If an index never arrives (a dropped
     *            callback, or a shutdown-timing edge), the lowest buffered item
     *            is emitted once more than `cap` items are held, so the display
     *            can never stall permanently. In steady state the buffer holds
     *            fewer items than the in-flight depth, so the cap is not hit.
     */
    explicit FrameReorderBuffer(size_t cap) : cap_(cap) {}

    /**
     * @brief Buffer one item, then emit everything that is now in order.
     * @param emit Callable invoked as ``emit(T&)`` per emitted item.
     */
    template <typename EmitFn>
    void push(T&& item, EmitFn&& emit) {
        const uint64_t index = item.frame_index;
        buffer_.emplace(index, std::move(item));
        while (!buffer_.empty()) {
            auto it = buffer_.begin();
            // next_ starts at 0 — the first index the producers assign. Seeding
            // it from the first *arriving* item instead would emit that item
            // immediately and then stall on its lower-numbered predecessors
            // until the cap force-flushed them.
            if (it->first != next_ && buffer_.size() <= cap_) break;
            emit(it->second);
            next_ = it->first + 1;
            buffer_.erase(it);
        }
    }

    /** @brief Emit every item still buffered, in submit order. */
    template <typename EmitFn>
    void drain(EmitFn&& emit) {
        for (auto& kv : buffer_) emit(kv.second);
        buffer_.clear();
    }

private:
    std::map<uint64_t, T> buffer_;
    uint64_t next_ = 0;
    size_t cap_;
};

}  // namespace dxapp

#endif  // DXAPP_FRAME_REORDER_HPP
