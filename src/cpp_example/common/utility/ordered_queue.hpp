/**
 * @file ordered_queue.hpp
 * @brief Hand-over queue that delivers items in submit (ticket) order even
 *        though producers hand them over in any order.
 *
 * WHY. An async pipeline submits frames from one thread but gets them back
 * on dxrt's pool of completion threads (4 on x86), so frame N+1 can finish
 * before frame N, while display, save, DXAPP_VERIFY and any stateful step
 * must see frames in input order. This queue solves that with ticketed,
 * blocking hand-over. No runner uses it at present: the async runners
 * reorder with FrameReorderBuffer (frame_reorder.hpp) instead. It is kept
 * as a tested utility (common_unit_test) and is one of the headers
 * scripts/check_cxx14.sh compiles standalone.
 *
 * CONTRACT.
 *   - issueTicket() is called once per submitted frame, in submit order
 *     (tickets are 0, 1, 2, ...).
 *   - Each ticket is handed over exactly once, by push() or skip().
 *   - One consumer calls try_pop(); it gets ticket N only after N-1.
 *   - push() blocks while its ticket is `capacity` or more ahead of the next
 *     ticket to deliver (the back-pressure a bounded FIFO gives), so at most
 *     `capacity` items are ever buffered.
 *   - If the next ticket is missing for `gap_timeout` while later frames or
 *     blocked producers wait behind it, it is given up with one warning, so a
 *     frame that never comes back cannot stall the stream or deadlock the
 *     producers.
 *   - After shutdown(), push() drops its item and returns false; try_pop()
 *     still returns the buffered items that continue the ticket sequence,
 *     then drops the rest. So a runner's output is always an unbroken prefix
 *     of its input (apart from a frame given up mid-stream, which warns), and
 *     DXAPP_VERIFY's per-frame index equals the ticket.
 */
#ifndef DXAPP_ORDERED_QUEUE_HPP
#define DXAPP_ORDERED_QUEUE_HPP

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <utility>

namespace dxapp {

template <typename T>
class OrderedQueue {
public:
    explicit OrderedQueue(std::size_t capacity = 100,
                          std::chrono::milliseconds gap_timeout = std::chrono::milliseconds(10000))
        : capacity_(capacity == 0 ? 1 : capacity), gap_timeout_(gap_timeout) {}

    OrderedQueue(const OrderedQueue&) = delete;
    OrderedQueue& operator=(const OrderedQueue&) = delete;

    /** Next ticket, in submit order. Call once per submitted frame. */
    std::uint64_t issueTicket() {
        std::lock_guard<std::mutex> lock(mutex_);
        return issued_++;
    }

    /** Hand over the item for `ticket`. Blocks for back-pressure (see file
     *  comment). Returns false, dropping the item, after shutdown() or when
     *  the ticket was already given up. */
    bool push(std::uint64_t ticket, T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto admitted = [&] { return stopped_ || ticket < next_ + capacity_; };
        if (!admitted()) {
            ++waiting_producers_;
            consumer_cv_.notify_all();  // the consumer times the gap it waits behind
            producer_cv_.wait(lock, admitted);
            --waiting_producers_;
        }
        if (stopped_ || ticket < next_ || skipped_.count(ticket) != 0) return false;
        buffer_.emplace(ticket, std::move(item));
        consumer_cv_.notify_all();
        return true;
    }

    /** `ticket` will never be pushed (e.g. its submission failed). */
    void skip(std::uint64_t ticket) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ticket >= next_) skipped_.insert(ticket);
        consumer_cv_.notify_all();
    }

    /** The item for the next ticket, waiting up to `timeout`. */
    bool try_pop(T& out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (;;) {
            advancePastSkipped();
            if (!buffer_.empty() && buffer_.begin()->first == next_) {
                auto it = buffer_.begin();
                out = std::move(it->second);
                next_ = it->first + 1;
                buffer_.erase(it);
                gap_started_ = false;
                producer_cv_.notify_all();
                return true;
            }
            if (stopped_) {
                // Shut down: what continues the sequence was delivered above;
                // frames behind a missing one are dropped, so what a runner
                // delivers is always an unbroken prefix of what it submitted.
                buffer_.clear();
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            auto wake = deadline;
            if (!buffer_.empty() || waiting_producers_ > 0) {
                // Frames wait behind a missing next_: time the gap.
                if (!gap_started_) {
                    gap_started_ = true;
                    gap_since_ = now;
                } else if (now - gap_since_ >= gap_timeout_) {
                    std::cerr << "[DXAPP] [WARN] frame " << next_ << " did not come back within "
                              << gap_timeout_.count() << " ms; continuing without it" << std::endl;
                    ++next_;
                    gap_started_ = false;
                    producer_cv_.notify_all();
                    continue;
                }
                const auto gap_deadline = gap_since_ + gap_timeout_;
                if (gap_deadline < wake) wake = gap_deadline;
            } else {
                gap_started_ = false;
            }
            if (now >= deadline) return false;
            consumer_cv_.wait_until(lock, wake);
        }
    }

    /** True when no handed-over item is waiting to be delivered. */
    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty();
    }

    /** Wake everyone; see the file comment for what changes afterwards. */
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        producer_cv_.notify_all();
        consumer_cv_.notify_all();
    }

private:
    void advancePastSkipped() {
        while (!skipped_.empty() && *skipped_.begin() <= next_) {
            if (*skipped_.begin() == next_) {
                ++next_;
                gap_started_ = false;
                producer_cv_.notify_all();
            }
            skipped_.erase(skipped_.begin());
        }
    }

    const std::size_t capacity_;
    const std::chrono::milliseconds gap_timeout_;
    std::mutex mutex_;
    std::condition_variable producer_cv_;
    std::condition_variable consumer_cv_;
    std::map<std::uint64_t, T> buffer_;
    std::set<std::uint64_t> skipped_;
    std::uint64_t issued_ = 0;
    std::uint64_t next_ = 0;
    int waiting_producers_ = 0;
    bool stopped_ = false;
    bool gap_started_ = false;
    std::chrono::steady_clock::time_point gap_since_;
};

}  // namespace dxapp

#endif  // DXAPP_ORDERED_QUEUE_HPP
