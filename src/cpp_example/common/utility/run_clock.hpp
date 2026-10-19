/**
 * @file run_clock.hpp
 * @brief Where an async run's Total Time ends.
 *
 * The async runners took their end time after ie.Wait(), the display-queue drain, thread
 * shutdown and (for images) the frame viewer. That added a fixed ~0.2 s that is not processing
 * to Total Time, so Overall FPS read low on short runs (30 images at 405 FPS read 107). The run
 * now ends at the last completed frame, so Overall FPS agrees with the inference throughput and
 * with [PROGRESS] avg_fps.
 *
 * Contract: tests/progress/test_run_clock.cpp
 */
#ifndef DXAPP_RUN_CLOCK_HPP
#define DXAPP_RUN_CLOCK_HPP

#include <chrono>
#include <mutex>

namespace dxapp {

/** End of an async run: the last completed frame (AsyncProfilingMetrics::infer_last_ts), or now
 *  when no frame completed. Takes the metrics lock the callbacks write under. */
template <class Metrics>
inline std::chrono::high_resolution_clock::time_point asyncRunEnd(Metrics& m) {
    std::lock_guard<std::mutex> lock(m.metrics_mutex);
    return m.infer_completed > 0 ? m.infer_last_ts : std::chrono::high_resolution_clock::now();
}

}  // namespace dxapp

#endif  // DXAPP_RUN_CLOCK_HPP
