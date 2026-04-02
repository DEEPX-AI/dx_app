/**
 * @file profiling.hpp
 * @brief Profiling metrics and performance measurement utilities
 * 
 * Part of DX-APP v3.0.0 refactoring for independent build capability.
 */

#ifndef DXAPP_PROFILING_HPP
#define DXAPP_PROFILING_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>

namespace dxapp {

/**
 * @brief Profiling metrics structure for synchronous pipeline
 * 
 * Tracks timing metrics for each stage of the inference pipeline.
 */
struct ProfilingMetrics {
    double sum_read{0.0};
    double sum_preprocess{0.0};
    double sum_inference{0.0};
    double sum_postprocess{0.0};
    double sum_render{0.0};
    int frame_count{0};

    void reset() {
        sum_read = 0.0;
        sum_preprocess = 0.0;
        sum_inference = 0.0;
        sum_postprocess = 0.0;
        sum_render = 0.0;
        frame_count = 0;
    }

    void addRead(double ms) { sum_read += ms; }
    void addPreprocess(double ms) { sum_preprocess += ms; }
    void addInference(double ms) { sum_inference += ms; }
    void addPostprocess(double ms) { sum_postprocess += ms; }
    void addRender(double ms) { sum_render += ms; }
    void incrementFrame() { frame_count++; }
};

/**
 * @brief Profiling metrics for asynchronous pipeline
 * 
 * Thread-safe metrics with inflight tracking for async operations.
 */
struct AsyncProfilingMetrics {
    double sum_read{0.0};
    double sum_preprocess{0.0};
    double sum_inference{0.0};
    double sum_postprocess{0.0};
    double sum_render{0.0};
    int infer_completed{0};

    // Inflight tracking
    std::chrono::high_resolution_clock::time_point infer_first_ts;
    std::chrono::high_resolution_clock::time_point infer_last_ts;
    std::chrono::high_resolution_clock::time_point inflight_last_ts;
    int inflight_current{0};
    int inflight_max{0};
    double inflight_time_sum{0.0};
    bool first_inference{true};

    std::mutex metrics_mutex;

    void reset() {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        sum_read = 0.0;
        sum_preprocess = 0.0;
        sum_inference = 0.0;
        sum_postprocess = 0.0;
        sum_render = 0.0;
        infer_completed = 0;
        inflight_current = 0;
        inflight_max = 0;
        inflight_time_sum = 0.0;
        first_inference = true;
    }
};

/**
 * @brief RAII-style timer for automatic timing measurement
 */
class ScopedTimer {
public:
    ScopedTimer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsedSec() const {
        return elapsedMs() / 1000.0;
    }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Print performance summary for synchronous pipeline
 * @param metrics Profiling metrics
 * @param total_frames Total number of frames processed
 * @param total_time_sec Total elapsed time in seconds
 * @param display_on Whether display metrics should be shown
 */
inline void printPerformanceSummary(const ProfilingMetrics& metrics,
                                    int total_frames,
                                    double total_time_sec,
                                    bool display_on = true) {
    if (total_frames == 0) return;

    double avg_read = metrics.sum_read / total_frames;
    double avg_pre = metrics.sum_preprocess / total_frames;
    double avg_inf = metrics.sum_inference / total_frames;
    double avg_post = metrics.sum_postprocess / total_frames;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
    double inf_fps = avg_inf > 0 ? 1000.0 / avg_inf : 0.0;
    double post_fps = avg_post > 0 ? 1000.0 / avg_post : 0.0;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Read" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_read << " ms     " << std::setw(6)
              << std::setprecision(1) << read_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Preprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_pre << " ms     " << std::setw(6)
              << std::setprecision(1) << pre_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Inference" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_inf << " ms     " << std::setw(6)
              << std::setprecision(1) << inf_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Postprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_post << " ms     " << std::setw(6)
              << std::setprecision(1) << post_fps << " FPS" << std::endl;

    if (display_on) {
        double avg_render = metrics.sum_render / total_frames;
        double render_fps = avg_render > 0 ? 1000.0 / avg_render : 0.0;
        std::cout << " " << std::left << std::setw(15) << "Display" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(2) << avg_render << " ms     " << std::setw(6)
                  << std::setprecision(1) << render_fps << " FPS" << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;

    double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;

    std::cout << " " << std::left << std::setw(19) << "Total Frames"
              << " :    " << total_frames << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Time"
              << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
              << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Overall FPS"
              << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
              << std::endl;
    std::cout << "==================================================" << std::endl;
}

/**
 * @brief Print performance summary for asynchronous pipeline
 */
inline void printAsyncPerformanceSummary(const AsyncProfilingMetrics& metrics,
                                         int total_frames,
                                         double total_time_sec,
                                         bool display_on = true) {
    if (metrics.infer_completed == 0) {
        std::cout << "[WARNING] No frames were processed." << std::endl;
        return;
    }

    double avg_read = metrics.sum_read / metrics.infer_completed;
    double avg_pre = metrics.sum_preprocess / metrics.infer_completed;
    double avg_inf = metrics.sum_inference / metrics.infer_completed;
    double avg_post = metrics.sum_postprocess / metrics.infer_completed;

    auto inflight_time_window = std::chrono::duration<double>(
        metrics.infer_last_ts - metrics.infer_first_ts).count();
    double infer_tp = inflight_time_window > 0 
        ? metrics.infer_completed / inflight_time_window : 0.0;
    double inflight_avg = inflight_time_window > 0 
        ? metrics.inflight_time_sum / inflight_time_window : 0.0;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
    double post_fps = avg_post > 0 ? 1000.0 / avg_post : 0.0;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Read" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_read << " ms     " << std::setw(6)
              << std::setprecision(1) << read_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Preprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_pre << " ms     " << std::setw(6)
              << std::setprecision(1) << pre_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Inference" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_inf << " ms     " << std::setw(6)
              << std::setprecision(1) << infer_tp << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Postprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_post << " ms     " << std::setw(6)
              << std::setprecision(1) << post_fps << " FPS" << std::endl;

    if (display_on) {
        double avg_render = metrics.sum_render / metrics.infer_completed;
        double render_fps = avg_render > 0 ? 1000.0 / avg_render : 0.0;
        std::cout << " " << std::left << std::setw(15) << "Display" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(2) << avg_render << " ms     " << std::setw(6)
                  << std::setprecision(1) << render_fps << " FPS" << std::endl;
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Inflight Avg" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << inflight_avg << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Inflight Max" << std::right << std::setw(8)
              << metrics.inflight_max << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;

    std::cout << " " << std::left << std::setw(19) << "Total Frames"
              << " :    " << total_frames << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Time"
              << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
              << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Overall FPS"
              << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
              << std::endl;
    std::cout << "==================================================" << std::endl;
}

}  // namespace dxapp

#endif  // DXAPP_PROFILING_HPP
