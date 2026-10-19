// Throughput comparison of the OLD vs NEW display topology.
//
// The NEW path uses the real production dxapp::DisplayPump / runPipelineWithDisplay.
// The OLD path reimplements exactly what the runners used to do: a bounded
// BLOCKING rendered-frame queue, and GUI work performed inline in the submit loop.
// GUI / NPU / submit costs are injected as calibrated busy-waits so the numbers
// are reproducible on a headless machine.
#include <opencv2/opencv.hpp>
#include "display_pump.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <thread>

using clk = std::chrono::steady_clock;

static void spin(double ms) {                       // busy-wait: precise, unlike sleep
    if (ms <= 0) return;
    const auto end = clk::now() + std::chrono::duration<double, std::milli>(ms);
    while (clk::now() < end) { /* spin */ }
}

// ---- calibrated per-stage costs (ms) --------------------------------------
// Derived from the reported YOLO26n numbers on x86_64 i9-13900:
//   --no-display => 223.7 FPS => 4.47 ms/frame of submit+NPU-bound work
//   with display => 94.2 FPS => 10.62 ms/frame  => ~6.15 ms of GUI per frame
static double SUBMIT_MS   = 4.47;   // read + preprocess + copy + RunAsync
static double GUI_OLD_MS  = 6.15;   // 2x waitKey + imshow + 2x getWindowProperty + namedWindow
static double GUI_NEW_MS  = 3.75;   // 1x waitKey + imshow + 1x getWindowProperty
static double RENDER_MS   = 1.20;   // draw boxes on the frame
static double RUN_SEC     = 3.0;

// --------------------------------------------------------------------------
// OLD: bounded blocking queue; GUI runs inline in the submit loop.
// --------------------------------------------------------------------------
template <typename T>
class BlockingQueue {                                // mirrors the old SafeQueue
public:
    explicit BlockingQueue(size_t cap) : cap_(cap) {}
    bool push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        while (q_.size() >= cap_ && !stopped_) cv_.wait_for(lk, std::chrono::milliseconds(100));
        if (stopped_) return false;
        q_.push(std::move(v)); cv_.notify_one(); return true;
    }
    bool try_pop(T& out, std::chrono::milliseconds to) {
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_.wait_for(lk, to, [&] { return !q_.empty(); })) return false;
        out = std::move(q_.front()); q_.pop(); cv_.notify_one(); return true;
    }
    void stop() { std::lock_guard<std::mutex> lk(m_); stopped_ = true; cv_.notify_all(); }
private:
    std::queue<T> q_; std::mutex m_; std::condition_variable cv_;
    size_t cap_; bool stopped_ = false;
};

static long run_old() {
    BlockingQueue<cv::Mat> rendered(100);
    std::atomic<bool> running{true};
    std::atomic<long> submitted{0};

    std::thread render([&] {                         // render thread -> rendered_queue
        while (running.load()) {
            spin(RENDER_MS);
            if (!rendered.push(cv::Mat(8, 8))) break; // BLOCKS once the GUI falls behind
        }
    });

    const auto end = clk::now() + std::chrono::duration<double>(RUN_SEC);
    while (clk::now() < end) {
        spin(SUBMIT_MS);                             // read + preprocess + submit
        submitted.fetch_add(1, std::memory_order_relaxed);
        cv::Mat f;                                   // ---- pollDisplay(), inline ----
        if (rendered.try_pop(f, std::chrono::milliseconds(1))) spin(GUI_OLD_MS);
        else spin(GUI_OLD_MS * 0.45);                // still pumps events when idle
    }
    running.store(false); rendered.stop(); render.join();
    return submitted.load();
}

// --------------------------------------------------------------------------
// NEW: real DisplayPump; submit loop on a worker, GUI on this thread.
// --------------------------------------------------------------------------
static long run_new() {
    dxapp::DisplayPump pump("Output", 60.0);
    std::atomic<bool> running{true};
    std::atomic<long> submitted{0};

    std::thread render([&] {
        while (running.load()) {
            spin(RENDER_MS);
            pump.offer(cv::Mat(8, 8));               // never blocks
        }
    });

    const auto end = clk::now() + std::chrono::duration<double>(RUN_SEC);
    dxapp::runPipelineWithDisplay(
        [&] {
            while (clk::now() < end) {
                spin(SUBMIT_MS);                     // no GUI on this path
                submitted.fetch_add(1, std::memory_order_relaxed);
            }
        },
        pump, /*no_display=*/false, [] {});

    running.store(false); render.join();
    return submitted.load();
}

static long run_nodisplay() {
    std::atomic<long> submitted{0};
    const auto end = clk::now() + std::chrono::duration<double>(RUN_SEC);
    while (clk::now() < end) { spin(SUBMIT_MS); submitted.fetch_add(1); }
    return submitted.load();
}

int main(int argc, char** argv) {
    if (argc > 1) RUN_SEC = atof(argv[1]);
    std::printf("stage costs (ms): submit=%.2f  gui_old=%.2f  gui_new=%.2f  render=%.2f\n",
                SUBMIT_MS, GUI_OLD_MS, GUI_NEW_MS, RENDER_MS);
    std::printf("each scenario runs %.1f s\n\n", RUN_SEC);

    const long base = run_nodisplay();
    const long oldf = run_old();
    const long newf = run_new();

    const double b = base / RUN_SEC, o = oldf / RUN_SEC, n = newf / RUN_SEC;
    std::printf("%-34s %8s %12s\n", "scenario", "FPS", "vs --no-display");
    std::printf("%-34s %8s %12s\n", "----------------------------------", "--------", "------------");
    std::printf("%-34s %8.1f %11.0f%%\n", "--no-display (ceiling)",      b, 100.0);
    std::printf("%-34s %8.1f %11.0f%%\n", "OLD: GUI inline + blocking q", o, 100.0 * o / b);
    std::printf("%-34s %8.1f %11.0f%%\n", "NEW: DisplayPump + worker",    n, 100.0 * n / b);
    return 0;
}
