// Contract tests for dxapp::asyncRunEnd (src/cpp_example/common/utility/run_clock.hpp).
// Build: g++ -std=c++17 -I src/cpp_example/common/utility tests/progress/test_run_clock.cpp -lpthread
// (tests/python_example/unit/test_progress_contract.py builds and runs it)
//
// The async runners took Total Time after ie.Wait(), the display drain, thread shutdown and (for
// images) the viewer, so a fixed ~0.2 s that is not processing was in Overall FPS: 30 images at
// 405 FPS read 107. The run now ends at the last completed frame.
#include "run_clock.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

static int g_failures = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_failures; } \
        else         { std::cout << "  ok  : " << (msg) << "\n"; }              \
    } while (0)

// The fields the runners' AsyncProfilingMetrics has.
struct FakeAsyncMetrics {
    int infer_completed = 0;
    std::chrono::high_resolution_clock::time_point infer_last_ts;
    std::mutex metrics_mutex;
};

int main() {
    using clk = std::chrono::high_resolution_clock;

    std::cout << "[test] the run ends at the last completed frame, not at teardown\n";
    {
        FakeAsyncMetrics m;
        m.infer_completed = 30;
        m.infer_last_ts = clk::now() - std::chrono::milliseconds(200);  // teardown took 200 ms since
        const auto end = dxapp::asyncRunEnd(m);
        CHECK(end == m.infer_last_ts, "end == infer_last_ts");
    }

    std::cout << "[test] no frame completed: the run ends now\n";
    {
        FakeAsyncMetrics m;
        const auto before = clk::now();
        const auto end = dxapp::asyncRunEnd(m);
        CHECK(end >= before, "end is the current time");
    }

    std::cout << "[test] it takes the metrics lock\n";
    {
        FakeAsyncMetrics m;
        m.infer_completed = 1;
        m.infer_last_ts = clk::now();
        std::unique_lock<std::mutex> held(m.metrics_mutex);
        bool returned = false;
        std::thread t([&] { dxapp::asyncRunEnd(m); returned = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(!returned, "waits while a callback holds metrics_mutex");
        held.unlock();
        t.join();
        CHECK(returned, "returns once the lock is free");
    }

    if (g_failures) { std::cerr << "\n" << g_failures << " CHECK(s) FAILED\n"; return 1; }
    std::cout << "\nAll run clock contract tests PASSED\n";
    return 0;
}
