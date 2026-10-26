// Contract tests for dxapp::Progress (src/cpp_example/common/utility/progress.hpp).
// Build: g++ -std=c++17 -I src/cpp_example/common/utility tests/progress/test_progress.cpp -lpthread
// (tests/python_example/unit/test_progress_contract.py builds and runs it)
//
// A front end that shows a running example (DX AI Studio's live view) could not tell
// how many frames had been processed: most runners print nothing per frame, and
// [DET]/[OBB]/[ISEG] print one line per OBJECT. With DXAPP_PROGRESS=1 every runner
// now prints "[PROGRESS] frames=N ..." at most once per interval and once at the end,
// with the rates and stage latencies the runner measures itself.
#include "progress.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_failures; } \
        else         { std::cout << "  ok  : " << (msg) << "\n"; }              \
    } while (0)

template <class F>
static std::string capture(F fn) {
    std::ostringstream buf;
    auto* old = std::cout.rdbuf(buf.rdbuf());
    fn();
    std::cout.rdbuf(old);
    return buf.str();
}

static int count(const std::string& s, const std::string& what) {
    int n = 0;
    for (size_t p = s.find(what); p != std::string::npos; p = s.find(what, p + 1)) ++n;
    return n;
}

static std::vector<std::string> lines(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

static bool has(const std::string& line, const std::string& field) {
    return (" " + line + " ").find(" " + field + " ") != std::string::npos;
}

// A clock the test moves by hand (nanoseconds).
static int64_t g_now = 0;
static int64_t fakeClock() { return g_now; }
static const int64_t kSec = 1000000000LL;

// The runners' metric structs share these field names (SyncProfilingMetrics, AsyncProfilingMetrics).
struct FakeMetrics {
    double sum_read = 0, sum_preprocess = 0, sum_inference = 0, sum_postprocess = 0;
};

int main() {
    std::cout << "[test] off unless DXAPP_PROGRESS is set\n";
    unsetenv("DXAPP_PROGRESS");
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            for (int i = 0; i < 10; ++i) p.tick();
            p.finish();
        });
        CHECK(out.empty(), "nothing printed without the env var");
    }

    std::cout << "[test] every tick with interval 0, final line on finish\n";
    setenv("DXAPP_PROGRESS", "1", 1);
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            for (int i = 0; i < 3; ++i) p.tick();
            p.finish();
        });
        CHECK(count(out, "[PROGRESS] frames=") == 4, "three ticks + finish -> four lines");
        CHECK(out.find("[PROGRESS] frames=3 ") != std::string::npos ||
              out.find("[PROGRESS] frames=3\n") != std::string::npos, "the count is the number of frames");
    }

    std::cout << "[test] rate limited\n";
    {
        auto out = capture([] {
            dxapp::Progress p(3600.0);
            for (int i = 0; i < 1000; ++i) p.tick();
            p.finish();
        });
        CHECK(count(out, "[PROGRESS]") <= 2, "a long interval prints at most the first tick and finish");
        CHECK(out.find("frames=1000") != std::string::npos, "finish reports the total");
    }

    std::cout << "[test] fps is the live rate, avg_fps excludes everything before the first frame\n";
    {
        auto out = capture([] {
            g_now = 10 * kSec;  // model load etc. happened before this
            dxapp::Progress p(1.0, fakeClock);
            p.tick();                                   // frame 1 at t=10.0
            for (int i = 1; i <= 10; ++i) {             // frames 2..11, 10 per second
                g_now = 10 * kSec + i * kSec / 10;
                p.tick();
            }
            for (int i = 1; i <= 20; ++i) {             // frames 12..31, 20 per second
                g_now = 11 * kSec + i * kSec / 20;
                p.tick();
            }
            p.finish();
        });
        auto ls = lines(out);
        CHECK(ls.size() == 4, "first frame, t=11, t=12, finish -> four lines");
        if (ls.size() == 4) {
            CHECK(ls[0] == "[PROGRESS] frames=1", "the first line has nothing to measure yet");
            CHECK(has(ls[1], "frames=11") && has(ls[1], "fps=10.0") && has(ls[1], "avg_fps=10.0"),
                  "after one second at 10/s: fps=10.0 avg_fps=10.0 (" + ls[1] + ")");
            CHECK(has(ls[2], "frames=31") && has(ls[2], "fps=20.0") && has(ls[2], "avg_fps=15.0"),
                  "then 20/s: fps=20.0, avg_fps=30 frames/2 s=15.0 (" + ls[2] + ")");
            CHECK(has(ls[3], "frames=31") && has(ls[3], "fps=15.0") && has(ls[3], "avg_fps=15.0"),
                  "the final line carries the run average (" + ls[3] + ")");
            CHECK(ls[1].find("_ms=") == std::string::npos, "no stage fields without metrics");
        }
    }

    std::cout << "[test] stage latency: the window average live, the run average at the end\n";
    {
        auto out = capture([] {
            g_now = 5 * kSec;
            dxapp::Progress p(0.0, fakeClock);
            FakeMetrics m;
            for (int i = 1; i <= 4; ++i) {
                m.sum_read += 1.0;
                m.sum_preprocess += 2.0;
                m.sum_inference += (i <= 2 ? 10.0 : 30.0);   // frames 3, 4 are slower
                m.sum_postprocess += 3.0;
                g_now += kSec / 10;
                p.tick(dxapp::StageSums{m.sum_read, m.sum_preprocess, m.sum_inference, m.sum_postprocess});
            }
            p.finish();
        });
        auto ls = lines(out);
        CHECK(ls.size() == 5, "four ticks + finish");
        if (ls.size() == 5) {
            CHECK(ls[0].find("_ms=") == std::string::npos, "the first line has no window yet (" + ls[0] + ")");
            CHECK(has(ls[1], "read_ms=1.00") && has(ls[1], "pre_ms=2.00") &&
                  has(ls[1], "infer_ms=10.00") && has(ls[1], "post_ms=3.00"),
                  "frame 2's window: 1/2/10/3 ms (" + ls[1] + ")");
            CHECK(has(ls[3], "infer_ms=30.00"), "frame 4's window is the slow one (" + ls[3] + ")");
            CHECK(has(ls[4], "frames=4") && has(ls[4], "infer_ms=20.00") && has(ls[4], "read_ms=1.00"),
                  "the final line: run averages, (10+10+30+30)/4 = 20 (" + ls[4] + ")");
        }
    }

    std::cout << "[test] sums that restart (new metrics for a new loop) keep adding up\n";
    {
        auto out = capture([] {
            g_now = 5 * kSec;
            dxapp::Progress p(0.0, fakeClock);
            const double infer[] = {10.0, 20.0, 10.0, 20.0};  // the second loop starts from zero
            for (double v : infer) {
                g_now += kSec / 10;
                p.tick(dxapp::StageSums{0.0, 0.0, v, 0.0});
            }
            p.finish();
        });
        auto ls = lines(out);
        CHECK(ls.size() == 5, "four ticks + finish");
        if (ls.size() == 5) {
            CHECK(has(ls[2], "infer_ms=10.00"), "the restart is not a negative window (" + ls[2] + ")");
            CHECK(has(ls[4], "infer_ms=10.00"), "the run average covers both loops (" + ls[4] + ")");
        }
    }

    std::cout << "[test] progressTick(metrics) reads the runners' metric fields\n";
    {
        FakeMetrics m;
        m.sum_inference = 7.0;
        auto out = capture([&m] { dxapp::progressTick(m); });
        CHECK(out.rfind("[PROGRESS] frames=1", 0) == 0, "the process-wide counter takes a metrics struct");
    }

    std::cout << "[test] concurrent ticks are all counted, lines never torn\n";
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            std::vector<std::thread> ts;
            for (int t = 0; t < 8; ++t)
                ts.emplace_back([&p] {
                    for (int i = 0; i < 500; ++i) p.tick(dxapp::StageSums{1.0 * i, 2.0 * i, 3.0 * i, 4.0 * i});
                });
            for (auto& t : ts) t.join();
            p.finish();
        });
        CHECK(out.find("frames=4000") != std::string::npos, "8 threads x 500 ticks = 4000");
        bool torn = false;
        for (const auto& line : lines(out))
            if (line.rfind("[PROGRESS] frames=", 0) != 0 || count(line, "[PROGRESS]") != 1) torn = true;
        CHECK(!torn, "every line is a whole [PROGRESS] line");
    }

    std::cout << "[test] node_ms: the window average live, the run average at the end\n";
    {
        auto out = capture([] {
            g_now = 5 * kSec;
            dxapp::Progress p(0.0, fakeClock);
            const double od[] = {10.0, 20.0};
            const double seg[] = {4.0, 6.0};
            for (int i = 0; i < 2; ++i) {
                g_now += kSec / 10;
                p.tickNodes(dxapp::NodeMs{{"od", od[i]}, {"seg", seg[i]}});
            }
            p.finish();
        });
        auto ls = lines(out);
        CHECK(ls.size() == 3, "two ticks + finish");
        if (ls.size() == 3) {
            CHECK(ls[0] == "[PROGRESS] frames=1", "the first line has no window yet (" + ls[0] + ")");
            CHECK(has(ls[1], "node_ms=od:20.00,seg:6.00"), "frame 2's window (" + ls[1] + ")");
            CHECK(has(ls[2], "frames=2") && has(ls[2], "node_ms=od:15.00,seg:5.00"),
                  "the final line: run averages (" + ls[2] + ")");
            CHECK(ls[1].find("read_ms=") == std::string::npos, "no runner stage fields on a node tick");
        }
    }

    std::cout << "[test] node_ms: a node is averaged over the frames that ran it\n";
    {
        auto out = capture([] {
            g_now = 5 * kSec;
            dxapp::Progress p(0.0, fakeClock);
            g_now += kSec / 10;
            p.tickNodes(dxapp::NodeMs{{"a", 10.0}, {"b", 2.0}});
            g_now += kSec / 10;
            p.tickNodes(dxapp::NodeMs{{"a", 10.0}});
            g_now += kSec / 10;
            p.tickNodes(dxapp::NodeMs{{"a", 10.0}, {"b", 4.0}});
            p.finish();
        });
        auto ls = lines(out);
        CHECK(ls.size() == 4, "three ticks + finish");
        if (ls.size() == 4) {
            CHECK(has(ls[1], "node_ms=a:10.00"), "b did not run in frame 2: left out (" + ls[1] + ")");
            CHECK(has(ls[2], "node_ms=a:10.00,b:4.00"), "frame 3 (" + ls[2] + ")");
            CHECK(has(ls[3], "node_ms=a:10.00,b:3.00"), "b over its own two frames (" + ls[3] + ")");
        }
    }

    std::cout << "[test] node_ms: ids that would break the field are percent-encoded\n";
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            p.tickNodes(dxapp::NodeMs{{"cam 1:x,y", 5.0}});
            p.tickNodes(dxapp::NodeMs{{"cam 1:x,y", 5.0}});
            p.finish();
        });
        CHECK(out.find(" node_ms=cam%201%3Ax%2Cy:5.00") != std::string::npos,
              "space, colon and comma are %20 %3A %2C (" + out + ")");
    }

    std::cout << "[test] node_ms: concurrent node ticks are all counted\n";
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            std::vector<std::thread> ts;
            for (int t = 0; t < 8; ++t)
                ts.emplace_back([&p] {
                    for (int i = 0; i < 500; ++i) p.tickNodes(dxapp::NodeMs{{"x", 1.0}});
                });
            for (auto& t : ts) t.join();
            p.finish();
        });
        auto ls = lines(out);
        CHECK(!ls.empty() && has(ls.back(), "frames=4000") && has(ls.back(), "node_ms=x:1.00"),
              "8 threads x 500 node ticks, run average 1.00");
    }

    std::cout << "[test] start(): the rates count from it, so a burst of reports is not a burst rate\n";
    {
        // A graph whose input is shorter than its in-flight cap hands every report
        // back in one burst; frames after the first over the burst's span read 114k fps.
        auto out = capture([] {
            g_now = 5 * kSec;
            dxapp::Progress p(1.0, fakeClock);
            p.start();
            g_now += kSec;
            for (int i = 0; i < 8; ++i) {
                p.tickNodes(dxapp::NodeMs{{"od", 1.0}});
                g_now += 100;  // all 8 within 1 us
            }
            p.finish();
        });
        auto ls = lines(out);
        CHECK(!ls.empty() && has(ls.back(), "frames=8") && has(ls.back(), "avg_fps=8.0") && has(ls.back(), "fps=8.0"),
              "8 frames 1 s after start(): fps = avg_fps = 8.0 (" + (ls.empty() ? std::string() : ls.back()) + ")");
    }
    unsetenv("DXAPP_PROGRESS");

    if (g_failures) { std::cerr << "\n" << g_failures << " CHECK(s) FAILED\n"; return 1; }
    std::cout << "\nAll Progress contract tests PASSED\n";
    return 0;
}
