// Contract tests for dxapp::DisplayPump (src/cpp_example/common/utility/display_pump.hpp).
// Compiled against the mock OpenCV in tests/display/mock_include.
#include <opencv2/opencv.hpp>
#include "display_pump.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

static int g_failures = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_failures; } \
        else         { std::cout << "  ok  : " << (msg) << "\n"; }            \
    } while (0)

using dxapp::DisplayPump;

// Drop mode (--drop-frames): producers never block, only the newest frame survives.
static void test_drop_mode_is_lossy_and_latest_wins() {
    std::cout << "[test] drop mode: offer() is lossy, depth 1, latest-wins\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);  // 0 => no rate limit, show every pump
    pump.setDropStale(true);

    cv::Mat a(8, 8), b(8, 8), c(8, 8);
    pump.offer(a);
    pump.offer(b);
    pump.offer(c);
    CHECK(pump.dropped() == 2, "two stale frames dropped before any pump");

    CHECK(pump.pump(), "pump() returns true (no quit)");
    CHECK(pump.shown() == 1, "exactly one frame shown");
    CHECK(cvmock::counters().imshow.load() == 1, "imshow called once");
}

// I2: offer() must not deep-copy pixels (cv::Mat is refcounted).
static void test_offer_does_not_copy_pixels() {
    std::cout << "[test] offer() does not deep-copy pixels (both modes)\n";
    cvmock::counters().reset();
    {
        DisplayPump pump("Output", 0.0);
        pump.setDropStale(true);
        int before = cv::Mat::clone_count();
        cv::Mat f(1080, 1920);
        for (int i = 0; i < 500; ++i) pump.offer(f);
        CHECK(cv::Mat::clone_count() == before, "drop mode: 500 offers caused zero clone()");
    }
    {
        DisplayPump pump("Output", 0.0);   // lossless default: offer+pump in turn
        int before = cv::Mat::clone_count();
        cv::Mat f(1080, 1920);
        for (int i = 0; i < 500; ++i) { pump.offer(f); pump.pump(); }
        CHECK(cv::Mat::clone_count() == before, "lossless: 500 offer/pump pairs caused zero clone()");
    }
}

// I3: exactly one event pump per tick, window created once.
static void test_single_event_pump_per_tick() {
    std::cout << "[test] one waitKey per pump, namedWindow once\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    for (int i = 0; i < 10; ++i) { pump.offer(cv::Mat(8, 8)); pump.pump(); }
    CHECK(cvmock::counters().wait_key.load() == 10, "waitKey called exactly 10x for 10 pumps");
    CHECK(cvmock::counters().named_window.load() == 1, "namedWindow called exactly once");
    CHECK(cvmock::counters().imshow.load() == 10, "imshow called 10x");
}

// Quit handling must survive: 'q' / ESC / window closed.
static void test_quit_detection() {
    std::cout << "[test] quit detection\n";
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.offer(cv::Mat(8, 8)); pump.pump();
        cvmock::counters().next_key = 'q';
        CHECK(!pump.pump(), "'q' stops the pump");
        CHECK(pump.stopped(), "pump reports stopped after 'q'");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.offer(cv::Mat(8, 8)); pump.pump();
        cvmock::counters().next_key = 27;
        CHECK(!pump.pump(), "ESC stops the pump");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.offer(cv::Mat(8, 8)); pump.pump();
        cvmock::counters().window_visible = 0.0;  // user closed the window
        CHECK(!pump.pump(), "window close stops the pump");
    }
}

// A backend that always returns -1 for WND_PROP_VISIBLE (e.g. GTK2) must not
// be mistaken for "user closed the window".
static void test_backend_without_window_property() {
    std::cout << "[test] backend returning -1 for WND_PROP_VISIBLE is tolerated\n";
    cvmock::counters().reset();
    cvmock::counters().window_visible = -1.0;
    DisplayPump pump("Output", 0.0);
    pump.offer(cv::Mat(8, 8));
    CHECK(pump.pump(), "first pump survives probe");
    pump.offer(cv::Mat(8, 8));
    CHECK(pump.pump(), "second pump still alive (probe disabled the check)");
}

// Drop mode: wouldShow() reserves one preview slot so producers do not all draw.
static void test_would_show_reserves_slot() {
    std::cout << "[test] drop mode: wouldShow reserves a single preview slot\n";
    if (!dxapp::hasDisplay()) {
        std::cout << "  ok  : skipped (no display server)\n";
        return;
    }
    DisplayPump pump("Output", 50.0);  // 20 ms between admitted frames
    pump.setDropStale(true);
    CHECK(pump.wouldShow(), "first slot is open");
    CHECK(!pump.wouldShow(), "second call inside the interval is refused");
}

// I3: rate limiting caps imshow frequency without capping the pump loop.
static void test_rate_limit() {
    std::cout << "[test] drop mode: rate limit caps imshow, not the event pump\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 50.0);  // 50 Hz => >= 20 ms between imshow
    pump.setDropStale(true);
    auto t0 = std::chrono::steady_clock::now();
    int pumps = 0;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(100)) {
        pump.offer(cv::Mat(8, 8));
        pump.pump();
        ++pumps;
    }
    int shown = pump.shown();
    CHECK(pumps > shown, "pump loop ran more often than imshow");
    CHECK(shown <= 8, "imshow rate-limited to ~50 Hz over 100 ms (got <=8)");
    CHECK(shown >= 2, "imshow still happened (got >=2)");
}

// I1: the headline property — a slow GUI must not slow the producer down.
static void test_slow_display_does_not_throttle_producer() {
    std::cout << "[test] drop mode: slow display does not throttle the producer\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    pump.setDropStale(true);
    std::atomic<bool> done{false};
    std::atomic<long> produced{0};

    std::thread producer([&] {
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(200)) {
            pump.offer(cv::Mat(64, 64));
            produced.fetch_add(1, std::memory_order_relaxed);
        }
        done.store(true);
    });

    // Consumer deliberately far slower than the producer (10 ms per frame).
    while (!done.load()) {
        pump.pump();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    producer.join();

    long n = produced.load();
    std::cout << "    produced=" << n << " shown=" << pump.shown()
              << " dropped=" << pump.dropped() << "\n";
    CHECK(n > 5000, "producer ran freely (>5000 offers in 200 ms despite a 10 ms display)");
    CHECK(pump.shown() < 40, "display only kept up with its own rate");
    CHECK(pump.dropped() > 0, "stale frames were dropped, not queued");
}

// runPipelineWithDisplay(): the control-flow primitive the runners now rely on.
// The pipeline must run to completion on a worker thread while this thread pumps,
// and a user quit must tear the pipeline down rather than leak the worker.
static void test_run_pipeline_with_display() {
    std::cout << "[test] runPipelineWithDisplay drives pipeline off the GUI thread\n";
    cvmock::counters().reset();
    {
        DisplayPump pump("Output", 0.0);
        std::atomic<long> iters{0};
        std::atomic<bool> finished{false};
        std::thread::id gui_tid = std::this_thread::get_id();
        std::atomic<bool> ran_on_other_thread{false};

        dxapp::runPipelineWithDisplay(
            [&] {
                ran_on_other_thread.store(std::this_thread::get_id() != gui_tid);
                for (int i = 0; i < 2000; ++i) {
                    pump.offer(cv::Mat(32, 32));
                    iters.fetch_add(1, std::memory_order_relaxed);
                }
                finished.store(true);
            },
            pump, /*no_display=*/false, [] {});

        CHECK(ran_on_other_thread.load(), "pipeline ran on a worker thread, not the GUI thread");
        CHECK(finished.load(), "pipeline ran to completion");
        CHECK(iters.load() == 2000, "every iteration executed");
        CHECK(pump.shown() == 2000, "lossless default: GUI thread showed every frame");
        CHECK(pump.dropped() == 0, "lossless default: nothing dropped");
    }
    {   // user quit must stop the pipeline and still join the worker
        DisplayPump pump("Output", 0.0);
        std::atomic<bool> stop_requested{false};
        std::atomic<bool> worker_exited{false};
        cvmock::counters().next_key = 'q';

        dxapp::runPipelineWithDisplay(
            [&] {
                while (!stop_requested.load()) {
                    pump.offer(cv::Mat(32, 32));
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                worker_exited.store(true);
            },
            pump, /*no_display=*/false,
            [&] { stop_requested.store(true); });

        CHECK(stop_requested.load(), "user quit propagated to the pipeline");
        CHECK(worker_exited.load(), "worker was joined, not leaked");
    }
    {   // --no-display: no GUI calls at all
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        std::atomic<bool> done{false};
        dxapp::runPipelineWithDisplay([&] { done.store(true); }, pump,
                                      /*no_display=*/true, [] {});
        CHECK(done.load(), "pipeline ran with no_display");
        CHECK(cvmock::counters().imshow.load() == 0, "no imshow in --no-display");
        CHECK(cvmock::counters().wait_key.load() == 0, "no waitKey in --no-display");
    }
}

// Headless: the pump must go inert WITHOUT reporting a user quit, or
// runPipelineWithDisplay() would tear the inference pipeline down.
static void test_headless_is_inert_not_quit() {
    std::cout << "[test] headless pump is inert, not a quit signal\n";
    cvmock::counters().reset();
    unsetenv("DISPLAY");
    unsetenv("WAYLAND_DISPLAY");
    CHECK(!dxapp::hasDisplay(), "hasDisplay() false with no DISPLAY/WAYLAND_DISPLAY");
    {
        DisplayPump pump("Output", 0.0);
        CHECK(!pump.guiAvailable(), "pump reports no GUI");
        for (int i = 0; i < 20; ++i) { pump.offer(cv::Mat(8, 8)); }
        bool alive = true;
        for (int i = 0; i < 20; ++i) alive = alive && pump.pump();
        CHECK(alive, "pump() keeps returning true (NOT a user quit)");
        CHECK(!pump.stopped(), "pump is not 'stopped'");
        CHECK(cvmock::counters().named_window.load() == 0, "namedWindow never called");
        CHECK(cvmock::counters().imshow.load() == 0, "imshow never called");
        CHECK(cvmock::counters().wait_key.load() == 0, "waitKey never called");
        CHECK(pump.shown() == 0, "nothing shown");
    }
    // A headless pipeline must still run to completion.
    {
        DisplayPump pump("Output", 0.0);
        std::atomic<bool> finished{false};
        dxapp::runPipelineWithDisplay([&] { finished.store(true); }, pump,
                                      /*no_display=*/false, [] {});
        CHECK(finished.load(), "pipeline completed even with display requested but absent");
    }
    setenv("DISPLAY", ":99", 1);   // restore for the GUI tests
    CHECK(dxapp::hasDisplay(), "hasDisplay() true again once DISPLAY is set");
}


// ---------------------------------------------------------------------------
// Lossless default (2026-10-06): every offered frame is shown, in order.
// ---------------------------------------------------------------------------
static void test_lossless_is_default() {
    std::cout << "[test] lossless is the default, drop mode is opt-in\n";
    DisplayPump pump("Output", 0.0);
    CHECK(!pump.dropStale(), "dropStale() is false by default");
    pump.setDropStale(true);
    CHECK(pump.dropStale(), "setDropStale(true) turns drop mode on");
    pump.setDropStale(false);
    CHECK(!pump.dropStale(), "setDropStale(false) turns it back off");
}

static void test_lossless_shows_every_frame_in_order() {
    std::cout << "[test] lossless: a producer faster than the display loses nothing\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    std::atomic<bool> done{false};
    std::thread producer([&] {
        for (int i = 0; i < 300; ++i) pump.offer(cv::Mat(16, 16));
        done.store(true);
    });
    // Consumer slower than the producer: 1 ms per frame.
    while (!done.load() ) {
        pump.pump();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pump.pump();  // the last offered frame may still be pending
    producer.join();
    CHECK(pump.shown() == 300, "all 300 frames were shown (got the exact count)");
    CHECK(pump.dropped() == 0, "no frame was dropped");
    CHECK(cvmock::counters().imshow.load() == 300, "imshow called 300x");
}

static void test_lossless_offer_blocks_until_pumped() {
    std::cout << "[test] lossless: offer() waits for the pending slot\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    pump.offer(cv::Mat(8, 8));                 // fills the single slot
    std::atomic<bool> second_returned{false};
    std::thread t([&] { pump.offer(cv::Mat(8, 8)); second_returned.store(true); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!second_returned.load(), "second offer() is still blocked after 50 ms");
    CHECK(pump.pump(), "pump() shows the first frame");
    t.join();
    CHECK(second_returned.load(), "second offer() returned once the slot was freed");
    CHECK(pump.pump(), "pump() shows the second frame");
    CHECK(pump.shown() == 2, "both frames shown");
    CHECK(pump.dropped() == 0, "nothing dropped");
}

static void test_lossless_would_show_is_always_true() {
    std::cout << "[test] lossless: wouldShow() never refuses a frame\n";
    if (!dxapp::hasDisplay()) {
        std::cout << "  ok  : skipped (no display server)\n";
        return;
    }
    DisplayPump pump("Output", 50.0);   // a rate limit must not apply in lossless mode
    bool all = true;
    for (int i = 0; i < 20; ++i) all = all && pump.wouldShow();
    CHECK(all, "20 consecutive wouldShow() calls all true");
}

static void test_lossless_ignores_rate_limit() {
    std::cout << "[test] lossless: every pump shows its frame regardless of max_fps\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 10.0);   // 10 Hz would allow ~1 frame in 20 ms
    for (int i = 0; i < 10; ++i) { pump.offer(cv::Mat(8, 8)); pump.pump(); }
    CHECK(pump.shown() == 10, "10 frames shown back-to-back (no rate limit)");
}

static void test_stop_unblocks_a_waiting_producer() {
    std::cout << "[test] lossless: stop() releases a producer blocked in offer()\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    pump.offer(cv::Mat(8, 8));
    std::atomic<bool> returned{false};
    std::thread t([&] { pump.offer(cv::Mat(8, 8)); returned.store(true); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(!returned.load(), "producer is blocked before stop()");
    pump.stop();
    auto t0 = std::chrono::steady_clock::now();
    t.join();
    const auto waited = std::chrono::steady_clock::now() - t0;
    CHECK(returned.load(), "producer returned after stop()");
    CHECK(waited < std::chrono::seconds(1), "release was immediate (< 1 s)");
    CHECK(pump.shown() == 0, "no frame shown after stop()");
}

// ---------------------------------------------------------------------------
// Initial window size == Python show_image(): fit into half the screen,
// aspect preserved, int() truncation. Explicit setInitialWindowSize() wins.
// ---------------------------------------------------------------------------
static void test_window_fits_half_screen_on_creation() {
    std::cout << "[test] window is sized to half the screen, aspect preserved\n";
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.setScreenSize(1920, 1080);
        pump.offer(cv::Mat(1080, 1920)); pump.pump();
        CHECK(cvmock::counters().resize_window.load() == 1, "resizeWindow called once on creation");
        CHECK(cvmock::counters().last_resize_w.load() == 960 &&
              cvmock::counters().last_resize_h.load() == 540, "1920x1080 frame -> 960x540 window");
        pump.offer(cv::Mat(1080, 1920)); pump.pump();
        CHECK(cvmock::counters().resize_window.load() == 1, "later frames do not resize again");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.setScreenSize(1920, 1080);
        pump.offer(cv::Mat(576, 768)); pump.pump();
        CHECK(cvmock::counters().last_resize_w.load() == 720 &&
              cvmock::counters().last_resize_h.load() == 540, "768x576 frame -> 720x540 window (min scale, truncated)");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);   // no setScreenSize(): 1920x1080 default, like Python's fallback
        pump.offer(cv::Mat(480, 640)); pump.pump();
        CHECK(cvmock::counters().last_resize_w.load() == 720 &&
              cvmock::counters().last_resize_h.load() == 540, "640x480 frame -> 720x540 (upscaled to the box, like Python)");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.setScreenSize(1920, 1080);
        pump.setInitialWindowSize(400, 300);
        pump.offer(cv::Mat(1080, 1920)); pump.pump();
        CHECK(cvmock::counters().last_resize_w.load() == 400 &&
              cvmock::counters().last_resize_h.load() == 300, "explicit setInitialWindowSize() wins");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.setScreenSize(0, 0);          // unknown screen: fall back to the 1920x1080 default
        pump.offer(cv::Mat(1080, 1920)); pump.pump();
        CHECK(cvmock::counters().last_resize_w.load() == 960 &&
              cvmock::counters().last_resize_h.load() == 540, "non-positive screen size falls back to 1920x1080");
    }
}


// Lossless tail: frames the display thread offers after the pipeline finished
// must still be shown. pumpUntilJoined() keeps pumping until that thread exits.
static void test_pump_until_joined_drains_the_display_thread() {
    std::cout << "[test] pumpUntilJoined shows every frame offered before the thread exits\n";
    cvmock::counters().reset();
    {
        DisplayPump pump("Output", 0.0);
        std::thread producer([&] {
            for (int i = 0; i < 50; ++i) pump.offer(cv::Mat(8, 8));   // each blocks until taken
        });
        dxapp::pumpUntilJoined(pump, producer);
        CHECK(!producer.joinable(), "display thread was joined");
        CHECK(pump.shown() == 50, "all 50 tail frames shown");
        CHECK(pump.dropped() == 0, "nothing dropped");
    }
    {   // after a user quit the pump is stopped: just join, never spin
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.stop();
        std::thread producer([&] { for (int i = 0; i < 50; ++i) pump.offer(cv::Mat(8, 8)); });
        auto t0 = std::chrono::steady_clock::now();
        dxapp::pumpUntilJoined(pump, producer);
        CHECK(!producer.joinable(), "joined after stop()");
        CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1), "returned promptly");
        CHECK(cvmock::counters().imshow.load() == 0, "nothing shown after stop()");
    }
}

// GTK prints a dbind warning when the at-spi socket is missing. The helper
// sets NO_AT_BRIDGE once and does not clobber a value the user chose.
static void test_prepare_gui_backend_skips_accessibility_bridge() {
    std::cout << "[test] prepareGuiBackend sets NO_AT_BRIDGE without overriding\n";
    unsetenv("NO_AT_BRIDGE");
    dxapp::prepareGuiBackend();
    const char* bridged = std::getenv("NO_AT_BRIDGE");
    CHECK(bridged && std::string(bridged) == "1", "NO_AT_BRIDGE defaults to 1");

    setenv("NO_AT_BRIDGE", "0", 1);
    dxapp::prepareGuiBackend();
    const char* kept = std::getenv("NO_AT_BRIDGE");
    CHECK(kept && std::string(kept) == "0", "an explicit NO_AT_BRIDGE is kept");
    unsetenv("NO_AT_BRIDGE");
}

// Copy an environment value. getenv()'s pointer is invalidated by setenv().
static std::string envOrEmpty(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
}

static void restoreEnv(const char* name, bool was_set, const std::string& value) {
    if (was_set) setenv(name, value.c_str(), 1);
    else unsetenv(name);
}

// ibus + the Wayland Qt platform constructs a QFileSystemWatcher before the
// main thread has an event dispatcher. compose does not, and anything other
// than that pair is left alone.
static void test_prepare_gui_backend_replaces_ibus_on_wayland() {
    std::cout << "[test] prepareGuiBackend replaces ibus only on Wayland Qt\n";
    const bool had_im = std::getenv("QT_IM_MODULE") != nullptr;
    const bool had_platform = std::getenv("QT_QPA_PLATFORM") != nullptr;
    const bool had_session = std::getenv("XDG_SESSION_TYPE") != nullptr;
    const bool had_wayland = std::getenv("WAYLAND_DISPLAY") != nullptr;
    const std::string saved_im = envOrEmpty("QT_IM_MODULE");
    const std::string saved_platform = envOrEmpty("QT_QPA_PLATFORM");
    const std::string saved_session = envOrEmpty("XDG_SESSION_TYPE");
    const std::string saved_wayland = envOrEmpty("WAYLAND_DISPLAY");

    setenv("QT_IM_MODULE", "ibus", 1);
    unsetenv("QT_QPA_PLATFORM");
    setenv("XDG_SESSION_TYPE", "wayland", 1);
    dxapp::prepareGuiBackend();
    CHECK(envOrEmpty("QT_IM_MODULE") == "compose", "ibus on a Wayland session becomes compose");

    setenv("QT_IM_MODULE", "ibus", 1);
    setenv("QT_QPA_PLATFORM", "xcb", 1);
    setenv("XDG_SESSION_TYPE", "wayland", 1);
    dxapp::prepareGuiBackend();
    CHECK(envOrEmpty("QT_IM_MODULE") == "ibus", "an explicit xcb platform keeps ibus");

    setenv("QT_IM_MODULE", "ibus", 1);
    unsetenv("QT_QPA_PLATFORM");
    setenv("XDG_SESSION_TYPE", "x11", 1);
    dxapp::prepareGuiBackend();
    CHECK(envOrEmpty("QT_IM_MODULE") == "ibus", "an X11 session keeps ibus");

    setenv("QT_IM_MODULE", "xim", 1);
    unsetenv("QT_QPA_PLATFORM");
    setenv("XDG_SESSION_TYPE", "wayland", 1);
    dxapp::prepareGuiBackend();
    CHECK(envOrEmpty("QT_IM_MODULE") == "xim", "a non-ibus module is kept");

    unsetenv("QT_IM_MODULE");
    unsetenv("QT_QPA_PLATFORM");
    unsetenv("XDG_SESSION_TYPE");
    setenv("WAYLAND_DISPLAY", "wayland-0", 1);
    dxapp::prepareGuiBackend();
    CHECK(envOrEmpty("QT_IM_MODULE").empty(), "an unset module stays unset");

    restoreEnv("QT_IM_MODULE", had_im, saved_im);
    restoreEnv("QT_QPA_PLATFORM", had_platform, saved_platform);
    restoreEnv("XDG_SESSION_TYPE", had_session, saved_session);
    restoreEnv("WAYLAND_DISPLAY", had_wayland, saved_wayland);
}

int main() {
    // The GUI contract tests below exercise the display path, so pin DISPLAY
    // on: otherwise results would depend on whether the machine running the
    // tests happens to have a display server.
    setenv("DISPLAY", ":99", 1);

    test_lossless_is_default();
    test_lossless_shows_every_frame_in_order();
    test_lossless_offer_blocks_until_pumped();
    test_lossless_would_show_is_always_true();
    test_lossless_ignores_rate_limit();
    test_stop_unblocks_a_waiting_producer();
    test_window_fits_half_screen_on_creation();
    test_pump_until_joined_drains_the_display_thread();
    test_drop_mode_is_lossy_and_latest_wins();
    test_offer_does_not_copy_pixels();
    test_single_event_pump_per_tick();
    test_quit_detection();
    test_backend_without_window_property();
    test_would_show_reserves_slot();
    test_rate_limit();
    test_slow_display_does_not_throttle_producer();
    test_run_pipeline_with_display();
    test_headless_is_inert_not_quit();
    test_prepare_gui_backend_skips_accessibility_bridge();
    test_prepare_gui_backend_replaces_ibus_on_wayland();

    if (g_failures) { std::cerr << "\n" << g_failures << " CHECK(s) FAILED\n"; return 1; }
    std::cout << "\nAll DisplayPump contract tests PASSED\n";
    return 0;
}
