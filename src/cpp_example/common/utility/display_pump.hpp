/**
 * @file display_pump.hpp
 * @brief Lossy, rate-limited display sink that never paces the inference pipeline.
 *
 * ## Why this exists
 *
 * A preview window is a *best-effort* output: the user cannot perceive more than
 * their monitor's refresh rate, and a frame that is already stale is worthless.
 * Inference throughput, by contrast, is the number the sample application exists
 * to demonstrate. Whenever the display is allowed to pace or block the pipeline,
 * the reported FPS stops measuring the NPU and starts measuring HighGUI.
 *
 * The previous design violated that in two ways:
 *
 *   1. `imshow()` / `waitKey()` ran **inside the frame-submission loop**, so every
 *      frame paid the GUI cost before the next frame could be submitted.
 *   2. The display queue was a *bounded blocking* queue. Once the GUI fell behind,
 *      back-pressure propagated backwards through the render thread and into the
 *      DXRT completion callback, stalling the whole pipeline.
 *
 * `DisplayPump` fixes both by construction:
 *
 *   - **I1 — No GUI on the submit path.** All HighGUI calls happen in `pump()`,
 *     which is driven by the GUI thread only. Producers only ever call `offer()`.
 *   - **I2 — Depth-1 sink, lossless by default.** `offer()` hands one frame to
 *     the GUI thread and waits for the slot to free (every frame is shown, so a
 *     slow window paces the producer). With `setDropStale(true)` (`--drop-frames`)
 *     it instead takes a mutex, replaces the pending
 *     frame and returns. `cv::Mat` assignment is a refcount bump, so this is O(1)
 *     with no pixel copy, and it can never block. Stale frames are dropped, and
 *     the drop count is reported so the behaviour stays visible.
 *   - **I3 — One event pump per tick.** Exactly one `cv::waitKey()` per `pump()`,
 *     and `namedWindow()` once for the lifetime of the window — not per frame.
 *
 * ## Threading contract
 *
 * `offer()` is safe from any thread. `pump()` MUST be called from one thread only,
 * and that thread MUST be the process main thread: OpenCV's Qt backend constructs
 * its `QApplication` on whichever thread first creates a window, and Qt requires
 * that to be the main thread. Use `runPipelineWithDisplay()` to get this right.
 */

#ifndef DXAPP_DISPLAY_PUMP_HPP
#define DXAPP_DISPLAY_PUMP_HPP

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace dxapp {

/**
 * @brief Is a graphical display server available to this process?
 *
 * Needed because HighGUI's failure mode is backend-dependent and can be fatal
 * when there is no display. With the GTK backend of OpenCV >= 4.5
 * `cv::namedWindow()` throws a catchable `cv::Exception`, but OpenCV < 4.5
 * (apt 4.2 on Ubuntu 20.04) calls `gtk_init()`, which prints "cannot open
 * display" and `exit(1)`s. With the **Qt** backend Qt fails to load its `xcb`
 * platform plugin and calls `qFatal()`, which `abort()`s the process. No
 * try/catch can rescue either, so the only safe option is to look before we
 * leap.
 *
 * Mirrors `_has_display()` in the Python runners.
 */
inline bool hasDisplay() {
#if defined(_WIN32) || defined(__APPLE__)
    return true;                       // native window system, always present
#else
    const char* x11 = std::getenv("DISPLAY");
    const char* wl  = std::getenv("WAYLAND_DISPLAY");
    return (x11 && *x11) || (wl && *wl);
#endif
}

/**
 * @brief True when Qt will load its Wayland platform plugin.
 *
 * An explicit QT_QPA_PLATFORM wins. Otherwise Qt follows XDG_SESSION_TYPE,
 * and a pure Wayland session (WAYLAND_DISPLAY set, no session type) too.
 */
inline bool qtWillUseWayland() {
    // Platform the user pinned, e.g. "xcb" or "wayland".
    const char* platform = std::getenv("QT_QPA_PLATFORM");
    if (platform != nullptr && platform[0] != '\0') {
        return std::strstr(platform, "wayland") != nullptr;
    }
    // Session type the desktop set ("wayland", "x11", "tty", ...).
    const char* session = std::getenv("XDG_SESSION_TYPE");
    if (session != nullptr && session[0] != '\0') {
        return std::strcmp(session, "wayland") == 0;
    }
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    return wayland_display != nullptr && wayland_display[0] != '\0';
}

/**
 * @brief Quiet HighGUI before the first window is created.
 *
 * GTK loads the AT-SPI bridge during `cv::namedWindow()`. When
 * `/run/user/<uid>/at-spi/bus` is missing, dbind prints
 * "Couldn't connect to accessibility bus". `NO_AT_BRIDGE=1` skips that
 * bridge. A value already set by the user is left unchanged.
 *
 * On Wayland, Qt loads the ibus input-method plugin while QApplication is
 * still starting, before the main thread has an event dispatcher. That plugin
 * builds a QFileSystemWatcher and Qt prints "QSocketNotifier: Can only be
 * used with threads started with QThread". The preview does not take text
 * input, so ibus is replaced with the compose plugin in that one case.
 *
 * Call this before every first `cv::namedWindow()` in the process.
 */
inline void prepareGuiBackend() {
#ifndef _WIN32
    // Skip the accessibility bridge when the at-spi socket is not there.
    if (std::getenv("NO_AT_BRIDGE") == nullptr) {
        ::setenv("NO_AT_BRIDGE", "1", 0);
    }
    // Input method the desktop asked Qt to load. Only "ibus" trips the warning.
    const char* input_module = std::getenv("QT_IM_MODULE");
    if (input_module != nullptr && std::strcmp(input_module, "ibus") == 0 &&
        qtWillUseWayland()) {
        ::setenv("QT_IM_MODULE", "compose", 1);
    }
#endif
}

/** Default preview rate. Beyond this the user cannot perceive the difference. */
constexpr double DISPLAY_PUMP_DEFAULT_FPS = 60.0;

class DisplayPump {
public:
    /**
     * @param winname  HighGUI window name.
     * @param max_fps  Upper bound on imshow frequency. 0 disables rate limiting.
     */
    explicit DisplayPump(std::string winname = "Output",
                         double max_fps = DISPLAY_PUMP_DEFAULT_FPS)
        : winname_(std::move(winname)),
          min_interval_(max_fps > 0.0
                            ? std::chrono::duration<double>(1.0 / max_fps)
                            : std::chrono::duration<double>(0.0)),
          gui_available_(hasDisplay()) {}

    DisplayPump(const DisplayPump&) = delete;
    DisplayPump& operator=(const DisplayPump&) = delete;

    /**
     * @brief Publish a frame for display. Never copies pixels.
     *
     * Lossless (default): if a previously offered frame has not been taken by
     * the GUI thread yet, wait until it is (or until stop()), so every frame is
     * shown in order and a slow window paces the producer.
     * Drop mode (setDropStale(true)): never blocks; a pending frame is discarded
     * and counted -- showing the newest frame beats showing a backlog.
     */
    void offer(const cv::Mat& frame) {
        if (!gui_available_) return;   // headless: nothing will ever show it
        if (frame.empty() || stopped_.load(std::memory_order_acquire)) return;
        std::unique_lock<std::mutex> lock(mutex_);
        if (has_pending_) {
            if (drop_stale_.load(std::memory_order_acquire)) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
            } else {
                slot_free_.wait(lock, [this] {
                    return !has_pending_ || stopped_.load(std::memory_order_acquire);
                });
                if (stopped_.load(std::memory_order_acquire)) return;
            }
        }
        pending_ = frame;  // refcount bump only
        has_pending_ = true;
        frame_ready_.notify_one();
    }

    /**
     * @brief Wait up to @p timeout for a frame to be offered (or for stop()).
     *
     * The GUI thread's idle gap: in lossless mode a frame must be shown as soon
     * as it arrives, or the waiting producer would be paced by the gap itself.
     */
    template <class Rep, class Period>
    void waitForFrame(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        frame_ready_.wait_for(lock, timeout, [this] {
            return has_pending_ || stopped_.load(std::memory_order_acquire);
        });
    }

    /**
     * @brief Service the GUI once: show the newest frame (subject to the rate
     *        limit) and pump the event loop exactly once.
     *
     * @return false if the user asked to quit ('q' / ESC) or closed the window.
     *         Once false, it stays false.
     */
    bool pump() {
        // Headless: do nothing, but report "no quit requested". Returning false
        // here would be read by runPipelineWithDisplay() as the user quitting
        // and would tear the inference pipeline down.
        if (!gui_available_) return true;
        if (stopped_.load(std::memory_order_acquire)) return false;

        cv::Mat frame;
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // The preview rate limit only exists to skip frames; lossless shows all.
            const bool due = !drop_stale_.load(std::memory_order_acquire) ||
                             !window_created_ || min_interval_.count() <= 0.0 ||
                             (now - last_show_) >= min_interval_;
            if (has_pending_ && due) {
                frame = std::move(pending_);
                pending_ = cv::Mat();
                has_pending_ = false;
                last_show_ = now;
            }
        }
        if (!frame.empty()) slot_free_.notify_all();   // producer may offer the next one

        if (!frame.empty() && !showFrame(frame)) return false;

        // Nothing on screen yet: no window exists, so there are no events to
        // pump and no window property to interrogate.
        if (!window_created_) return true;

        return pumpEvents();
    }

    /** Stop the pump. Idempotent; subsequent pump() calls return false. */
    void stop() {
        stopped_.store(true, std::memory_order_release);
        { std::lock_guard<std::mutex> lock(mutex_); }   // no lost wake-up vs. the predicates
        slot_free_.notify_all();
        frame_ready_.notify_all();
    }

    /**
     * @brief Drop stale frames instead of pacing the producer (`--drop-frames`).
     *
     * Off by default: every frame is shown and a slow window slows the pipeline.
     * On: depth-1 lossy sink with the preview rate limit, as before 2026-10-06.
     */
    void setDropStale(bool on) { drop_stale_.store(on, std::memory_order_release); }
    bool dropStale() const { return drop_stale_.load(std::memory_order_acquire); }

    /**
     * @brief Screen size used to fit the window on creation. Non-positive values
     *        fall back to 1920x1080, the same fallback the Python runners use.
     */
    void setScreenSize(int w, int h) {
        screen_w_ = (w > 0 && h > 0) ? w : 1920;
        screen_h_ = (w > 0 && h > 0) ? h : 1080;
    }

    bool stopped() const { return stopped_.load(std::memory_order_acquire); }

    /** False when no display server exists; the pump is then inert. */
    bool guiAvailable() const { return gui_available_; }

    /**
     * @brief Reserve the next preview slot, if one is due.
     *
     * A true result consumes the slot: the caller must render that frame, and
     * further callers within @c min_interval_ get false. That is what keeps a
     * 200+ FPS pipeline from drawing an overlay it is about to throw away.
     * Saving is NOT this function's concern -- the caller renders every frame
     * when the pixels are written to disk, and only consults this for a live
     * preview.
     *
     * The slot is reserved here, not when the GUI thread later calls imshow, so
     * frames queued before the window exists are not all drawn.
     */
    bool wouldShow() const {
        if (!gui_available_ || stopped_.load(std::memory_order_acquire)) return false;
        if (!drop_stale_.load(std::memory_order_acquire)) return true;   // lossless: render all
        std::lock_guard<std::mutex> lock(mutex_);
        if (min_interval_.count() <= 0.0) return true;
        const auto now = std::chrono::steady_clock::now();
        if ((now - last_admit_) < min_interval_) return false;
        last_admit_ = now;
        return true;
    }

    /** Frames actually shown. */
    int shown() const { return shown_.load(std::memory_order_relaxed); }

    /** Frames superseded before they could be shown (healthy, not an error). */
    int dropped() const { return dropped_.load(std::memory_order_relaxed); }

    /** Mean wall time spent inside imshow, in milliseconds. */
    double avgShowMs() const {
        const int n = shown_.load(std::memory_order_relaxed);
        return n > 0 ? sum_show_ms_.load(std::memory_order_relaxed) / n : 0.0;
    }

    /**
     * @brief Size the window once, on creation. Ignored if either value is <= 0.
     *
     * Kept as a caller-supplied hint so this header stays dependency-free and
     * unit-testable; the runners compute it from the screen resolution.
     */
    void setInitialWindowSize(int w, int h) { win_w_ = w; win_h_ = h; }

private:
    bool showFrame(const cv::Mat& frame) {
        if (!window_created_) {
            prepareGuiBackend();
            try {
                cv::namedWindow(winname_, cv::WINDOW_NORMAL);
            } catch (const cv::Exception&) {
                // No GUI available (headless). Degrade to no display rather
                // than killing the inference run.
                stop();
                return false;
            }
            window_created_ = true;
            int w = win_w_, h = win_h_;
            if ((w <= 0 || h <= 0) && frame.cols > 0 && frame.rows > 0) {
                // Same rule as the Python runners' show_image(): fit the frame
                // into half the screen width and height, aspect preserved,
                // truncating like int().
                const double target_w = static_cast<double>(screen_w_ / 2);
                const double target_h = static_cast<double>(screen_h_ / 2);
                const double scale = std::min(target_w / frame.cols, target_h / frame.rows);
                w = static_cast<int>(frame.cols * scale);
                h = static_cast<int>(frame.rows * scale);
            }
            if (w > 0 && h > 0) {
                try { cv::resizeWindow(winname_, w, h); }
                catch (const cv::Exception&) { /* non-fatal */ }
            }
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        try {
            cv::imshow(winname_, frame);
        } catch (const cv::Exception&) {
            stop();
            return false;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        sum_show_ms_.store(
            sum_show_ms_.load(std::memory_order_relaxed) +
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::memory_order_relaxed);
        shown_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    /** Exactly one waitKey per tick, plus user-quit / window-closed detection. */
    bool pumpEvents() {
        int key = -1;
        try {
            key = cv::waitKey(1);
        } catch (const cv::Exception&) {
            stop();
            return false;
        }
        if (key == 'q' || key == 27) {
            stop();
            return false;
        }

        // Some backends (e.g. GTK2) always return -1 for WND_PROP_VISIBLE even
        // for a live window. Probe once and disable the check if unsupported,
        // so -1 is never mistaken for "the user closed the window".
        if (!probed_) {
            probed_ = true;
            try {
                const double probe = cv::getWindowProperty(winname_, cv::WND_PROP_VISIBLE);
                if (probe < -0.5) prop_supported_ = false;
            } catch (const cv::Exception&) {
                prop_supported_ = false;
            }
            return true;
        }

        if (prop_supported_) {
            try {
                if (cv::getWindowProperty(winname_, cv::WND_PROP_VISIBLE) <= 0.0) {
                    stop();
                    return false;
                }
            } catch (const cv::Exception&) {
                stop();
                return false;
            }
        }
        return true;
    }

    std::string winname_;
    std::chrono::duration<double> min_interval_;
    bool gui_available_;

    mutable std::mutex mutex_;
    std::condition_variable slot_free_;    // pending slot taken by the GUI thread
    std::condition_variable frame_ready_;  // a frame was offered
    std::atomic<bool> drop_stale_{false};  // --drop-frames; default lossless
    cv::Mat pending_;
    bool has_pending_ = false;
    std::chrono::steady_clock::time_point last_show_{};
    // Last time a producer was told to render a preview. Independent of
    // last_show_ (when imshow actually ran) so the render thread can skip
    // frames the pump has not displayed yet.
    mutable std::chrono::steady_clock::time_point last_admit_{};

    // Touched only by the pump thread.
    bool window_created_ = false;
    bool probed_ = false;
    bool prop_supported_ = true;
    int win_w_ = 0;
    int win_h_ = 0;
    int screen_w_ = 1920;
    int screen_h_ = 1080;

    std::atomic<bool> stopped_{false};
    std::atomic<int> shown_{0};
    std::atomic<int> dropped_{0};
    std::atomic<double> sum_show_ms_{0.0};
};

/**
 * @brief True when this frame's pixels must be produced.
 *
 * `--no-display` means do not render. `-s` / a per-frame save path /
 * `DXAPP_SAVE_IMAGE` still render, because the file on disk is the visualized
 * frame. A live preview renders only the slot @c DisplayPump::wouldShow
 * reserved -- drawing every frame fills the bounded display queue, the DXRT
 * callback blocks on push, and end-to-end FPS measures the renderer.
 */
inline bool mustRenderFrame(bool no_display, bool save_on,
                            const std::string& save_path,
                            const DisplayPump& pump) {
    if (save_on || !save_path.empty()) return true;
    const char* env_save = std::getenv("DXAPP_SAVE_IMAGE");
    if (env_save != nullptr && *env_save != '\0') return true;
    if (no_display) return false;
    return pump.wouldShow();
}

/** Sync preview rate. imshow is on the frame loop, so this stays below the async 60. */
constexpr double SYNC_PREVIEW_FPS = 10.0;

/** One pump per process. Sync runners share it; only one runner executes. */
inline DisplayPump& syncPreviewPump() {
    static DisplayPump pump{"Output", SYNC_PREVIEW_FPS};
    return pump;
}

/**
 * @brief Split "draw the overlay" from "paint the window" for a sync frame.
 *
 * persist: -s, a save path, or DXAPP_SAVE_IMAGE. Every such frame is drawn.
 * preview: a live-window slot. wouldShow() is called only when a window is
 * requested, so --no-display does not consume a slot.
 */
struct FrameRenderPlan {
    bool persist = false;
    bool preview = false;
    bool need_render = false;
};

inline FrameRenderPlan planFrameRender(bool no_display, bool save_on,
                                       const std::string& save_path,
                                       DisplayPump& pump) {
    const char* env_save = std::getenv("DXAPP_SAVE_IMAGE");
    FrameRenderPlan plan;
    plan.persist = save_on || !save_path.empty()
        || (env_save != nullptr && *env_save != '\0');
    plan.preview = !no_display && pump.wouldShow();
    plan.need_render = plan.persist || plan.preview;
    return plan;
}

/**
 * @brief Run the inference pipeline on a worker thread while the calling thread
 *        owns the GUI.
 *
 * This is the piece that makes display cost disappear from the critical path.
 * `pipeline` performs read -> preprocess -> RunAsync submission and must never
 * touch HighGUI; the calling thread (which must be the main thread) does nothing
 * but service the window. With `no_display` the caller simply waits.
 *
 * If the user quits, `pump()` returns false, `stop()` has already been latched,
 * and `on_user_quit` is invoked so the runner can tear the pipeline down; the
 * worker is then joined, so the pipeline is never left running.
 */
template <typename PipelineFn, typename QuitFn>
void runPipelineWithDisplay(PipelineFn&& pipeline, DisplayPump& pump,
                            bool no_display, QuitFn&& on_user_quit) {
    std::atomic<bool> done{false};
    std::thread worker([&] {
        pipeline();
        done.store(true, std::memory_order_release);
    });

    if (no_display) {
        while (!done.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    } else {
        // Idle between pumps. A tight loop spends the whole run inside
        // waitKey(), and HighGUI holds the OpenCV lock while it does -- that
        // stalls preprocess/render on the worker and caps end-to-end FPS near
        // the GUI rate (~90 FPS on the YOLO26 async runs) no matter how fast
        // the NPU is. The preview is already rate-limited; this is just the
        // event-poll gap.
        constexpr auto kGuiPollIdle = std::chrono::milliseconds(8);
        bool user_quit = false;
        while (!done.load(std::memory_order_acquire)) {
            if (!pump.pump()) {
                user_quit = true;
                on_user_quit();
                break;
            }
            if (pump.dropStale()) {
                std::this_thread::sleep_for(kGuiPollIdle);
            } else {
                pump.waitForFrame(kGuiPollIdle);   // lossless: show frames as they arrive
            }
        }
        // The idle gap can outlast a short pipeline: the last offered frame is
        // still pending when the worker sets done. Show it once. A user quit
        // already stopped the pump, so do not service the window again.
        if (!user_quit) {
            pump.pump();
        }
    }
    worker.join();
}

/**
 * @brief Keep servicing the window until @p thread exits, then join it.
 *
 * The lossless tail: when the pipeline finishes, the display thread may still
 * be rendering and offering its last frames. Stopping the pump first would
 * leave them unshown (offer() returns at once after stop()). After a user quit
 * the pump is already stopped, so this only joins -- producers are released.
 */
template <typename Thread>
void pumpUntilJoined(DisplayPump& pump, Thread& thread) {
    if (!thread.joinable()) return;
    std::atomic<bool> joined{false};
    std::thread joiner([&] { thread.join(); joined.store(true, std::memory_order_release); });
    constexpr auto kGuiPollIdle = std::chrono::milliseconds(8);
    while (!joined.load(std::memory_order_acquire)) {
        if (pump.stopped()) break;
        pump.pump();
        pump.waitForFrame(kGuiPollIdle);
    }
    joiner.join();
}

}  // namespace dxapp

#endif  // DXAPP_DISPLAY_PUMP_HPP
