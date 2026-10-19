/**
 * @file progress.hpp
 * @brief Machine-readable progress for front ends:
 *        "[PROGRESS] frames=N fps=F avg_fps=A read_ms=R pre_ms=P infer_ms=I post_ms=Q",
 *        or, from the multi-model graph binaries,
 *        "[PROGRESS] frames=N fps=F avg_fps=A node_ms=<id>:<ms>,<id>:<ms>".
 *
 * A front end showing a running example (DX AI Studio's live view) cannot tell how
 * many frames have been processed: most runners print nothing per frame, and the
 * [DET] / [OBB] / [ISEG] tags print one line per OBJECT. With DXAPP_PROGRESS=1 every
 * runner calls progressTick(metrics) once per completed frame and this prints, at most
 * once per second plus once when the process ends, what the runner measured itself:
 *
 *   frames   frames completed so far (always first, always present)
 *   fps      frames since the previous line / seconds since it -- the live rate
 *   avg_fps  frames after the first / seconds since the first -- start-up and model
 *            load are not in it, unlike frames / process time
 *   *_ms     the read / preprocess / inference / postprocess average over the same
 *            window, from the runner's own sums (async inference is the submit-to-
 *            callback turnaround the runner's summary prints)
 *
 * A field that cannot be computed yet is left out: the first line is "frames=1" only,
 * and a tick without metrics prints no *_ms. The final line carries run averages
 * (fps = avg_fps, *_ms = sum / frames) up to the last frame. Without the variable
 * nothing is printed and the cost is one branch.
 *
 * A graph runner ticks with tickNodes(): ONE frame's time per node, in the
 * order the line should list them. node_ms is the average per node over the
 * same window - over the frames of that window that ran the node, so a node
 * of one stream is not diluted by the other stream's frames - and the run
 * average on the final line. A node with no frame in the window is left out.
 * Ids are percent-encoded outside [A-Za-z0-9_.-] so ',' ':' '=' and spaces
 * cannot break the field.
 *
 * A runner may call start() when its first frame enters the pipeline; avg_fps
 * (and the final fps) are then frames / seconds since start(). The graph CLI
 * does: an input shorter than its in-flight cap returns every report in one
 * burst, and frames after the first over that burst's span read ~100k fps.
 * Runners that never call start() keep the first-tick rule above.
 *
 * stdoutMutex() is shared by everything that writes pipeline-parsed lines, so lines
 * from concurrent async callbacks are written whole ("3.50273.5913" was two scores
 * from two callbacks interleaved).
 *
 * Contract: tests/progress/test_progress.cpp
 */
#ifndef DXAPP_PROGRESS_HPP
#define DXAPP_PROGRESS_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace dxapp {

/** Serialises writes of pipeline-parsed stdout lines across threads.
 *  Never destroyed: progress()'s Progress prints its last line from its destructor at exit, and it is built before
 *  this mutex is first used, so a static mutex would be destroyed first and then locked. */
inline std::mutex& stdoutMutex() {
    static std::mutex* m = new std::mutex;
    return *m;
}

/** Running stage-time sums in milliseconds, as the runners' metric structs keep them. */
struct StageSums {
    double read_ms = 0.0;
    double pre_ms = 0.0;
    double infer_ms = 0.0;
    double post_ms = 0.0;
};

/** One frame's processing time per graph node, in the order a line lists them. */
typedef std::vector<std::pair<std::string, double> > NodeMs;

class Progress {
public:
    using Clock = int64_t (*)();  // nanoseconds; tests pass their own

    explicit Progress(double interval_s = 1.0, Clock clock = nullptr)
        : interval_ns_(static_cast<int64_t>(interval_s * 1e9)),
          enabled_(envEnabled()),
          clock_(clock ? clock : &steadyNs) {}
    ~Progress() { finish(); }
    Progress(const Progress&) = delete;
    Progress& operator=(const Progress&) = delete;

    /** One frame finished. Thread-safe. */
    void tick() { tickImpl(nullptr); }

    /** One frame finished; @p sums is the runner's running stage sums after it. Thread-safe. */
    void tick(const StageSums& sums) { tickImpl(&sums); }

    /** One frame finished; @p node_ms is THAT frame's time per node (not a running sum). Thread-safe. */
    void tickNodes(const NodeMs& node_ms) { tickImpl(nullptr, &node_ms); }

    /** Final line with the run averages; printed once. */
    void finish() {
        if (!enabled_ || finished_.exchange(true)) return;
        std::lock_guard<std::mutex> lock(state_mutex_);
        const long n = frames_.load();
        std::string line = "[PROGRESS] frames=" + std::to_string(n);
        const double avg = avgFps(n, last_tick_ns_);
        if (avg >= 0) {
            appendField(line, "fps", avg, 1);
            appendField(line, "avg_fps", avg, 1);
        }
        if (has_sums_ && n > 0) appendStages(line, last_sums_, StageSums{}, static_cast<double>(n));
        appendNodes(line, false);
        print(line);
    }

    /** The run starts now: avg_fps counts every frame from here. Thread-safe. */
    void start() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        start_ns_ = clock_();
        has_start_ = true;
    }

    long frames() const { return frames_.load(); }

private:
    void tickImpl(const StageSums* sums, const NodeMs* nodes = nullptr) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(state_mutex_);
        const long n = ++frames_;
        const int64_t t = clock_();
        if (n == 1) first_ns_ = t;
        last_tick_ns_ = t;
        if (sums) {
            // A runner that starts new metrics (a new loop) restarts its sums from zero:
            // carry what it had, so the window and the run average keep adding up.
            if (has_sums_ && (sums->read_ms < raw_sums_.read_ms || sums->pre_ms < raw_sums_.pre_ms ||
                              sums->infer_ms < raw_sums_.infer_ms || sums->post_ms < raw_sums_.post_ms))
                base_sums_ = plus(base_sums_, raw_sums_);
            raw_sums_ = *sums;
            last_sums_ = plus(base_sums_, raw_sums_);
            has_sums_ = true;
        }
        if (nodes) addNodes(*nodes);
        if (printed_ && t - prev_ns_ < interval_ns_) return;

        std::string line = "[PROGRESS] frames=" + std::to_string(n);
        if (printed_) {
            const long df = n - prev_frames_;
            const double dt = static_cast<double>(t - prev_ns_) / 1e9;
            if (df > 0 && dt > 0) appendField(line, "fps", df / dt, 1);
            const double avg = avgFps(n, t);
            if (avg >= 0) appendField(line, "avg_fps", avg, 1);
            if (sums && prev_has_sums_ && df > 0)
                appendStages(line, last_sums_, prev_sums_, static_cast<double>(df));
            if (nodes) appendNodes(line, true);
        }
        print(line);
        printed_ = true;
        prev_ns_ = t;
        prev_frames_ = n;
        prev_sums_ = last_sums_;
        prev_has_sums_ = has_sums_;
        for (NodeSum& s : node_sums_) {
            s.prev_sum_ms = s.sum_ms;
            s.prev_frames = s.frames;
        }
    }

    /** Per node: the run's sum and frame count, and both as of the previous line. */
    struct NodeSum {
        std::string id;
        double sum_ms = 0.0;
        long frames = 0;
        double prev_sum_ms = 0.0;
        long prev_frames = 0;
    };

    void addNodes(const NodeMs& nodes) {
        for (const auto& node : nodes) {
            NodeSum* sum = nullptr;
            for (NodeSum& s : node_sums_) {
                if (s.id == node.first) { sum = &s; break; }
            }
            if (!sum) {
                node_sums_.push_back(NodeSum());
                sum = &node_sums_.back();
                sum->id = node.first;
            }
            sum->sum_ms += node.second;
            ++sum->frames;
        }
    }

    /** " node_ms=<id>:<ms>,..." over the window (live) or the run (final); omitted when empty. */
    void appendNodes(std::string& line, bool window) const {
        std::string field;
        for (const NodeSum& s : node_sums_) {
            const long frames = window ? s.frames - s.prev_frames : s.frames;
            if (frames <= 0) continue;
            const double ms = (window ? s.sum_ms - s.prev_sum_ms : s.sum_ms) / frames;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f", ms);
            if (!field.empty()) field += ",";
            field += encodeId(s.id);
            field += ":";
            field += buf;
        }
        if (!field.empty()) line += " node_ms=" + field;
    }

    /** Every byte outside [A-Za-z0-9_.-] as %XX. */
    static std::string encodeId(const std::string& id) {
        static const char kHex[] = "0123456789ABCDEF";
        std::string out;
        for (unsigned char c : id) {
            const bool plain = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                               (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
            if (plain) {
                out += static_cast<char>(c);
            } else {
                out += '%';
                out += kHex[c >> 4];
                out += kHex[c & 15];
            }
        }
        return out;
    }

    static StageSums plus(const StageSums& a, const StageSums& b) {
        return StageSums{a.read_ms + b.read_ms, a.pre_ms + b.pre_ms, a.infer_ms + b.infer_ms, a.post_ms + b.post_ms};
    }

    /** Frames since start() over the time since it, or, without start(), frames
     *  after the first over the time since it; -1 when there is no span yet. */
    double avgFps(long n, int64_t t) const {
        if (has_start_) {
            if (n < 1 || t <= start_ns_) return -1.0;
            return n / (static_cast<double>(t - start_ns_) / 1e9);
        }
        if (n < 2 || t <= first_ns_) return -1.0;
        return (n - 1) / (static_cast<double>(t - first_ns_) / 1e9);
    }

    static void appendField(std::string& line, const char* key, double value, int decimals) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " %s=%.*f", key, decimals, value);
        line += buf;
    }

    /** (now - before) / frames for each stage; a stage that went backwards is left out. */
    static void appendStages(std::string& line, const StageSums& now, const StageSums& before, double frames) {
        const struct { const char* key; double v; } stages[] = {
            {"read_ms", now.read_ms - before.read_ms},
            {"pre_ms", now.pre_ms - before.pre_ms},
            {"infer_ms", now.infer_ms - before.infer_ms},
            {"post_ms", now.post_ms - before.post_ms},
        };
        for (const auto& s : stages)
            if (s.v >= 0) appendField(line, s.key, s.v / frames, 2);
    }

    static bool envEnabled() {
        const char* v = std::getenv("DXAPP_PROGRESS");
        return v && *v && std::strcmp(v, "0") != 0;
    }
    static int64_t steadyNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static void print(const std::string& line) {
        std::lock_guard<std::mutex> lock(stdoutMutex());
        std::cout << line << '\n' << std::flush;
    }

    const int64_t interval_ns_;
    const bool enabled_;
    const Clock clock_;
    std::atomic<long> frames_{0};
    std::atomic<bool> finished_{false};

    std::mutex state_mutex_;  // everything below
    bool printed_ = false;
    int64_t first_ns_ = 0;
    int64_t start_ns_ = 0;
    bool has_start_ = false;
    int64_t last_tick_ns_ = 0;
    int64_t prev_ns_ = 0;
    long prev_frames_ = 0;
    StageSums raw_sums_;   // as the runner passed them last
    StageSums base_sums_;  // what earlier metrics had when the runner restarted them
    StageSums last_sums_;  // base + raw: the run so far
    StageSums prev_sums_;
    bool has_sums_ = false;
    bool prev_has_sums_ = false;
    std::vector<NodeSum> node_sums_;  // first-seen order
};

/** The process-wide counter the runners tick. */
inline Progress& progress() {
    static Progress p(1.0);
    return p;
}

inline void progressTick() { progress().tick(); }

/** Tick with a runner's metrics (SyncProfilingMetrics / AsyncProfilingMetrics: sum_* in ms).
 *  Async runners call it under their metrics mutex, so the sums are a consistent snapshot. */
template <class Metrics>
inline void progressTick(const Metrics& m) {
    progress().tick(StageSums{m.sum_read, m.sum_preprocess, m.sum_inference, m.sum_postprocess});
}

}  // namespace dxapp

#endif  // DXAPP_PROGRESS_HPP
