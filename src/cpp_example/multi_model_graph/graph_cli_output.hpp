/**
 * @file graph_cli_output.hpp
 * @brief Where the multi-model-graph CLI sends a rendered frame: --output
 *        (one video, or one image per frame) and --display (a window).
 *
 * The decisions are pure functions so graph_engine_test can pin them; the
 * two sinks are RAII classes. Header-only, like graph_cli_interrupt.hpp.
 *
 * Why each rule is what it is:
 *
 *  - Output kind by extension, case-insensitively: .mp4, .avi and .mkv are
 *    one video; anything else - including no extension - is image output,
 *    as before (<stem>_<index><ext> on a multi-frame source, exactly
 *    --output on a single image; NumberedPath appends .png when there is no
 *    extension). An extension is only the part after the last dot of the
 *    last path component, so "dir.mp4/out" is an image.
 *
 *  - A live source (camera:, rtsp:// - the prefixes consumer::OpenInput
 *    dispatches on) with image output and no --frames writes images until
 *    the disk is full. Main refuses that as a usage error before
 *    PrepareGraph, so no model is loaded and no device is opened to find
 *    out, and the answer does not depend on having an NPU, a camera or a
 *    reachable RTSP server. The message offers .mkv and .avi first: a run
 *    that is killed leaves an .mp4 with no index, which nothing can read.
 *
 *  - An --output whose directory does not exist fails in Main, before
 *    PrepareGraph (OutputDirectoryProblem), not at the first rendered
 *    frame behind the models' load and GStreamer's log lines. Exit 1 with
 *    "could not write <path>: ...", as --report's unwritable path.
 *
 *  - Codecs, probed on this OpenCV (4.6.0, FFMPEG and GStreamer on): .mp4
 *    and .mkv -> mp4v, .avi -> MJPG. All three open, write and read back
 *    every frame. avc1 also works here, but it depends on the H.264 encoder
 *    the FFMPEG build has; mp4v and MJPG are in every FFMPEG build.
 *
 *  - fps: the source's when it is finite and within [1, 240], else 30.
 *    Cameras and images report 0, and RTSP streams often report the 90 kHz
 *    RTP clock (90000) as their fps.
 *
 *  - cv::VideoWriter silently drops a frame whose size differs from the
 *    size it was opened with. VideoOutput opens on the first frame, at its
 *    size, and resizes any later frame of another size, so the video holds
 *    every reported frame.
 *
 *  - A Qt HighGUI with no display calls qFatal (an abort, not a
 *    cv::Exception). So --display is refused up front when none of DISPLAY,
 *    WAYLAND_DISPLAY, QT_QPA_PLATFORM is set (Windows always has a display).
 *    A variable that is set but names nothing reachable (DISPLAY=:99 with
 *    no server there, QT_QPA_PLATFORM=xcb with no DISPLAY) aborts the same
 *    way, so Main then opens a window once in a forked child
 *    (DisplayOpens): a child that dies is a refusal before any model loads,
 *    instead of an abort after they have, with --report left unfinished.
 *    A cv::Exception from the window itself turns display off for the rest
 *    of the run with one warning. Closing the window does not stop the run
 *    (the next frame reopens it): detecting a closed window depends on the
 *    backend, so only q, Q and ESC stop it.
 */
#ifndef DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_OUTPUT_HPP
#define DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_OUTPUT_HPP

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "common/utility/display_pump.hpp"

namespace dxapp {
namespace graph {
namespace cli {

enum class OutputKind { kNone, kImagePerFrame, kVideo };

/// ".MP4" -> ".mp4"; "" when the last path component has no dot.
inline std::string LowerExtension(const std::string& path) {
    const std::size_t dot = path.rfind('.');
    const std::size_t slash = path.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return std::string();
    }
    std::string extension = path.substr(dot);
    for (std::size_t i = 0; i < extension.size(); ++i) {
        extension[i] = static_cast<char>(
            std::tolower(static_cast<unsigned char>(extension[i])));
    }
    return extension;
}

/// .mp4 / .avi / .mkv: one video. "": no output. Anything else: images.
inline OutputKind ClassifyOutput(const std::string& path) {
    if (path.empty()) return OutputKind::kNone;
    const std::string extension = LowerExtension(path);
    if (extension == ".mp4" || extension == ".avi" || extension == ".mkv") {
        return OutputKind::kVideo;
    }
    return OutputKind::kImagePerFrame;
}

/// "out.png" + index 3 -> "out_000003.png". Used only when the source has
/// more than one frame, so a single image writes exactly --output.
inline std::string NumberedPath(const std::string& path, std::size_t index) {
    const std::size_t dot = path.rfind('.');
    const std::size_t slash = path.find_last_of("/\\");
    const bool has_extension =
        dot != std::string::npos &&
        (slash == std::string::npos || dot > slash);
    std::ostringstream out;
    out << (has_extension ? path.substr(0, dot) : path) << "_"
        << std::setw(6) << std::setfill('0') << index
        << (has_extension ? path.substr(dot) : std::string(".png"));
    return out.str();
}

/// `path` with "_<tag>" before its extension: ("run.v1/out.mp4", "cam1") ->
/// "run.v1/out_cam1.mp4"; ("dir.x/out", "cam1") -> "dir.x/out_cam1".
inline std::string TaggedPath(const std::string& path, const std::string& tag) {
    const std::size_t dot = path.rfind('.');
    const std::size_t slash = path.find_last_of("/\\");
    const bool has_extension = dot != std::string::npos &&
                               (slash == std::string::npos || dot > slash);
    return has_extension ? path.substr(0, dot) + "_" + tag + path.substr(dot)
                         : path + "_" + tag;
}

/// --output's video for one stream: `output` with one source, else
/// <stem>_<source><ext> (spec R6).
inline std::string StreamVideoPath(const std::string& output, const std::string& source,
                                   bool several) {
    return several ? TaggedPath(output, source) : output;
}

/// One frame's --output image. One source: `output` for a single-frame
/// source, else NumberedPath(output, index), as before SP2. Several:
/// <stem>_<source>_<index6><ext>, always numbered (spec R6).
inline std::string StreamImagePath(const std::string& output, const std::string& source,
                                   bool several, bool single_frame, std::size_t index) {
    if (several) return NumberedPath(TaggedPath(output, source), index);
    return single_frame ? output : NumberedPath(output, index);
}

/// "multi_model_graph" with one source; "multi_model_graph: <source>" with
/// several. The CLI no longer opens a window per stream: with several
/// sources it shows one window, "multi_model_graph", holding TileFrames'
/// grid, so one capture of it shows every stream.
inline std::string WindowName(const std::string& source, bool several) {
    return several ? "multi_model_graph: " + source : std::string("multi_model_graph");
}

/// --display with several sources: the grid's columns (width) and rows
/// (height) for `count` cells. 2 side by side, 3-4 in 2x2, more in
/// ceil(sqrt(count)) columns and as many rows as they need.
inline cv::Size GridShape(std::size_t count) {
    if (count <= 1) return cv::Size(1, 1);
    int columns = 1;
    while (static_cast<std::size_t>(columns) * columns < count) ++columns;
    const int rows = static_cast<int>((count + columns - 1) / columns);
    return cv::Size(columns, rows);
}

/// The cells of GridShape(count), left to right then top to bottom, each
/// `cell` in size - all scaled down alike (aspect kept) when the grid
/// would be wider or taller than `max_canvas`. *canvas: the grid's size.
inline std::vector<cv::Rect> GridCells(std::size_t count, cv::Size cell, cv::Size max_canvas,
                                       cv::Size* canvas) {
    const cv::Size shape = GridShape(count);
    double scale = 1.0;
    if (cell.width * shape.width > max_canvas.width) {
        scale = static_cast<double>(max_canvas.width) / (cell.width * shape.width);
    }
    if (cell.height * shape.height * scale > max_canvas.height) {
        scale = static_cast<double>(max_canvas.height) / (cell.height * shape.height);
    }
    const cv::Size scaled(std::max(1, static_cast<int>(cell.width * scale)),
                          std::max(1, static_cast<int>(cell.height * scale)));
    std::vector<cv::Rect> cells;
    for (std::size_t i = 0; i < count; ++i) {
        const int column = static_cast<int>(i % shape.width);
        const int row = static_cast<int>(i / shape.width);
        cells.push_back(cv::Rect(column * scaled.width, row * scaled.height, scaled.width,
                                 scaled.height));
    }
    if (canvas != NULL) *canvas = cv::Size(scaled.width * shape.width, scaled.height * shape.height);
    return cells;
}

/// The window's image with several sources: frame i (each stream's latest
/// rendered frame) fitted into cell i of GridCells(frames.size(), cell,
/// max_canvas), letterboxed in black when its aspect differs, and names[i]
/// (its source id) written at the cell's top left. An empty frame (a
/// stream with no frame yet) leaves its cell dark grey, still labelled.
inline cv::Mat TileFrames(const std::vector<cv::Mat>& frames,
                          const std::vector<std::string>& names, cv::Size cell,
                          cv::Size max_canvas = cv::Size(1920, 1080)) {
    cv::Size size;
    const std::vector<cv::Rect> cells = GridCells(frames.size(), cell, max_canvas, &size);
    cv::Mat canvas = cv::Mat::zeros(size, CV_8UC3);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const cv::Rect& box = cells[i];
        cv::Mat target = canvas(box);
        if (frames[i].empty()) {
            target.setTo(cv::Scalar(48, 48, 48));
        } else {
            const double scale = std::min(static_cast<double>(box.width) / frames[i].cols,
                                          static_cast<double>(box.height) / frames[i].rows);
            const cv::Size fitted(
                std::max(1, std::min(box.width, static_cast<int>(frames[i].cols * scale + 0.5))),
                std::max(1, std::min(box.height, static_cast<int>(frames[i].rows * scale + 0.5))));
            cv::Mat resized;
            cv::resize(frames[i], resized, fitted, 0, 0, cv::INTER_AREA);
            resized.copyTo(target(cv::Rect((box.width - fitted.width) / 2,
                                           (box.height - fitted.height) / 2, fitted.width,
                                           fitted.height)));
        }
        if (i < names.size()) {
            // Readable once a window scales the grid down: larger in larger cells.
            const double font = std::max(0.6, box.height / 500.0);
            const int thickness = font >= 1.0 ? 2 : 1;
            int baseline = 0;
            const cv::Size text = cv::getTextSize(names[i], cv::FONT_HERSHEY_SIMPLEX, font,
                                                  thickness, &baseline);
            cv::rectangle(target, cv::Rect(0, 0, std::min(box.width, text.width + 12),
                                           std::min(box.height, text.height + baseline + 10)),
                          cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(target, names[i], cv::Point(6, text.height + 5), cv::FONT_HERSHEY_SIMPLEX,
                        font, cv::Scalar(255, 255, 255), thickness);
        }
    }
    return canvas;
}

/// One --output-layers file: <stem>[_<source>]_<frame, 6 digits>_<part>.png,
/// `part` a node id, "base" (the clean frame) or "events" (the panel).
inline std::string LayerFileName(const std::string& stem, const std::string& source,
                                 bool several, std::size_t frame, const std::string& part) {
    std::ostringstream name;
    name << stem << (several ? "_" + source : std::string()) << "_" << std::setw(6)
         << std::setfill('0') << frame << "_" << part << ".png";
    return name.str();
}

/// The <stem> of --output-layers' files: --output's file name without its
/// extension, or the graph file's when there is no --output.
inline std::string LayersStem(const std::string& output, const std::string& graph_path) {
    const std::string& path = output.empty() ? graph_path : output;
    const std::size_t slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const std::size_t dot = name.rfind('.');
    if (dot != std::string::npos && dot > 0) name = name.substr(0, dot);
    return name.empty() ? std::string("frame") : name;
}

/// The prefixes consumer::OpenInput dispatches on.
inline bool IsLiveUri(const std::string& uri) {
    return uri.compare(0, 7, "camera:") == 0 || uri.compare(0, 7, "rtsp://") == 0;
}

/// "" when --output, the effective source uri and --frames go together;
/// otherwise the usage error for image output from a live source.
inline std::string OutputProblem(const std::string& output, const std::string& uri,
                                 std::size_t frames) {
    if (ClassifyOutput(output) != OutputKind::kImagePerFrame || frames != 0 ||
        !IsLiveUri(uri)) {
        return std::string();
    }
    return "--output " + output + " writes one image per frame, and \"" + uri +
           "\" is a live source that never ends: write a video instead (--output "
           "<stem>.mkv, .avi or .mp4) or stop after N frames (--frames N)";
}

/// .avi: MJPG. .mp4, .mkv: mp4v.
inline int VideoFourcc(const std::string& path) {
    return LowerExtension(path) == ".avi" ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                                          : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
}

/// The four characters of a fourcc code, for messages.
inline std::string FourccText(int fourcc) {
    std::string text(4, ' ');
    for (int i = 0; i < 4; ++i) text[i] = static_cast<char>((fourcc >> (8 * i)) & 0xFF);
    return text;
}

/// Why the writer for `path` did not open. The directory was checked before
/// the run (OutputDirectoryProblem), so what is left is a directory that
/// refuses the file or a codec this OpenCV cannot encode. ".avi" is offered
/// only when it is not what already failed.
inline std::string VideoOpenError(const std::string& path, int fourcc) {
    std::string text = "could not open a video writer for " + path + " (fourcc " +
                       FourccText(fourcc) + "): the file cannot be created there, or "
                       "this OpenCV build cannot encode it";
    if (LowerExtension(path) != ".avi") text += "; try .avi";
    return text;
}

/// The directory `path` would be written into: "." when it names none.
inline std::string ParentDirectory(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return ".";
    if (slash == 0) return path.substr(0, 1);
    return path.substr(0, slash);
}

/// "" when the directory --output writes into exists; otherwise the error.
/// Images and video alike: every file the run writes goes there.
inline std::string OutputDirectoryProblem(const std::string& path) {
    if (path.empty()) return std::string();
    const std::string directory = ParentDirectory(path);
    struct stat info;
    if (::stat(directory.c_str(), &info) != 0) {
        return "could not write " + path + ": directory " + directory + " does not exist";
    }
    if ((info.st_mode & S_IFMT) != S_IFDIR) {
        return "could not write " + path + ": " + directory + " is not a directory";
    }
    return std::string();
}

/// The source's fps when it is finite and within [1, 240], else 30.
inline double VideoFps(double source_fps) {
    return (std::isfinite(source_fps) && source_fps >= 1.0 && source_fps <= 240.0)
               ? source_fps : 30.0;
}

/// q, Q or ESC in cv::waitKey's result (the low byte; the bits above it are
/// modifiers on some backends). -1 is "no key".
inline bool IsStopKey(int key) {
    if (key < 0) return false;
    const int low = key & 0xFF;
    return low == 'q' || low == 'Q' || low == 27;
}

/// Whether a HighGUI window can open without Qt aborting the process.
inline bool DisplayAvailable() {
#ifdef _WIN32
    return true;
#else
    const char* names[] = {"DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"};
    for (std::size_t i = 0; i < 3; ++i) {
        const char* value = std::getenv(names[i]);
        if (value != NULL && *value != '\0') return true;
    }
    return false;
#endif
}

/// "DISPLAY=:99, QT_QPA_PLATFORM=xcb": the window-system variables that are
/// set, for the message when DisplayOpens() says no.
inline std::string DisplayEnvironment() {
    const char* names[] = {"DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"};
    std::string text;
    for (std::size_t i = 0; i < 3; ++i) {
        const char* value = std::getenv(names[i]);
        if (value == NULL || *value == '\0') continue;
        if (!text.empty()) text += ", ";
        text += std::string(names[i]) + "=" + value;
    }
    return text;
}

/**
 * @brief Whether a HighGUI window opens here, without risking this process.
 *
 * A child opens and destroys one window and _exit(0)s. Qt's qFatal kills
 * the child instead of the run; a cv::Exception is caught and still exits
 * 0, because the in-process DisplayWindow turns that into its one warning.
 * The child's output is discarded: Qt's own text ("Reinstalling the
 * application may fix this problem") points the wrong way. A child that
 * has not finished in 10 s is killed and counts as no display. When the
 * probe itself cannot run (fork fails) the answer is yes, as before.
 * Called before any model loads, so no NPU work is in flight.
 */
inline bool DisplayOpens() {
#ifdef _WIN32
    return true;
#else
    std::fflush(stdout);  // or the child's copy of the buffers is written twice
    std::fflush(stderr);
    const pid_t child = ::fork();
    if (child < 0) return true;
    if (child == 0) {
        const int null_fd = ::open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
            ::dup2(null_fd, 1);
            ::dup2(null_fd, 2);
        }
        try {
            prepareGuiBackend();
            cv::namedWindow("multi_model_graph_probe", cv::WINDOW_NORMAL);
            cv::destroyWindow("multi_model_graph_probe");
        } catch (...) {
        }
        ::_exit(0);
    }
    int status = 0;
    for (int waited_ms = 0;; waited_ms += 10) {
        const pid_t done = ::waitpid(child, &status, WNOHANG);
        if (done == child) break;
        if (done < 0 && errno != EINTR) return true;  // not ours to judge
        if (waited_ms >= 10000) {
            ::kill(child, SIGKILL);
            ::waitpid(child, &status, 0);
            return false;
        }
        struct timespec slice = {0, 10 * 1000 * 1000};
        ::nanosleep(&slice, NULL);
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

/// --output <stem>.mp4|.avi|.mkv: one video of every rendered frame. Opens
/// on the first frame; the destructor (or Close) finalizes the container.
class VideoOutput {
 public:
    VideoOutput(const std::string& path, double source_fps)
        : path_(path), fps_(VideoFps(source_fps)), opened_(false), frames_(0) {}
    ~VideoOutput() { Close(); }

    /// False (with *error set) when the writer cannot open or is closed.
    bool Write(const cv::Mat& frame, std::string* error) {
        if (!opened_) {
            opened_ = true;
            size_ = frame.size();
            const int fourcc = VideoFourcc(path_);
            if (!writer_.open(path_, fourcc, fps_, size_)) {
                *error = VideoOpenError(path_, fourcc);
                return false;
            }
        }
        if (!writer_.isOpened()) {
            *error = "could not write " + path_;
            return false;
        }
        if (frame.size() == size_) {
            writer_.write(frame);
        } else {
            cv::Mat resized;  // VideoWriter silently drops a frame of another size
            cv::resize(frame, resized, size_);
            writer_.write(resized);
        }
        ++frames_;
        return true;
    }

    void Close() {
        if (writer_.isOpened()) writer_.release();
    }
    std::size_t frames_written() const { return frames_; }

 private:
    VideoOutput(const VideoOutput&) = delete;
    VideoOutput& operator=(const VideoOutput&) = delete;
    cv::VideoWriter writer_;
    std::string path_;
    double fps_;
    bool opened_;
    cv::Size size_;
    std::size_t frames_;
};

/// --display: shows each rendered frame in the window `name` (WindowName);
/// the destructor destroys the window.
class DisplayWindow {
 public:
    explicit DisplayWindow(const std::string& name = "multi_model_graph")
        : opened_(false), broken_(false), name_(name) {}
    ~DisplayWindow() {
        if (!opened_) return;
        try {
            cv::destroyWindow(name_);
        } catch (const cv::Exception&) {
        }
    }

    /// True when q, Q or ESC was pressed.
    bool Show(const cv::Mat& canvas) {
        if (broken_) return false;
        try {
            if (!opened_) {
                prepareGuiBackend();
                cv::namedWindow(name_, cv::WINDOW_NORMAL);
                opened_ = true;
            }
            cv::imshow(name_, canvas);
            return IsStopKey(cv::waitKey(1));
        } catch (const cv::Exception& error) {
            std::fprintf(stderr, "warning: --display is off for the rest of this run: %s\n",
                         error.what());
            broken_ = true;
            return false;
        }
    }

 private:
    DisplayWindow(const DisplayWindow&) = delete;
    DisplayWindow& operator=(const DisplayWindow&) = delete;
    bool opened_;
    bool broken_;
    std::string name_;
};

}  // namespace cli
}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_OUTPUT_HPP
