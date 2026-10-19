/**
 * @file common_util.hpp
 * @brief Common utility functions and macros
 */

#ifndef DXAPP_COMMON_UTIL_HPP
#define DXAPP_COMMON_UTIL_HPP

#include <dxrt/device_info_status.h>
#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <array>
#include <cstring>
#include <utility>

#include "common/utility/interrupt_flag.hpp"
#include "common/utility/display_pump.hpp"
#include "common/utility/camera.hpp"
#include "common/utility/repo_path.hpp"

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// Color codes for console output
static constexpr const char* DXAPP_RED    = "\033[1;31m";
static constexpr const char* DXAPP_YELLOW = "\033[1;33m";
static constexpr const char* DXAPP_GREEN  = "\033[1;32m";
static constexpr const char* DXAPP_RESET  = "\033[0m";

// Logging macros
#define LOG_INFO(msg) std::cout << "[DXAPP] [INFO] " << msg << std::endl
#define LOG_WARN(msg) std::cout << DXAPP_YELLOW << "[DXAPP] [WARN] " << msg << DXAPP_RESET << std::endl
#define LOG_ERROR(msg) std::cerr << DXAPP_RED << "[DXAPP] [ERROR] " << msg << DXAPP_RESET << std::endl

#include <stdexcept>

namespace dxapp {

/**
 * @brief Replace spaces with underscores in a string (for pipeline-parseable output).
 * @param name  Input string
 * @return Sanitized copy
 */
inline std::string sanitize_name(const std::string& name) {
    std::string s = name;
    std::replace(s.begin(), s.end(), ' ', '_');
    return s;
}

/**
 * @brief Terminate with a descriptive error (replaces exit(1) for RAII safety).
 *
 * The thrown exception propagates to the DXRT_TRY_CATCH_END guard,
 * ensuring all destructors run before the process exits.
 *
 * @param msg Error message (printed by the top-level catch)
 */
[[noreturn]] inline void fatal_error(const std::string& msg) {
    throw std::runtime_error(msg);
}

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @return Sigmoid output
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Softmax function for a vector
 * @param input Input vector
 * @return Softmax output vector
 */
inline std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }
    
    return output;
}

/**
 * @brief Get argmax of a float array
 * @param data Pointer to float array
 * @param size Array size
 * @return Index of maximum value
 */
inline int argmax(const float* data, int size) {
    int max_idx = 0;
    float max_val = data[0];
    
    for (int i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/**
 * @brief Check if file exists
 * @param path File path
 * @return true if file exists
 */
inline bool fileExists(const std::string& path) {
    return fs::exists(path);
}

/**
 * @brief Get file extension in lowercase
 * @param path File path
 * @return Lowercase extension without dot
 */
inline std::string getFileExtension(const std::string& path) {
    size_t dot_pos = path.rfind('.');
    if (dot_pos == std::string::npos) return "";
    
    std::string ext = path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ext;
}

/**
 * @brief Compare two version strings (e.g., "3.0.0" >= "3.0.0")
 * @param v1 First version string
 * @param v2 Second version string
 * @return true if v1 >= v2
 */
inline bool isVersionGreaterOrEqual(const std::string& v1, const std::string& v2) {
    std::istringstream s1(v1), s2(v2);
    int num1 = 0, num2 = 0;
    char dot;

    while (s1.good() || s2.good()) {
        if (s1.good()) s1 >> num1;
        if (s2.good()) s2 >> num2;

        if (num1 < num2) return false;
        if (num1 > num2) return true;

        num1 = num2 = 0;
        if (s1.good()) s1 >> dot;
        if (s2.good()) s2 >> dot;
    }
    return true;
}

/**
 * @brief Check minimum version compatibility for RT and Compiler
 * 
 * Validates that the DXRT library version is >= 3.5.0 and
 * the compiled model version is >= v7 (matching Legacy behavior).
 * Model Zoo 2_5_0 .dxnn files are container format v9, which DX-RT 3.5.0 parses.
 * 
 * @param ie Pointer to InferenceEngine
 * @return true if versions are compatible
 */
inline bool minversionforRTandCompiler(dxrt::InferenceEngine* ie) {
    if (!ie) return false;

    std::string rt_version = dxrt::Configuration::GetInstance().GetVersion();
    std::string compiler_version = ie->GetModelVersion();

    if (isVersionGreaterOrEqual(rt_version, "3.5.0")) {
        if (isVersionGreaterOrEqual(compiler_version, "v7")) {
            return true;
        } else {
            std::cerr << "[DXAPP] [ERROR] Compiler version is too low. (required: "
                         ">= 7, current: "
                      << compiler_version << ")" << std::endl;
            std::cerr << DXAPP_GREEN << "[HINT] Model/compiler version mismatch. "
                         "Please download updated models: python3 scripts/download_models.py --models <model_name>"
                      << DXAPP_RESET << std::endl;
        }
    } else {
        std::cerr << "[DXAPP] [ERROR] DXRT library version is too low. (required: "
                     ">= 3.5.0, current: "
                  << rt_version << ")" << std::endl;
        std::cerr << DXAPP_GREEN << "[HINT] Model Zoo 2_5_0 .dxnn files are container v9. "
                     "Update DX-RT (and the NPU driver) to >= 3.5.0."
                  << DXAPP_RESET << std::endl;
    }
    return false;
}

/** Save frame to path specified by DXAPP_SAVE_IMAGE env var (debug/test hook). */
inline void saveDebugImage(const cv::Mat& frame) {
    const char* path = std::getenv("DXAPP_SAVE_IMAGE");
    if (path && *path && !frame.empty()) cv::imwrite(path, frame);
}

/** "<stem>_output_only<ext>" sibling of `path` ("" in -> "" out). */
inline std::string outputOnlyPath(const std::string& path) {
    if (path.empty()) return "";
    std::size_t dot = path.find_last_of('.');
    return (dot == std::string::npos)
        ? path + "_output_only"
        : path.substr(0, dot) + "_output_only" + path.substr(dot);
}

/**
 * @brief Save the panel-free output next to BOTH the run-dir image and the
 *        caller's DXAPP_SAVE_IMAGE image.
 *
 * `runDirPath` is the runner's own per-image output path ("" when not saving).
 * The env path belongs to the caller (CI, AI Studio) and is never overwritten
 * by the runner, so both files are produced when both are requested.
 */
inline void saveOutputOnlyImage(const std::string& runDirPath, const cv::Mat& frame) {
    if (frame.empty()) return;
    const std::string run_out = outputOnlyPath(runDirPath);
    if (!run_out.empty()) cv::imwrite(run_out, frame);
    const char* env = std::getenv("DXAPP_SAVE_IMAGE");
    if (env && *env) {
        const std::string env_out = outputOnlyPath(env);
        if (env_out != run_out) cv::imwrite(env_out, frame);
    }
}

/**
 * @brief Shared flag: true once the display window has been closed by the user.
 *
 * Once set, showOutput() will no longer recreate the window, and
 * windowShouldClose() will return true immediately.
 */
inline bool& _displayClosed() {
    static bool closed = false;
    return closed;
}

/**
 * @brief Handle window events and check whether the display window was closed or user requested quit.
 *
 * - Pressing 'q' sets the global interrupt flag and returns true.
 * - A pending interrupt (g_interrupted(), e.g. from SIGINT/SIGTERM) returns true.
 * - If the window named `winname` is closed (getWindowProperty <= 0), this returns true
 *   to signal the caller to stop displaying and proceed to next model.
 */
inline bool windowShouldClose(const std::string& winname = "Output") {
    if (_displayClosed()) return true;
    // SIGINT/SIGTERM (run_dir.hpp's handler) end the wait too: the image-mode
    // "hold the window open" loop in every sync runner is
    //     while (!windowShouldClose("Output")) sleep;
    // and a window nobody can close (QT_QPA_PLATFORM=offscreen, a remote
    // session) otherwise kept the process alive after the signal.
    if (g_interrupted().load()) return true;
    try {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            _displayClosed() = true;
            g_interrupted().store(true);
            return true;
        }
    } catch (const cv::Exception&) {
        // Backend throws if no window exists (e.g. Qt)
        _displayClosed() = true;
        return true;
    }
    // getWindowProperty returns -1 when window was destroyed (user closed),
    // 0 during initial creation on some backends, and 1 when fully visible.
    // Some backends (e.g. GTK2) always return -1 even for valid windows,
    // so we probe once and disable the check if the backend doesn't support it.
    static bool probed = false;
    static bool prop_supported = true;
    if (!probed) {
        probed = true;
        try {
            double probe = cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE);
            if (probe < -0.5) {
                prop_supported = false;
            }
        } catch (const cv::Exception&) {
            prop_supported = false;
        }
    }
    if (prop_supported) {
        try {
            double visible = cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE);
            if (visible <= 0.0) {
                _displayClosed() = true;
                return true;
            }
        } catch (const cv::Exception&) {
            _displayClosed() = true;
            return true;
        }
    }
    return false;
}

/**
 * @brief Resize image to exactly max_w × max_h with letterbox padding.
 *        Scales down to fit within max_w×max_h (aspect-ratio preserved), then
 *        pads with black bars so the output is always exactly max_w×max_h.
 *        This ensures the result always matches the VideoWriter's declared frame size.
 * @param src Input image
 * @param dst Output image (always max_w × max_h)
 * @param max_w Output width (default 960)
 * @param max_h Output height (default 540)
 */
inline void displayResize(const cv::Mat &src, cv::Mat &dst, int max_w = 960, int max_h = 540) {
    (void)max_w; (void)max_h;
    dst = src;
}

/**
 * @brief Query the primary screen resolution.
 *
 * Tries (in order):
 *   1. Environment variables DXAPP_SCREEN_W / DXAPP_SCREEN_H
 *   2. xdpyinfo (X11) parsing "dimensions: WxH"
 * Falls back to 1920×1080 if detection fails.
 */
inline std::pair<int, int> getScreenResolution() {
    // 1. Env override
    const char* env_w = std::getenv("DXAPP_SCREEN_W");
    const char* env_h = std::getenv("DXAPP_SCREEN_H");
    if (env_w && env_h) {
        int w = std::atoi(env_w);
        int h = std::atoi(env_h);
        if (w > 0 && h > 0) return {w, h};
    }
#ifndef _WIN32
    // 2. xdpyinfo
    FILE* pipe = popen("xdpyinfo 2>/dev/null | grep dimensions", "r");
    if (pipe) {
        char buf[256];
        if (fgets(buf, sizeof(buf), pipe)) {
            int w = 0, h = 0;
            if (sscanf(buf, " dimensions: %dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                pclose(pipe);
                return {w, h};
            }
        }
        pclose(pipe);
    }
#endif
    return {1920, 1080};
}

/**
 * @brief Apply the run-time display choices to a DisplayPump, once per run.
 *
 * - The window is sized on creation like the Python runners' show_image():
 *   fitted into half the screen (getScreenResolution(), 1920x1080 fallback).
 * - `--drop-frames` turns the depth-1 lossy preview back on; by default the
 *   pump is lossless and a slow window paces the pipeline (2026-10-06).
 */
inline void configureDisplayPump(DisplayPump& pump, bool drop_frames) {
    const std::pair<int, int> screen = getScreenResolution();
    pump.setScreenSize(screen.first, screen.second);
    pump.setDropStale(drop_frames);
}

/**
 * @brief Display frame in a resizable window, sized to ~1/4 screen area on first call.
 *
 * On the first frame, detects screen resolution and sets the window to
 * half-screen width × half-screen height, preserving the frame's aspect ratio.
 * Subsequent frames reuse the same window without re-querying.
 */
inline void showOutput(const cv::Mat& frame) {
    if (_displayClosed()) return;

    static bool window_ever_opened = false;
    static bool headless_warned = false;

    // After the window has been opened at least once, process pending GUI
    // events (e.g. X-button close) and verify it is still alive BEFORE
    // calling namedWindow/imshow which would recreate a destroyed window.
    if (window_ever_opened) {
        int key = -1;
        try { key = cv::waitKey(1); } catch (const cv::Exception&) {
            _displayClosed() = true;
            return;
        }
        if (key == 'q' || key == 27) {
            _displayClosed() = true;
            g_interrupted().store(true);
            return;
        }
        // Some backends (e.g. GTK2) return -1 for WND_PROP_VISIBLE even
        // when the window is alive.  Probe once and disable the check if
        // the backend does not support it — same guard as windowShouldClose().
        static bool show_probed = false;
        static bool show_prop_supported = true;
        if (!show_probed) {
            show_probed = true;
            try {
                double probe = cv::getWindowProperty("Output", cv::WND_PROP_VISIBLE);
                if (probe < -0.5) {
                    show_prop_supported = false;
                }
            } catch (const cv::Exception&) {
                show_prop_supported = false;
            }
        }
        if (show_prop_supported) {
            try {
                double v = cv::getWindowProperty("Output", cv::WND_PROP_VISIBLE);
                if (v <= 0.0) { _displayClosed() = true; return; }
            } catch (const cv::Exception&) {
                _displayClosed() = true;
                return;
            }
        }
    }

    static bool window_sized = false;
    try {
        prepareGuiBackend();
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
    } catch (const cv::Exception& e ) {
        if (!headless_warned) {
            std::cerr << DXAPP_YELLOW
                      << "[DXAPP] [WARN] Display not available. Use --no-display for headless mode."
                      << DXAPP_RESET << std::endl;
            headless_warned = true;
        }
        _displayClosed() = true;
        return;
    }
    window_ever_opened = true;

    if (!window_sized && !frame.empty()) {
        const std::pair<int, int> screen_res = getScreenResolution();
        int target_w = screen_res.first / 2;
        int target_h = screen_res.second / 2;

        // Fit frame aspect ratio within target_w × target_h
        double scale = std::min(
            static_cast<double>(target_w) / frame.cols,
            static_cast<double>(target_h) / frame.rows);
        int win_w = static_cast<int>(frame.cols * scale);
        int win_h = static_cast<int>(frame.rows * scale);

        cv::resizeWindow("Output", win_w, win_h);
        window_sized = true;
    }

    cv::imshow("Output", frame);
}

/**
 * @brief Write frame to video, resizing to the writer's frame size first.
 *
 * `expected_w`/`expected_h` MUST be the size the writer was opened with (they
 * have no defaults on purpose). cv::VideoWriter SILENTLY discards any frame
 * whose size differs from that, and `writer.get(CAP_PROP_FRAME_WIDTH/HEIGHT)`
 * returns 0 on several OpenCV builds (e.g. the GStreamer backend), so a
 * size-less call used to produce an empty video with no error at all.
 */
inline void writeToVideo(cv::VideoWriter& writer, const cv::Mat& frame,
                         int expected_w, int expected_h) {
    if (!writer.isOpened() || frame.empty()) return;
    int w = static_cast<int>(writer.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(writer.get(cv::CAP_PROP_FRAME_HEIGHT));
    // CAP_PROP_FRAME_WIDTH/HEIGHT may return 0 on some OpenCV builds
    if (w <= 0 || h <= 0) { w = expected_w; h = expected_h; }
    if (w <= 0 || h <= 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[DXAPP] [WARN] Output video frame size is unknown; "
                         "frames may be dropped by OpenCV." << std::endl;
        }
        writer << frame;
    } else if (frame.cols == w && frame.rows == h) {
        writer << frame;
    } else {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(w, h));
        writer << resized;
    }
}

/**
 * @brief Build a per-image save path under a run directory.
 *
 * If `runDir` is empty, returns empty string. If `imagePath` is a file,
 * uses its filename as a subdirectory name to avoid collisions when saving
 * multiple images from the same source directory.
 */
/**
 * @brief Resolve a factory's companion .dxnn files beside the primary model.
 *
 * `declared` is what the factory's getCompanionModels() returned: (role, filename)
 * pairs, where a relative filename is resolved in the primary model's own directory --
 * how a model set stays together on disk.
 *
 * A missing companion throws. Falling back to the primary alone would produce a
 * heatmap that still looks like a heatmap while meaning something else entirely, which
 * is exactly the failure a single-network "EfficientAD" example already demonstrated.
 *
 * Mirrors resolve_companion_models() in the Python runner.
 */
inline std::vector<std::pair<std::string, std::string>> resolveCompanionModels(
    const std::vector<std::pair<std::string, std::string>>& declared,
    const std::string& primaryPath) {
    std::vector<std::pair<std::string, std::string>> resolved;
    if (declared.empty()) return resolved;

    const fs::path base = fs::absolute(fs::path(primaryPath)).parent_path();
    for (const auto& item : declared) {
        fs::path candidate(item.second);
        if (!candidate.is_absolute()) candidate = base / candidate;
        if (!fs::is_regular_file(candidate)) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] companion model for role '" << item.first
                << "' not found: " << candidate.string() << "\n"
                << "  This factory needs it alongside the primary model ("
                << fs::path(primaryPath).filename().string() << ").\n"
                << "  This model set must be downloaded together; running the primary "
                   "network alone would produce a different measurement under the same "
                   "name.\n";
            throw std::runtime_error(msg.str());
        }
        resolved.emplace_back(item.first, candidate.string());
    }
    return resolved;
}

inline std::string buildPerImageSavePath(const std::string& runDir,
                                        const std::string& modelName,
                                        const std::string& imagePath,
                                        int img_idx = 0,
                                        bool createDirs = true) {
    if (runDir.empty()) return std::string();
    fs::path base(runDir);
    fs::path model_dir = base / (modelName);

    fs::path fname = fs::path(imagePath).filename();
    std::string stem = fname.stem().string();
    if (stem.empty()) stem = "image" + std::to_string(img_idx);
    fs::path out = model_dir / (stem + std::string("_output.jpg"));
    // `createDirs = false` is for the callers whose visualizer may legitimately render
    // nothing -- the embedding comparison, whose first image is only the reference.
    // Creating the directory up front left an EMPTY <model>_sync/ next to run_info.txt,
    // which reads exactly like an output that failed to be written.
    if (createDirs) fs::create_directories(out.parent_path());
    return out.string();
}

/**
 * @brief Normalise a model name to lowercase alphanumerics.
 *
 * Factories report model names in inconsistent casing ("ESPCN-x4", "Espcn_x3",
 * "Realesrgan X4"), so per-model lookups match against this normalised form.
 */
inline std::string normalizeModelKey(const std::string& modelName) {
    std::string key;
    key.reserve(modelName.size());
    for (char c : modelName) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) key += static_cast<char>(std::tolower(uc));
    }
    return key;
}

/**
 * @brief Get default sample image path for a given task type.
 *
 * When no input source is specified, returns a bundled sample image
 * appropriate for the task. `modelName` (optional) lets a model override the
 * task default: super-resolution needs a genuinely low-resolution input, and
 * the right size depends on the scale factor — ESPCN (x2/x3/x4) reads a
 * 275x150 crop, Real-ESRGAN (x2/x4/x8) a smaller 165x90 one so the x8 output
 * stays a sane size.
 */
inline std::string getDefaultSampleImage(const std::string& taskType,
                                         const std::string& modelName = "") {
    if (taskType == "super_resolution") {
        const std::string key = normalizeModelKey(modelName);
        if (key.compare(0, 10, "realesrgan") == 0) return "sample/img/sample_lowres165x90.png";
        return "sample/img/sample_lowres275x150.png";
    }
    if (taskType == "object_detection")       return "sample/img/sample_street.jpg";
    if (taskType == "mainbody_detection")     return "sample/img/sample_street.jpg";
    if (taskType == "face_detection")         return "sample/img/sample_face.jpg";
    if (taskType == "obb_detection")          return "sample/img/sample_airport_satellite_view.png";
    if (taskType == "pose_estimation")        return "sample/img/sample_people.jpg";
    if (taskType == "hand_landmark")          return "sample/img/sample_hand.jpg";
    if (taskType == "hand_detection")         return "sample/img/sample_hand.jpg";
    if (taskType == "face_alignment")         return "sample/img/sample_face_a1.jpg";
    if (taskType == "instance_segmentation")  return "sample/img/sample_street.jpg";
    if (taskType == "semantic_segmentation")  return "sample/img/sample_parking.jpg";
    if (taskType == "classification" || taskType == "image_classification")
        return "sample/img/sample_dog.jpg";
    if (taskType == "zero_shot_image_classification") return "sample/img/sample_dog.jpg";
    if (taskType == "oriented_object_detection") return "sample/img/sample_airport_satellite_view.png";
    if (taskType == "face_recognition")       return "sample/img/face_pair";
    if (taskType == "face_landmark")          return "sample/img/sample_face_a1.jpg";
    if (taskType == "face_attribute" || taskType == "person_attribute")
        return "sample/img/sample_person_a1.jpg";
    if (taskType == "low_light_enhancement")  return "sample/img/sample_lowlight.jpg";
    if (taskType == "object_pose_estimation") return "sample/dope/000000.png";
    if (taskType == "anomaly_detection")      return "sample/img/sample_parking.jpg";
    if (taskType == "zero_shot_instance_segmentation") return "sample/img/sample_street.jpg";
    if (taskType == "depth_estimation")       return "sample/img/sample_parking.jpg";
    if (taskType == "image_denoising")        return "sample/img/sample_denoising.jpg";
    if (taskType == "image_enhancement")      return "sample/img/sample_lowlight.jpg";
    if (taskType == "embedding")              return "sample/img/face_pair";
    // The four task categories DX Model Zoo added in 2_5_0. The three retrieval ones
    // rank a query against a committed gallery (sample/gallery/*.bin), so their default
    // is a query held OUT of that gallery -- a query that is also a gallery member
    // scores a meaningless 1.0000 self-match.
    if (taskType == "image_retrieval")        return "sample/img/sample_person_a2.jpg";
    if (taskType == "visual_place_recognition") return "sample/vpr/queries/q1.jpg";
    if (taskType == "person_reid")            return "sample/reid/queries/sample_person_a2.jpg";
    // Trimap-free matting needs a subject separable from its background; measured, the
    // beach portrait mattes cleanly where a white shirt on a white backdrop does not.
    if (taskType == "image_matting")          return "sample/img/sample_person_b.jpg";
    if (taskType == "attribute_recognition")  return "sample/img/sample_person_a1.jpg";
    if (taskType == "reid")                   return "sample/img/person_pair";
    if (taskType == "ppu")                    return "sample/img/sample_street.jpg";
    if (taskType == "3d_object_detection")
        return "sample/kitti/velodyne/000049.bin";
    return "sample/img/sample_street.jpg";
}

/**
 * @brief Get default sample video path for a given task type.
 *
 * Returns empty string for image-only tasks (embedding, attribute_recognition, reid).
 */
/// True for the tasks that compare one image against a stored set, so a video or
/// camera source has nothing to compare frame-by-frame. Defined once here because four
/// runners asked the same question and had drifted to four copies of the answer.
inline bool isComparisonOnlyTask(const std::string& taskType) {
    return taskType == "embedding" || taskType == "reid"
        || taskType == "attribute_recognition" || taskType == "image_retrieval"
        || taskType == "visual_place_recognition" || taskType == "person_reid"
        || taskType == "face_recognition" || taskType == "face_attribute"
        || taskType == "person_attribute";
}

inline std::string getDefaultSampleVideo(const std::string& taskType) {
    if (isComparisonOnlyTask(taskType))       return "";
    if (taskType == "image_matting")          return "assets/videos/person-pair-hallway.mp4";
    if (taskType == "object_detection")       return "assets/videos/snowboard.mp4";
    if (taskType == "mainbody_detection")     return "assets/videos/snowboard.mp4";
    if (taskType == "face_detection")         return "assets/videos/dance-group.mov";
    if (taskType == "obb_detection")          return "assets/videos/obb.mp4";
    if (taskType == "pose_estimation")        return "assets/videos/dance-solo.mov";
    if (taskType == "hand_landmark")          return "assets/videos/hand.mp4";
    if (taskType == "hand_detection")         return "assets/videos/hand.mp4";
    if (taskType == "face_alignment")         return "assets/videos/face-alignment-closeup.mp4";
    if (taskType == "instance_segmentation")  return "assets/videos/dogs.mp4";
    if (taskType == "semantic_segmentation")  return "assets/videos/blackbox-city-road.mp4";
    if (taskType == "classification")         return "assets/videos/dogs.mp4";
    if (taskType == "depth_estimation")       return "assets/videos/blackbox-city-road.mp4";
    if (taskType == "image_denoising")        return "assets/videos/noisy_hand.mp4";
    if (taskType == "super_resolution")       return "assets/videos/lowres-drone-city-road.mp4";
    if (taskType == "image_enhancement")      return "assets/videos/lowlight.mp4";
    if (taskType == "ppu")                    return "assets/videos/snowboard.mp4";
    return "";  // image-only tasks (embedding, attribute_recognition, reid)
}

/**
 * @brief Attempt to auto-download a missing model via setup_sample_models.sh.
 * @return true if download succeeded and file now exists.
 */
inline bool autoDownloadModel(const std::string& modelPath,
                               const std::string& registryName = "") {
    // setup.sh matches the manifest display name, which lines up with model_name
    // (yolov8n), not the .dxnn stem (yolov8-n_640x640).
    std::string selector = registryName.empty()
        ? fs::path(modelPath).stem().string() : registryName;
    std::string modelsDir = fs::path(modelPath).parent_path().string();
    if (modelsDir.empty()) modelsDir = "./assets/models";
    std::cout << "[DXAPP] [INFO] Model not found: " << modelPath
              << " — attempting auto-download..." << std::endl;
    std::string cmd = "./setup_sample_models.sh --output=" + modelsDir
                    + " --models " + selector;
    int ret = std::system(cmd.c_str());
    return (ret == 0) && fs::exists(modelPath);
}

/**
 * @brief Attempt to auto-download sample videos via setup_sample_videos.sh.
 * @return true if download succeeded.
 */
inline bool autoDownloadVideos() {
    std::cout << "[DXAPP] [INFO] Videos not found — attempting auto-download..." << std::endl;
    int ret = std::system("./setup_sample_videos.sh --output=./assets/videos");
    return ret == 0;
}

/**
 * @brief Derive the example key from argv[0] by stripping the variant suffix.
 *
 * The build names each binary "<variant>_sync" / "<variant>_async"
 * (optionally with a "_cpp_postprocess" tail). The basename with that suffix
 * removed is the registry "variant" (for example yolov8-n_640x640), not the
 * legacy "model_name" (yolov8n).
 */
inline std::string exampleKeyFromArgv0(const std::string& argv0) {
    std::string name = fs::path(argv0).filename().string();
    const char* suffixes[] = {"_async_cpp_postprocess", "_sync_cpp_postprocess",
                              "_cpp_postprocess", "_async", "_sync"};
    for (const char* suf : suffixes) {
        size_t slen = std::strlen(suf);
        if (name.size() > slen && name.compare(name.size() - slen, slen, suf) == 0) {
            return name.substr(0, name.size() - slen);
        }
    }
    return name;
}

/**
 * @brief Resolve this example's default .dxnn path from model_registry.json.
 *
 * Looks the example key up as "variant" first, then "model_name". When
 * several rows match (an alias_of row shares its target's variant), the
 * canonical row (alias_of null) wins. dxnn_file sits before variant in each
 * object, so the match is bounded to the enclosing object. Returns
 * "assets/models/<dxnn_file>" plus the model_name used by
 * python3 scripts/download_models.py --models. path is empty when the key is absent.
 */
struct ExampleModelRef {
    std::string path;
    std::string modelName;
};

inline size_t findJsonStringField(const std::string& content, const std::string& field,
                                   const std::string& value, size_t from) {
    const std::string spaced = "\"" + field + "\": \"" + value + "\"";
    const std::string tight = "\"" + field + "\":\"" + value + "\"";
    size_t spacedPos = content.find(spaced, from);
    size_t tightPos = content.find(tight, from);
    if (spacedPos == std::string::npos) return tightPos;
    if (tightPos == std::string::npos) return spacedPos;
    return std::min(spacedPos, tightPos);
}

inline size_t enclosingObjectStart(const std::string& content, size_t pos) {
    int depth = 0;
    size_t index = pos;
    while (index > 0) {
        --index;
        if (content[index] == '}') {
            ++depth;
        } else if (content[index] == '{') {
            if (depth == 0) return index;
            --depth;
        }
    }
    return std::string::npos;
}

inline size_t enclosingObjectEnd(const std::string& content, size_t start) {
    int depth = 0;
    for (size_t index = start; index < content.size(); ++index) {
        if (content[index] == '{') {
            ++depth;
        } else if (content[index] == '}') {
            --depth;
            if (depth == 0) return index;
        }
    }
    return std::string::npos;
}

inline std::string jsonStringValue(const std::string& content, size_t fieldPos, size_t limit) {
    size_t colon = content.find(':', fieldPos);
    if (colon == std::string::npos || colon > limit) return "";
    size_t openQuote = content.find('"', colon + 1);
    if (openQuote == std::string::npos || openQuote > limit) return "";
    size_t closeQuote = content.find('"', openQuote + 1);
    if (closeQuote == std::string::npos || closeQuote > limit) return "";
    return content.substr(openQuote + 1, closeQuote - openQuote - 1);
}

// True unless the object [start, end] carries a non-null "alias_of".
inline bool isCanonicalRegistryRow(const std::string& content, size_t start, size_t end) {
    size_t field = content.find("\"alias_of\"", start);
    if (field == std::string::npos || field > end) return true;
    size_t colon = content.find(':', field);
    if (colon == std::string::npos || colon > end) return true;
    size_t value = content.find_first_not_of(" \t\r\n", colon + 1);
    return value != std::string::npos && content.compare(value, 4, "null") == 0;
}

// Bounds of the registry row whose `field` is `value`: the first canonical
// one, else the first. False when no row matches.
inline bool findRegistryRow(const std::string& content, const std::string& field,
                            const std::string& value, size_t& start, size_t& end) {
    bool found = false;
    for (size_t match = findJsonStringField(content, field, value, 0);
         match != std::string::npos;
         match = findJsonStringField(content, field, value, match + 1)) {
        size_t objectStart = enclosingObjectStart(content, match);
        size_t objectEnd = (objectStart == std::string::npos)
            ? std::string::npos : enclosingObjectEnd(content, objectStart);
        if (objectStart == std::string::npos || objectEnd == std::string::npos) continue;
        if (isCanonicalRegistryRow(content, objectStart, objectEnd)) {
            start = objectStart;
            end = objectEnd;
            return true;
        }
        if (!found) {
            start = objectStart;
            end = objectEnd;
            found = true;
        }
    }
    return found;
}

inline ExampleModelRef resolveExampleModel(const std::string& argv0) {
    ExampleModelRef ref;
    std::string key = exampleKeyFromArgv0(argv0);
    if (key.empty()) return ref;
    fs::path reg = fs::path(::dxapp::RuntimeRoot()) / "config" / "model_registry.json";
    std::ifstream registryFile(reg);
    if (!registryFile.is_open()) return ref;
    std::string content((std::istreambuf_iterator<char>(registryFile)),
                        std::istreambuf_iterator<char>());

    size_t objectStart = std::string::npos;
    size_t objectEnd = std::string::npos;
    if (!findRegistryRow(content, "variant", key, objectStart, objectEnd) &&
        !findRegistryRow(content, "model_name", key, objectStart, objectEnd)) {
        return ref;
    }

    size_t dxnnPos = content.find("\"dxnn_file\"", objectStart);
    if (dxnnPos == std::string::npos || dxnnPos > objectEnd) return ref;
    std::string dxnn = jsonStringValue(content, dxnnPos, objectEnd);
    if (dxnn.empty()) return ref;

    size_t namePos = content.find("\"model_name\"", objectStart);
    if (namePos != std::string::npos && namePos < objectEnd) {
        ref.modelName = jsonStringValue(content, namePos, objectEnd);
    }
    ref.path = "assets/models/" + dxnn;
    return ref;
}

inline std::string resolveDefaultModelPath(const std::string& argv0) {
    return resolveExampleModel(argv0).path;
}

/**
 * @brief Resolve and validate the model path (SDKREQ-529 policy).
 *
 * - `-m` omitted  : resolve this example's default model from the registry and
 *                   auto-download it if missing (convenience path).
 * - `-m <path>`   : an explicit path is a contract — if the file is missing we
 *                   error out immediately and do NOT run the auto-downloader.
 *
 * @param modelPath in/out — filled with the resolved default when `-m` omitted.
 * @param argv0     program path (argv[0]) used to resolve the default.
 */
inline void resolveAndValidateModel(std::string& modelPath, const std::string& argv0) {
    if (modelPath.empty()) {
        ExampleModelRef ref = resolveExampleModel(argv0);
        if (ref.path.empty()) {
            std::string key = exampleKeyFromArgv0(argv0);
            fatal_error("[DXAPP] [ERROR] Model path is required. Use -m or --model_path option.\n"
                "        -> Download:  python3 scripts/download_models.py --models " + key + "\n"
                "        -> Or use:    ./run_demo.sh  (auto-downloads demo models)\n"
                "Use -h or --help for usage information.");
        }
        modelPath = ref.path;
        std::cout << "[DXAPP] [INFO] No model specified (-m). Using example default: "
                  << modelPath << std::endl;
        if (!fileExists(modelPath)) {
            std::string selector = ref.modelName.empty()
                ? fs::path(modelPath).stem().string() : ref.modelName;
            if (!autoDownloadModel(modelPath, selector)) {
                fatal_error("[DXAPP] [ERROR] Model file not found: " + modelPath + "\n"
                    "        -> Download:  python3 scripts/download_models.py --models " + selector + "\n"
                    "        -> Or use:    ./run_demo.sh  (auto-downloads demo models)");
            }
            std::cout << "[DXAPP] [INFO] Model downloaded successfully: " << modelPath << std::endl;
        }
        return;
    }
    // Explicit -m: no auto-download — a wrong path is a user error.
    if (!fileExists(modelPath)) {
        fatal_error("[DXAPP] [ERROR] Model file not found: " + modelPath + "\n"
            "        -> Check the path, or omit -m to use this example's default model.\n"
            "        -> Download:  python3 scripts/download_models.py --models " + fs::path(modelPath).stem().string());
    }
}

/**
 * @brief Require an explicitly-given input file to exist (SDKREQ-529 policy).
 *
 * A wrong `-i`/`-v` path errors out immediately — we never silently fall back
 * to a default sample. (An empty path means "no input given" and is handled by
 * the default-sample logic before this call.)
 */
inline void requireInputExists(const std::string& path) {
    if (!path.empty() && !fileExists(path) && !fs::is_directory(path)) {
        fatal_error("[DXAPP] [ERROR] Input file not found: " + path);
    }
}

/**
 * @brief Require a LiDAR point-cloud input to be a .bin file (SDKREQ-529 policy).
 *
 * 3D-detection examples consume raw point clouds; an existing-but-wrong-format
 * file (e.g. a .jpg) must error rather than be fed to the model. Accepts a
 * directory (batch of .bin files) — per-file extensions are checked on read.
 */
inline void requireBinInput(const std::string& path) {
    if (path.empty() || fs::is_directory(path)) return;
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != ".bin") {
        fatal_error("[DXAPP] [ERROR] This example requires a LiDAR point-cloud .bin input "
                    "(-i / --image_path). Got: " + path);
    }
}

}  // namespace dxapp

/**
 * @brief Detect if a 4D input shape is NHWC layout.
 *
 * Heuristic: if the last dimension is small (≤4, typical for image channels)
 * and the second dimension is larger, it's likely NHWC [N,H,W,C].
 * Otherwise, assume NCHW [N,C,H,W].
 */
inline bool isInputNHWC(const std::vector<int64_t>& shape) {
    if (shape.size() >= 4) {
        return (shape[3] <= 4 && shape[1] > shape[3]);
    }
    return false;
}

/**
 * @brief Parse model input shape to get spatial dimensions (H, W).
 *
 * Handles NCHW [N,C,H,W] and NHWC [N,H,W,C] tensor layouts automatically.
 * For 3D shapes [N/C, H, W], uses shape[1] and shape[2].
 * For 2D shapes [H, W], uses shape[0] and shape[1].
 */
inline void parseInputShape(const std::vector<int64_t>& shape, int& width, int& height) {
    if (shape.size() >= 4) {
        if (isInputNHWC(shape)) {
            // NHWC: [N, H, W, C]
            height = static_cast<int>(shape[1]);
            width  = static_cast<int>(shape[2]);
        } else {
            // NCHW: [N, C, H, W]
            height = static_cast<int>(shape[2]);
            width  = static_cast<int>(shape[3]);
        }
    } else if (shape.size() == 3) {
        height = static_cast<int>(shape[1]);
        width  = static_cast<int>(shape[2]);
    } else if (shape.size() >= 2) {
        height = static_cast<int>(shape[0]);
        width  = static_cast<int>(shape[1]);
    } else {
        height = 0;
        width  = 0;
    }
}

/**
 * @brief Fill a flat float buffer from a single-channel uint8 image.
 */
inline void fillGrayscaleBuffer(const cv::Mat& img, int h, int w,
                                std::vector<float>& buf) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            buf[y * w + x] = img.at<uint8_t>(y, x) / 255.0f;
}

/**
 * @brief Fill a flat float buffer from a 3-channel image in NHWC (HWC) layout.
 */
inline void fillNHWCBuffer(const cv::Mat& img, int h, int w, int c,
                           std::vector<float>& buf) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                buf[y * w * c + x * c + ch] = img.at<cv::Vec3b>(y, x)[ch] / 255.0f;
}

/**
 * @brief Fill a flat float buffer from a 3-channel image in NCHW (CHW) layout.
 */
inline void fillNCHWBuffer(const cv::Mat& img, int h, int w, int c,
                           std::vector<float>& buf) {
    for (int ch = 0; ch < c; ++ch)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                buf[ch * h * w + y * w + x] = img.at<cv::Vec3b>(y, x)[ch] / 255.0f;
}

/**
 * @brief Convert uint8 image to float32 buffer with specified layout.
 * @param img Input image (CV_8UC3 or CV_8UC1)
 * @param nhwc If true, output is HWC layout; if false, CHW layout
 * @return Float buffer normalized to [0, 1]
 */
inline std::vector<float> convertToFloatBuffer(const cv::Mat& img, bool nhwc) {
    int h = img.rows, w = img.cols, c = img.channels();
    std::vector<float> buf(h * w * c);
    if (c == 1) {
        fillGrayscaleBuffer(img, h, w, buf);
    } else if (nhwc) {
        fillNHWCBuffer(img, h, w, c, buf);
    } else {
        fillNCHWBuffer(img, h, w, c, buf);
    }
    return buf;
}

/**
 * @brief Fill a flat float buffer from a 3-channel image in NHWC layout,
 *        applying (pixel/255 - mean) / std per channel.
 *
 * @param mean Per-channel mean in the image's channel order (post /255 scale).
 * @param stdv Per-channel std  in the image's channel order.
 */
inline void fillNHWCBufferNormalized(const cv::Mat& img, int h, int w, int c,
                                     const std::array<float, 3>& mean,
                                     const std::array<float, 3>& stdv,
                                     std::vector<float>& buf) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                buf[y * w * c + x * c + ch] =
                    (img.at<cv::Vec3b>(y, x)[ch] / 255.0f - mean[ch]) / stdv[ch];
}

/**
 * @brief Fill a flat float buffer from a 3-channel image in NCHW layout,
 *        applying (pixel/255 - mean) / std per channel.
 */
inline void fillNCHWBufferNormalized(const cv::Mat& img, int h, int w, int c,
                                     const std::array<float, 3>& mean,
                                     const std::array<float, 3>& stdv,
                                     std::vector<float>& buf) {
    for (int ch = 0; ch < c; ++ch)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                buf[ch * h * w + y * w + x] =
                    (img.at<cv::Vec3b>(y, x)[ch] / 255.0f - mean[ch]) / stdv[ch];
}

/**
 * @brief Convert a uint8 image to a float32 buffer with mean/std normalization.
 *
 * Applies (pixel/255 - mean) / std per channel and lays the result out in
 * either CHW (NCHW) or HWC (NHWC) order. mean/std must be in the same channel
 * order as @p img (RGB if the preprocessor already applied BGR->RGB).
 *
 * @param img  Input image (CV_8UC3).
 * @param nhwc If true, output is HWC layout; if false, CHW layout.
 * @param mean Per-channel mean (post /255 scale).
 * @param stdv Per-channel std.
 * @return Normalized float buffer.
 */
inline std::vector<float> convertToFloatBufferNormalized(
        const cv::Mat& img, bool nhwc,
        const std::array<float, 3>& mean,
        const std::array<float, 3>& stdv) {
    int h = img.rows, w = img.cols, c = img.channels();
    std::vector<float> buf(h * w * c);
    if (c == 1) {
        // Single channel: normalize with channel-0 mean/std.
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                buf[y * w + x] = (img.at<uint8_t>(y, x) / 255.0f - mean[0]) / stdv[0];
    } else if (nhwc) {
        fillNHWCBufferNormalized(img, h, w, c, mean, stdv, buf);
    } else {
        fillNCHWBufferNormalized(img, h, w, c, mean, stdv, buf);
    }
    return buf;
}

/**
 * @brief Run synchronous inference, feeding a buffer that matches the model's
 *        input dtype.
 *
 * If the model input is float32 and the preprocessed image is 8-bit, the image
 * is converted to a float32 buffer (plain /255 normalization) in the model's
 * channel layout before inference. Otherwise the preprocessed buffer is passed
 * as-is (uint8 models, or preprocessors that already emit float, e.g. SFA3D).
 *
 * This prevents feeding a uint8 buffer to a float-input model (ViT/DeiT/CLIP,
 * Depth Anything, ...), which reads 4x the bytes -> out-of-bounds read/garbage.
 * Models that additionally need mean/std normalization not baked into the .dxnn
 * should handle it explicitly (see depth runners + factory getInputNormalization).
 */
inline dxrt::TensorPtrs runSyncInferenceTyped(dxrt::InferenceEngine& ie,
                                              const cv::Mat& preprocessed) {
    // GetInputs() returns Tensors by value: keep the vector alive in a local (binding
    // a reference straight to .front() dangles), and keep it non-const so this builds
    // against dxrt < v3.3.0, where Tensor::type() has no const overload.
    auto inputs = ie.GetInputs();
    auto& input = inputs.front();
    if (input.type() == dxrt::DataType::FLOAT && !preprocessed.empty()
            && preprocessed.depth() == CV_8U) {
        std::vector<float> fb = convertToFloatBuffer(preprocessed, isInputNHWC(input.shape()));
        return ie.Run(fb.data(), nullptr, nullptr);
    }
    return ie.Run(preprocessed.data, nullptr, nullptr);
}

/**
 * @brief Fill a pre-allocated model-input byte buffer from a preprocessed image,
 *        matching the model's input dtype (async path).
 *
 * Same dtype logic as runSyncInferenceTyped(): float32 models get a converted
 * float buffer (from an 8-bit image), everything else is copied verbatim. The
 * copy is size-clamped to the destination buffer.
 */
inline void fillModelInputBuffer(dxrt::InferenceEngine& ie,
                                 std::vector<uint8_t>& buf,
                                 const cv::Mat& preprocessed) {
    // Same as runSyncInferenceTyped(): own the Tensors vector locally, non-const.
    auto inputs = ie.GetInputs();
    auto& input = inputs.front();
    if (input.type() == dxrt::DataType::FLOAT && !preprocessed.empty()
            && preprocessed.depth() == CV_8U) {
        std::vector<float> fb = convertToFloatBuffer(preprocessed, isInputNHWC(input.shape()));
        size_t bytes = std::min(buf.size(), fb.size() * sizeof(float));
        std::memcpy(buf.data(), fb.data(), bytes);
    } else {
        size_t bytes = std::min(buf.size(), preprocessed.total() * preprocessed.elemSize());
        std::memcpy(buf.data(), preprocessed.data, bytes);
    }
}

// Platform-specific setup file paths
#ifndef SETUP_FILE_PATH
#if _WIN32
constexpr const char* SETUP_FILE_PATH = "setup.bat";
#else
constexpr const char* SETUP_FILE_PATH = "setup.sh --force";
#endif
#endif

// Exception handling macros (matching Legacy format)
#ifndef DXRT_EXCEPTION_UTIL
#define DXRT_EXCEPTION_UTIL

#define DXRT_TRY_CATCH_BEGIN try {

#define DXRT_TRY_CATCH_END                                                                       \
    }                                                                                            \
    catch (const dxrt::Exception& e) {                                                           \
        std::cerr << DXAPP_RED << e.what() << " error-code=" << e.code() << DXAPP_RESET          \
                  << std::endl;                                                                  \
        fs::path dx_app_dir(fs::canonical(PROJECT_ROOT_DIR));                                    \
        fs::path setup_script = dx_app_dir / SETUP_FILE_PATH;                                    \
        std::cerr << "dx_app_dir: " << dx_app_dir.string() << std::endl;                         \
        if (e.code() == 257) {                                                                   \
            if (dx_app_dir != fs::canonical(fs::current_path())) {                               \
                std::cerr << DXAPP_GREEN << "[HINT] The current directory is '"                  \
                          << fs::current_path().string() << "'. Please move to '"                \
                          << dx_app_dir.string() << "' before running the application."          \
                          << DXAPP_RESET << std::endl;                                           \
            } else {                                                                             \
                std::cerr << DXAPP_GREEN << "[HINT] Please run '"                               \
                          << setup_script.string()                                               \
                          << "' to set up the model and input video files "                      \
                             "before running the application again."                             \
                          << DXAPP_RESET << std::endl;                                           \
            }                                                                                    \
        }                                                                                        \
        return -1;                                                                               \
    }                                                                                            \
    catch (const std::exception& e) {                                                            \
        const std::string _dxapp_msg(e.what());                                                  \
        std::cerr << DXAPP_RED << _dxapp_msg << DXAPP_RESET << std::endl;                        \
        /* Image-only examples (embedding, ReID, attribute recognition, …) do    */              \
        /* not register the stream flags, so cxxopts throws "Option '<flag>' does */              \
        /* not exist" for -v/-c/-r. Surface an explicit image-only note in that   */              \
        /* case instead of the generic usage hint.                               */              \
        const bool _dxapp_no_opt = _dxapp_msg.find("does not exist") != std::string::npos;       \
        const bool _dxapp_stream_flag =                                                          \
            _dxapp_msg.find("'video'") != std::string::npos ||                                   \
            _dxapp_msg.find("'v'") != std::string::npos ||                                       \
            _dxapp_msg.find("'camera'") != std::string::npos ||                                  \
            _dxapp_msg.find("'c'") != std::string::npos ||                                       \
            _dxapp_msg.find("'rtsp'") != std::string::npos ||                                    \
            _dxapp_msg.find("'r'") != std::string::npos;                                         \
        if (_dxapp_no_opt && _dxapp_stream_flag) {                                               \
            std::cerr << DXAPP_GREEN                                                             \
                      << "[HINT] This example is image-only: video/camera/RTSP input "          \
                         "(-v/--video, -c/--camera, -r/--rtsp) is not supported. "              \
                         "Use -i (--image_path) to provide an image file or directory."         \
                      << DXAPP_RESET << std::endl;                                               \
        } else {                                                                                 \
            std::cerr << DXAPP_GREEN << "[HINT] Use -h or --help for usage information."         \
                      << DXAPP_RESET << std::endl;                                               \
        }                                                                                        \
        return -1;                                                                               \
    }

#endif  // DXRT_EXCEPTION_UTIL

#endif  // DXAPP_COMMON_UTIL_HPP
