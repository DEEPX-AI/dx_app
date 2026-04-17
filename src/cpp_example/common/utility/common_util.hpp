/**
 * @file common_util.hpp
 * @brief Common utility functions and macros
 */

#ifndef DXAPP_COMMON_UTIL_HPP
#define DXAPP_COMMON_UTIL_HPP

#include <dxrt/device_info_status.h>
#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <atomic>

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
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_WARN(msg) std::cout << DXAPP_YELLOW << "[WARN] " << msg << DXAPP_RESET << std::endl
#define LOG_ERROR(msg) std::cerr << DXAPP_RED << "[ERROR] " << msg << DXAPP_RESET << std::endl

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
 * Validates that the DXRT library version is >= 3.0.0 and
 * the compiled model version is >= v7 (matching Legacy behavior).
 * 
 * @param ie Pointer to InferenceEngine
 * @return true if versions are compatible
 */
inline bool minversionforRTandCompiler(dxrt::InferenceEngine* ie) {
    if (!ie) return false;

    std::string rt_version = dxrt::Configuration::GetInstance().GetVersion();
    std::string compiler_version = ie->GetModelVersion();

    if (isVersionGreaterOrEqual(rt_version, "3.0.0")) {
        if (isVersionGreaterOrEqual(compiler_version, "v7")) {
            return true;
        } else {
            std::cerr << "[DXAPP] [ER] Compiler version is too low. (required: "
                         ">= 7, current: "
                      << compiler_version << ")" << std::endl;
        }
    } else {
        std::cerr << "[DXAPP] [ER] DXRT library version is too low. (required: "
                     ">= 3.0.0, current: "
                  << rt_version << ")" << std::endl;
    }
    return false;
}

/** Save frame to path specified by DXAPP_SAVE_IMAGE env var (debug/test hook). */
inline void saveDebugImage(const cv::Mat& frame) {
    const char* path = std::getenv("DXAPP_SAVE_IMAGE");
    if (path && *path && !frame.empty()) cv::imwrite(path, frame);
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
 * - If the window named `winname` is closed (getWindowProperty <= 0), this returns true
 *   to signal the caller to stop displaying and proceed to next model.
 */
// Forward declaration: g_interrupted is defined in run_dir.hpp. Provide a
// declaration here to avoid build ordering issues when this header is
// included without run_dir.hpp.
inline std::atomic<bool>& g_interrupted();

inline bool windowShouldClose(const std::string& winname = "Output") {
    if (_displayClosed()) return true;
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
 * @brief Display frame in a resizable window, sized to ~1/4 screen area on first call.
 *
 * On the first frame, detects screen resolution and sets the window to
 * half-screen width × half-screen height, preserving the frame's aspect ratio.
 * Subsequent frames reuse the same window without re-querying.
 */
inline void showOutput(const cv::Mat& frame) {
    if (_displayClosed()) return;

    static bool window_ever_opened = false;

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
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    window_ever_opened = true;

    if (!window_sized && !frame.empty()) {
        auto [screen_w, screen_h] = getScreenResolution();
        int target_w = screen_w / 2;
        int target_h = screen_h / 2;

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
 * @brief Write frame to video, auto-resizing if frame size differs from writer.
 */
inline void writeToVideo(cv::VideoWriter& writer, const cv::Mat& frame) {
    if (!writer.isOpened() || frame.empty()) return;
    int w = static_cast<int>(writer.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(writer.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (frame.cols == w && frame.rows == h) {
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
inline std::string buildPerImageSavePath(const std::string& runDir,
                                        const std::string& modelName,
                                        const std::string& imagePath,
                                        int img_idx = 0) {
    if (runDir.empty()) return std::string();
    fs::path base(runDir);
    fs::path model_dir = base / (modelName);
    fs::create_directories(model_dir);

    fs::path fname = fs::path(imagePath).filename();
    std::string stem = fname.stem().string();
    if (stem.empty()) stem = "image" + std::to_string(img_idx);
    fs::path out = model_dir / (stem + std::string("_output.jpg"));
    fs::create_directories(out.parent_path());
    return out.string();
}

/**
 * @brief Get default sample image path for a given task type.
 *
 * When no input source is specified, returns a bundled sample image
 * appropriate for the task.
 */
inline std::string getDefaultSampleImage(const std::string& taskType) {
    if (taskType == "object_detection")       return "sample/img/sample_street.jpg";
    if (taskType == "face_detection")         return "sample/img/sample_face.jpg";
    if (taskType == "obb_detection")          return "sample/dota8_test/P0284.png";
    if (taskType == "pose_estimation")        return "sample/img/sample_people.jpg";
    if (taskType == "hand_landmark")          return "sample/img/sample_hand.jpg";
    if (taskType == "face_alignment")         return "sample/img/sample_face_a1.jpg";
    if (taskType == "instance_segmentation")  return "sample/img/sample_street.jpg";
    if (taskType == "semantic_segmentation")  return "sample/img/sample_parking.jpg";
    if (taskType == "classification")         return "sample/img/sample_dog.jpg";
    if (taskType == "depth_estimation")       return "sample/img/sample_parking.jpg";
    if (taskType == "image_denoising")        return "sample/img/sample_denoising.jpg";
    if (taskType == "super_resolution")       return "sample/img/sample_superresolution.png";
    if (taskType == "image_enhancement")      return "sample/img/sample_lowlight.jpg";
    if (taskType == "embedding")              return "sample/img/face_pair";
    if (taskType == "attribute_recognition")  return "sample/img/sample_person_a1.jpg";
    if (taskType == "reid")                   return "sample/img/person_pair";
    if (taskType == "ppu")                    return "sample/img/sample_street.jpg";
    return "sample/img/sample_street.jpg";
}

/**
 * @brief Get default sample video path for a given task type.
 *
 * Returns empty string for image-only tasks (embedding, attribute_recognition, reid).
 */
inline std::string getDefaultSampleVideo(const std::string& taskType) {
    if (taskType == "object_detection")       return "assets/videos/snowboard.mp4";
    if (taskType == "face_detection")         return "assets/videos/dance-group.mov";
    if (taskType == "obb_detection")          return "assets/videos/obb.mp4";
    if (taskType == "pose_estimation")        return "assets/videos/dance-solo.mov";
    if (taskType == "hand_landmark")          return "assets/videos/hand.mp4";
    if (taskType == "face_alignment")         return "assets/videos/face-alignment-closeup.mp4";
    if (taskType == "instance_segmentation")  return "assets/videos/dogs.mp4";
    if (taskType == "semantic_segmentation")  return "assets/videos/blackbox-city-road.mp4";
    if (taskType == "classification")         return "assets/videos/dogs.mp4";
    if (taskType == "depth_estimation")       return "assets/videos/blackbox-city-road.mp4";
    if (taskType == "image_denoising")        return "assets/videos/noisy_hand.mp4";
    if (taskType == "super_resolution")       return "assets/videos/dance-group.mov";
    if (taskType == "image_enhancement")      return "assets/videos/lowlight.mp4";
    if (taskType == "ppu")                    return "assets/videos/snowboard.mp4";
    return "";  // image-only tasks (embedding, attribute_recognition, reid)
}

/**
 * @brief Attempt to auto-download a missing model via setup_sample_models.sh.
 * @return true if download succeeded and file now exists.
 */
inline bool autoDownloadModel(const std::string& modelPath) {
    std::string stem = fs::path(modelPath).stem().string();
    std::string modelsDir = fs::path(modelPath).parent_path().string();
    if (modelsDir.empty()) modelsDir = "./assets/models";
    std::cout << "[INFO] Model not found: " << modelPath
              << " — attempting auto-download..." << std::endl;
    std::string cmd = "./setup_sample_models.sh --output=" + modelsDir
                    + " --models " + stem;
    int ret = std::system(cmd.c_str());
    return (ret == 0) && fs::exists(modelPath);
}

/**
 * @brief Attempt to auto-download sample videos via setup_sample_videos.sh.
 * @return true if download succeeded.
 */
inline bool autoDownloadVideos() {
    std::cout << "[INFO] Videos not found — attempting auto-download..." << std::endl;
    int ret = std::system("./setup_sample_videos.sh --output=./assets/videos");
    return ret == 0;
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
        std::cerr << e.what() << std::endl;                                                      \
        return -1;                                                                               \
    }

#endif  // DXRT_EXCEPTION_UTIL

#endif  // DXAPP_COMMON_UTIL_HPP
