/**
 * @file common_util.hpp
 * @brief Common utility functions and macros
 * 
 * Part of DX-APP v3.0.0 refactoring for independent build capability.
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
#define LOG_VALUE(var) std::cout << #var << " = " << var << std::endl

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
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
        g_interrupted().store(true);
        return true;
    }
    // getWindowProperty returns -1 when window was destroyed (user closed),
    // 0 during initial creation on some backends, and 1 when fully visible.
    // Use < 0 to avoid false positives on newly created windows.
    double visible = cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE);
    if (visible < 0.0) {
        return true;
    }
    return false;
}

/**
 * @brief Resize image for display preserving aspect ratio only when larger than max size.
 *        Matches Python SyncRunner._display_resize behaviour.
 * @param src Input image
 * @param dst Output image (resized or original)
 * @param max_w Maximum display width (default 960)
 * @param max_h Maximum display height (default 640)
 */
inline void displayResize(const cv::Mat &src, cv::Mat &dst, int max_w = 960, int max_h = 640) {
    if (src.empty()) { dst = src; return; }
    int w = src.cols;
    int h = src.rows;
    float scale = 1.0f;
    if (w > max_w || h > max_h) {
        float sx = static_cast<float>(max_w) / w;
        float sy = static_cast<float>(max_h) / h;
        scale = std::min(sx, sy);
    }
    if (scale >= 1.0f) {
        dst = src;
    } else {
        cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_AREA);
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
