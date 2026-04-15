/**
 * @file run_dir.hpp
 * @brief Structured run-directory utilities for C++ example runners
 *
 * Provides timestamped output directories, run_info.txt metadata,
 * tensor dump, VideoWriter codec fallback, and signal handling.
 *
 * Ported from original yolov5_sync.cpp / yolov5_async.cpp features
 * to the common runner infrastructure.
 */

#ifndef DXAPP_RUN_DIR_HPP
#define DXAPP_RUN_DIR_HPP

#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include <dxrt/dxrt_api.h>

namespace dxapp {
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

// ---------------------------------------------------------------
// Global signal handler for graceful Ctrl+C shutdown
// ---------------------------------------------------------------

/**
 * @brief Global flag set by SIGINT/SIGTERM handler for graceful shutdown.
 * All runner loops should check this flag in their iteration condition.
 */
inline std::atomic<bool>& g_interrupted() {
    static std::atomic<bool> flag{false};
    return flag;
}

/**
 * @brief Signal handler that sets g_interrupted to true.
 * Register with std::signal(SIGINT, signalHandler) in main.
 */
inline void signalHandler(int /*signum*/) {
    g_interrupted().store(true);
}

/**
 * @brief Install SIGINT and SIGTERM signal handlers for graceful shutdown.
 * Should be called at the beginning of run().
 */
inline void installSignalHandlers() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
}

// ---------------------------------------------------------------
// Run directory creation
// ---------------------------------------------------------------

/**
 * @brief Create a timestamped run directory.
 * @param baseDir   Base directory (e.g., "artifacts/cpp_example")
 * @param scriptTag Short tag for the binary (e.g., "yolov5s_sync")
 * @param runKind   "image", "image-dir", or "stream"
 * @param runName   Source name (e.g., "bus.jpg", "camera0")
 * @return Full path like: {baseDir}/{scriptTag}-{runKind}-{runName}-{YYYYMMDD-HHMMSS}
 */
inline std::string makeRunDir(const std::string& baseDir,
                              const std::string& scriptTag,
                              const std::string& runKind,
                              const std::string& runName) {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    std::string timestamp(20, '\0');
    auto len = std::strftime(&timestamp[0], timestamp.size(), "%Y%m%d-%H%M%S", &tm_buf);
    timestamp.resize(len);
    return baseDir + "/" + scriptTag + "-" + runKind + "-" + runName + "-" + timestamp;
}

// ---------------------------------------------------------------
// Run info metadata
// ---------------------------------------------------------------

/**
 * @brief Write run_info.txt with execution metadata into the given directory.
 */
inline void writeRunInfo(const std::string& runDir,
                         const std::string& binaryPath,
                         const std::string& modelPath,
                         const std::string& inputSource) {
    std::ofstream ofs(runDir + "/run_info.txt");
    if (!ofs) return;
    ofs << "script: " << binaryPath << "\n";
    ofs << "model: " << modelPath << "\n";
    ofs << "input: " << inputSource << "\n";
}

// ---------------------------------------------------------------
// Tensor dump
// ---------------------------------------------------------------

/** Write a single cv::Mat to a binary file (for tensor dump or error diagnostics). */
inline void writeInputTensor(const std::string& filePath, const cv::Mat& image) {
    std::ofstream ofs(filePath, std::ios::binary);
    if (ofs && !image.empty()) {
        size_t byte_size = image.total() * image.elemSize();
        ofs.write(static_cast<const char*>(static_cast<const void*>(image.data)),
                  static_cast<std::streamsize>(byte_size));
    }
}

/**
 * @brief Dump input and output tensors as raw binary files.
 * Files: input_tensor.bin, output_tensor_0.bin, ...
 */
inline void dumpTensorsToFiles(
    const std::string& targetDir,
    const cv::Mat& preprocessedImage,
    const std::vector<std::shared_ptr<dxrt::Tensor>>& outputs) {

    fs::create_directories(targetDir);

    // Input tensor
    writeInputTensor(targetDir + "/input_tensor.bin", preprocessedImage);

    // Output tensors
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& shape = outputs[i]->shape();
        size_t num_elements = 1;
        for (auto s : shape) num_elements *= static_cast<size_t>(s);
        size_t byte_size = num_elements * sizeof(float);
        std::ofstream ofs(targetDir + "/output_tensor_" + std::to_string(i) + ".bin",
                          std::ios::binary);
        if (ofs) {
            ofs.write(static_cast<const char*>(static_cast<const void*>(outputs[i]->data())),
                      static_cast<std::streamsize>(byte_size));
        }
    }
}

/**
 * @brief Compute per-frame dump directory path.
 * @param baseDumpDir Base dump directory (e.g., runDir + "/dump_tensors")
 * @param frameIdx    0-based frame index
 * @param perFrame    If true, creates frame000001/ subdirectory
 */
inline std::string frameDumpDir(const std::string& baseDumpDir,
                                int frameIdx, bool perFrame) {
    if (!perFrame) return baseDumpDir;
    std::string buf(16, '\0');
    auto len = std::snprintf(&buf[0], buf.size(), "frame%06d", frameIdx);
    buf.resize(static_cast<size_t>(len));
    return baseDumpDir + "/" + buf;
}

// ---------------------------------------------------------------
// VideoWriter with mp4v → XVID fallback
// ---------------------------------------------------------------

/**
 * @brief Open a VideoWriter trying mp4v first, then XVID/AVI fallback.
 * @param saveDir   Directory to place the output file
 * @param fps       Frames per second
 * @param frameSize Output frame size
 * @param[out] resolvedPath  Absolute path of the successfully opened file
 * @return Opened cv::VideoWriter (check .isOpened())
 */
inline cv::VideoWriter initVideoWriter(const std::string& saveDir,
                                       double fps,
                                       const cv::Size& frameSize,
                                       std::string& resolvedPath) {
    fs::create_directories(saveDir);
    double useFps = (fps > 0) ? fps : 30.0;

    std::string mp4Path = saveDir + "/output.mp4";
    cv::VideoWriter writer;

    // Try H.264 (avc1) first — universally playable on all modern players/OS
    writer.open(mp4Path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                useFps, frameSize);
    if (writer.isOpened()) {
        resolvedPath = fs::absolute(mp4Path).string();
        std::cout << "[INFO] Saving output video: " << resolvedPath << std::endl;
        return writer;
    }
    writer.release();

    // Fallback: mp4v (MPEG-4 Part 2) — requires codec pack on some systems
    std::cerr << "[WARN] avc1 codec failed, retrying with mp4v..." << std::endl;
    writer.open(mp4Path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                useFps, frameSize);
    if (writer.isOpened()) {
        resolvedPath = fs::absolute(mp4Path).string();
        std::cout << "[INFO] Saving output video: " << resolvedPath
                  << " (mp4v fallback)" << std::endl;
        return writer;
    }
    writer.release();

    // Fallback: XVID/AVI
    std::cerr << "[WARN] mp4v codec failed, retrying with XVID/AVI..." << std::endl;
    std::string aviPath = saveDir + "/output.avi";
    writer.open(aviPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                useFps, frameSize);
    if (writer.isOpened()) {
        resolvedPath = fs::absolute(aviPath).string();
        std::cout << "[INFO] Saving output video: " << resolvedPath
                  << " (XVID fallback)" << std::endl;
        return writer;
    }
    writer.release();

    std::cerr << "[ERROR] Failed to open VideoWriter for output." << std::endl;
    resolvedPath.clear();
    return writer;
}

// ---------------------------------------------------------------
// Input source string helper
// ---------------------------------------------------------------

/**
 * @brief Build a human-readable input source string for run_info.txt.
 */
inline std::string buildInputSourceString(const std::string& imagePath,
                                          const std::string& videoFile,
                                          int cameraIndex,
                                          const std::string& rtspUrl) {
    if (!imagePath.empty()) return imagePath;
    if (cameraIndex >= 0) return "camera:" + std::to_string(cameraIndex);
    if (!rtspUrl.empty()) return rtspUrl;
    return videoFile;
}

}  // namespace dxapp

#endif  // DXAPP_RUN_DIR_HPP
