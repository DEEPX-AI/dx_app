/**
 * @file camera.hpp
 * @brief Open a camera by index: USB webcams and the Raspberry Pi CSI camera.
 *
 * On current Raspberry Pi OS the CSI camera is managed by libcamera, not the
 * legacy V4L2 capture interface, so OpenCV has to read it through the
 * libcamerasrc GStreamer element. openCameraCapture() takes that path on a
 * Raspberry Pi and is plain cv::VideoCapture::open(index) everywhere else.
 *
 * Mirrors src/python_example/common/utility/camera.py - keep the two in sync.
 */

#ifndef DXAPP_UTILITY_CAMERA_HPP
#define DXAPP_UTILITY_CAMERA_HPP

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/wait.h>
#endif

namespace dxapp {

/// libcamera capture size when the caller does not ask for one.
constexpr int kLibcameraDefaultWidth = 1280;
constexpr int kLibcameraDefaultHeight = 720;

namespace camera_detail {

inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace camera_detail

/** True when a device-tree model string names a Raspberry Pi. */
inline bool isRaspberryPiModel(std::string model) {
    // device-tree strings are NUL-terminated; drop any embedded/trailing NULs.
    model.erase(std::remove(model.begin(), model.end(), '\0'), model.end());
    return camera_detail::toLower(model).find("raspberry pi") != std::string::npos;
}

/** True when the host is a Raspberry Pi (device-tree model). Safe elsewhere. */
inline bool isRaspberryPi() {
    const char* model_paths[] = {"/proc/device-tree/model",
                                 "/sys/firmware/devicetree/base/model"};
    for (const char* p : model_paths) {
        std::ifstream f(p, std::ios::binary);
        if (!f.is_open()) continue;
        const std::string model((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
        if (isRaspberryPiModel(model)) return true;
    }
    return false;
}

/** V4L2 driver/card name of /dev/video<index>, lower-cased ("" if unknown). */
inline std::string v4l2DeviceName(int index) {
    std::ifstream f("/sys/class/video4linux/video" + std::to_string(index) + "/name");
    if (!f.is_open()) return "";
    std::string name;
    std::getline(f, name);
    return camera_detail::toLower(name);
}

/**
 * True for the V4L2 name of a Raspberry Pi CSI/ISP node (rp1-cfe, unicam,
 * pispbe, bcm2835-*). These nodes emit raw/processing streams that OpenCV's
 * V4L2 backend cannot consume; they must go through libcamera. USB webcams
 * (uvcvideo, etc.) report other names.
 */
inline bool isRpiCsiDeviceName(const std::string& name) {
    const std::string lowered = camera_detail::toLower(name);
    // The legacy bcm2835-v4l2 capture driver works through OpenCV's V4L2
    // backend, so it is NOT a libcamera-only node despite the bcm2835 prefix.
    if (lowered.find("bcm2835-v4l2") != std::string::npos) return false;
    const char* tags[] = {"rp1-cfe", "unicam", "pispbe", "bcm2835"};
    for (const char* tag : tags) {
        if (lowered.find(tag) != std::string::npos) return true;
    }
    return false;
}

/** True when /dev/video<index> is a Raspberry Pi CSI/ISP node. */
inline bool isRpiCsiVideoNode(int index) {
    return isRpiCsiDeviceName(v4l2DeviceName(index));
}

/**
 * @brief libcamera ID of camera @p index in `rpicam-hello --list-cameras` output.
 *
 * Camera lines look like
 *   "0 : imx708 [4608x2592 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a)"
 * and the ID is the last parenthesised field. Mode lines ("Modes: ...",
 * "2304x1296 [...]") never match. Returns "" when camera @p index is not listed.
 */
inline std::string parseLibcameraCameraId(const std::string& listing, int index) {
    std::istringstream in(listing);
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t end = line.find_last_not_of(" \t\r");
        if (end == std::string::npos || line[end] != ')') continue;
        const std::size_t first = line.find_first_not_of(" \t");
        std::size_t digits_end = first;
        while (digits_end < line.size() &&
               std::isdigit(static_cast<unsigned char>(line[digits_end]))) {
            ++digits_end;
        }
        if (digits_end == first || digits_end - first > 4) continue;
        const std::size_t colon = line.find_first_not_of(" \t", digits_end);
        if (colon == std::string::npos || line[colon] != ':') continue;
        if (std::stoi(line.substr(first, digits_end - first)) != index) continue;
        const std::size_t open = line.rfind('(', end);
        if (open == std::string::npos || open < colon) continue;
        return line.substr(open + 1, end - open - 1);
    }
    return "";
}

/** True when OpenCV build information reports "GStreamer: YES". */
inline bool buildInfoHasGStreamer(const std::string& buildInfo) {
    std::istringstream in(buildInfo);
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t key = line.find("GStreamer:");
        if (key == std::string::npos) continue;
        return line.find("YES", key) != std::string::npos;
    }
    return false;
}

/** True when this OpenCV can open GStreamer pipelines. */
inline bool opencvHasGStreamer() {
    return buildInfoHasGStreamer(cv::getBuildInformation());
}

/**
 * @brief libcamerasrc pipeline string for OpenCV (BGR appsink).
 *
 * An empty @p cameraId lets libcamerasrc take its first camera. The caps ask
 * for a processed (ISP) NV12 stream: without an explicit format libcamerasrc
 * negotiates a raw Bayer stream that videoconvert cannot consume ("Internal
 * data stream error", seen on a Raspberry Pi). videoconvert then produces the
 * BGR frames OpenCV's appsink expects.
 */
inline std::string buildLibcameraPipeline(const std::string& cameraId,
                                          int width = kLibcameraDefaultWidth,
                                          int height = kLibcameraDefaultHeight) {
    std::ostringstream pipe;
    pipe << "libcamerasrc";
    // camera-name takes the libcamera ID, not an index.
    if (!cameraId.empty()) pipe << " camera-name=\"" << cameraId << "\"";
    pipe << " ! video/x-raw,width=" << width << ",height=" << height << ",format=NV12"
         << " ! videoconvert ! video/x-raw,format=BGR"
         << " ! appsink drop=true max-buffers=2";
    return pipe.str();
}

#ifndef _WIN32

/**
 * True when `timeout 30 <tool> --list-cameras` gave a listing: the pclose() /
 * system() status of a run that was neither killed by the timeout (124) nor
 * unable to start (125-127). Mirrors camera.py, where TimeoutExpired / OSError
 * make list_libcamera_cameras() return None.
 */
inline bool cameraListingCompleted(int status) {
    if (status == -1) return false;
    if (!WIFEXITED(status)) return true;
    const int code = WEXITSTATUS(status);
    return code < 124 || code > 127;
}

/**
 * Output of `rpicam-hello --list-cameras` (`libcamera-hello` on older
 * Raspberry Pi OS) in @p listing. False when the listing is unavailable (no
 * tool installed, or the tool could not run / timed out), so the caller falls
 * back to libcamerasrc's first camera for index 0 instead of failing hard.
 */
inline bool listLibcameraCameras(std::string& listing) {
    listing.clear();
    const char* tools[] = {"rpicam-hello", "libcamera-hello"};
    for (const char* tool : tools) {
        const std::string probe = std::string("command -v ") + tool + " > /dev/null 2>&1";
        if (std::system(probe.c_str()) != 0) continue;
        // Wrap in `timeout` so a stuck tool cannot block popen() indefinitely
        // (the Python mirror uses subprocess timeout=30).
        const std::string cmd = std::string("timeout 30 ") + tool + " --list-cameras 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (pipe == nullptr) return false;
        char buf[512];
        while (std::fgets(buf, sizeof(buf), pipe) != nullptr) listing += buf;
        if (!cameraListingCompleted(pclose(pipe))) {
            listing.clear();
            return false;
        }
        return true;
    }
    return false;
}

/** True when the libcamerasrc GStreamer element is installed. */
inline bool hasLibcameraGstElement() {
    // `timeout` guards against gst-inspect-1.0 hanging on a broken registry
    // (same 30 s bound as the Python mirror).
    return std::system("timeout 30 gst-inspect-1.0 libcamerasrc > /dev/null 2>&1") == 0;
}

/** Open camera @p cameraIndex through libcamerasrc; prints why on failure. */
inline bool openLibcameraCapture(cv::VideoCapture& video, int cameraIndex,
                                 int width, int height) {
    if (!opencvHasGStreamer()) {
        std::cerr << "[DXAPP] [ERROR] Raspberry Pi camera (index " << cameraIndex
                  << ") is read through GStreamer, but this OpenCV was built without it\n"
                  << "    (cv::getBuildInformation() shows \"GStreamer: NO\").\n"
                  << "    Use an OpenCV built with GStreamer (e.g. apt libopencv-dev)."
                  << std::endl;
        return false;
    }
    // Fail fast if the libcamera GStreamer element is missing: opening a pipeline
    // that references an unknown element otherwise blocks for tens of seconds.
    if (!hasLibcameraGstElement()) {
        std::cerr << "[DXAPP] [ERROR] Raspberry Pi camera (index " << cameraIndex
                  << ") needs the libcamera GStreamer plugin, which is not installed.\n"
                  << "    Install it with: sudo apt-get install -y gstreamer1.0-libcamera\n"
                  << "    (or re-run dx_app/install.sh), then verify: rpicam-hello --list-cameras"
                  << std::endl;
        return false;
    }

    std::string cameraId;
    std::string listing;
    if (listLibcameraCameras(listing)) {
        cameraId = parseLibcameraCameraId(listing, cameraIndex);
        if (cameraId.empty()) {
            std::cerr << "[DXAPP] [ERROR] libcamera has no camera " << cameraIndex
                      << ". rpicam-hello --list-cameras reports:\n" << listing << std::endl;
            return false;
        }
    } else if (cameraIndex > 0) {
        std::cerr << "[DXAPP] [ERROR] Selecting Raspberry Pi camera " << cameraIndex
                  << " needs rpicam-hello (or libcamera-hello) to look up its libcamera ID.\n"
                  << "    Install it with: sudo apt-get install -y rpicam-apps-lite" << std::endl;
        return false;
    }

    const std::string pipeline = buildLibcameraPipeline(cameraId, width, height);
    std::cout << "[DXAPP] [INFO] Using Raspberry Pi libcamera pipeline for camera "
              << cameraIndex << ":\n    " << pipeline << std::endl;
    if (video.open(pipeline, cv::CAP_GSTREAMER) && video.isOpened()) {
        std::cout << "[DXAPP] [INFO] Raspberry Pi camera opened via libcamera." << std::endl;
        return true;
    }
    video.release();
    std::cerr << "[DXAPP] [ERROR] Failed to open Raspberry Pi camera (index "
              << cameraIndex << ") through libcamera.\n"
              << "    Check that it is detected: rpicam-hello --list-cameras" << std::endl;
    return false;
}

#endif  // _WIN32

/**
 * @brief Open camera @p cameraIndex, transparently supporting the Raspberry Pi camera.
 *
 * Off a Raspberry Pi this is cv::VideoCapture::open(index) (USB webcams,
 * capture cards). On a Raspberry Pi the backend follows the V4L2 node, so a
 * CSI raw node is never probed with the wrong backend:
 *   - CSI/ISP node, or no /dev/video<index> -> libcamerasrc GStreamer pipeline
 *   - any other node (USB webcam)           -> V4L2, then libcamera if that fails
 * @p width / @p height size the libcamera stream (0: 1280x720); on the V4L2
 * path callers set those properties after opening, as before.
 */
inline bool openCameraCapture(cv::VideoCapture& video, int cameraIndex,
                              int width = 0, int height = 0) {
#ifdef _WIN32
    (void)width;
    (void)height;
    video.open(cameraIndex);
    return video.isOpened();
#else
    if (!isRaspberryPi()) {
        video.open(cameraIndex);  // Standard path - unchanged off Raspberry Pi.
        return video.isOpened();
    }
    const std::string node = "/dev/video" + std::to_string(cameraIndex);
    struct stat node_stat;
    const bool use_libcamera =
        isRpiCsiVideoNode(cameraIndex) || stat(node.c_str(), &node_stat) != 0;
    if (!use_libcamera) {
        video.open(cameraIndex);  // USB webcam / capture card on the Pi.
        if (video.isOpened()) return true;
        video.release();
    }
    return openLibcameraCapture(video, cameraIndex,
                                width > 0 ? width : kLibcameraDefaultWidth,
                                height > 0 ? height : kLibcameraDefaultHeight);
#endif
}

}  // namespace dxapp

#endif  // DXAPP_UTILITY_CAMERA_HPP
