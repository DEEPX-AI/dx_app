/**
 * @file kitti_calib.hpp
 * @brief KITTI calibration parsing for SFA3D multi-view visualization
 */

#ifndef DXAPP_KITTI_CALIB_HPP
#define DXAPP_KITTI_CALIB_HPP

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

#include <opencv2/opencv.hpp>

namespace dxapp {
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

struct KittiCalib {
    cv::Mat p2;             // 3x4 CV_64F
    cv::Mat r0_rect;        // 3x3 CV_64F
    cv::Mat tr_velo_to_cam; // 3x4 CV_64F
    cv::Mat velo_to_image;  // 3x4 CV_64F
};

inline cv::Mat parseMatrixLine(const std::string& line) {
    const auto colon = line.find(':');
    std::string values = (colon == std::string::npos) ? line : line.substr(colon + 1);
    std::istringstream iss(values);
    std::vector<double> nums;
    double v = 0.0;
    while (iss >> v) {
        nums.push_back(v);
    }
    return cv::Mat(static_cast<int>(nums.size()), 1, CV_64F, nums.data()).clone();
}

inline bool loadKittiCalib(const std::string& path, KittiCalib& out) {
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }

    cv::Mat p2, r0, tr;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("P2:") == 0) {
            p2 = parseMatrixLine(line).reshape(1, 3);
        } else if (line.find("R0_rect:") == 0) {
            r0 = parseMatrixLine(line).reshape(1, 3);
        } else if (line.find("Tr_velo_to_cam:") == 0) {
            tr = parseMatrixLine(line).reshape(1, 3);
        }
    }
    if (p2.empty() || r0.empty() || tr.empty()) {
        return false;
    }

    cv::Mat r0_4 = cv::Mat::eye(4, 4, CV_64F);
    r0.convertTo(r0_4(cv::Rect(0, 0, 3, 3)), CV_64F);

    cv::Mat tr_4 = cv::Mat::eye(4, 4, CV_64F);
    tr.convertTo(tr_4(cv::Rect(0, 0, 4, 3)), CV_64F);

    cv::Mat p_4 = cv::Mat::eye(4, 4, CV_64F);
    p2.convertTo(p_4(cv::Rect(0, 0, 4, 3)), CV_64F);

    out.p2 = p2;
    out.r0_rect = r0;
    out.tr_velo_to_cam = tr;
    out.velo_to_image = p_4 * r0_4 * tr_4;
    out.velo_to_image = out.velo_to_image(cv::Rect(0, 0, 4, 3)).clone();
    return true;
}

inline std::string& kittiCalibDirOverride() {
    static std::string dir;
    return dir;
}

inline std::string& kittiImage2DirOverride() {
    static std::string dir;
    return dir;
}

inline void setCalibDirOverride(const std::string& dir) {
    kittiCalibDirOverride() = dir;
}

inline void setImage2DirOverride(const std::string& dir) {
    kittiImage2DirOverride() = dir;
}

inline std::vector<std::string> candidateCalibPaths(const std::string& bin_path) {
    const fs::path path(bin_path);
    const std::string stem = path.stem().string();
    const fs::path parent = path.parent_path();
    std::vector<std::string> candidates;
    if (!kittiCalibDirOverride().empty()) {
        candidates.push_back((fs::path(kittiCalibDirOverride()) / (stem + ".txt")).string());
    }
    candidates.push_back((parent / (stem + ".txt")).string());
    candidates.push_back((parent.parent_path() / "calib" / (stem + ".txt")).string());
    if (parent.filename() == "velodyne") {
        candidates.push_back((parent.parent_path() / "calib" / (stem + ".txt")).string());
    }
    return candidates;
}

inline bool findCalibPath(const std::string& bin_path, std::string& out_path) {
    for (const auto& candidate : candidateCalibPaths(bin_path)) {
        if (fs::is_regular_file(candidate)) {
            out_path = candidate;
            return true;
        }
    }
    return false;
}

inline std::vector<std::string> candidateImagePaths(const std::string& bin_path) {
    fs::path path(bin_path);
    const std::string stem = path.stem().string();
    fs::path root = path.parent_path();
    if (root.filename() == "velodyne") {
        root = root.parent_path();
    }
    std::vector<std::string> out;
    const char* exts[] = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"};
    if (!kittiImage2DirOverride().empty()) {
        for (const char* ext : exts) {
            out.push_back((fs::path(kittiImage2DirOverride()) / (stem + ext)).string());
        }
    }
    const char* subdirs[] = {"image_2", "image_02", "image", "images"};
    for (const char* sub : subdirs) {
        for (const char* ext : exts) {
            out.push_back((root / sub / (stem + ext)).string());
        }
    }
    return out;
}

inline bool findImagePath(const std::string& bin_path, std::string& out_path) {
    for (const auto& candidate : candidateImagePaths(bin_path)) {
        if (fs::is_regular_file(candidate)) {
            out_path = candidate;
            return true;
        }
    }
    return false;
}

inline cv::Size defaultCameraCanvasSize(const KittiCalib& calib) {
    const double cx = calib.p2.at<double>(0, 2);
    const double cy = calib.p2.at<double>(1, 2);
    const int width = std::max(1242, static_cast<int>(std::round(cx * 2.0)));
    const int height = std::max(375, static_cast<int>(std::round(cy * 2.0)));
    return cv::Size(width, height);
}

}  // namespace dxapp

#endif  // DXAPP_KITTI_CALIB_HPP
