/**
 * @file lidar_util.hpp
 * @brief KITTI LiDAR .bin loading and BEV conversion for SFA3D
 */

#ifndef DXAPP_LIDAR_UTIL_HPP
#define DXAPP_LIDAR_UTIL_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace dxapp {

inline int clampInt(int value, int lo, int hi) {
    return std::max(lo, std::min(value, hi));
}

inline float clampFloat(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

inline bool isLidarInputPath(const std::string& path) {
    if (path.size() < 4) return false;
    return path.compare(path.size() - 4, 4, ".bin") == 0;
}

inline cv::Mat loadKittiPointCloud(const std::string& bin_path) {
    std::ifstream ifs(bin_path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open LiDAR file: " + bin_path);
    }
    const std::streamsize bytes = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (bytes <= 0 || (bytes % static_cast<std::streamsize>(sizeof(float) * 4)) != 0) {
        throw std::runtime_error("Invalid KITTI .bin size: " + bin_path);
    }
    const size_t num_points = static_cast<size_t>(bytes / (sizeof(float) * 4));
    std::vector<float> data(static_cast<size_t>(bytes / sizeof(float)));
    ifs.read(reinterpret_cast<char*>(data.data()), bytes);
    return cv::Mat(static_cast<int>(num_points), 4, CV_32F, data.data()).clone();
}

inline cv::Mat pointcloudToBEV(const cv::Mat& points,
                               int input_height = 608,
                               int input_width = 608,
                               float x_min = 0.0f, float x_max = 50.0f,
                               float y_min = -25.0f, float y_max = 25.0f,
                               float z_min = -2.5f, float z_max = 1.0f) {
    cv::Mat bev = cv::Mat::zeros(input_height, input_width, CV_8UC3);
    if (points.empty() || points.cols < 4) {
        return bev;
    }

    const float x_res = (x_max - x_min) / static_cast<float>(input_height);
    const float y_res = (y_max - y_min) / static_cast<float>(input_width);

    cv::Mat intensity_map = cv::Mat::zeros(input_height, input_width, CV_32F);
    cv::Mat height_map(input_height, input_width, CV_32F, cv::Scalar(z_min));
    cv::Mat density_map = cv::Mat::zeros(input_height, input_width, CV_32F);

    for (int i = 0; i < points.rows; ++i) {
        const float x = points.at<float>(i, 0);
        const float y = points.at<float>(i, 1);
        const float z = points.at<float>(i, 2);
        const float intensity = points.at<float>(i, 3);
        if (x < x_min || x >= x_max || y < y_min || y >= y_max ||
            z < z_min || z >= z_max) {
            continue;
        }
        const int row = clampInt(
            static_cast<int>((x_max - x) / x_res), 0, input_height - 1);
        const int col = clampInt(
            static_cast<int>((y - y_min) / y_res), 0, input_width - 1);

        if (intensity > intensity_map.at<float>(row, col)) {
            intensity_map.at<float>(row, col) = intensity;
        }
        if (z > height_map.at<float>(row, col)) {
            height_map.at<float>(row, col) = z;
        }
        density_map.at<float>(row, col) += 1.0f;
    }

    double int_max = 0.0;
    cv::minMaxLoc(intensity_map, nullptr, &int_max);
    cv::Mat intensity_u8(input_height, input_width, CV_8U);
    if (int_max > 0.0) {
        intensity_map.convertTo(intensity_u8, CV_8U, 255.0 / int_max);
    }

    cv::Mat height_u8(input_height, input_width, CV_8U);
    const float h_range = z_max - z_min;
    height_map.convertTo(height_u8, CV_8U, 255.0 / h_range, -z_min * 255.0 / h_range);

    double dens_max = 0.0;
    cv::minMaxLoc(density_map, nullptr, &dens_max);
    cv::Mat density_u8 = cv::Mat::zeros(input_height, input_width, CV_8U);
    if (dens_max > 0.0) {
        for (int r = 0; r < input_height; ++r) {
            for (int c = 0; c < input_width; ++c) {
                const float d = density_map.at<float>(r, c);
                const float norm = std::log1p(d) / std::log1p(static_cast<float>(dens_max));
                density_u8.at<uchar>(r, c) =
                    static_cast<uchar>(clampFloat(norm * 255.0f, 0.0f, 255.0f));
            }
        }
    }

    std::vector<cv::Mat> channels = {intensity_u8, height_u8, density_u8};
    cv::merge(channels, bev);
    return bev;
}

inline cv::Mat loadDisplayFrame(const std::string& path,
                                int input_height = 608,
                                int input_width = 608) {
    if (isLidarInputPath(path)) {
        const cv::Mat points = loadKittiPointCloud(path);
        return pointcloudToBEV(points, input_height, input_width);
    }
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load input: " + path);
    }
    return img;
}

}  // namespace dxapp

#endif  // DXAPP_LIDAR_UTIL_HPP
