/**
 * @file sfa3d_geometry.hpp
 * @brief 3D box geometry and multi-view rendering for SFA3D
 */

#ifndef DXAPP_SFA3D_GEOMETRY_HPP
#define DXAPP_SFA3D_GEOMETRY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/base/i_processor.hpp"
#include "common/utility/kitti_calib.hpp"

namespace dxapp {

constexpr double kXMin = 0.0, kXMax = 50.0;
constexpr double kYMin = -25.0, kYMax = 25.0;
constexpr double kZMin = -2.5, kZMax = 1.5;

inline cv::Scalar sfa3dClassColor(int class_id) {
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 128, 0),  // Pedestrian
        cv::Scalar(0, 255, 0),    // Car
        cv::Scalar(0, 128, 255),  // Cyclist
    };
    return colors[static_cast<size_t>(class_id) % colors.size()];
}

inline bool cameraBoxIsVisible(const std::array<cv::Point2d, 8>& corners2d,
                               const cv::Size& image_size, int margin = 8) {
    int inside = 0;
    double sum_x = 0.0, sum_y = 0.0;
    int count = 0;
    for (const auto& pt : corners2d) {
        if (pt.x < -1e8) continue;
        ++count;
        sum_x += pt.x;
        sum_y += pt.y;
        if (pt.x >= margin && pt.x < image_size.width - margin
            && pt.y >= margin && pt.y < image_size.height - margin) {
            ++inside;
        }
    }
    if (inside >= 2) return true;
    if (count == 0) return false;
    const double cx = sum_x / count;
    const double cy = sum_y / count;
    return cx >= -margin && cx < image_size.width + margin
        && cy >= -margin && cy < image_size.height + margin
        && inside >= 1;
}

inline std::array<cv::Point3d, 8> boxCornersVelo(const Detection3DResult& det) {
    const double cx = det.x3d, cy = det.y3d, cz = det.z3d;
    const double h = det.dim_h, w = det.dim_w, l = det.dim_l;
    const double cos_a = std::cos(det.yaw), sin_a = std::sin(det.yaw);
    // z3d is bottom-center height in KITTI / SFA3D convention
    const std::array<cv::Point3d, 8> local = {{
        {l / 2, w / 2, 0.0}, {l / 2, -w / 2, 0.0},
        {-l / 2, -w / 2, 0.0}, {-l / 2, w / 2, 0.0},
        {l / 2, w / 2, h}, {l / 2, -w / 2, h},
        {-l / 2, -w / 2, h}, {-l / 2, w / 2, h},
    }};
    std::array<cv::Point3d, 8> out{};
    for (size_t i = 0; i < local.size(); ++i) {
        out[i].x = cx + local[i].x * cos_a - local[i].y * sin_a;
        out[i].y = cy + local[i].x * sin_a + local[i].y * cos_a;
        out[i].z = cz + local[i].z;
    }
    return out;
}

inline cv::Point2d worldXyToBevPx(double x, double y, int width, int height, bool display = true) {
    const double ySpan = std::max(kYMax - kYMin, 1e-3);
    const double col = display
        ? (kYMax - y) / ySpan * width
        : (y - kYMin) / ySpan * width;
    const double row = (kXMax - x) / std::max(kXMax - kXMin, 1e-3) * height;
    return cv::Point2d(col, row);
}

inline std::array<cv::Point2d, 4> bevBoxCorners(const Detection3DResult& det, int width, int height) {
    const double cx = det.x3d, cy = det.y3d;
    const double l = det.dim_l, w = det.dim_w;
    const double cos_a = std::cos(det.yaw), sin_a = std::sin(det.yaw);
    const std::array<std::pair<double, double>, 4> local = {{
        {l / 2, w / 2}, {l / 2, -w / 2}, {-l / 2, -w / 2}, {-l / 2, w / 2},
    }};
    std::array<cv::Point2d, 4> out{};
    for (size_t i = 0; i < local.size(); ++i) {
        const double wx = cx + local[i].first * cos_a - local[i].second * sin_a;
        const double wy = cy + local[i].first * sin_a + local[i].second * cos_a;
        out[i] = worldXyToBevPx(wx, wy, width, height);
    }
    return out;
}

inline cv::Mat drawBevBoxes(cv::Mat frame, const std::vector<Detection3DResult>& detections,
                            int width, int height,
                            const std::vector<std::string>& class_names = {}) {
    if (frame.empty()) return frame;
    cv::Mat output = frame.clone();
    if (output.type() != CV_8UC3) cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    cv::Mat fill_layer = output.clone();

    for (const auto& det : detections) {
        const cv::Scalar color = sfa3dClassColor(det.class_id);
        const auto corners = bevBoxCorners(det, width, height);
        std::vector<cv::Point> pts;
        pts.reserve(4);
        for (const auto& c : corners) {
            pts.emplace_back(static_cast<int>(std::round(c.x)), static_cast<int>(std::round(c.y)));
        }
        const std::vector<std::vector<cv::Point>> poly{pts};
        cv::fillPoly(fill_layer, poly, color, cv::LINE_AA);
        cv::polylines(output, pts, true, color, 2, cv::LINE_AA);

        const double front_x = det.x3d + (det.dim_l * 0.5) * std::cos(det.yaw);
        const double front_y = det.y3d + (det.dim_l * 0.5) * std::sin(det.yaw);
        const cv::Point2d center = worldXyToBevPx(det.x3d, det.y3d, width, height);
        const cv::Point2d front = worldXyToBevPx(front_x, front_y, width, height);
        cv::arrowedLine(output,
                        cv::Point(static_cast<int>(std::round(center.x)),
                                  static_cast<int>(std::round(center.y))),
                        cv::Point(static_cast<int>(std::round(front.x)),
                                  static_cast<int>(std::round(front.y))),
                        color, 2, cv::LINE_AA, 0, 0.25);

        const std::string label = det.class_name + " " + cv::format("%.2f", det.confidence);
        int lx = std::max(0, pts[0].x);
        int ly = std::max(15, pts[0].y - 4);
        for (const auto& p : pts) {
            lx = std::min(lx, p.x);
            ly = std::min(ly, std::max(15, p.y - 4));
        }
        cv::putText(output, label, cv::Point(lx, ly), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::putText(output, label, cv::Point(lx, ly), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv::LINE_AA);
    }

    cv::addWeighted(fill_layer, 0.20, output, 0.80, 0.0, output);
    int y0 = 48;
    for (size_t i = 0; i < class_names.size(); ++i) {
        cv::putText(output, class_names[i], cv::Point(10, y0 + static_cast<int>(i) * 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    sfa3dClassColor(static_cast<int>(i)), 1, cv::LINE_AA);
    }
    return output;
}

inline cv::Point2d projectVeloPoint(const cv::Point3d& pt, const KittiCalib& calib, bool& valid) {
    cv::Mat hom = (cv::Mat_<double>(4, 1) << pt.x, pt.y, pt.z, 1.0);
    cv::Mat proj = calib.velo_to_image * hom;
    const double depth = proj.at<double>(2, 0);
    valid = depth > 0.1;
    if (!valid) return cv::Point2d(0.0, 0.0);
    return cv::Point2d(proj.at<double>(0, 0) / depth, proj.at<double>(1, 0) / depth);
}

inline void drawBoxWireframe(cv::Mat& canvas, const std::array<cv::Point2d, 8>& corners,
                             const cv::Scalar& color, int thickness = 2) {
    static const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (const auto& edge : edges) {
        const cv::Point2d& p0 = corners[static_cast<size_t>(edge[0])];
        const cv::Point2d& p1 = corners[static_cast<size_t>(edge[1])];
        if (p0.x < -1e8 || p1.x < -1e8) continue;
        cv::line(canvas,
                 cv::Point(static_cast<int>(std::round(p0.x)), static_cast<int>(std::round(p0.y))),
                 cv::Point(static_cast<int>(std::round(p1.x)), static_cast<int>(std::round(p1.y))),
                 color, thickness, cv::LINE_AA);
    }
}

inline cv::Mat fitPanel(const cv::Mat& image, int width, int height) {
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(16, 16, 20));
    if (image.empty()) return canvas;
    cv::Mat frame = image.clone();
    if (frame.type() != CV_8UC3) cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    const double scale = std::min(
        static_cast<double>(width) / std::max(frame.cols, 1),
        static_cast<double>(height) / std::max(frame.rows, 1));
    const int new_w = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    const int x0 = (width - new_w) / 2;
    const int y0 = (height - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x0, y0, new_w, new_h)));
    return canvas;
}

inline void putPanelLabel(cv::Mat& tile, const std::string& label) {
    cv::putText(tile, label, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(tile, label, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

inline cv::Mat renderSyntheticCameraImage(const cv::Mat& points, const KittiCalib& calib) {
    const cv::Size sz = defaultCameraCanvasSize(calib);
    cv::Mat intensity = cv::Mat::zeros(sz, CV_32F);
    for (int i = 0; i < points.rows; ++i) {
        const cv::Point3d pt(points.at<float>(i, 0), points.at<float>(i, 1), points.at<float>(i, 2));
        bool valid = false;
        const cv::Point2d uv = projectVeloPoint(pt, calib, valid);
        if (!valid) continue;
        const int xi = static_cast<int>(std::round(uv.x));
        const int yi = static_cast<int>(std::round(uv.y));
        if (xi < 0 || yi < 0 || xi >= sz.width || yi >= sz.height) continue;
        const float val = points.cols > 3 ? points.at<float>(i, 3) : 0.5f;
        intensity.at<float>(yi, xi) = std::max(intensity.at<float>(yi, xi), val);
    }
    cv::Mat intensity_u8;
    if (cv::countNonZero(intensity > 0) > 0) {
        cv::Mat norm;
        cv::normalize(intensity, norm, 0, 255, cv::NORM_MINMAX);
        norm.convertTo(intensity_u8, CV_8U);
    } else {
        intensity_u8 = cv::Mat::zeros(sz, CV_8U);
    }
    cv::Mat bgr;
    cv::cvtColor(intensity_u8, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

inline cv::Mat renderCameraOverlay(cv::Mat image, const std::vector<Detection3DResult>& detections,
                                   const KittiCalib& calib) {
    if (image.empty()) {
        image = cv::Mat(defaultCameraCanvasSize(calib), CV_8UC3, cv::Scalar(16, 12, 12));
    }
    cv::Mat output = image.clone();
    if (output.type() != CV_8UC3) cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    for (const auto& det : detections) {
        const cv::Scalar color = sfa3dClassColor(det.class_id);
        const auto corners3d = boxCornersVelo(det);
        std::array<cv::Point2d, 8> corners2d{};
        bool any_valid = false;
        for (size_t i = 0; i < corners3d.size(); ++i) {
            bool valid = false;
            corners2d[i] = projectVeloPoint(corners3d[i], calib, valid);
            if (!valid) corners2d[i] = cv::Point2d(-1e9, -1e9);
            else any_valid = true;
        }
        if (!any_valid) continue;
        if (!cameraBoxIsVisible(corners2d, output.size())) continue;
        drawBoxWireframe(output, corners2d, color, 2);

        std::vector<cv::Point> bottom;
        bottom.reserve(4);
        bool bottom_ok = true;
        for (size_t i = 0; i < 4; ++i) {
            if (corners2d[i].x < -1e8) bottom_ok = false;
            bottom.emplace_back(static_cast<int>(std::round(corners2d[i].x)),
                                static_cast<int>(std::round(corners2d[i].y)));
        }
        if (bottom_ok) {
            cv::polylines(output, bottom, true, color, 3, cv::LINE_AA);
        }

        const std::string label = det.class_name + " " + cv::format("%.2f", det.confidence);
        int lx = output.cols - 1, ly = output.rows - 1;
        for (size_t i = 0; i < corners2d.size(); ++i) {
            if (corners2d[i].x < -1e8) continue;
            lx = std::min(lx, static_cast<int>(std::round(corners2d[i].x)));
            ly = std::min(ly, static_cast<int>(std::round(corners2d[i].y)));
        }
        if (lx >= 0 && ly >= 15) {
            cv::putText(output, label, cv::Point(lx, ly - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
            cv::putText(output, label, cv::Point(lx, ly - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv::LINE_AA);
        }
    }
    return output;
}

inline cv::Point2d sideViewProject(const cv::Point3d& pt, int width, int height, int margin = 28) {
    const int usable_w = std::max(width - 2 * margin, 1);
    const int usable_h = std::max(height - 2 * margin, 1);
    const double px = margin + (pt.x - kXMin) / std::max(kXMax - kXMin, 1e-3) * usable_w;
    const double py = margin + (1.0 - (pt.z - kZMin) / std::max(kZMax - kZMin, 1e-3)) * usable_h;
    return cv::Point2d(px, py);
}

inline cv::Mat renderPointCloud3DView(const cv::Mat& points,
                                      const std::vector<Detection3DResult>& detections,
                                      int width = 608, int height = 608) {
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(18, 18, 24));
    if (points.empty() || points.cols < 3) return canvas;

    const int step = std::max(1, points.rows / 30000);
    double y_min = 1e9, y_max = -1e9;
    for (int i = 0; i < points.rows; i += step) {
        const float x = points.at<float>(i, 0);
        const float y = points.at<float>(i, 1);
        const float z = points.at<float>(i, 2);
        if (x < kXMin || x >= kXMax || y < kYMin || y >= kYMax || z < kZMin || z >= kZMax) continue;
        y_min = std::min(y_min, static_cast<double>(y));
        y_max = std::max(y_max, static_cast<double>(y));
    }
    const double span = std::max(y_max - y_min, 1e-3);

    for (int i = 0; i < points.rows; i += step) {
        const float x = points.at<float>(i, 0);
        const float y = points.at<float>(i, 1);
        const float z = points.at<float>(i, 2);
        if (x < kXMin || x >= kXMax || y < kYMin || y >= kYMax || z < kZMin || z >= kZMax) continue;
        const cv::Point2d px = sideViewProject(cv::Point3d(x, y, z), width, height);
        const int ix = static_cast<int>(std::round(px.x));
        const int iy = static_cast<int>(std::round(px.y));
        if (ix < 0 || iy < 0 || ix >= width || iy >= height) continue;
        const double t = (y - y_min) / span;
        cv::circle(canvas, cv::Point(ix, iy), 1,
                   cv::Scalar(255 * (1.0 - t), 60 + 140 * t, 255 * t), -1, cv::LINE_AA);
    }

    for (const auto& det : detections) {
        const auto corners3d = boxCornersVelo(det);
        std::array<cv::Point2d, 8> corners2d{};
        for (size_t j = 0; j < corners3d.size(); ++j) {
            corners2d[j] = sideViewProject(corners3d[j], width, height);
        }
        drawBoxWireframe(canvas, corners2d, sfa3dClassColor(det.class_id), 2);
    }

    const int ground_y = static_cast<int>(std::round(
        28 + (1.0 - (0.0 - kZMin) / std::max(kZMax - kZMin, 1e-3)) * std::max(height - 56, 1)));
    cv::line(canvas, cv::Point(28, ground_y), cv::Point(width - 28, ground_y),
             cv::Scalar(60, 60, 80), 1, cv::LINE_AA);
    return canvas;
}

inline cv::Mat composeRowPanels(const std::vector<cv::Mat>& panels,
                                const std::vector<std::string>& labels,
                                int panel_size = 608, int gap = 4) {
    if (panels.empty()) {
        return cv::Mat(panel_size, panel_size, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    std::vector<cv::Mat> tiles;
    tiles.reserve(panels.size());
    for (size_t i = 0; i < panels.size(); ++i) {
        cv::Mat tile = fitPanel(panels[i], panel_size, panel_size);
        if (i < labels.size()) putPanelLabel(tile, labels[i]);
        tiles.push_back(tile);
    }
    if (tiles.size() == 1) return tiles.front();

    const cv::Mat sep_v(panel_size, gap, CV_8UC3, cv::Scalar(32, 32, 32));
    cv::Mat out = tiles.front().clone();
    for (size_t i = 1; i < tiles.size(); ++i) {
        cv::Mat with_sep, merged;
        cv::hconcat(out, sep_v, with_sep);
        cv::hconcat(with_sep, tiles[i], merged);
        out = merged;
    }
    return out;
}

}  // namespace dxapp

#endif  // DXAPP_SFA3D_GEOMETRY_HPP
