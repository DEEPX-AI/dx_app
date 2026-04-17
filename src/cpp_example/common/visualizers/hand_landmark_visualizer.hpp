/**
 * @file hand_landmark_visualizer.hpp
 * @brief Visualizer for hand landmark detection models (MediaPipe Hands, etc.)
 *
 * Draws 21-point hand skeleton on the image.
 */

#ifndef DXAPP_HAND_LANDMARK_VISUALIZER_HPP
#define DXAPP_HAND_LANDMARK_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include <opencv2/opencv.hpp>

namespace dxapp {

class HandLandmarkVisualizer : public IVisualizer<HandLandmarkResult> {
public:
    void setParameters(int /*line_thickness*/ = 2,
                       double /*font_scale*/ = 0.5,
                       float /*alpha*/ = 0.6f) override { /* unused */ }

    cv::Mat draw(const cv::Mat& image,
                 const std::vector<HandLandmarkResult>& results,
                 const PreprocessContext& ctx) override {
        cv::Mat output = image.clone();
        float disp_scale = 1.0f;
        if (ctx.original_width > 0 && ctx.original_height > 0 &&
            (ctx.original_width > output.cols || ctx.original_height > output.rows)) {
            disp_scale = std::min(static_cast<float>(output.cols) / ctx.original_width,
                                  static_cast<float>(output.rows) / ctx.original_height);
        }
        const float x_off = (ctx.original_width > 0)
            ? (output.cols - ctx.original_width * disp_scale) / 2.0f : 0.0f;
        const float y_off = (ctx.original_height > 0)
            ? (output.rows - ctx.original_height * disp_scale) / 2.0f : 0.0f;
        for (const auto& result : results) {
            if (result.landmarks.size() < 21) continue;
            drawConnections(output, result.landmarks, disp_scale, x_off, y_off);
            drawKeypoints(output, result.landmarks, disp_scale, x_off, y_off);
            drawLabel(output, result, disp_scale, x_off, y_off);
        }
        return output;
    }

private:
    static void drawConnections(cv::Mat& output,
                                const std::vector<Keypoint>& lmks,
                                float disp_scale, float x_off, float y_off) {
        static const int connections[][2] = {
            {0,1}, {1,2}, {2,3}, {3,4},           // Thumb
            {0,5}, {5,6}, {6,7}, {7,8},            // Index
            {0,9}, {9,10}, {10,11}, {11,12},        // Middle
            {0,13}, {13,14}, {14,15}, {15,16},      // Ring
            {0,17}, {17,18}, {18,19}, {19,20},      // Pinky
            {5,9}, {9,13}, {13,17}                  // Palm
        };
        static const cv::Scalar finger_colors[] = {
            {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
            {255, 255, 0}, {255, 0, 255}, {200, 200, 200}
        };
        for (size_t i = 0; i < 23; ++i) {
            int i1 = connections[i][0];
            int i2 = connections[i][1];
            int color_idx = (i < 4) ? 0 : (i < 8) ? 1 : (i < 12) ? 2 :
                            (i < 16) ? 3 : (i < 20) ? 4 : 5;
            cv::Point pt1(static_cast<int>(lmks[i1].x * disp_scale + x_off), static_cast<int>(lmks[i1].y * disp_scale + y_off));
            cv::Point pt2(static_cast<int>(lmks[i2].x * disp_scale + x_off), static_cast<int>(lmks[i2].y * disp_scale + y_off));
            cv::line(output, pt1, pt2, finger_colors[color_idx], 2, cv::LINE_AA);
        }
    }

    static void drawKeypoints(cv::Mat& output,
                              const std::vector<Keypoint>& lmks,
                              float disp_scale, float x_off, float y_off) {
        for (size_t i = 0; i < 21; ++i) {
            cv::circle(output,
                       cv::Point(static_cast<int>(lmks[i].x * disp_scale + x_off), static_cast<int>(lmks[i].y * disp_scale + y_off)),
                       3, cv::Scalar(0, 255, 255), -1);
        }
    }

    static void drawLabel(cv::Mat& output, const HandLandmarkResult& result, float disp_scale, float x_off, float y_off) {
        if (result.handedness.empty()) return;
        const auto& lmks = result.landmarks;
        char buf[64];
        snprintf(buf, sizeof(buf), "%s (%.2f)",
                 result.handedness.c_str(), result.confidence);
        cv::putText(output, buf,
                    cv::Point(static_cast<int>(lmks[0].x * disp_scale + x_off),
                              static_cast<int>(lmks[0].y * disp_scale + y_off) - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
};

}  // namespace dxapp

#endif  // DXAPP_HAND_LANDMARK_VISUALIZER_HPP
