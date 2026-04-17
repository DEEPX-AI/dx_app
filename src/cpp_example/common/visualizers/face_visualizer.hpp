/**
 * @file face_visualizer.hpp
 * @brief Face detection result visualizer with landmarks
 */

#ifndef FACE_VISUALIZER_HPP
#define FACE_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"

namespace dxapp {

/**
 * @brief Visualizer for face detection results with landmarks
 */
class FaceVisualizer : public IVisualizer<FaceDetectionResult> {
public:
    FaceVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<FaceDetectionResult>& results,
                 const PreprocessContext& ctx) override {
        cv::Mat output = frame.clone();
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

        for (const auto& face : results) {
            if (face.box.size() < 4) continue;

            // Draw bounding box
            cv::Point pt1(static_cast<int>(face.box[0] * disp_scale + x_off), static_cast<int>(face.box[1] * disp_scale + y_off));
            cv::Point pt2(static_cast<int>(face.box[2] * disp_scale + x_off), static_cast<int>(face.box[3] * disp_scale + y_off));
            cv::rectangle(output, pt1, pt2, cv::Scalar(0, 255, 0), line_thickness_);

            // Draw confidence
            std::string label = "Face: " + std::to_string(static_cast<int>(face.confidence * 100)) + "%";
            int baseline;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                   font_scale_, 1, &baseline);
            cv::Point label_pt(pt1.x, std::max(pt1.y - 5, label_size.height));
            cv::putText(output, label, label_pt, cv::FONT_HERSHEY_SIMPLEX,
                       font_scale_, cv::Scalar(0, 255, 0), 1);

            // Draw landmarks (5 points: left eye, right eye, nose, left mouth, right mouth)
            for (size_t i = 0; i < face.landmarks.size(); ++i) {
                const auto& lm = face.landmarks[i];
                cv::Point pt(static_cast<int>(lm.x * disp_scale + x_off), static_cast<int>(lm.y * disp_scale + y_off));
                cv::Scalar color;
                switch (i) {
                    case 0: color = cv::Scalar(255, 0, 0); break;   // Left eye - Blue
                    case 1: color = cv::Scalar(0, 0, 255); break;   // Right eye - Red
                    case 2: color = cv::Scalar(0, 255, 0); break;   // Nose - Green
                    case 3: color = cv::Scalar(255, 255, 0); break; // Left mouth - Cyan
                    case 4: color = cv::Scalar(255, 0, 255); break; // Right mouth - Magenta
                    default: color = cv::Scalar(255, 255, 255); break;
                }
                cv::circle(output, pt, 3, color, -1);
            }
        }

        return output;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        line_thickness_ = line_thickness;
        font_scale_ = font_scale;
        alpha_ = alpha;
    }

private:
    int line_thickness_{2};
    double font_scale_{0.5};
    float alpha_{0.6f};
};

}  // namespace dxapp

#endif  // FACE_VISUALIZER_HPP
