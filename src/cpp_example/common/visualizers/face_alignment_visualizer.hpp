/**
 * @file face_alignment_visualizer.hpp
 * @brief Visualizer for 3D face alignment models (3DDFA v2, etc.)
 *
 * Draws 68-point facial landmarks and pose axes on the image.
 * Part of DX-APP v3.0.0 refactoring.
 */

#ifndef DXAPP_FACE_ALIGNMENT_VISUALIZER_HPP
#define DXAPP_FACE_ALIGNMENT_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include <opencv2/opencv.hpp>

namespace dxapp {

class FaceAlignmentVisualizer : public IVisualizer<FaceAlignmentResult> {
public:
    void setParameters(int /*line_thickness*/ = 2,
                       double /*font_scale*/ = 0.5,
                       float /*alpha*/ = 0.6f) override {
        /* No-op: FaceAlignmentVisualizer does not use these drawing parameters */
    }

    cv::Mat draw(const cv::Mat& image,
                 const std::vector<FaceAlignmentResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat output = image.clone();
        if (results.empty()) return output;

        const auto& result = results[0];
        const auto& lmks = result.landmarks_2d;

        // 68-point face landmark connectivity
        struct FacePart {
            std::vector<int> indices;
            cv::Scalar color;
        };

        std::vector<FacePart> parts = {
            {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},          {200,200,200}},  // contour
            {{17,18,19,20,21},                                       {0,255,0}},      // left eyebrow
            {{22,23,24,25,26},                                       {0,255,0}},      // right eyebrow
            {{27,28,29,30},                                          {255,200,0}},    // nose bridge
            {{31,32,33,34,35},                                       {255,200,0}},    // nose bottom
            {{36,37,38,39,40,41,36},                                 {255,0,0}},      // left eye (closed)
            {{42,43,44,45,46,47,42},                                 {255,0,0}},      // right eye (closed)
            {{48,49,50,51,52,53,54,55,56,57,58,59,48},              {0,0,255}},      // outer lip
            {{60,61,62,63,64,65,66,67,60},                           {0,100,255}},    // inner lip
        };

        for (const auto& part : parts) {
            for (size_t i = 0; i + 1 < part.indices.size(); ++i) {
                int i1 = part.indices[i];
                int i2 = part.indices[i + 1];
                if (i1 < static_cast<int>(lmks.size()) && i2 < static_cast<int>(lmks.size())) {
                    cv::Point pt1(static_cast<int>(lmks[i1].x), static_cast<int>(lmks[i1].y));
                    cv::Point pt2(static_cast<int>(lmks[i2].x), static_cast<int>(lmks[i2].y));
                    cv::line(output, pt1, pt2, part.color, 1, cv::LINE_AA);
                }
            }
        }

        // Draw landmark points
        for (size_t i = 0; i < std::min(static_cast<size_t>(68), lmks.size()); ++i) {
            cv::circle(output, cv::Point(static_cast<int>(lmks[i].x),
                                         static_cast<int>(lmks[i].y)),
                       2, cv::Scalar(0, 255, 255), -1);
        }

        // Display pose text
        if (result.pose.size() >= 3) {
            char buf[128];
            snprintf(buf, sizeof(buf), "Yaw: %.1f Pitch: %.1f Roll: %.1f",
                     result.pose[0], result.pose[1], result.pose[2]);
            cv::putText(output, buf, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        return output;
    }
};

}  // namespace dxapp

#endif  // DXAPP_FACE_ALIGNMENT_VISUALIZER_HPP
