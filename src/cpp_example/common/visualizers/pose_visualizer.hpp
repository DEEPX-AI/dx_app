/**
 * @file pose_visualizer.hpp
 * @brief Pose estimation result visualizer with skeleton
 */

#ifndef POSE_VISUALIZER_HPP
#define POSE_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"

namespace dxapp {

// COCO pose skeleton connections (17 keypoints, 19 connections — matching original)
// 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
// 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
// 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
// 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
static const std::vector<std::pair<int, int>> POSE_SKELETON = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},  // Legs + torso
    {5, 11}, {6, 12}, {5, 6},                           // Torso → shoulders
    {5, 7}, {6, 8}, {7, 9}, {8, 10},                    // Arms
    {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4},             // Head
    {3, 5}, {4, 6}                                       // Ears → shoulders
};

// CenterPose 3D bounding box skeleton (8 keypoints, 12 edges)
// Corners: 0-3 front face, 4-7 back face
static const std::vector<std::pair<int, int>> BBOX3D_SKELETON = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Front face
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Back face
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Connecting edges
};

// Limb colors (matching original: blue/magenta/orange/green pattern)
static const std::vector<cv::Scalar> POSE_LIMB_COLORS = {
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(255, 51, 255), cv::Scalar(255, 51, 255),
    cv::Scalar(255, 51, 255), cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0)
};

// Keypoint colors (matching original: green/orange/blue per body part)
static const std::vector<cv::Scalar> POSE_KPT_COLORS = {
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255)
};

/**
 * @brief Visualizer for pose estimation results with skeleton
 */
class PoseVisualizer : public IVisualizer<PoseResult> {
public:
    PoseVisualizer() = default;

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<PoseResult>& results,
                 const PreprocessContext& ctx) override {
        (void)ctx;
        cv::Mat output = frame.clone();
        for (const auto& pose : results) {
            drawBoundingBox(output, pose);
            drawSkeleton(output, pose);
            drawKeypoints(output, pose);
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

    void setKeypointThreshold(float threshold) {
        kp_threshold_ = threshold;
    }

private:
    // Draw bounding box and confidence label for one pose
    void drawBoundingBox(cv::Mat& output, const PoseResult& pose) const {
        if (pose.box.size() < 4) return;
        cv::Point pt1(static_cast<int>(pose.box[0]), static_cast<int>(pose.box[1]));
        cv::Point pt2(static_cast<int>(pose.box[2]), static_cast<int>(pose.box[3]));
        cv::rectangle(output, pt1, pt2, cv::Scalar(0, 255, 0), line_thickness_);

        std::string conf_text = "Person: " +
            std::to_string(static_cast<int>(pose.confidence * 100)) + "%";
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(conf_text, cv::FONT_HERSHEY_SIMPLEX,
                                             0.5, 2, &baseline);
        cv::Point text_pos(pt1.x, pt1.y - 10);
        cv::rectangle(output,
            cv::Point(text_pos.x, text_pos.y - text_size.height),
            cv::Point(text_pos.x + text_size.width, text_pos.y + baseline),
            cv::Scalar(0, 0, 0), -1);
        cv::putText(output, conf_text, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(255, 255, 255), 2);
    }

    // Draw skeleton connections for one pose
    void drawSkeleton(cv::Mat& output, const PoseResult& pose) const {
        const auto& skeleton = (pose.keypoints.size() <= 8) ? BBOX3D_SKELETON : POSE_SKELETON;
        cv::Scalar bbox3d_color(0, 255, 128);

        for (size_t i = 0; i < skeleton.size(); ++i) {
            int idx1 = skeleton[i].first;
            int idx2 = skeleton[i].second;
            if (idx1 >= static_cast<int>(pose.keypoints.size()) ||
                idx2 >= static_cast<int>(pose.keypoints.size())) continue;

            const auto& kp1 = pose.keypoints[idx1];
            const auto& kp2 = pose.keypoints[idx2];
            if (kp1.confidence < kp_threshold_ || kp2.confidence < kp_threshold_) continue;

            cv::Point pt1(static_cast<int>(kp1.x), static_cast<int>(kp1.y));
            cv::Point pt2(static_cast<int>(kp2.x), static_cast<int>(kp2.y));
            cv::Scalar color = (pose.keypoints.size() <= 8) ? bbox3d_color
                : (i < POSE_LIMB_COLORS.size() ? POSE_LIMB_COLORS[i] : bbox3d_color);
            cv::line(output, pt1, pt2, color, line_thickness_, cv::LINE_AA);
        }
    }

    // Draw individual keypoints for one pose
    void drawKeypoints(cv::Mat& output, const PoseResult& pose) const {
        for (size_t i = 0; i < pose.keypoints.size(); ++i) {
            const auto& kp = pose.keypoints[i];
            if (kp.confidence < kp_threshold_) continue;
            cv::Point pt(static_cast<int>(kp.x), static_cast<int>(kp.y));
            cv::circle(output, pt, 3,
                       POSE_KPT_COLORS[i % POSE_KPT_COLORS.size()], -1, cv::LINE_AA);
        }
    }

    int line_thickness_{2};
    double font_scale_{0.5};
    float alpha_{0.6f};
    float kp_threshold_{0.3f};  // Matching original (0.3)
};

}  // namespace dxapp

#endif  // POSE_VISUALIZER_HPP
