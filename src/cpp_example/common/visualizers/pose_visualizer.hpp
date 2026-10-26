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

// MediaPipe hand skeleton (21 keypoints, 23 bones): 0 wrist, then four
// joints per digit from base to tip - thumb 1-4, index 5-8, middle 9-12,
// ring 13-16, pinky 17-20 - plus the palm's knuckle line.
static const size_t HAND_KEYPOINT_COUNT = 21;
static const std::vector<std::pair<int, int>> HAND_SKELETON = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},          // Thumb
    {0, 5}, {5, 6}, {6, 7}, {7, 8},          // Index
    {0, 9}, {9, 10}, {10, 11}, {11, 12},     // Middle
    {0, 13}, {13, 14}, {14, 15}, {15, 16},   // Ring
    {0, 17}, {17, 18}, {18, 19}, {19, 20},   // Pinky
    {5, 9}, {9, 13}, {13, 17}                // Palm
};

// One BGR color per digit, then the palm (HandLandmarkVisualizer's colors).
static const std::vector<cv::Scalar> HAND_BONE_COLORS = {
    cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(200, 200, 200)
};

// 68-point face landmarks (iBUG 300-W order, as 3DDFA emits them): jaw
// 0-16, brows 17-21 / 22-26, nose bridge 27-30, nostrils 31-35, eyes
// 36-41 / 42-47 (closed), outer lip 48-59 and inner lip 60-67 (closed).
// One color per part, FaceAlignmentVisualizer's.
static const size_t FACE68_KEYPOINT_COUNT = 68;
struct Face68Part { int first; int last; bool closed; cv::Scalar color; };
static const std::vector<Face68Part> FACE68_PARTS = {
    {0, 16, false, cv::Scalar(200, 200, 200)},   // jaw
    {17, 21, false, cv::Scalar(0, 255, 0)},      // left brow
    {22, 26, false, cv::Scalar(0, 255, 0)},      // right brow
    {27, 30, false, cv::Scalar(255, 200, 0)},    // nose bridge
    {31, 35, false, cv::Scalar(255, 200, 0)},    // nostrils
    {36, 41, true, cv::Scalar(255, 0, 0)},       // left eye
    {42, 47, true, cv::Scalar(255, 0, 0)},       // right eye
    {48, 59, true, cv::Scalar(0, 0, 255)},       // outer lip
    {60, 67, true, cv::Scalar(0, 100, 255)}      // inner lip
};

inline std::vector<std::pair<int, int>> Face68Bones(std::vector<cv::Scalar>* colors) {
    std::vector<std::pair<int, int>> bones;
    for (const auto& part : FACE68_PARTS) {
        for (int i = part.first; i < part.last; ++i) {
            bones.emplace_back(i, i + 1);
            if (colors) colors->push_back(part.color);
        }
        if (part.closed) {
            bones.emplace_back(part.last, part.first);
            if (colors) colors->push_back(part.color);
        }
    }
    return bones;
}
static const std::vector<std::pair<int, int>> FACE68_SKELETON = Face68Bones(nullptr);
static const std::vector<cv::Scalar> FACE68_BONE_COLORS = [] {
    std::vector<cv::Scalar> colors;
    Face68Bones(&colors);
    return colors;
}();

static const size_t COCO_KEYPOINT_COUNT = 17;
static const size_t BBOX3D_KEYPOINT_COUNT = 8;

/// How PoseVisualizer joins a pose's keypoints (see layoutOf()).
enum class KeypointLayout { kCoco17, kHand21, kFace68, kBox3d, kPoints };

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
        for (const auto& pose : results) {
            drawBoundingBox(output, pose, disp_scale, x_off, y_off);
            drawSkeleton(output, pose, disp_scale, x_off, y_off);
            drawKeypoints(output, pose, disp_scale, x_off, y_off);
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

    /// Keypoints with no order or connectivity (SuperPoint): drawn as points
    /// only, whatever their count, and all of them.
    void setUnorderedKeypoints(bool unordered) {
        unordered_ = unordered;
    }

private:
    // Draw bounding box and confidence label for one pose
    void drawBoundingBox(cv::Mat& output, const PoseResult& pose, float disp_scale, float x_off, float y_off) const {
        if (pose.box.size() < 4) return;
        cv::Point pt1(static_cast<int>(pose.box[0] * disp_scale + x_off), static_cast<int>(pose.box[1] * disp_scale + y_off));
        cv::Point pt2(static_cast<int>(pose.box[2] * disp_scale + x_off), static_cast<int>(pose.box[3] * disp_scale + y_off));
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
    void drawSkeleton(cv::Mat& output, const PoseResult& pose, float disp_scale, float x_off, float y_off) const {
        const KeypointLayout layout = layoutOf(pose);
        const std::vector<std::pair<int, int>>* found = skeletonOf(layout);
        if (found == nullptr) return;  // points only
        const auto& skeleton = *found;

        for (size_t i = 0; i < skeleton.size(); ++i) {
            int idx1 = skeleton[i].first;
            int idx2 = skeleton[i].second;
            if (idx1 >= static_cast<int>(pose.keypoints.size()) ||
                idx2 >= static_cast<int>(pose.keypoints.size())) continue;

            const auto& kp1 = pose.keypoints[idx1];
            const auto& kp2 = pose.keypoints[idx2];
            if (kp1.confidence < kp_threshold_ || kp2.confidence < kp_threshold_) continue;

            cv::Point pt1(static_cast<int>(kp1.x * disp_scale + x_off), static_cast<int>(kp1.y * disp_scale + y_off));
            cv::Point pt2(static_cast<int>(kp2.x * disp_scale + x_off), static_cast<int>(kp2.y * disp_scale + y_off));
            cv::line(output, pt1, pt2, boneColor(layout, i),
                     layout == KeypointLayout::kFace68 ? 1 : line_thickness_, cv::LINE_AA);
        }
    }

    // Draw individual keypoints for one pose
    void drawKeypoints(cv::Mat& output, const PoseResult& pose, float disp_scale, float x_off, float y_off) const {
        const KeypointLayout layout = layoutOf(pose);
        const bool fine_marks = layout == KeypointLayout::kFace68 || layout == KeypointLayout::kPoints;
        for (size_t i = 0; i < pose.keypoints.size(); ++i) {
            const auto& kp = pose.keypoints[i];
            // Unordered points (SuperPoint) are each a detection that already
            // passed the model's own threshold; a pose's joint is drawn only
            // when it is confident.
            if (!unordered_ && kp.confidence < kp_threshold_) continue;
            cv::Point pt(static_cast<int>(kp.x * disp_scale + x_off), static_cast<int>(kp.y * disp_scale + y_off));
            const cv::Scalar color =
                layout == KeypointLayout::kFace68 ? cv::Scalar(0, 255, 255)
                : layout == KeypointLayout::kPoints ? cv::Scalar(0, 255, 0)
                : POSE_KPT_COLORS[i % POSE_KPT_COLORS.size()];
            cv::circle(output, pt, fine_marks ? 2 : 3, color, -1, cv::LINE_AA);
        }
    }

    /// What one pose's keypoints are, from their count: COCO body (17),
    /// MediaPipe hand (21), 68-point face, CenterPose 3D box (8). Any other
    /// count, and every set marked unordered, is drawn as points only.
    KeypointLayout layoutOf(const PoseResult& pose) const {
        if (unordered_) return KeypointLayout::kPoints;
        switch (pose.keypoints.size()) {
            case COCO_KEYPOINT_COUNT: return KeypointLayout::kCoco17;
            case HAND_KEYPOINT_COUNT: return KeypointLayout::kHand21;
            case FACE68_KEYPOINT_COUNT: return KeypointLayout::kFace68;
            case BBOX3D_KEYPOINT_COUNT: return KeypointLayout::kBox3d;
            default: return KeypointLayout::kPoints;
        }
    }

    static const std::vector<std::pair<int, int>>* skeletonOf(KeypointLayout layout) {
        switch (layout) {
            case KeypointLayout::kCoco17: return &POSE_SKELETON;
            case KeypointLayout::kHand21: return &HAND_SKELETON;
            case KeypointLayout::kFace68: return &FACE68_SKELETON;
            case KeypointLayout::kBox3d: return &BBOX3D_SKELETON;
            default: return nullptr;
        }
    }

    static cv::Scalar boneColor(KeypointLayout layout, size_t bone) {
        const cv::Scalar bbox3d_color(0, 255, 128);
        switch (layout) {
            case KeypointLayout::kHand21:
                return HAND_BONE_COLORS[std::min<size_t>(bone / 4, HAND_BONE_COLORS.size() - 1)];
            case KeypointLayout::kFace68: return FACE68_BONE_COLORS[bone];
            case KeypointLayout::kCoco17:
                return bone < POSE_LIMB_COLORS.size() ? POSE_LIMB_COLORS[bone] : bbox3d_color;
            default: return bbox3d_color;
        }
    }

    int line_thickness_{2};
    double font_scale_{0.5};
    float alpha_{0.6f};
    float kp_threshold_{0.3f};  // Matching original (0.3)
    bool unordered_{false};
};

/**
 * @brief Keypoint-only visualizer — draws dots without skeleton connections.
 *
 * Intended for models like SuperPoint that produce unstructured keypoints
 * with no meaningful ordering or connectivity between them.
 */
class KeypointOnlyVisualizer : public IVisualizer<PoseResult> {
public:
    explicit KeypointOnlyVisualizer(int radius = 3,
                                    cv::Scalar color = cv::Scalar(0, 255, 0),
                                    float conf_threshold = 0.015f)
        : radius_(radius), color_(color), conf_threshold_(conf_threshold) {}

    cv::Mat draw(const cv::Mat& image,
                 const std::vector<PoseResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat output = image.clone();

        int total = 0;
        for (const auto& pose : results) {
            for (const auto& kp : pose.keypoints) {
                if (kp.confidence < conf_threshold_) continue;
                int r = std::max(1, static_cast<int>(radius_ * kp.confidence * 3));
                r = std::min(r, radius_ + 2);
                cv::circle(output,
                           cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                           r, color_, -1, cv::LINE_AA);
                ++total;
            }
        }

        cv::putText(output,
                    "Keypoints: " + std::to_string(total),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2);
        return output;
    }

    void setParameters(int /*line_thickness*/ = 2,
                       double /*font_scale*/ = 0.5,
                       float /*alpha*/ = 0.6f) override {}

private:
    int radius_;
    cv::Scalar color_;
    float conf_threshold_;
};

}  // namespace dxapp

#endif  // POSE_VISUALIZER_HPP
