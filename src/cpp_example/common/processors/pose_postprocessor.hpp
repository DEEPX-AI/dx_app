/**
 * @file pose_postprocessor.hpp
 * @brief Unified Pose Estimation Postprocessors for v3 interface
 * 
 * Groups all pose estimation postprocessors:
 *   - YOLOv5Pose (6 args): obj_threshold included
 */

#ifndef POSE_POSTPROCESSOR_HPP
#define POSE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/result_converters.hpp"

// Postprocess headers
#include "anchor_pose_postprocessor.hpp"

namespace dxapp {

namespace detail {

inline void scalePoseResults(std::vector<PoseResult>& results,
                             const PreprocessContext& ctx) {
    for (auto& pose : results) {
        scaleBox(pose.box, ctx);
        for (auto& kp : pose.keypoints) {
            scaleKeypoint(kp, ctx);
        }
    }
}

}  // namespace detail

// ============================================================================
// YOLOv5Pose Postprocessor (6 args: obj_threshold included)
// ============================================================================
class YOLOv5PosePostprocessor : public IPostprocessor<PoseResult> {
public:
    YOLOv5PosePostprocessor(int input_width = 640, int input_height = 640,
                            float obj_threshold = 0.5f,
                            float score_threshold = 0.5f,
                            float nms_threshold = 0.45f,
                            bool is_ort_configured = false)
        : impl_(input_width, input_height, obj_threshold, score_threshold,
                nms_threshold, is_ort_configured) {}

    std::vector<PoseResult> process(const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        auto results = convertAllWith(legacy_results,
            [](const YOLOv5PoseResult& s) { return convertToPose(s); });
        detail::scalePoseResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv5Pose"; }

private:
    YOLOv5PosePostProcess impl_;
};

// ============================================================================
// YOLOv8Pose Postprocessor (anchor-free, transposed output, no objectness)
// Output: [1, 56, N] -> transpose -> [N, 56] = [cx, cy, w, h, score, kp_x*17, kp_y*17, kp_conf*17]
// ============================================================================
class YOLOv8PosePostprocessor : public IPostprocessor<PoseResult> {
public:
    YOLOv8PosePostprocessor(int input_width = 640, int input_height = 640,
                            float score_threshold = 0.3f,
                            float nms_threshold = 0.45f,
                            int num_keypoints = 17,
                            bool is_ort_configured = false)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          num_keypoints_(num_keypoints) {}

    std::vector<PoseResult> process(const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) override {
        std::vector<PoseResult> results;
        if (outputs.empty()) return results;

        const auto& tensor = outputs[0];
        auto shape = tensor->shape();
        const float* raw_data = static_cast<const float*>(tensor->data());

        // Determine dimensions - output is [1, 56, N] (YOLOv8) or [1, N, 57] (YOLO26 post-NMS)
        int channels, N;
        bool needs_transpose;
        if (shape.size() == 3) {
            // Heuristic: expected_channels = 5 + num_keypoints*3 (e.g. 56 or 57)
            // If shape[1] < shape[2], tensor is [1, C, N] (transposed, needs transpose)
            // If shape[1] > shape[2], tensor is [1, N, C] (row-major, already correct)
            if (shape[1] <= shape[2]) {
                channels = static_cast<int>(shape[1]);
                N = static_cast<int>(shape[2]);
                needs_transpose = true;
            } else {
                N = static_cast<int>(shape[1]);
                channels = static_cast<int>(shape[2]);
                needs_transpose = false;
            }
        } else {
            if (shape[0] <= shape[1]) {
                channels = static_cast<int>(shape[0]);
                N = static_cast<int>(shape[1]);
                needs_transpose = true;
            } else {
                N = static_cast<int>(shape[0]);
                channels = static_cast<int>(shape[1]);
                needs_transpose = false;
            }
        }

        // Get row-major data: [N, channels]
        std::vector<float> transposed;
        const float* row_data;
        if (needs_transpose) {
            transposed.resize(N * channels);
            for (int c = 0; c < channels; ++c) {
                for (int n = 0; n < N; ++n) {
                    transposed[n * channels + c] = raw_data[c * N + n];
                }
            }
            row_data = transposed.data();
        } else {
            row_data = raw_data;
        }

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_indices;

        for (int i = 0; i < N; ++i) {
            const float* row = row_data + i * channels;
            float score = row[4];
            if (score < score_threshold_) continue;

            float x1, y1, bw, bh;
            if (needs_transpose) {
                // YOLOv8 pre-NMS: [cx, cy, w, h] → convert to corner
                float cx = row[0], cy = row[1], w = row[2], h = row[3];
                x1 = cx - w * 0.5f;
                y1 = cy - h * 0.5f;
                bw = w;
                bh = h;
            } else {
                // YOLO26 post-NMS: [x1, y1, x2, y2] already corners
                x1 = row[0];
                y1 = row[1];
                bw = row[2] - row[0];
                bh = row[3] - row[1];
            }

            nms_boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                        static_cast<int>(bw), static_cast<int>(bh)));
            nms_scores.push_back(score);
            nms_indices.push_back(i);
        }

        if (nms_boxes.empty()) return results;

        // NMS
        std::vector<int> keep;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, keep);

        for (int k : keep) {
            int i = nms_indices[k];
            const float* row = row_data + i * channels;

            float x1, y1, x2, y2;
            if (needs_transpose) {
                float cx = row[0], cy = row[1], w = row[2], h = row[3];
                x1 = (cx - w * 0.5f - ctx.pad_x) / ctx.scale;
                y1 = (cy - h * 0.5f - ctx.pad_y) / ctx.scale;
                x2 = (cx + w * 0.5f - ctx.pad_x) / ctx.scale;
                y2 = (cy + h * 0.5f - ctx.pad_y) / ctx.scale;
            } else {
                x1 = (row[0] - ctx.pad_x) / ctx.scale;
                y1 = (row[1] - ctx.pad_y) / ctx.scale;
                x2 = (row[2] - ctx.pad_x) / ctx.scale;
                y2 = (row[3] - ctx.pad_y) / ctx.scale;
            }

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(ctx.original_width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(ctx.original_height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(ctx.original_width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(ctx.original_height)));

            PoseResult pose;
            pose.box = {x1, y1, x2, y2};
            pose.confidence = nms_scores[k];

            // Parse keypoints: x, y, conf triplets
            // For transposed (pre-NMS) layout: [cx,cy,w,h,score, kp...] → kp at index 5
            // For row-major (post-NMS) layout: [x1,y1,x2,y2,score,class_id, kp...] → kp at index 6
            int kp_offset = needs_transpose ? 5 : 6;
            const float* kp_data = row + kp_offset;
            for (int kp = 0; kp < num_keypoints_; ++kp) {
                float kp_x = (kp_data[kp * 3] - ctx.pad_x) / ctx.scale;
                float kp_y = (kp_data[kp * 3 + 1] - ctx.pad_y) / ctx.scale;
                float kp_conf = kp_data[kp * 3 + 2];
                kp_x = std::max(0.0f, std::min(kp_x, static_cast<float>(ctx.original_width)));
                kp_y = std::max(0.0f, std::min(kp_y, static_cast<float>(ctx.original_height)));
                pose.keypoints.emplace_back(kp_x, kp_y, kp_conf);
            }

            results.push_back(std::move(pose));
        }

        return results;
    }

    std::string getModelName() const override { return "YOLOv8Pose"; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    int num_keypoints_;
};

}  // namespace dxapp

#endif  // POSE_POSTPROCESSOR_HPP
