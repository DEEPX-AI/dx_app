/**
 * @file ulfg_postprocessor.hpp
 * @brief ULFG (Ultra-Light-Fast-Generic) face detection postprocessor
 * 
 * Ported from Python ulfg_postprocessor.py.
 * 
 * SSD-style lightweight face detector without keypoints.
 * Output: 2 tensors
 *   - scores: [1, N, 2]  (background/face softmax)
 *   - boxes:  [1, N, 4]  (normalized [x1, y1, x2, y2] or raw SSD deltas)
 * 
 * Algorithm:
 *   1. Identify scores (last dim=2) and boxes (last dim=4) tensors
 *   2. Extract face scores from softmax index 1
 *   3. Auto-detect if boxes are normalized coords or SSD deltas
 *   4. Top-K filtering + NMS
 *   5. Scale to original coordinates
 */

#ifndef ULFG_POSTPROCESSOR_HPP
#define ULFG_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

class ULFGPostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    ULFGPostprocessor(int input_width = 320, int input_height = 240,
                      float score_threshold = 0.7f,
                      float nms_threshold = 0.3f)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                              const PreprocessContext& ctx) override {
        std::vector<FaceDetectionResult> results;
        if (outputs.size() < 2) return results;

        // Identify tensors by last dimension: scores(2), boxes(4)
        const dxrt::TensorPtr* scores_t = nullptr;
        const dxrt::TensorPtr* boxes_t = nullptr;
        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            if (last_dim == 2) scores_t = &t;
            else if (last_dim == 4) boxes_t = &t;
        }
        if (!scores_t || !boxes_t) return results;

        auto scores_shape = (*scores_t)->shape();
        int N = static_cast<int>(scores_shape.size() == 3 ? scores_shape[1] : scores_shape[0]);

        const float* scores_data = static_cast<const float*>((*scores_t)->data());
        const float* boxes_data = static_cast<const float*>((*boxes_t)->data());

        // Auto-detect normalized vs pixel coordinates
        bool is_normalized = checkIsNormalized(boxes_data, N);

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<std::array<float, 4>> float_boxes;

        collectCandidates(N, scores_data, boxes_data, is_normalized,
                          nms_boxes, nms_scores, float_boxes);

        if (nms_boxes.empty()) return results;

        std::vector<int> indices;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, indices);

        for (int idx : indices) {
            float x1 = float_boxes[idx][0];
            float y1 = float_boxes[idx][1];
            float x2 = float_boxes[idx][2];
            float y2 = float_boxes[idx][3];

            scaleBoxCoords(ctx, x1, y1, x2, y2);

            FaceDetectionResult face;
            face.box = {x1, y1, x2, y2};
            face.confidence = nms_scores[idx];
            // ULFG has no landmarks
            results.push_back(face);
        }
        return results;
    }

    std::string getModelName() const override { return "ULFG"; }

private:
    // Helper: return true if first 100 boxes are all within [-2, 2] (normalized)
    bool checkIsNormalized(const float* boxes_data, int N) const {
        for (int i = 0; i < std::min(N, 100); ++i) {
            for (int j = 0; j < 4; ++j) {
                if (std::abs(boxes_data[i * 4 + j]) > 2.0f) return false;
            }
        }
        return true;
    }

    // Helper: build NMS candidate lists from raw score/box tensors
    void collectCandidates(int N, const float* scores_data, const float* boxes_data,
                           bool is_normalized,
                           std::vector<cv::Rect>& nms_boxes,
                           std::vector<float>& nms_scores,
                           std::vector<std::array<float, 4>>& float_boxes) const {
        for (int i = 0; i < N; ++i) {
            float face_score = scores_data[i * 2 + 1];
            if (face_score < score_threshold_) continue;

            float x1 = boxes_data[i * 4 + 0];
            float y1 = boxes_data[i * 4 + 1];
            float x2 = boxes_data[i * 4 + 2];
            float y2 = boxes_data[i * 4 + 3];

            if (is_normalized) {
                x1 *= input_width_;  y1 *= input_height_;
                x2 *= input_width_;  y2 *= input_height_;
            }
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(input_width_)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(input_height_)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(input_width_)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(input_height_)));

            nms_boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                         static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            float_boxes.push_back({x1, y1, x2, y2});
            nms_scores.push_back(face_score);
        }
    }

    // Helper: scale a box from input space to original image space
    void scaleBoxCoords(const PreprocessContext& ctx,
                        float& x1, float& y1, float& x2, float& y2) const {
        if (ctx.pad_x == 0 && ctx.pad_y == 0) {
            float sx = static_cast<float>(ctx.original_width)  / input_width_;
            float sy = static_cast<float>(ctx.original_height) / input_height_;
            x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy;
        } else {
            x1 = (x1 - ctx.pad_x) / ctx.scale;
            y1 = (y1 - ctx.pad_y) / ctx.scale;
            x2 = (x2 - ctx.pad_x) / ctx.scale;
            y2 = (y2 - ctx.pad_y) / ctx.scale;
        }
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(ctx.original_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(ctx.original_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(ctx.original_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(ctx.original_height)));
    }

    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // ULFG_POSTPROCESSOR_HPP
