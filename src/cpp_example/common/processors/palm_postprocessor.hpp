/**
 * @file palm_postprocessor.hpp
 * @brief Palm Detection (MediaPipe) postprocessor
 * 
 * Ported from Python palm_postprocessor.py.
 * 
 * SSD-style palm detector with MediaPipe anchor generation.
 * Output: 2 tensors
 *   - regression: [1, N, 18]  (cx_off, cy_off, w, h, 7 keypoints × 2)
 *   - scores:     [1, N, 1]   (sigmoid logit)
 * 
 * Algorithm:
 *   1. Generate anchors using MediaPipe SSD anchor spec
 *   2. Apply sigmoid to raw scores
 *   3. Decode boxes: cx=anchor_cx + off*anchor_w, etc.
 *   4. NMS
 *   5. Scale to original coordinates
 */

#ifndef PALM_POSTPROCESSOR_HPP
#define PALM_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

class PalmDetectionPostprocessor : public IPostprocessor<DetectionResult> {
public:
    PalmDetectionPostprocessor(int input_width = 192, int input_height = 192,
                               float score_threshold = 0.5f,
                               float nms_threshold = 0.3f)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold) {
        generateAnchors();
    }

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        if (outputs.size() < 2) return results;

        // Identify regression (last_dim=18) and scores (last_dim=1)
        const dxrt::TensorPtr* reg_t = nullptr;
        const dxrt::TensorPtr* score_t = nullptr;
        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            if (last_dim == 18 || last_dim >= 8) reg_t = &t;
            else if (last_dim == 1) score_t = &t;
        }
        if (!reg_t || !score_t) return results;

        auto reg_shape = (*reg_t)->shape();
        int N = static_cast<int>(reg_shape.size() == 3 ? reg_shape[1] : reg_shape[0]);
        int reg_cols = static_cast<int>(reg_shape.back());
        N = std::min(N, static_cast<int>(anchors_.size()));

        const float* reg_data = static_cast<const float*>((*reg_t)->data());
        const float* score_data = static_cast<const float*>((*score_t)->data());

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<std::array<float, 4>> float_boxes;

        for (int i = 0; i < N; ++i) {
            float raw_score = score_data[i];
            float score = 1.0f / (1.0f + std::exp(-raw_score)); // sigmoid
            if (score < score_threshold_) continue;

            const float* reg = reg_data + i * reg_cols;
            float anchor_cx = anchors_[i][0];
            float anchor_cy = anchors_[i][1];
            float anchor_w = anchors_[i][2];
            float anchor_h = anchors_[i][3];

            float cx = anchor_cx + reg[0] * anchor_w;
            float cy = anchor_cy + reg[1] * anchor_h;
            float w = reg[2] * anchor_w;
            float h = reg[3] * anchor_h;

            float x1 = (cx - w * 0.5f) * input_width_;
            float y1 = (cy - h * 0.5f) * input_height_;
            float x2 = (cx + w * 0.5f) * input_width_;
            float y2 = (cy + h * 0.5f) * input_height_;

            nms_boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            float_boxes.push_back({x1, y1, x2, y2});
            nms_scores.push_back(score);
        }

        if (nms_boxes.empty()) return results;

        std::vector<int> indices;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, indices);

        for (int idx : indices) {
            float x1 = float_boxes[idx][0];
            float y1 = float_boxes[idx][1];
            float x2 = float_boxes[idx][2];
            float y2 = float_boxes[idx][3];

            // Scale to original image coordinates
            if (ctx.pad_x == 0 && ctx.pad_y == 0) {
                float sx = static_cast<float>(ctx.original_width) / input_width_;
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

            DetectionResult det;
            det.box = {x1, y1, x2, y2};
            det.confidence = nms_scores[idx];
            det.class_id = 0; // Palm = single class
            results.push_back(det);
        }
        return results;
    }

    std::string getModelName() const override { return "PalmDetection"; }

private:
    void generateAnchors() {
        // MediaPipe SSD anchor generation for palm detection
        // strides: [8, 16, 16, 16] with 2 anchors per location
        int strides[] = {8, 16, 16, 16};
        for (int stride : strides) {
            int grid_h = input_height_ / stride;
            int grid_w = input_width_ / stride;
            for (int y = 0; y < grid_h; ++y) {
                for (int x = 0; x < grid_w; ++x) {
                    float cx = (x + 0.5f) / grid_w;
                    float cy = (y + 0.5f) / grid_h;
                    float anchor_w = 1.0f;
                    float anchor_h = 1.0f;
                    anchors_.push_back({cx, cy, anchor_w, anchor_h});
                    anchors_.push_back({cx, cy, anchor_w, anchor_h});
                }
            }
        }
    }

    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    std::vector<std::array<float, 4>> anchors_; // cx, cy, w, h (normalized)
};

}  // namespace dxapp

#endif  // PALM_POSTPROCESSOR_HPP
