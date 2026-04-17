/**
 * @file ulfg_face_postprocessor.hpp
 * @brief ULFG (Ultra-Light-Fast-Generic) Face Detection Postprocessor
 *
 * SSD-style lightweight face detection without keypoints.
 * Output format: 2 tensors
 *   - scores: [1, N, 2] (background/face softmax)
 *   - boxes:  [1, N, 4] (face bounding boxes, normalized or pixel coords)
 *
 * Auto-detects tensor order and box format.
 */

#ifndef ULFG_FACE_POSTPROCESSOR_HPP
#define ULFG_FACE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/face_postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dxapp {

class ULFGFacePostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    ULFGFacePostprocessor(int input_width = 320, int input_height = 240,
                          float score_threshold = 0.7f,
                          float nms_threshold = 0.3f,
                          int top_k = 200)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          top_k_(top_k) {}

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                             const PreprocessContext& ctx) override {
        std::vector<FaceDetectionResult> results;
        if (outputs.size() < 2) return results;

        // Identify which tensor is scores (last dim == 2) and which is boxes (last dim == 4)
        const dxrt::TensorPtr* scores_ptr = nullptr;
        const dxrt::TensorPtr* boxes_ptr = nullptr;
        identifyTensors_(outputs, scores_ptr, boxes_ptr);
        if (!scores_ptr || !boxes_ptr) return results;

        auto scores_shape = (*scores_ptr)->shape();
        auto boxes_shape = (*boxes_ptr)->shape();

        int num_proposals = 1;
        for (size_t i = 0; i + 1 < scores_shape.size(); ++i) {
            num_proposals *= static_cast<int>(scores_shape[i]);
        }

        const float* scores_data = static_cast<const float*>((*scores_ptr)->data());
        const float* boxes_data = static_cast<const float*>((*boxes_ptr)->data());
        int score_cols = static_cast<int>(scores_shape.back());

        // Collect faces above threshold
        std::vector<cv::Rect2f> nms_boxes;
        std::vector<float> nms_scores;

        // Auto-detect box format once from data
        box_format_ = detectBoxFormat_(boxes_data, num_proposals);

        for (int i = 0; i < num_proposals; ++i) {
            // Face score is column 1 (column 0 = background)
            float face_score = (score_cols >= 2) ? scores_data[i * score_cols + 1]
                                                 : scores_data[i];
            if (face_score < score_threshold_) continue;

            const float* box = boxes_data + i * 4;
            float x1, y1, x2, y2;
            decodeBox_(box, i, x1, y1, x2, y2);

            nms_scores.push_back(face_score);
            nms_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);  // cv::Rect2f uses x,y,w,h
        }

        if (nms_boxes.empty()) return results;

        // Top-K filter
        if (static_cast<int>(nms_scores.size()) > top_k_) {
            std::vector<int> indices(nms_scores.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_k_, indices.end(),
                [&](int a, int b) { return nms_scores[a] > nms_scores[b]; });
            indices.resize(top_k_);
            std::vector<cv::Rect2f> filtered_boxes;
            std::vector<float> filtered_scores;
            for (int idx : indices) {
                filtered_boxes.push_back(nms_boxes[idx]);
                filtered_scores.push_back(nms_scores[idx]);
            }
            nms_boxes = std::move(filtered_boxes);
            nms_scores = std::move(filtered_scores);
        }

        // NMS
        std::vector<cv::Rect> nms_int_boxes;
        nms_int_boxes.reserve(nms_boxes.size());
        for (const auto& b : nms_boxes) {
            nms_int_boxes.emplace_back(
                static_cast<int>(b.x), static_cast<int>(b.y),
                static_cast<int>(b.width), static_cast<int>(b.height));
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(nms_int_boxes, nms_scores, score_threshold_, nms_threshold_, keep);

        // Build results
        for (int idx : keep) {
            FaceDetectionResult face;
            float bx1 = nms_boxes[idx].x;
            float by1 = nms_boxes[idx].y;
            float bx2 = bx1 + nms_boxes[idx].width;
            float by2 = by1 + nms_boxes[idx].height;
            face.box = {bx1, by1, bx2, by2};
            face.confidence = nms_scores[idx];
            // No landmarks for ULFG
            results.push_back(std::move(face));
        }

        // Scale to original image coordinates
        detail::scaleFaceResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "ULFG"; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    int top_k_;

    // SSD prior cache
    mutable std::vector<std::array<float, 4>> priors_;
    // SSD prior variance
    static constexpr float kVariance0 = 0.1f;
    static constexpr float kVariance1 = 0.2f;
    // Whether boxes are SSD deltas (detected once per run)
    mutable int box_format_ = -1;  // -1=unknown, 0=normalized, 1=pixel, 2=ssd_deltas

    void identifyTensors_(const dxrt::TensorPtrs& outputs,
                          const dxrt::TensorPtr*& scores,
                          const dxrt::TensorPtr*& boxes) {
        auto last_dim = [](const dxrt::TensorPtr& t) -> int {
            auto s = t->shape();
            return s.empty() ? 0 : static_cast<int>(s.back());
        };
        int d0 = last_dim(outputs[0]);
        int d1 = last_dim(outputs[1]);
        if (d0 == 2 && d1 == 4) {
            scores = &outputs[0]; boxes = &outputs[1];
        } else if (d0 == 4 && d1 == 2) {
            scores = &outputs[1]; boxes = &outputs[0];
        } else {
            // Fallback: smaller last dim = scores
            scores = (d0 <= d1) ? &outputs[0] : &outputs[1];
            boxes = (d0 <= d1) ? &outputs[1] : &outputs[0];
        }
    }

    // Generate SSD prior boxes for ULFG face detector (matching Python impl)
    void generatePriors_() const {
        if (!priors_.empty()) return;
        // ULFG prior box configuration: min_sizes per feature map level
        const int min_sizes[][3] = {{10, 16, 24}, {32, 48, 0}, {64, 96, 0}, {128, 192, 256}};
        const int num_sizes[]    = {3, 2, 2, 3};
        const int strides[]      = {8, 16, 32, 64};
        const int num_levels     = 4;

        for (int k = 0; k < num_levels; ++k) {
            int fh = static_cast<int>(std::ceil(static_cast<float>(input_height_) / strides[k]));
            int fw = static_cast<int>(std::ceil(static_cast<float>(input_width_) / strides[k]));
            for (int i = 0; i < fh; ++i) {
                for (int j = 0; j < fw; ++j) {
                    for (int s = 0; s < num_sizes[k]; ++s) {
                        float cx = (j + 0.5f) * strides[k] / input_width_;
                        float cy = (i + 0.5f) * strides[k] / input_height_;
                        float w  = static_cast<float>(min_sizes[k][s]) / input_width_;
                        float h  = static_cast<float>(min_sizes[k][s]) / input_height_;
                        priors_.push_back({cx, cy, w, h});
                    }
                }
            }
        }
    }

    // Detect box format from data range
    int detectBoxFormat_(const float* boxes_data, int num_proposals) const {
        int neg_count = 0;
        int sample = std::min(num_proposals * 4, 4000);
        for (int i = 0; i < sample; ++i) {
            if (boxes_data[i] < 0.0f) neg_count++;
        }
        float neg_ratio = static_cast<float>(neg_count) / sample;
        if (neg_ratio > 0.1f) return 2;  // SSD deltas

        // Check if normalized [0,1] or pixel coordinates
        float abs_sum = 0.0f;
        for (int i = 0; i < std::min(sample, 1000); ++i)
            abs_sum += std::abs(boxes_data[i]);
        float avg_abs = abs_sum / std::min(sample, 1000);
        return (avg_abs < 2.0f) ? 0 : 1;  // 0=normalized, 1=pixel
    }

    void decodeBox_(const float* raw, int box_idx, float& x1, float& y1, float& x2, float& y2) const {
        if (box_format_ == 2) {
            // SSD delta decoding: [dcx, dcy, dw, dh] relative to prior
            generatePriors_();
            if (box_idx < static_cast<int>(priors_.size())) {
                float pcx = priors_[box_idx][0], pcy = priors_[box_idx][1];
                float pw  = priors_[box_idx][2], ph  = priors_[box_idx][3];
                float cx = pcx + raw[0] * kVariance0 * pw;
                float cy = pcy + raw[1] * kVariance0 * ph;
                float bw = pw * std::exp(raw[2] * kVariance1);
                float bh = ph * std::exp(raw[3] * kVariance1);
                x1 = (cx - bw * 0.5f) * input_width_;
                y1 = (cy - bh * 0.5f) * input_height_;
                x2 = (cx + bw * 0.5f) * input_width_;
                y2 = (cy + bh * 0.5f) * input_height_;
                return;
            }
        }
        if (box_format_ == 0) {
            // Normalized [0,1] → scale to input size
            x1 = raw[0] * input_width_;
            y1 = raw[1] * input_height_;
            x2 = raw[2] * input_width_;
            y2 = raw[3] * input_height_;
        } else {
            // Pixel coordinates — pass through
            x1 = raw[0]; y1 = raw[1]; x2 = raw[2]; y2 = raw[3];
        }
    }
};

}  // namespace dxapp

#endif  // ULFG_FACE_POSTPROCESSOR_HPP
