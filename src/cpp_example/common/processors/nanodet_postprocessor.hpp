/**
 * @file nanodet_postprocessor.hpp
 * @brief NanoDet Detection Postprocessor for v3 interface (v3-native, no legacy lib)
 * 
 * Handles NanoDet/NanoDet-Plus output with Distribution Focal Loss (DFL):
 *   - output[0]: [1, N, num_classes + 4 * (reg_max + 1)]
 *     Combined tensor with class logits and bbox distribution values.
 */

#ifndef NANODET_POSTPROCESSOR_HPP
#define NANODET_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dxapp {

class NanoDetPostprocessor : public IPostprocessor<DetectionResult> {
public:
    NanoDetPostprocessor(int input_width = 416, int input_height = 416,
                         float conf_threshold = 0.3f,
                         float nms_threshold = 0.45f,
                         int num_classes = 80,
                         int reg_max = 10,
                         bool is_ort_configured = false)
        : input_width_(input_width), input_height_(input_height),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          num_classes_(num_classes), reg_max_(reg_max) {
        // Pre-compute DFL weights [0, 1, ..., reg_max]
        dfl_weights_.resize(reg_max_ + 1);
        for (int i = 0; i <= reg_max_; ++i) {
            dfl_weights_[i] = static_cast<float>(i);
        }
        // Pre-compute anchor centers and strides
        buildAnchors();
    }

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        if (outputs.empty()) return results;

        const auto& tensor = outputs[0];
        auto shape = tensor->shape();

        // Determine N and cols
        int N, cols;
        if (shape.size() == 3) {
            N = static_cast<int>(shape[1]);
            cols = static_cast<int>(shape[2]);
        } else if (shape.size() == 2) {
            N = static_cast<int>(shape[0]);
            cols = static_cast<int>(shape[1]);
        } else {
            return results;
        }

        const float* data = static_cast<const float*>(tensor->data());
        int bins = reg_max_ + 1;
        int expected_cols = num_classes_ + 4 * bins;

        // Auto-detect reg_max from actual tensor columns if mismatch
        if (cols != expected_cols) {
            int reg_cols = cols - num_classes_;
            if (reg_cols > 0 && reg_cols % 4 == 0) {
                bins = reg_cols / 4;
                reg_max_ = bins - 1;
                // Rebuild DFL weights for new reg_max
                dfl_weights_.resize(bins);
                for (int i = 0; i < bins; ++i) {
                    dfl_weights_[i] = static_cast<float>(i);
                }
            }
        }

        bool need_sigmoid = needsSigmoid_(data, N, cols);

        // Collect detections
        std::vector<cv::Rect> nms_boxes;
        std::vector<std::array<float, 4>> nms_fboxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;

        for (int i = 0; i < N && i < static_cast<int>(anchor_cx_.size()); ++i) {
            const float* row = data + i * cols;

            auto [max_score, max_cls] = findMaxClassScore_(row, num_classes_, need_sigmoid);
            if (max_score < conf_threshold_) continue;

            const float* reg = row + num_classes_;
            float distances[4];
            decodeDFL_(reg, bins, anchor_strides_[i], distances);

            // Convert to x1y1x2y2
            float x1 = anchor_cx_[i] - distances[0];
            float y1 = anchor_cy_[i] - distances[1];
            float x2 = anchor_cx_[i] + distances[2];
            float y2 = anchor_cy_[i] + distances[3];

            nms_boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                        static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            nms_fboxes.push_back({x1, y1, x2, y2});
            nms_scores.push_back(max_score);
            nms_class_ids.push_back(max_cls);
        }

        if (nms_boxes.empty()) return results;

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, conf_threshold_, nms_threshold_, indices);

        for (int idx : indices) {
            float x1 = nms_fboxes[idx][0];
            float y1 = nms_fboxes[idx][1];
            float x2 = nms_fboxes[idx][2];
            float y2 = nms_fboxes[idx][3];

            // Scale to original coordinates
            x1 = std::max(0.0f, std::min((x1 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            y1 = std::max(0.0f, std::min((y1 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));
            x2 = std::max(0.0f, std::min((x2 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            y2 = std::max(0.0f, std::min((y2 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));

            DetectionResult det;
            det.box = {x1, y1, x2, y2};
            det.confidence = nms_scores[idx];
            det.class_id = nms_class_ids[idx];
            det.class_name = dxapp::common::get_coco_class_name(det.class_id);
            results.push_back(det);
        }

        return results;
    }

    std::string getModelName() const override { return "NanoDet"; }

private:
    void buildAnchors() {
        const int strides[] = {8, 16, 32};
        for (int s : strides) {
            int h = input_height_ / s;
            int w = input_width_ / s;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    anchor_cx_.push_back((x + 0.5f) * s);
                    anchor_cy_.push_back((y + 0.5f) * s);
                    anchor_strides_.push_back(static_cast<float>(s));
                }
            }
        }
    }

    // Helper: check if sigmoid is needed for class scores
    bool needsSigmoid_(const float* data, int N, int cols) const {
        int sample_count = std::min(N, 100);
        for (int i = 0; i < sample_count; ++i) {
            const float* row = data + i * cols;
            for (int c = 0; c < num_classes_; ++c) {
                if (row[c] < 0.0f || row[c] > 1.0f) return true;
            }
        }
        return false;
    }

    // Helper: find max class score for a single detection row
    static std::pair<float, int> findMaxClassScore_(const float* row, int num_classes,
                                                     bool need_sigmoid) {
        float max_score = 0.0f;
        int max_cls = 0;
        for (int c = 0; c < num_classes; ++c) {
            float s = need_sigmoid
                ? (1.0f / (1.0f + std::exp(-row[c])))
                : row[c];
            if (s > max_score) {
                max_score = s;
                max_cls = c;
            }
        }
        return {max_score, max_cls};
    }

    // Helper: DFL decode for 4 distance values
    void decodeDFL_(const float* reg, int bins, float stride, float* distances) const {
        for (int side = 0; side < 4; ++side) {
            const float* side_data = reg + side * bins;
            float max_val = *std::max_element(side_data, side_data + bins);
            float sum_exp = 0.0f;
            std::vector<float> exp_vals(bins);
            for (int b = 0; b < bins; ++b) {
                exp_vals[b] = std::exp(side_data[b] - max_val);
                sum_exp += exp_vals[b];
            }
            float dist = 0.0f;
            for (int b = 0; b < bins; ++b) {
                dist += (exp_vals[b] / sum_exp) * dfl_weights_[b];
            }
            distances[side] = dist * stride;
        }
    }

    int input_width_;
    int input_height_;
    float conf_threshold_;
    float nms_threshold_;
    int num_classes_;
    int reg_max_;
    std::vector<float> dfl_weights_;
    std::vector<float> anchor_cx_;
    std::vector<float> anchor_cy_;
    std::vector<float> anchor_strides_;
};

}  // namespace dxapp

#endif  // NANODET_POSTPROCESSOR_HPP
