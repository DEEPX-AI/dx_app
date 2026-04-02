/**
 * @file ssd_postprocessor.hpp
 * @brief SSD Detection Postprocessor for v3 interface (v3-native, no legacy lib)
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Handles SSD (MobileNetV1/V2-Lite) output format:
 *   - output[0]: [1, N, num_classes+1] class scores (softmax, class 0 = background)
 *   - output[1]: [1, N, 4] bounding boxes (normalized [ymin, xmin, ymax, xmax])
 */

#ifndef SSD_POSTPROCESSOR_HPP
#define SSD_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dxapp {

class SSDPostprocessor : public IPostprocessor<DetectionResult> {
public:
    SSDPostprocessor(int input_width = 300, int input_height = 300,
                     float conf_threshold = 0.3f,
                     float nms_threshold = 0.45f,
                     int num_classes = 20,
                     bool has_background = true,
                     const std::string& label_set = "voc")
        : input_width_(input_width), input_height_(input_height),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          num_classes_(num_classes), has_background_(has_background),
          label_set_(label_set) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        if (outputs.size() < 2) return results;

        auto shape0 = outputs[0]->shape();
        auto shape1 = outputs[1]->shape();

        const dxrt::TensorPtr* scores_ptr = &outputs[0];
        const dxrt::TensorPtr* boxes_ptr = &outputs[1];
        if (shape0.back() == 4) {
            boxes_ptr = &outputs[0];
            scores_ptr = &outputs[1];
        } else if (shape1.back() == 4) {
            scores_ptr = &outputs[0];
            boxes_ptr = &outputs[1];
        }

        auto scores_shape = (*scores_ptr)->shape();
        auto boxes_shape = (*boxes_ptr)->shape();

        int num_proposals = 1;
        if (scores_shape.size() >= 2) {
            num_proposals = static_cast<int>(scores_shape.size() == 3 ? scores_shape[1] : scores_shape[0]);
        }
        int score_cols = static_cast<int>(scores_shape.back());

        const float* scores_data = static_cast<const float*>((*scores_ptr)->data());
        const float* boxes_data = static_cast<const float*>((*boxes_ptr)->data());

        int fg_offset = has_background_ ? 1 : 0;

        // Collect detections above threshold
        std::vector<cv::Rect> nms_boxes;
        std::vector<std::array<float, 4>> nms_fboxes;  // float coords for accurate scaling
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;

        for (int i = 0; i < num_proposals; ++i) {
            const float* row_scores = scores_data + i * score_cols;

            auto [max_score, max_cls] = findMaxFgClass_(row_scores, score_cols, fg_offset);
            if (max_score < conf_threshold_) continue;

            const float* box = boxes_data + i * 4;
            // Defer box decoding — collect raw values for format auto-detection
            nms_fboxes.push_back({box[0], box[1], box[2], box[3]});
            nms_scores.push_back(max_score);
            nms_class_ids.push_back(max_cls);
        }

        if (nms_fboxes.empty()) return results;

        // Auto-detect box format: [ymin,xmin,ymax,xmax] vs [xmin,ymin,xmax,ymax]
        // by checking which interpretation yields more sensible (positive w/h) boxes.
        auto decodeBoxes = [&](bool swap_xy) {
            std::vector<std::array<float, 4>> decoded;
            for (const auto& raw : nms_fboxes) {
                float x1, y1, x2, y2;
                if (std::abs(raw[0]) < 5.0f && std::abs(raw[1]) < 5.0f &&
                    std::abs(raw[2]) < 5.0f && std::abs(raw[3]) < 5.0f) {
                    if (swap_xy) {
                        // [ymin, xmin, ymax, xmax] → decode with swap
                        y1 = raw[0] * input_height_; x1 = raw[1] * input_width_;
                        y2 = raw[2] * input_height_; x2 = raw[3] * input_width_;
                    } else {
                        // [xmin, ymin, xmax, ymax] → no swap
                        x1 = raw[0] * input_width_;  y1 = raw[1] * input_height_;
                        x2 = raw[2] * input_width_;  y2 = raw[3] * input_height_;
                    }
                } else {
                    if (swap_xy) {
                        y1 = raw[0]; x1 = raw[1]; y2 = raw[2]; x2 = raw[3];
                    } else {
                        x1 = raw[0]; y1 = raw[1]; x2 = raw[2]; y2 = raw[3];
                    }
                }
                decoded.push_back({x1, y1, x2, y2});
            }
            return decoded;
        };

        auto sensibleCount = [&](const std::vector<std::array<float, 4>>& boxes) {
            int count = 0;
            for (const auto& b : boxes) {
                float w = b[2] - b[0], h = b[3] - b[1];
                if (w > 1.0f && h > 1.0f &&
                    w < input_width_ * 1.2f && h < input_height_ * 1.2f)
                    ++count;
            }
            return count;
        };

        auto decoded_a = decodeBoxes(true);   // SSD standard [ymin,xmin,ymax,xmax]
        auto decoded_b = decodeBoxes(false);   // DXNN typical  [xmin,ymin,xmax,ymax]
        int score_a = sensibleCount(decoded_a);
        int score_b = sensibleCount(decoded_b);
        // Prefer [xmin,ymin,xmax,ymax] (DXNN typical) unless swap produces strictly more sensible boxes
        auto& decoded = (score_a > score_b) ? decoded_a : decoded_b;

        // Build NMS input from decoded boxes
        for (size_t i = 0; i < decoded.size(); ++i) {
            float x1 = decoded[i][0], y1 = decoded[i][1];
            float x2 = decoded[i][2], y2 = decoded[i][3];
            nms_boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                        static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, conf_threshold_, nms_threshold_, indices);

        // Auto-detect label set from actual model output dimensions
        int inferred_num_classes = score_cols - fg_offset;

        for (int idx : indices) {
            float x1 = decoded[idx][0];
            float y1 = decoded[idx][1];
            float x2 = decoded[idx][2];
            float y2 = decoded[idx][3];

            // Scale to original image (SSD uses simple resize, no letterbox)
            if (ctx.pad_x == 0 && ctx.pad_y == 0) {
                float sx = static_cast<float>(ctx.original_width) / input_width_;
                float sy = static_cast<float>(ctx.original_height) / input_height_;
                x1 = std::max(0.0f, std::min(x1 * sx, static_cast<float>(ctx.original_width)));
                y1 = std::max(0.0f, std::min(y1 * sy, static_cast<float>(ctx.original_height)));
                x2 = std::max(0.0f, std::min(x2 * sx, static_cast<float>(ctx.original_width)));
                y2 = std::max(0.0f, std::min(y2 * sy, static_cast<float>(ctx.original_height)));
            } else {
                x1 = std::max(0.0f, std::min((x1 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
                y1 = std::max(0.0f, std::min((y1 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));
                x2 = std::max(0.0f, std::min((x2 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
                y2 = std::max(0.0f, std::min((y2 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));
            }

            DetectionResult det;
            det.box = {x1, y1, x2, y2};
            det.confidence = nms_scores[idx];
            det.class_id = nms_class_ids[idx];
            // Auto-detect label set: use VOC only for exactly 20 fg classes,
            // otherwise fall back to COCO (which covers 80/90-class models).
            if (inferred_num_classes == 20) {
                det.class_name = dxapp::common::get_voc_class_name(det.class_id);
            } else {
                det.class_name = dxapp::common::get_coco_class_name(det.class_id);
            }
            results.push_back(det);
        }

        return results;
    }

    std::string getModelName() const override { return "SSD"; }

private:
    // Helper: find max foreground class score
    static std::pair<float, int> findMaxFgClass_(const float* row_scores,
                                                  int score_cols, int fg_offset) {
        float max_score = 0.0f;
        int max_cls = 0;
        for (int c = fg_offset; c < score_cols; ++c) {
            if (row_scores[c] > max_score) {
                max_score = row_scores[c];
                max_cls = c - fg_offset;
            }
        }
        return {max_score, max_cls};
    }

    int input_width_;
    int input_height_;
    float conf_threshold_;
    float nms_threshold_;
    int num_classes_;
    bool has_background_;
    std::string label_set_;
};

}  // namespace dxapp

#endif  // SSD_POSTPROCESSOR_HPP
