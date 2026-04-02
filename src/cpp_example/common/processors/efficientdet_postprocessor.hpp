/**
 * @file efficientdet_postprocessor.hpp
 * @brief EfficientDet detection postprocessor
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Ported from Python efficientdet_postprocessor.py.
 * 
 * EfficientDet outputs 2~4 tensors:
 *   - TF format (4 tensors): [boxes, classes, scores, num_detections]
 *   - 2-tensor format: [1,N,4] boxes (normalized ymin,xmin,ymax,xmax) + [1,N,C] scores
 * 
 * Algorithm:
 *   1. Auto-detect tensor format (2 vs 4 output tensors)
 *   2. Extract boxes/scores/classes from appropriate tensors
 *   3. Convert [ymin,xmin,ymax,xmax] normalized → [x1,y1,x2,y2] pixel
 *   4. NMS with cv::dnn::NMSBoxes
 *   5. Scale to original coordinates
 */

#ifndef EFFICIENTDET_POSTPROCESSOR_HPP
#define EFFICIENTDET_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

class EfficientDetPostprocessor : public IPostprocessor<DetectionResult> {
public:
    EfficientDetPostprocessor(int input_width = 512, int input_height = 512,
                              float score_threshold = 0.3f,
                              float nms_threshold = 0.45f,
                              int num_classes = 90)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          num_classes_(num_classes) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        if (outputs.size() < 2) return results;

        if (outputs.size() >= 4) {
            return processTFFormat(outputs, ctx);
        }
        return process2Tensor(outputs, ctx);
    }

    std::string getModelName() const override { return "EfficientDet"; }

private:
    /**
     * @brief Process TF-style 4-tensor format:
     *   [boxes(1,N,4), classes(1,N), scores(1,N), num_detections(1)]
     */
    std::vector<DetectionResult> processTFFormat(
        const dxrt::TensorPtrs& outputs, const PreprocessContext& ctx) {
        std::vector<DetectionResult> results;

        // Sort tensors: boxes(last_dim=4), scores(last_dim=N), classes(last_dim=N), num_det(last_dim=1)
        const dxrt::TensorPtr* boxes_t = nullptr;
        const dxrt::TensorPtr* classes_t = nullptr;
        const dxrt::TensorPtr* scores_t = nullptr;
        const dxrt::TensorPtr* num_det_t = nullptr;

        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            if (last_dim == 4) { boxes_t = &t; }
            else if (last_dim == 1 && shape.size() <= 2) { num_det_t = &t; }
        }
        // Remaining tensors are classes and scores
        for (auto& t : outputs) {
            if (&t == boxes_t || &t == num_det_t) continue;
            auto shape = t->shape();
            if (!scores_t) { scores_t = &t; }
            else { classes_t = &t; }
        }
        if (!boxes_t || !scores_t) return results;

        const float* boxes = static_cast<const float*>((*boxes_t)->data());
        const float* scores = static_cast<const float*>((*scores_t)->data());

        int num_det = 0;
        if (num_det_t) {
            num_det = static_cast<int>(*static_cast<const float*>((*num_det_t)->data()));
        } else {
            auto shape = (*scores_t)->shape();
            num_det = static_cast<int>(shape.size() >= 2 ? shape[1] : shape[0]);
        }

        const float* classes_data = classes_t ? static_cast<const float*>((*classes_t)->data()) : nullptr;

        for (int i = 0; i < num_det; ++i) {
            float score = scores[i];
            if (score < score_threshold_) continue;

            // TF format: [ymin, xmin, ymax, xmax] normalized
            float ymin = boxes[i * 4 + 0];
            float xmin = boxes[i * 4 + 1];
            float ymax = boxes[i * 4 + 2];
            float xmax = boxes[i * 4 + 3];

            float x1 = xmin * input_width_;
            float y1 = ymin * input_height_;
            float x2 = xmax * input_width_;
            float y2 = ymax * input_height_;

            scaleCoords(x1, y1, x2, y2, ctx);

            DetectionResult det;
            det.box = {x1, y1, x2, y2};
            det.confidence = score;
            det.class_id = classes_data ? static_cast<int>(classes_data[i]) : 0;
            det.class_name = dxapp::common::get_coco_class_name(det.class_id);
            results.push_back(det);
        }
        return results;
    }

    /**
     * @brief Process 2-tensor format:
     *   [1,N,4] boxes (normalized) + [1,N,C] class scores
     */
    std::vector<DetectionResult> process2Tensor(
        const dxrt::TensorPtrs& outputs, const PreprocessContext& ctx) {
        std::vector<DetectionResult> results;

        // Identify boxes (last_dim=4) and scores
        const dxrt::TensorPtr* boxes_t = nullptr;
        const dxrt::TensorPtr* scores_t = nullptr;
        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            if (last_dim == 4) boxes_t = &t;
            else scores_t = &t;
        }
        if (!boxes_t || !scores_t) return results;

        auto boxes_shape = (*boxes_t)->shape();
        auto scores_shape = (*scores_t)->shape();
        int N = static_cast<int>(boxes_shape.size() == 3 ? boxes_shape[1] : boxes_shape[0]);
        int C = static_cast<int>(scores_shape.back());

        const float* boxes = static_cast<const float*>((*boxes_t)->data());
        const float* scores = static_cast<const float*>((*scores_t)->data());

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;
        std::vector<std::array<float, 4>> float_boxes;

        for (int i = 0; i < N; ++i) {
            // Find max class score
            float max_score = 0.0f;
            int max_cls = 0;
            for (int c = 0; c < C; ++c) {
                float s = scores[i * C + c];
                if (s > max_score) { max_score = s; max_cls = c; }
            }
            if (max_score < score_threshold_) continue;

            float ymin = boxes[i * 4 + 0];
            float xmin = boxes[i * 4 + 1];
            float ymax = boxes[i * 4 + 2];
            float xmax = boxes[i * 4 + 3];

            // Auto-detect normalized vs pixel coords
            float x1, y1, x2, y2;
            if (std::abs(ymin) < 2.0f && std::abs(xmin) < 2.0f) {
                x1 = xmin * input_width_;
                y1 = ymin * input_height_;
                x2 = xmax * input_width_;
                y2 = ymax * input_height_;
            } else {
                x1 = xmin; y1 = ymin; x2 = xmax; y2 = ymax;
            }

            nms_boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            float_boxes.push_back({x1, y1, x2, y2});
            nms_scores.push_back(max_score);
            nms_class_ids.push_back(max_cls);
        }

        if (nms_boxes.empty()) return results;

        std::vector<int> indices;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, indices);

        for (int idx : indices) {
            float x1 = float_boxes[idx][0];
            float y1 = float_boxes[idx][1];
            float x2 = float_boxes[idx][2];
            float y2 = float_boxes[idx][3];
            scaleCoords(x1, y1, x2, y2, ctx);

            DetectionResult det;
            det.box = {x1, y1, x2, y2};
            det.confidence = nms_scores[idx];
            det.class_id = nms_class_ids[idx];
            det.class_name = dxapp::common::get_coco_class_name(det.class_id);
            results.push_back(det);
        }
        return results;
    }

    void scaleCoords(float& x1, float& y1, float& x2, float& y2,
                     const PreprocessContext& ctx) {
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
        float w = static_cast<float>(ctx.original_width);
        float h = static_cast<float>(ctx.original_height);
        x1 = std::max(0.0f, std::min(x1, w));
        y1 = std::max(0.0f, std::min(y1, h));
        x2 = std::max(0.0f, std::min(x2, w));
        y2 = std::max(0.0f, std::min(y2, h));
    }

    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    int num_classes_;
};

}  // namespace dxapp

#endif  // EFFICIENTDET_POSTPROCESSOR_HPP
