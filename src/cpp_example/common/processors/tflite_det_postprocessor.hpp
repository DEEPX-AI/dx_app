/**
 * @file tflite_det_postprocessor.hpp
 * @brief TFLite Detection postprocessor (TF-style 4-tensor output)
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Ported from Python tflite_det_postprocessor.py.
 * 
 * TFLite detection models output 4 tensors:
 *   - boxes:          [1, N, 4] normalized [ymin, xmin, ymax, xmax]
 *   - class_ids:      [1, N]    class indices (float)
 *   - scores:         [1, N]    confidence scores
 *   - num_detections: [1]       number of valid detections
 * 
 * No NMS needed — already applied by TFLite detection output op.
 */

#ifndef TFLITE_DET_POSTPROCESSOR_HPP
#define TFLITE_DET_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"
#include <algorithm>
#include <cmath>

namespace dxapp {

class TFLiteDetPostprocessor : public IPostprocessor<DetectionResult> {
public:
    TFLiteDetPostprocessor(int input_width = 300, int input_height = 300,
                           float score_threshold = 0.3f,
                           bool one_based_class_ids = true)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold),
          one_based_class_ids_(one_based_class_ids) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        if (outputs.size() < 4) return results;

        // Identify tensors by last dimension:
        //   boxes(4), class_ids(N), scores(N), num_det(1)
        const dxrt::TensorPtr* boxes_t = nullptr;
        const dxrt::TensorPtr* num_det_t = nullptr;
        std::vector<const dxrt::TensorPtr*> score_like;  // class_ids and scores

        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            if (shape.size() >= 2 && last_dim == 4) {
                boxes_t = &t;
            } else if (shape.size() <= 2 && last_dim == 1) {
                num_det_t = &t;
            } else {
                score_like.push_back(&t);
            }
        }
        if (!boxes_t || !num_det_t || score_like.size() < 2) return results;

        // Heuristic: scores have values in [0,1], class_ids have integer-like values
        const float* d0 = static_cast<const float*>((*score_like[0])->data());
        const float* d1 = static_cast<const float*>((*score_like[1])->data());
        
        float sum0 = 0, sum1 = 0;
        int sample = std::min(10, static_cast<int>((*score_like[0])->shape().back()));
        for (int i = 0; i < sample; ++i) {
            sum0 += d0[i]; sum1 += d1[i];
        }
        
        const float* scores_data;
        const float* classes_data;
        // Scores are typically < 1.0 on average, class_ids are larger
        if (sum0 / sample <= 1.0f) {
            scores_data = d0; classes_data = d1;
        } else {
            scores_data = d1; classes_data = d0;
        }

        const float* boxes_data = static_cast<const float*>((*boxes_t)->data());
        int num_det = static_cast<int>(*static_cast<const float*>((*num_det_t)->data()));

        for (int i = 0; i < num_det; ++i) {
            float score = scores_data[i];
            if (score < score_threshold_) continue;

            // TFLite format: [ymin, xmin, ymax, xmax] normalized
            float ymin = boxes_data[i * 4 + 0];
            float xmin = boxes_data[i * 4 + 1];
            float ymax = boxes_data[i * 4 + 2];
            float xmax = boxes_data[i * 4 + 3];

            float x1 = xmin * input_width_;
            float y1 = ymin * input_height_;
            float x2 = xmax * input_width_;
            float y2 = ymax * input_height_;

            // Scale to original image
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
            det.confidence = score;
            int raw_cls = static_cast<int>(classes_data[i]);
            det.class_id = one_based_class_ids_ ? std::max(0, raw_cls - 1) : raw_cls;
            det.class_name = dxapp::common::get_coco_class_name(det.class_id);
            results.push_back(det);
        }
        return results;
    }

    std::string getModelName() const override { return "TFLiteDet"; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    bool one_based_class_ids_;
};

}  // namespace dxapp

#endif  // TFLITE_DET_POSTPROCESSOR_HPP
