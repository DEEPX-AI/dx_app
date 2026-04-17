/**
 * @file damoyolo_postprocessor.hpp
 * @brief DamoYOLO Detection Postprocessor for v3 interface (v3-native, no legacy lib)
 * 
 * Handles DamoYOLO split-head output format:
 *   - output[0]: [1, N, num_classes] class scores (already sigmoid'd)
 *   - output[1]: [1, N, 4] bounding boxes (x1, y1, x2, y2 in pixel scale)
 */

#ifndef DAMOYOLO_POSTPROCESSOR_HPP
#define DAMOYOLO_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dxapp {

class DamoYOLOPostprocessor : public IPostprocessor<DetectionResult> {
public:
    DamoYOLOPostprocessor(int input_width = 640, int input_height = 640,
                          float conf_threshold = 0.3f,
                          float nms_threshold = 0.45f,
                          int num_classes = 80,
                          bool is_ort_configured = false)
        : input_width_(input_width), input_height_(input_height),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          num_classes_(num_classes) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        if (outputs.size() < 2) return results;

        const auto& scores_tensor = outputs[0];
        const auto& boxes_tensor = outputs[1];

        auto scores_shape = scores_tensor->shape();

        int N;
        if (scores_shape.size() == 3) {
            N = static_cast<int>(scores_shape[1]);
        } else {
            N = static_cast<int>(scores_shape[0]);
        }

        const float* scores_data = static_cast<const float*>(scores_tensor->data());
        const float* boxes_data = static_cast<const float*>(boxes_tensor->data());

        std::vector<cv::Rect> nms_boxes;
        std::vector<std::array<float, 4>> nms_fboxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;

        for (int i = 0; i < N; ++i) {
            const float* row_scores = scores_data + i * num_classes_;
            const float* row_box = boxes_data + i * 4;

            // Find max class score (already sigmoid'd)
            float max_score = 0.0f;
            int max_cls = 0;
            for (int c = 0; c < num_classes_; ++c) {
                if (row_scores[c] > max_score) {
                    max_score = row_scores[c];
                    max_cls = c;
                }
            }

            if (max_score < conf_threshold_) continue;

            // Boxes are already in x1,y1,x2,y2 pixel coordinates
            float x1 = row_box[0];
            float y1 = row_box[1];
            float x2 = row_box[2];
            float y2 = row_box[3];

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

            // Scale to original coordinates (letterbox reverse)
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

    std::string getModelName() const override { return "DamoYOLO"; }

private:
    int input_width_;
    int input_height_;
    float conf_threshold_;
    float nms_threshold_;
    int num_classes_;
};

}  // namespace dxapp

#endif  // DAMOYOLO_POSTPROCESSOR_HPP
