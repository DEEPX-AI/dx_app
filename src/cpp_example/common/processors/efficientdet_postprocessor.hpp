/**
 * @file efficientdet_postprocessor.hpp
 * @brief EfficientDet detection postprocessor
 * 
 * Ported from Python efficientdet_postprocessor.py.
 * 
 * EfficientDet outputs 2~4 tensors:
 *   - TF format (4 tensors): [boxes, classes, scores, num_detections]
 *   - 2-tensor format: [1,N,4] boxes + [1,N,C] scores
 *     (YOLOv4: normalized [x1,y1,x2,y2]; EfficientDet: anchor deltas [ty,tx,th,tw])
 * 
 * Algorithm:
 *   1. Auto-detect tensor format (2 vs 4 output tensors)
 *   2. Extract boxes/scores/classes from appropriate tensors
 *   3. Auto-detect box format (xyxy, cxcywh, anchor-delta)
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

        // Identify box tensor (last_dim=4, 3D or 4D) and score tensor (last_dim=C)
        const dxrt::TensorPtr* boxes_t = nullptr;
        const dxrt::TensorPtr* scores_t = nullptr;
        const dxrt::TensorPtr* classes_t = nullptr;
        const dxrt::TensorPtr* num_det_t = nullptr;
        int boxes_N = 0;

        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            int ndim = static_cast<int>(shape.size());

            if (last_dim == 4 && ndim >= 3) {
                // Box tensor: [1,N,4] or [1,N,1,4]
                boxes_t = &t;
                boxes_N = static_cast<int>(shape[1]);
            } else if (last_dim == 1 && ndim <= 2) {
                num_det_t = &t;
            }
        }

        // Score tensor: 3D tensor with last_dim matching num_classes (or closest)
        for (auto& t : outputs) {
            if (&t == boxes_t || &t == num_det_t) continue;
            auto shape = t->shape();
            if (shape.size() == 3 && static_cast<int>(shape[1]) == boxes_N) {
                int last = static_cast<int>(shape[2]);
                if (last == num_classes_ || (!scores_t && last > 1)) {
                    scores_t = &t;
                }
            }
        }

        // If we have boxes+scores with matching N, use 2-tensor path
        if (boxes_t && scores_t) {
            return process2Tensor(*boxes_t, *scores_t, boxes_N, ctx);
        }

        // Fallback: TF 4-tensor format [boxes, classes, scores, num_det]
        if (outputs.size() >= 4) {
            return processTFFormat(outputs, ctx);
        }

        return results;
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

        const dxrt::TensorPtr* boxes_t = nullptr;
        const dxrt::TensorPtr* classes_t = nullptr;
        const dxrt::TensorPtr* scores_t = nullptr;
        const dxrt::TensorPtr* num_det_t = nullptr;

        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            if (last_dim == 4 && shape.size() >= 3) { boxes_t = &t; }
            else if (last_dim == 1 && shape.size() <= 2) { num_det_t = &t; }
        }
        for (auto& t : outputs) {
            if (&t == boxes_t || &t == num_det_t) continue;
            auto shape = t->shape();
            if (shape.size() < 2) continue;
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
     * @brief Process 2-tensor format given pre-identified box and score tensors.
     *   Supports: [1,N,4] / [1,N,1,4] boxes + [1,N,C] scores
     *   Auto-detects: normalized, pixel, or anchor-delta box format
     */
    std::vector<DetectionResult> process2Tensor(
        const dxrt::TensorPtr& boxes_tensor,
        const dxrt::TensorPtr& scores_tensor,
        int N,
        const PreprocessContext& ctx) {
        std::vector<DetectionResult> results;

        auto scores_shape = scores_tensor->shape();
        int C = static_cast<int>(scores_shape.back());

        // Handle 4D box shape [1,N,1,4] by treating as [N,4]
        const float* boxes = static_cast<const float*>(boxes_tensor->data());
        const float* scores = static_cast<const float*>(scores_tensor->data());

        // Detect box format by examining value ranges
        BoxFormat fmt = detectBoxFormat_(boxes, N);

        // Generate anchors if needed
        std::vector<std::array<float,4>> anchors;
        if (fmt == BoxFormat::ANCHOR_DELTA) {
            anchors = generateAnchors_(N);
        }

        // For NORMALIZED/PIXEL: auto-detect xyxy vs cxcywh (like Python)
        bool is_xyxy = true;
        if (fmt != BoxFormat::ANCHOR_DELTA) {
            int xyxy_count = 0;
            int total_check = 0;
            int sample = std::min(N, 500);
            for (int i = 0; i < sample; ++i) {
                float max_s = 0.0f;
                for (int c = 0; c < C; ++c) {
                    float s = scores[i * C + c]; if (s > max_s) max_s = s;
                }
                if (max_s < score_threshold_) continue;
                ++total_check;
                if (boxes[i*4+2] > boxes[i*4+0] && boxes[i*4+3] > boxes[i*4+1])
                    ++xyxy_count;
            }
            is_xyxy = (total_check == 0) || (static_cast<float>(xyxy_count) / total_check > 0.8f);
        }

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;
        std::vector<std::array<float, 4>> float_boxes;

        for (int i = 0; i < N; ++i) {
            float max_score = 0.0f;
            int max_cls = 0;
            for (int c = 0; c < C; ++c) {
                float s = scores[i * C + c];
                if (s > max_score) { max_score = s; max_cls = c; }
            }
            if (max_score < score_threshold_) continue;

            float v0 = boxes[i * 4 + 0];
            float v1 = boxes[i * 4 + 1];
            float v2 = boxes[i * 4 + 2];
            float v3 = boxes[i * 4 + 3];

            float x1, y1, x2, y2;
            if (fmt == BoxFormat::ANCHOR_DELTA && i < static_cast<int>(anchors.size())) {
                // Decode: [ty, tx, th, tw] with anchor [cy, cx, h, w]
                float ay = anchors[i][0], ax = anchors[i][1];
                float ah = anchors[i][2], aw = anchors[i][3];
                float cy = ay + v0 * ah;
                float cx = ax + v1 * aw;
                float h  = ah * std::exp(std::min(v2, 10.0f));
                float w  = aw * std::exp(std::min(v3, 10.0f));
                x1 = cx - w * 0.5f;
                y1 = cy - h * 0.5f;
                x2 = cx + w * 0.5f;
                y2 = cy + h * 0.5f;
            } else if (fmt == BoxFormat::NORMALIZED) {
                // YOLO-style: [col0, col1, col2, col3] with col0,col2=x, col1,col3=y
                if (is_xyxy) {
                    x1 = v0 * input_width_;
                    y1 = v1 * input_height_;
                    x2 = v2 * input_width_;
                    y2 = v3 * input_height_;
                } else {
                    // cxcywh
                    float cx = v0 * input_width_, cy = v1 * input_height_;
                    float bw = v2 * input_width_, bh = v3 * input_height_;
                    x1 = cx - bw * 0.5f; y1 = cy - bh * 0.5f;
                    x2 = cx + bw * 0.5f; y2 = cy + bh * 0.5f;
                }
            } else {
                // PIXEL format
                if (is_xyxy) {
                    x1 = v0; y1 = v1; x2 = v2; y2 = v3;
                } else {
                    x1 = v0 - v2 * 0.5f; y1 = v1 - v3 * 0.5f;
                    x2 = v0 + v2 * 0.5f; y2 = v1 + v3 * 0.5f;
                }
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

    enum class BoxFormat { NORMALIZED, PIXEL, ANCHOR_DELTA };

    /** Detect box format: normalized [0,1], pixel coords, or anchor deltas */
    BoxFormat detectBoxFormat_(const float* boxes, int N) const {
        if (N <= 0) return BoxFormat::NORMALIZED;
        int sample = std::min(N, 500);
        int neg_count = 0;
        float sum_abs = 0.0f;
        for (int i = 0; i < sample * 4; ++i) {
            float v = boxes[i];
            if (v < 0) ++neg_count;
            sum_abs += std::abs(v);
        }
        float neg_ratio = static_cast<float>(neg_count) / (sample * 4);
        float avg_abs = sum_abs / (sample * 4);
        // High negative ratio → anchor deltas (deltas can be large negative)
        if (neg_ratio > 0.15f || avg_abs > 5.0f) return BoxFormat::ANCHOR_DELTA;
        if (avg_abs < 2.0f) return BoxFormat::NORMALIZED;
        return BoxFormat::PIXEL;
    }

    /**
     * @brief Generate EfficientDet-style anchors for all feature levels.
     *   Levels P3-P7, strides [8,16,32,64,128], 9 anchors per location.
     *   Returns [cy, cx, h, w] for each anchor.
     */
    std::vector<std::array<float,4>> generateAnchors_(int expected_count) const {
        const float anchor_scale = 4.0f;
        const std::vector<int> strides = {8, 16, 32, 64, 128};
        const std::vector<float> octave_scales = {1.0f, 1.2599f, 1.5874f};  // 2^0, 2^(1/3), 2^(2/3)
        const std::vector<float> aspect_ratios = {1.0f, 2.0f, 0.5f};

        std::vector<std::array<float,4>> anchors;
        anchors.reserve(expected_count > 0 ? expected_count : 76725);

        for (int stride : strides) {
            int grid_h = input_height_ / stride;
            int grid_w = input_width_ / stride;
            float base_size = anchor_scale * stride;

            for (int y = 0; y < grid_h; ++y) {
                for (int x = 0; x < grid_w; ++x) {
                    float cy = (y + 0.5f) * stride;
                    float cx = (x + 0.5f) * stride;
                    for (float scale : octave_scales) {
                        for (float ratio : aspect_ratios) {
                            float h = base_size * scale / std::sqrt(ratio);
                            float w = base_size * scale * std::sqrt(ratio);
                            anchors.push_back({cy, cx, h, w});
                        }
                    }
                }
            }
        }
        return anchors;
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
