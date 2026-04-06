/**
 * @file yolact_postprocessor.hpp
 * @brief YOLACT instance segmentation postprocessor
 * 
 * Ported from Python yolact_postprocessor.py.
 * 
 * SSD-based instance segmentation with prototype masks.
 * Output: 4 tensors (sorted by channel/element count):
 *   - loc:        [1, N, 4]   SSD box regression
 *   - conf:       [1, N, C]   class confidences (softmax, class 0 = bg)
 *   - mask_coeff: [1, N, 32]  mask coefficient vectors
 *   - proto:      [1, H, W, 32]  prototype mask features
 * 
 * Algorithm:
 *   1. Auto-generate SSD anchors for feature map scales
 *   2. Decode SSD boxes using priors
 *   3. Filter + NMS per class
 *   4. Compute mask = sigmoid(coeff @ proto.T)
 *   5. Crop masks to detected bounding boxes
 *   6. Scale to original coordinates
 */

#ifndef YOLACT_POSTPROCESSOR_HPP
#define YOLACT_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

class YOLACTPostprocessor : public IPostprocessor<InstanceSegmentationResult> {
public:
    YOLACTPostprocessor(int input_width = 550, int input_height = 550,
                        float score_threshold = 0.3f,
                        float nms_threshold = 0.5f,
                        int num_classes = 81,
                        int num_protos = 32,
                        int top_k = 200)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          num_classes_(num_classes), num_protos_(num_protos), top_k_(top_k) {
        generatePriors();
    }

    std::vector<InstanceSegmentationResult> process(
        const dxrt::TensorPtrs& outputs, const PreprocessContext& ctx) override {
        std::vector<InstanceSegmentationResult> results;
        if (outputs.size() < 4) return results;

        // Identify tensors: proto (4D with H,W), loc (last=4), conf, mask_coeff (last=32)
        const dxrt::TensorPtr* loc_t = nullptr;
        const dxrt::TensorPtr* conf_t = nullptr;
        const dxrt::TensorPtr* mask_coeff_t = nullptr;
        const dxrt::TensorPtr* proto_t = nullptr;
        identifyTensors_(outputs, loc_t, conf_t, mask_coeff_t, proto_t);
        if (!loc_t || !conf_t || !mask_coeff_t || !proto_t) return results;

        auto conf_shape = (*conf_t)->shape();
        int N_model = static_cast<int>(conf_shape.size() == 3 ? conf_shape[1] : conf_shape[0]);
        int C = static_cast<int>(conf_shape.back());

        // Lazy rebuild priors to match model output anchor count
        if (priors_.empty() || static_cast<int>(priors_.size()) != N_model) {
            rebuildPriorsForN(N_model);
        }
        int N = std::min(N_model, static_cast<int>(priors_.size()));

        const float* loc = static_cast<const float*>((*loc_t)->data());
        const float* conf = static_cast<const float*>((*conf_t)->data());
        const float* coeff = static_cast<const float*>((*mask_coeff_t)->data());
        const float* proto = static_cast<const float*>((*proto_t)->data());

        auto proto_shape = (*proto_t)->shape();
        int proto_h = static_cast<int>(proto_shape[1]);
        int proto_w = static_cast<int>(proto_shape[2]);
        int proto_c = static_cast<int>(proto_shape[3]);

        // Collect detections and run NMS
        std::vector<Detection_> dets;
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;
        collectDetections_(conf, loc, N, C, dets, nms_boxes, nms_scores, nms_class_ids);
        if (nms_boxes.empty()) return results;

        // Per-class containment-aware NMS: max(IoU, 0.65 * IoMin) to suppress
        // contained boxes that standard IoU misses due to area disparity.
        std::vector<int> indices = containmentAwareNMS_(
            nms_boxes, nms_scores, nms_class_ids, score_threshold_, nms_threshold_);
        if (static_cast<int>(indices.size()) > top_k_) indices.resize(top_k_);

        for (int k : indices) {
            const auto& d = dets[k];

            cv::Mat mask = generateSigmoidMask_(coeff, d.idx, num_protos_,
                                                 proto, proto_h, proto_w, proto_c);
            cv::Mat resized_mask;
            cv::resize(mask, resized_mask, cv::Size(input_width_, input_height_));

            int bx1 = std::max(0, static_cast<int>(d.x1));
            int by1 = std::max(0, static_cast<int>(d.y1));
            int bx2 = std::min(input_width_, static_cast<int>(d.x2));
            int by2 = std::min(input_height_, static_cast<int>(d.y2));
            cv::Mat cropped_mask = cropAndThresholdMask_(resized_mask, bx1, by1, bx2, by2,
                                                          input_height_, input_width_);

            auto box = scaleBoxToOriginal_(d.x1, d.y1, d.x2, d.y2, ctx);
            InstanceSegmentationResult seg;
            seg.box = box;
            seg.confidence = d.score;
            seg.class_id = d.class_id;
            seg.class_name = dxapp::common::get_coco_class_name(seg.class_id);
            seg.mask = cropped_mask;
            results.push_back(seg);
        }
        return results;
    }

    std::string getModelName() const override { return "YOLACT"; }

private:
    struct Detection_ {
        float x1, y1, x2, y2, score;
        int class_id, idx;
    };

    // Containment-aware NMS: max(IoU, 0.65 * IoMin) as overlap metric.
    // Uses per-class NMS to prevent cross-class suppression.
    static std::vector<int> containmentAwareNMS_(
        const std::vector<cv::Rect>& boxes, const std::vector<float>& scores,
        const std::vector<int>& class_ids,
        float score_threshold, float nms_threshold) {
        const float containment_factor = 0.65f;
        int n = static_cast<int>(boxes.size());

        // Sort by score descending
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return scores[a] > scores[b]; });

        std::vector<bool> suppressed(n, false);
        std::vector<int> kept;

        for (int i = 0; i < n; ++i) {
            int idx_i = order[i];
            if (suppressed[idx_i] || scores[idx_i] < score_threshold) continue;
            kept.push_back(idx_i);

            float ax1 = static_cast<float>(boxes[idx_i].x);
            float ay1 = static_cast<float>(boxes[idx_i].y);
            float ax2 = ax1 + boxes[idx_i].width;
            float ay2 = ay1 + boxes[idx_i].height;
            float area_a = boxes[idx_i].width * boxes[idx_i].height;

            for (int j = i + 1; j < n; ++j) {
                int idx_j = order[j];
                if (suppressed[idx_j]) continue;
                // Only suppress within same class
                if (class_ids[idx_i] != class_ids[idx_j]) continue;

                float bx1 = static_cast<float>(boxes[idx_j].x);
                float by1 = static_cast<float>(boxes[idx_j].y);
                float bx2 = bx1 + boxes[idx_j].width;
                float by2 = by1 + boxes[idx_j].height;
                float area_b = boxes[idx_j].width * boxes[idx_j].height;

                float inter_x1 = std::max(ax1, bx1);
                float inter_y1 = std::max(ay1, by1);
                float inter_x2 = std::min(ax2, bx2);
                float inter_y2 = std::min(ay2, by2);
                float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                                   std::max(0.0f, inter_y2 - inter_y1);

                float iou = inter_area / (area_a + area_b - inter_area + 1e-6f);
                float iomin = (area_a > 0 && area_b > 0)
                    ? inter_area / (std::min(area_a, area_b) + 1e-6f)
                    : 0.0f;
                float overlap = std::max(iou, containment_factor * iomin);
                if (overlap > nms_threshold) suppressed[idx_j] = true;
            }
        }
        return kept;
    }

    // Identify the four output tensors by shape heuristics.
    void identifyTensors_(const dxrt::TensorPtrs& outputs,
                          const dxrt::TensorPtr*& loc_t,
                          const dxrt::TensorPtr*& conf_t,
                          const dxrt::TensorPtr*& mask_coeff_t,
                          const dxrt::TensorPtr*& proto_t) const {
        for (auto& t : outputs) {
            auto shape = t->shape();
            if (shape.size() == 4 && shape[2] > 1 && shape[3] > 1) {
                proto_t = &t;
            }
        }
        for (auto& t : outputs) {
            if (&t == proto_t) continue;
            auto shape = t->shape();
            auto last_dim = static_cast<int>(shape.back());
            if (last_dim == 4) loc_t = &t;
            else if (last_dim == num_protos_) mask_coeff_t = &t;
            else conf_t = &t;
        }
    }

    // First pass: collect above-threshold detections by decoding SSD boxes.
    void collectDetections_(const float* conf, const float* loc,
                            int N, int C,
                            std::vector<Detection_>& dets,
                            std::vector<cv::Rect>& nms_boxes,
                            std::vector<float>& nms_scores,
                            std::vector<int>& nms_class_ids) const {
        for (int i = 0; i < N; ++i) {
            float max_score = 0;
            int max_cls = 0;
            for (int c = 1; c < C; ++c) {
                float s = conf[i * C + c];
                if (s > max_score) { max_score = s; max_cls = c; }
            }
            if (max_score < score_threshold_) continue;

            float pcx = priors_[i][0], pcy = priors_[i][1];
            float pw = priors_[i][2], ph = priors_[i][3];
            float cx = pcx + loc[i * 4 + 0] * 0.1f * pw;
            float cy = pcy + loc[i * 4 + 1] * 0.1f * ph;
            float w = pw * std::exp(loc[i * 4 + 2] * 0.2f);
            float h = ph * std::exp(loc[i * 4 + 3] * 0.2f);

            float x1 = (cx - w * 0.5f) * input_width_;
            float y1 = (cy - h * 0.5f) * input_height_;
            float x2 = (cx + w * 0.5f) * input_width_;
            float y2 = (cy + h * 0.5f) * input_height_;

            // Clamp to valid input region
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(input_width_)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(input_height_)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(input_width_)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(input_height_)));

            // Reject degenerate or near-full-image boxes
            float bw = x2 - x1, bh = y2 - y1;
            if (bw < 1.0f || bh < 1.0f) continue;
            float img_area = static_cast<float>(input_width_ * input_height_);
            if ((bw * bh) / img_area > 0.8f) continue;

            dets.push_back({x1, y1, x2, y2, max_score, max_cls - 1, i});
            nms_boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            nms_scores.push_back(max_score);
            nms_class_ids.push_back(max_cls - 1);
        }
    }

    // Compute sigmoid(coeff @ proto.T) for a single detection.
    static cv::Mat generateSigmoidMask_(const float* coeff, int orig_idx, int num_protos,
                                         const float* proto,
                                         int proto_h, int proto_w, int proto_c) {
        cv::Mat mask(proto_h, proto_w, CV_32FC1);
        for (int y = 0; y < proto_h; ++y) {
            for (int x = 0; x < proto_w; ++x) {
                float val = 0;
                for (int c = 0; c < proto_c; ++c)
                    val += coeff[orig_idx * num_protos + c] *
                           proto[(y * proto_w + x) * proto_c + c];
                mask.at<float>(y, x) = 1.0f / (1.0f + std::exp(-val));
            }
        }
        return mask;
    }

    // Crop a sigmoid mask to the bounding box and binarize at threshold 0.5.
    static cv::Mat cropAndThresholdMask_(const cv::Mat& resized_mask,
                                          int bx1, int by1, int bx2, int by2,
                                          int mask_h, int mask_w) {
        // Clamp crop bounds to valid range to avoid OOB access
        int x0 = std::max(0, bx1);
        int y0 = std::max(0, by1);
        int x1 = std::min(mask_w, bx2);
        int y1 = std::min(mask_h, by2);

        cv::Mat cropped_mask = cv::Mat::zeros(mask_h, mask_w, CV_8UC1);
        if (x1 <= x0 || y1 <= y0) return cropped_mask;

        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                // Safety: if resized_mask is float, use at<float>
                float v = 0.0f;
                if (resized_mask.type() == CV_32FC1) v = resized_mask.at<float>(y, x);
                else if (resized_mask.type() == CV_8UC1) v = resized_mask.at<uchar>(y, x) / 255.0f;
                else if (resized_mask.type() == CV_64FC1) v = static_cast<float>(resized_mask.at<double>(y, x));
                if (v > 0.5f) cropped_mask.at<uint8_t>(y, x) = 255;
            }
        }
        return cropped_mask;
    }

    // Scale a box from model-input space to original image coordinates.
    std::vector<float> scaleBoxToOriginal_(float x1, float y1, float x2, float y2,
                                              const PreprocessContext& ctx) const {
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
        auto ow = static_cast<float>(ctx.original_width);
        auto oh = static_cast<float>(ctx.original_height);
        return {
            std::max(0.0f, std::min(x1, ow)),
            std::max(0.0f, std::min(y1, oh)),
            std::max(0.0f, std::min(x2, ow)),
            std::max(0.0f, std::min(y2, oh))
        };
    }

    void generatePriors() {
        // YOLACT SSD-style prior generation
        // Try multiple anchor configs and pick the one matching target_n.
        // The model output anchor count determines which config to use.
        priors_.clear();
        // Will be rebuilt in process() if target_n doesn't match initially.
    }

    // Rebuild priors matching the model's actual anchor count.
    void rebuildPriorsForN(int target_n) {
        int strides[] = {8, 16, 32, 64, 128};

        // Candidate configurations: (scales_per_level, aspect_ratios_per_level)
        struct AnchorConfig {
            std::vector<std::vector<float>> scales;
            std::vector<float> aspect_ratios;
        };
        std::vector<AnchorConfig> configs = {
            // 3 scales × 3 ARs = 9 anchors/cell (most YOLACT models)
            {{{24,48,96}, {48,96,192}, {96,192,384}, {192,384,768}, {384,768,1536}},
             {1.0f, 0.5f, 2.0f}},
            // 1 scale × 3 ARs = 3 anchors/cell (default YOLACT)
            {{{24}, {48}, {96}, {192}, {384}},
             {1.0f, 0.5f, 2.0f}},
            // 2 scales × 1 AR = 2 anchors/cell
            {{{24,48}, {48,96}, {96,192}, {192,384}, {384,768}},
             {1.0f}},
        };

        for (auto& cfg : configs) {
            int total = 0;
            for (int s = 0; s < 5; ++s) {
                int stride = strides[s];
                int fh = (input_height_ + stride - 1) / stride;
                int fw = (input_width_ + stride - 1) / stride;
                total += fh * fw * static_cast<int>(cfg.scales[s].size())
                                 * static_cast<int>(cfg.aspect_ratios.size());
            }
            if (target_n > 0 && total != target_n) continue;

            priors_.clear();
            for (int s = 0; s < 5; ++s) {
                int stride = strides[s];
                int fh = (input_height_ + stride - 1) / stride;
                int fw = (input_width_ + stride - 1) / stride;
                for (int y = 0; y < fh; ++y) {
                    for (int x = 0; x < fw; ++x) {
                        float cx = (x + 0.5f) / fw;
                        float cy = (y + 0.5f) / fh;
                        for (float sc : cfg.scales[s]) {
                            for (float ar : cfg.aspect_ratios) {
                                float pw = sc / input_width_ * std::sqrt(ar);
                                float ph = sc / input_height_ / std::sqrt(ar);
                                priors_.push_back({cx, cy, pw, ph});
                            }
                        }
                    }
                }
            }
            return;
        }
        // Fallback: use first config (9 anchors/cell)
        rebuildPriorsForN(0);
    }

    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    int num_classes_;
    int num_protos_;
    int top_k_;
    std::vector<std::array<float, 4>> priors_;
};

}  // namespace dxapp

#endif  // YOLACT_POSTPROCESSOR_HPP
