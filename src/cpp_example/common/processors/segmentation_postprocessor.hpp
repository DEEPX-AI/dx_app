/**
 * @file segmentation_postprocessor.hpp
 * @brief Unified Segmentation Postprocessors for v3 interface
 * 
 * Groups all segmentation postprocessors:
 *   - DeepLabv3 (Semantic Segmentation)
 *   - YOLOv8Seg (Instance Segmentation)
 */

#ifndef SEGMENTATION_POSTPROCESSOR_HPP
#define SEGMENTATION_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/result_converters.hpp"
#include "common_util.hpp"

#include <set>

// Postprocess headers
#include "argmax_semantic_seg_postprocessor.hpp"
#include "anchorless_instance_seg_postprocessor.hpp"

namespace dxapp {

// ============================================================================
// DeepLabv3 Semantic Segmentation Postprocessor
// ============================================================================
class DeepLabv3Postprocessor : public IPostprocessor<SegmentationResult> {
public:
    DeepLabv3Postprocessor(int input_width = 513, int input_height = 513,
                           bool upsample_to_input = false)
        : input_width_(input_width), input_height_(input_height),
          upsample_to_input_(upsample_to_input) {}

    std::vector<SegmentationResult> process(const dxrt::TensorPtrs& outputs,
                                            const PreprocessContext& ctx) override {
        std::vector<SegmentationResult> results;
        if (outputs.empty()) return results;

        const auto& tensor = outputs[0];
        auto shape = tensor->shape();
        size_t elem_size = tensor->elem_size();

        // Determine layout: NCHW [1,C,H,W] vs NHWC [1,H,W,C]
        // For segmentation, C (num classes) is typically much smaller than H,W.
        // Heuristic: the smaller of shape[1] vs shape[3] is the class dimension.
        int C, H, W;
        bool is_nhwc = false;
        if (shape.size() == 4) {
            if (shape[1] <= shape[3]) {
                // NCHW: [1,C,H,W] — C at dim[1] is smaller
                C = static_cast<int>(shape[1]);
                H = static_cast<int>(shape[2]);
                W = static_cast<int>(shape[3]);
            } else {
                // NHWC: [1,H,W,C] — C at dim[3] is smaller
                H = static_cast<int>(shape[1]);
                W = static_cast<int>(shape[2]);
                C = static_cast<int>(shape[3]);
                is_nhwc = true;
            }
        } else if (shape.size() == 3) {
            C = static_cast<int>(shape[0]);
            H = static_cast<int>(shape[1]);
            W = static_cast<int>(shape[2]);
        } else {
            return results;
        }

        // Target dimensions: upsample logits to input resolution before argmax
        // for smoother segmentation boundaries (e.g., SegFormer outputs at H/4, W/4).
        int out_h = H, out_w = W;
        bool do_upsample = upsample_to_input_ && (H < input_height_ || W < input_width_);
        if (do_upsample) {
            out_h = input_height_;
            out_w = input_width_;
        }

        SegmentationResult seg;
        seg.width = out_w;
        seg.height = out_h;
        seg.mask.resize(out_h * out_w);

        std::set<int> unique_classes;

        if (elem_size == 8) {
            // int64 pre-argmaxed (e.g. SegFormer h)
            fillMaskInt64(static_cast<const int64_t*>(tensor->data()),
                          C, H, W, seg.mask, unique_classes);
        } else if (elem_size == 2) {
            fillMaskInt16(static_cast<const int16_t*>(tensor->data()),
                          H, W, seg.mask, unique_classes);
        } else if (C == 1) {
            // Single channel float = pre-argmaxed class indices
            fillMaskSingleChannel(static_cast<const float*>(tensor->data()),
                                  H, W, seg.mask, unique_classes);
        } else if (do_upsample) {
            fillMaskFloatUpsampled(static_cast<const float*>(tensor->data()),
                                    C, H, W, out_h, out_w, is_nhwc,
                                    seg.mask, unique_classes);
        } else {
            fillMaskFloat(static_cast<const float*>(tensor->data()),
                          C, H, W, is_nhwc, seg.mask, unique_classes);
        }

        seg.class_ids.assign(unique_classes.begin(), unique_classes.end());
        results.push_back(std::move(seg));
        return results;
    }

    std::string getModelName() const override { return "DeepLabv3"; }

private:
    // Helper: copy int16 argmax-already indices into mask
    static void fillMaskInt16(const int16_t* data, int H, int W,
                              std::vector<int>& mask,
                              std::set<int>& unique_classes) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int cls = static_cast<int>(data[y * W + x]);
                mask[y * W + x] = cls;
                unique_classes.insert(cls);
            }
        }
    }

    // Helper: copy int64 argmax-already indices into mask (e.g. SegFormer h)
    // Shape: [1,1,H,W] or [1,H,W] — single channel containing class indices
    static void fillMaskInt64(const int64_t* data, int C, int H, int W,
                              std::vector<int>& mask,
                              std::set<int>& unique_classes) {
        const int64_t* p = data;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int cls = static_cast<int>(p[y * W + x]);
                mask[y * W + x] = cls;
                unique_classes.insert(cls);
            }
        }
    }

    // Helper: single-channel float → treat as pre-argmaxed class indices
    static void fillMaskSingleChannel(const float* data, int H, int W,
                                      std::vector<int>& mask,
                                      std::set<int>& unique_classes) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int cls = static_cast<int>(std::round(data[y * W + x]));
                mask[y * W + x] = cls;
                unique_classes.insert(cls);
            }
        }
    }

    // Helper: compute per-pixel argmax from float scores and fill mask
    static void fillMaskFloat(const float* data, int C, int H, int W, bool is_nhwc,
                              std::vector<int>& mask,
                              std::set<int>& unique_classes) {
        auto argmax_pixel = [&](int y, int x) {
            float max_val = -1e9f;
            int max_cls = 0;
            for (int c = 0; c < C; ++c) {
                float val = is_nhwc ? data[y * W * C + x * C + c]
                                    : data[c * H * W + y * W + x];
                if (val > max_val) { max_val = val; max_cls = c; }
            }
            return max_cls;
        };

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int cls = argmax_pixel(y, x);
                mask[y * W + x] = cls;
                unique_classes.insert(cls);
            }
        }
    }

    // Helper: bilinear-upsample logits per channel, then argmax at target resolution.
    // This produces smooth class boundaries similar to the Python pipeline.
    static void fillMaskFloatUpsampled(const float* data, int C, int H, int W,
                                        int out_h, int out_w, bool is_nhwc,
                                        std::vector<int>& mask,
                                        std::set<int>& unique_classes) {
        // Upsample each class channel to (out_h, out_w)
        std::vector<cv::Mat> upsampled(C);
        for (int c = 0; c < C; ++c) {
            cv::Mat ch;
            if (!is_nhwc) {
                ch = cv::Mat(H, W, CV_32FC1, const_cast<float*>(data + c * H * W));
            } else {
                ch = cv::Mat(H, W, CV_32FC1);
                float* dst = ch.ptr<float>();
                for (int i = 0; i < H * W; ++i)
                    dst[i] = data[i * C + c];
            }
            cv::resize(ch, upsampled[c], cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
        }

        // Argmax at upsampled resolution using ptr<> row access
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float max_val = -1e9f;
                int max_cls = 0;
                for (int c = 0; c < C; ++c) {
                    float val = upsampled[c].ptr<float>(y)[x];
                    if (val > max_val) { max_val = val; max_cls = c; }
                }
                mask[y * out_w + x] = max_cls;
                unique_classes.insert(max_cls);
            }
        }
    }

    int input_width_;
    int input_height_;
    bool upsample_to_input_;
};

// ============================================================================
// Instance Segmentation Base Template
// ============================================================================
namespace detail {

inline void scaleInstanceSegResults(std::vector<InstanceSegmentationResult>& results,
                                    const PreprocessContext& ctx) {
    for (auto& seg : results) {
        scaleBox(seg.box, ctx);
        // Crop padding from mask, then resize to original image size (matching original)
        if (seg.mask.empty() || ctx.original_width <= 0 || ctx.original_height <= 0) continue;
        cv::Mat cropped_mask = seg.mask;
        if (ctx.pad_x > 0 || ctx.pad_y > 0) {
            int unpad_w = seg.mask.cols - 2 * ctx.pad_x;
            int unpad_h = seg.mask.rows - 2 * ctx.pad_y;
            if (unpad_w > 0 && unpad_h > 0) {
                cv::Rect crop_region(ctx.pad_x, ctx.pad_y, unpad_w, unpad_h);
                cropped_mask = seg.mask(crop_region).clone();
            }
        }
        cv::resize(cropped_mask, seg.mask,
                  cv::Size(ctx.original_width, ctx.original_height));
    }
}

}  // namespace detail

// ============================================================================
// YOLOv8Seg Instance Segmentation Postprocessor
// ============================================================================
class YOLOv8SegPostprocessor : public IPostprocessor<InstanceSegmentationResult> {
public:
    YOLOv8SegPostprocessor(int input_width = 640, int input_height = 640,
                           float score_threshold = 0.45f, float nms_threshold = 0.4f,
                           bool is_ort_configured = false, int num_classes = 80)
        : impl_(input_width, input_height, score_threshold, nms_threshold,
                is_ort_configured, num_classes) {}

    std::vector<InstanceSegmentationResult> process(const dxrt::TensorPtrs& outputs,
                                                    const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        auto results = convertAllWith(legacy_results,
            [](const YOLOv8SegResult& s) { return convertToInstanceSeg(s); });
        detail::scaleInstanceSegResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv8-Seg"; }

private:
    YOLOv8SegPostProcess impl_;
};

// ============================================================================
// YOLOv5Seg Instance Segmentation Postprocessor (has objectness, not transposed)
// Output: [1, N, 4+1+C+32] detection + mask coefficients
//         [1, 32, mask_h, mask_w] prototype masks
// ============================================================================
class YOLOv5SegPostprocessor : public IPostprocessor<InstanceSegmentationResult> {
public:
    YOLOv5SegPostprocessor(int input_width = 640, int input_height = 640,
                           float obj_threshold = 0.25f,
                           float score_threshold = 0.3f,
                           float nms_threshold = 0.45f,
                           int num_classes = 80,
                           int num_masks = 32,
                           bool is_ort_configured = false)
        : input_width_(input_width), input_height_(input_height),
          obj_threshold_(obj_threshold), score_threshold_(score_threshold),
          nms_threshold_(nms_threshold), num_classes_(num_classes),
          num_masks_(num_masks) {}

    std::vector<InstanceSegmentationResult> process(const dxrt::TensorPtrs& outputs,
                                                    const PreprocessContext& ctx) override {
        std::vector<InstanceSegmentationResult> results;
        if (outputs.size() < 2) return results;

        const auto& det_tensor = outputs[0];
        const auto& proto_tensor = outputs[1];

        auto det_shape = det_tensor->shape();
        int N = static_cast<int>(det_shape.size() == 3 ? det_shape[1] : det_shape[0]);
        int cols = static_cast<int>(det_shape.back());

        const float* det_data = static_cast<const float*>(det_tensor->data());

        // Collect filtered detections
        std::vector<cv::Rect> nms_boxes;
        std::vector<std::array<float, 4>> nms_fboxes;  // x1, y1, w, h in float
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;
        std::vector<int> nms_orig_indices;
        std::vector<std::vector<float>> nms_mask_coefs;

        for (int i = 0; i < N; ++i) {
            const float* row = det_data + i * cols;
            float obj = row[4];
            if (obj < obj_threshold_) continue;

            // Find max class
            float max_cls_score = 0.0f;
            int max_cls = 0;
            for (int c = 0; c < num_classes_; ++c) {
                if (row[5 + c] > max_cls_score) {
                    max_cls_score = row[5 + c];
                    max_cls = c;
                }
            }

            float conf = obj * max_cls_score;
            if (conf < score_threshold_) continue;

            float cx = row[0], cy = row[1], w = row[2], h = row[3];
            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;

            nms_boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                        static_cast<int>(w), static_cast<int>(h)));
            nms_fboxes.push_back({x1, y1, w, h});
            nms_scores.push_back(conf);
            nms_class_ids.push_back(max_cls);
            nms_orig_indices.push_back(i);

            // Mask coefficients
            std::vector<float> coefs(num_masks_);
            for (int m = 0; m < num_masks_; ++m) {
                coefs[m] = row[5 + num_classes_ + m];
            }
            nms_mask_coefs.push_back(std::move(coefs));
        }

        if (nms_boxes.empty()) return results;

        // NMS
        std::vector<int> keep;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, keep);

        if (keep.empty()) return results;

        // Get prototype masks
        auto proto_shape = proto_tensor->shape();
        int proto_c = static_cast<int>(proto_shape.size() == 4 ? proto_shape[1] : proto_shape[0]);
        int proto_h = static_cast<int>(proto_shape.size() == 4 ? proto_shape[2] : proto_shape[1]);
        int proto_w = static_cast<int>(proto_shape.size() == 4 ? proto_shape[3] : proto_shape[2]);
        const float* proto_data = static_cast<const float*>(proto_tensor->data());

        for (int k : keep) {
            float x1 = nms_fboxes[k][0];
            float y1 = nms_fboxes[k][1];
            float bw = nms_fboxes[k][2];
            float bh = nms_fboxes[k][3];
            float x2 = x1 + bw;
            float y2 = y1 + bh;

            // Generate mask: coefs @ proto -> sigmoid
            cv::Mat mask(proto_h, proto_w, CV_32FC1, cv::Scalar(0));
            for (int ph = 0; ph < proto_h; ++ph) {
                for (int pw = 0; pw < proto_w; ++pw) {
                    float val = computeMaskDotProduct_(nms_mask_coefs[k], proto_data,
                                                       proto_c, proto_h, proto_w, ph, pw);
                    // Sigmoid activation
                    mask.at<float>(ph, pw) = 1.0f / (1.0f + std::exp(-val));
                }
            }

            // Resize mask to input size
            cv::Mat scaled_mask;
            cv::resize(mask, scaled_mask, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);

            // Crop to bbox
            int bx1 = std::max(0, static_cast<int>(x1));
            int by1 = std::max(0, static_cast<int>(y1));
            int bx2 = std::min(input_width_, static_cast<int>(x2));
            int by2 = std::min(input_height_, static_cast<int>(y2));
            scaled_mask(cv::Rect(0, 0, scaled_mask.cols, by1)).setTo(0);
            scaled_mask(cv::Rect(0, by2, scaled_mask.cols, scaled_mask.rows - by2)).setTo(0);
            scaled_mask(cv::Rect(0, 0, bx1, scaled_mask.rows)).setTo(0);
            scaled_mask(cv::Rect(bx2, 0, scaled_mask.cols - bx2, scaled_mask.rows)).setTo(0);

            // Remove padding and resize to original
            int unpad_h = static_cast<int>(std::round(ctx.original_height * ctx.scale));
            int unpad_w = static_cast<int>(std::round(ctx.original_width * ctx.scale));
            cv::Mat mask_crop = scaled_mask(cv::Rect(ctx.pad_x, ctx.pad_y, unpad_w, unpad_h)).clone();
            cv::Mat orig_mask;
            cv::resize(mask_crop, orig_mask, cv::Size(ctx.original_width, ctx.original_height), 0, 0, cv::INTER_LINEAR);

            // Binarize mask to 0/255
            cv::Mat binary_mask;
            orig_mask.convertTo(binary_mask, CV_8UC1, 255.0);
            cv::threshold(binary_mask, binary_mask, 127, 255, cv::THRESH_BINARY);

            // Scale box to original coords
            float fx1 = std::max(0.0f, std::min((x1 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            float fy1 = std::max(0.0f, std::min((y1 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));
            float fx2 = std::max(0.0f, std::min((x2 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            float fy2 = std::max(0.0f, std::min((y2 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));

            InstanceSegmentationResult seg;
            seg.box = {fx1, fy1, fx2, fy2};
            seg.confidence = nms_scores[k];
            seg.class_id = nms_class_ids[k];
            seg.class_name = dxapp::common::get_coco_class_name(seg.class_id);
            seg.mask = binary_mask;
            results.push_back(std::move(seg));
        }

        return results;
    }

    std::string getModelName() const override { return "YOLOv5-Seg"; }

private:
    int input_width_;
    int input_height_;
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
    int num_classes_;
    int num_masks_;

    // Compute dot-product of a single pixel's prototype features with mask
    // coefficients, returning the raw (pre-sigmoid) mask value.
    static float computeMaskDotProduct_(const std::vector<float>& coefs,
                                        const float* proto_data,
                                        int proto_c, int proto_h, int proto_w,
                                        int ph, int pw) {
        float val = 0.0f;
        for (int c = 0; c < proto_c; ++c)
            val += coefs[c] * proto_data[c * proto_h * proto_w + ph * proto_w + pw];
        return val;
    }
};

// ============================================================================
// BiseNetPostprocessor — DEPRECATED, use DeepLabv3Postprocessor instead.
// Kept as a type alias for backward compatibility.
// DeepLabv3Postprocessor is a strict superset (NCHW + NHWC, int16 + float).
// ============================================================================
using BiseNetPostprocessor = DeepLabv3Postprocessor;

}  // namespace dxapp

#endif  // SEGMENTATION_POSTPROCESSOR_HPP
