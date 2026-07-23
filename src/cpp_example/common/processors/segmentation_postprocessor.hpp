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
#include "common/processors/roi_instance_mask.hpp"
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
// Fast Semantic Segmentation Postprocessor
// ============================================================================
class FastSegmentationPostprocessor : public IPostprocessor<SegmentationResult> {
public:
    FastSegmentationPostprocessor(int input_width = 2048, int input_height = 1024)
        : input_width_(input_width), input_height_(input_height) {}

    std::vector<SegmentationResult> process(const dxrt::TensorPtrs& outputs,
                                            const PreprocessContext& ctx) override {
        std::vector<SegmentationResult> results;
        if (outputs.empty()) return results;

        const auto& tensor = outputs[0];
        cv::Mat class_map = toClassMap_(tensor->data(), tensor->shape(), tensor->elem_size());
        if (class_map.empty()) return results;

        class_map = resizeToOriginal_(class_map, ctx);

        SegmentationResult seg;
        seg.width = class_map.cols;
        seg.height = class_map.rows;
        seg.mask.resize(seg.width * seg.height);

        std::set<int> unique_classes;
        for (int y = 0; y < class_map.rows; ++y) {
            const int* row = class_map.ptr<int>(y);
            for (int x = 0; x < class_map.cols; ++x) {
                int cls = row[x];
                seg.mask[y * class_map.cols + x] = cls;
                unique_classes.insert(cls);
            }
        }
        seg.class_ids.assign(unique_classes.begin(), unique_classes.end());
        results.push_back(std::move(seg));
        return results;
    }

    std::string getModelName() const override { return "fast_segmentation"; }

private:
    static cv::Mat toClassMap_(const void* data, const std::vector<int64_t>& shape,
                               size_t elem_size) {
        if (shape.size() == 2) {
            return copyClassMap_(data, static_cast<int>(shape[0]), static_cast<int>(shape[1]), elem_size);
        }

        int C = 0, H = 0, W = 0;
        bool is_nhwc = false;
        bool single_channel = false;
        if (shape.size() == 4) {
            if (shape[1] <= shape[3]) {
                C = static_cast<int>(shape[1]);
                H = static_cast<int>(shape[2]);
                W = static_cast<int>(shape[3]);
            } else {
                H = static_cast<int>(shape[1]);
                W = static_cast<int>(shape[2]);
                C = static_cast<int>(shape[3]);
                is_nhwc = true;
            }
            single_channel = C == 1;
        } else if (shape.size() == 3) {
            if (shape[0] == 1) {
                C = 1;
                H = static_cast<int>(shape[1]);
                W = static_cast<int>(shape[2]);
                single_channel = true;
            } else if (shape[2] < shape[0] && shape[2] < shape[1]) {
                H = static_cast<int>(shape[0]);
                W = static_cast<int>(shape[1]);
                C = static_cast<int>(shape[2]);
                is_nhwc = true;
            } else {
                C = static_cast<int>(shape[0]);
                H = static_cast<int>(shape[1]);
                W = static_cast<int>(shape[2]);
            }
        } else {
            return cv::Mat();
        }

        if (single_channel || elem_size != sizeof(float)) {
            return copyClassMap_(data, H, W, elem_size);
        }
        return argmaxLogits_(static_cast<const float*>(data), C, H, W, is_nhwc);
    }

    static cv::Mat copyClassMap_(const void* data, int H, int W, size_t elem_size) {
        cv::Mat class_map(H, W, CV_32SC1);
        for (int y = 0; y < H; ++y) {
            int* dst = class_map.ptr<int>(y);
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;
                if (elem_size == sizeof(int64_t)) {
                    dst[x] = static_cast<int>(static_cast<const int64_t*>(data)[idx]);
                } else if (elem_size == sizeof(int16_t)) {
                    dst[x] = static_cast<int>(static_cast<const int16_t*>(data)[idx]);
                } else if (elem_size == sizeof(float)) {
                    dst[x] = static_cast<int>(std::round(static_cast<const float*>(data)[idx]));
                } else {
                    return cv::Mat();
                }
            }
        }
        return class_map;
    }

    static cv::Mat argmaxLogits_(const float* data, int C, int H, int W, bool is_nhwc) {
        cv::Mat class_map(H, W, CV_32SC1);
        for (int y = 0; y < H; ++y) {
            int* dst = class_map.ptr<int>(y);
            for (int x = 0; x < W; ++x) {
                float max_val = -1e9f;
                int max_cls = 0;
                for (int c = 0; c < C; ++c) {
                    float val = is_nhwc ? data[y * W * C + x * C + c]
                                        : data[c * H * W + y * W + x];
                    if (val > max_val) {
                        max_val = val;
                        max_cls = c;
                    }
                }
                dst[x] = max_cls;
            }
        }
        return class_map;
    }

    static cv::Mat resizeToOriginal_(const cv::Mat& class_map, const PreprocessContext& ctx) {
        if (ctx.original_width <= 0 || ctx.original_height <= 0) return class_map;
        if (ctx.pad_x == 0 && ctx.pad_y == 0) {
            if (class_map.cols == ctx.original_width && class_map.rows == ctx.original_height) {
                return class_map;
            }
            cv::Mat resized;
            cv::resize(class_map, resized, cv::Size(ctx.original_width, ctx.original_height),
                       0, 0, cv::INTER_NEAREST);
            return resized;
        }

        float gain = std::max(ctx.scale, 1e-6f);
        int unpad_w = static_cast<int>(std::round(ctx.original_width * gain));
        int unpad_h = static_cast<int>(std::round(ctx.original_height * gain));
        cv::Rect crop(ctx.pad_x, ctx.pad_y,
                      std::min(unpad_w, class_map.cols - ctx.pad_x),
                      std::min(unpad_h, class_map.rows - ctx.pad_y));
        if (crop.width <= 0 || crop.height <= 0) return class_map;
        cv::Mat resized;
        cv::resize(class_map(crop), resized, cv::Size(ctx.original_width, ctx.original_height),
                   0, 0, cv::INTER_NEAREST);
        return resized;
    }

    int input_width_;
    int input_height_;
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
                           bool is_ort_configured = false, int num_classes = 80,
                           const std::vector<std::string>& class_names = {})
        : impl_(input_width, input_height, score_threshold, nms_threshold,
                is_ort_configured, num_classes, class_names) {}

    std::vector<InstanceSegmentationResult> process(const dxrt::TensorPtrs& outputs,
                                                    const PreprocessContext& ctx) override {
        std::vector<InstanceSegmentationResult> results;

        // Align once, then decode detections WITHOUT materialising a full-frame
        // float mask per instance. The old path built an input-resolution float
        // mask in the impl AND resized every mask to the full original frame in
        // scaleInstanceSegResults() — the multi-GB peak-memory hotspot for
        // instance-heavy inputs (e.g. FastSAM 1024x1024).
        dxrt::TensorPtrs aligned = impl_.get_is_ort_configured()
                                       ? outputs : impl_.align_tensors(outputs);
        if (aligned.size() < 2) return results;

        auto detections = impl_.decode_detections(aligned);
        if (detections.empty()) return results;

        // Prototype masks: layout [1, C, H, W] (or [C, H, W]).
        const auto& proto = aligned[1];
        auto ps = proto->shape();
        int proto_c = static_cast<int>(ps.size() == 4 ? ps[1] : ps[0]);
        int proto_h = static_cast<int>(ps.size() == 4 ? ps[2] : ps[1]);
        int proto_w = static_cast<int>(ps.size() == 4 ? ps[3] : ps[2]);
        const float* proto_data = static_cast<const float*>(proto->data());

        const int in_w = impl_.get_input_width();
        const int in_h = impl_.get_input_height();

        results.reserve(detections.size());
        for (const auto& d : detections) {
            if (d.box.size() < 4 || d.seg_mask_coef.size() != static_cast<size_t>(proto_c))
                continue;

            // Box is in model-input (letterboxed) coordinates.
            const float x1 = d.box[0], y1 = d.box[1], x2 = d.box[2], y2 = d.box[3];

            // Box-ROI mask: sigmoid(coefs . proto) + a single aligned resize over
            // only the bbox region, directly to original resolution. Output is a
            // CV_8UC1 mask zeroed outside the bbox — matching the YOLOv5-Seg path
            // and what InstanceSegmentationVisualizer already expects.
            cv::Mat binary_mask = roiInstanceMask(
                d.seg_mask_coef, proto_data, proto_c, proto_h, proto_w,
                x1, y1, x2, y2, in_w, in_h, ctx);

            const float fx1 = std::max(0.0f, std::min((x1 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            const float fy1 = std::max(0.0f, std::min((y1 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));
            const float fx2 = std::max(0.0f, std::min((x2 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            const float fy2 = std::max(0.0f, std::min((y2 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));

            InstanceSegmentationResult seg;
            seg.box = {fx1, fy1, fx2, fy2};
            seg.confidence = d.confidence;
            seg.class_id = d.class_id;
            seg.class_name = d.class_name;
            seg.mask = binary_mask;
            results.push_back(std::move(seg));
        }
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
                           bool is_ort_configured = false,
                           const std::vector<std::string>& class_names = {})
        : input_width_(input_width), input_height_(input_height),
          obj_threshold_(obj_threshold), score_threshold_(score_threshold),
          nms_threshold_(nms_threshold), num_classes_(num_classes),
          num_masks_(num_masks), class_names_(class_names) {}

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

            // Box-ROI mask: sigmoid(coefs . proto) + single aligned resize over
            // only the bbox region, directly to original resolution (mirrors the
            // YOLOv8-Seg path). Replaces the full-prototype sigmoid + two
            // full-frame resizes; output matches at sub-pixel mask boundaries.
            cv::Mat binary_mask = roiInstanceMask(
                nms_mask_coefs[k], proto_data, proto_c, proto_h, proto_w,
                x1, y1, x2, y2, input_width_, input_height_, ctx);

            // Scale box to original coords
            float fx1 = std::max(0.0f, std::min((x1 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            float fy1 = std::max(0.0f, std::min((y1 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));
            float fx2 = std::max(0.0f, std::min((x2 - ctx.pad_x) / ctx.scale, static_cast<float>(ctx.original_width)));
            float fy2 = std::max(0.0f, std::min((y2 - ctx.pad_y) / ctx.scale, static_cast<float>(ctx.original_height)));

            InstanceSegmentationResult seg;
            seg.box = {fx1, fy1, fx2, fy2};
            seg.confidence = nms_scores[k];
            seg.class_id = nms_class_ids[k];
            seg.class_name = dxapp::common::resolve_class_name(seg.class_id, class_names_);
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
    std::vector<std::string> class_names_;
};

// ============================================================================
// BiseNetPostprocessor — DEPRECATED, use DeepLabv3Postprocessor instead.
// Kept as a type alias for backward compatibility.
// DeepLabv3Postprocessor is a strict superset (NCHW + NHWC, int16 + float).
// ============================================================================
using BiseNetPostprocessor = DeepLabv3Postprocessor;

}  // namespace dxapp

#endif  // SEGMENTATION_POSTPROCESSOR_HPP
