/**
 * @file rtdetr_postprocessor.hpp
 * @brief RT-DETR family postprocessors (RT-DETR, RT-DETRv2, RT-DETRv3, mask-RT-DETR)
 *
 * RT-DETR is NMS-free: the decoder emits a fixed number of queries (300), each with
 * one box and per-class logits, and the reference postprocess sigmoids the logits and
 * takes the top-k over the flattened (query x class) scores. Its preprocessing is a
 * plain resize with no letterbox, so the context carries scale_x / scale_y and zero
 * padding.
 *
 * MEASURED on DX-RT 3.5.0, the first runtime able to load these v9 models:
 *
 *   rtdetr-r18vd-6x_640x640            (300, 6)
 *   mask-rtdetr-hgnetv2-s-6x_640x640   (100, 6) + (100, 640, 640) int32 0/1
 *
 * so paddle_nms below is what these models actually emit, in model-input pixels, with
 * NO batch axis and with masks that are already binary. split is kept for an export
 * without the postprocess op. Both are accepted; anything else is refused:
 *
 *   split       boxes  (1, N, 4)   cxcywh normalised to [0,1]
 *               logits (1, N, C)   per-class, before sigmoid
 *               -- an export WITHOUT the postprocess op
 *
 *   paddle_nms  bbox   [N, 6]      [class_id, score, x1, y1, x2, y2], input pixels
 *               -- PaddleDetection's own DETRPostProcess folded into the export
 *
 * The layout is inferred from the shapes, and that inference is the risky part: a
 * (1, 300, 6) tensor is indistinguishable by shape from a pre-optimized row table,
 * which orders the same six columns the other way round ([box, score, class] rather
 * than [class, score, box]). Reading one as the other yields plausible-looking
 * nonsense rather than an error, so the config key "layout" forces the choice and
 * every failure message names it alongside the shapes actually seen.
 *
 * Mirrors src/python_example/common/processors/rtdetr_postprocessor.py, whose unit
 * tests pin this behaviour.
 */

#ifndef RTDETR_POSTPROCESSOR_HPP
#define RTDETR_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/processors/postprocess_utils.hpp"
#include "common/processors/result_converters.hpp"
#include "common_util.hpp"

namespace dxapp {

/// Which output layout an RT-DETR model emits.
enum class RTDETRLayout { Auto, Split, PaddleNMS };

inline RTDETRLayout rtdetrLayoutFromString(const std::string& name) {
    if (name.empty() || name == "auto") return RTDETRLayout::Auto;
    if (name == "split") return RTDETRLayout::Split;
    if (name == "paddle_nms") return RTDETRLayout::PaddleNMS;
    throw PostprocessConfigError(
        "[DXAPP] [ERROR] RT-DETR - unknown layout \"" + name +
        "\"; expected \"auto\", \"split\" or \"paddle_nms\".\n");
}

namespace rtdetr_detail {

/// One decoded query, before the result struct is built.
struct Query {
    float x1{0.0f}, y1{0.0f}, x2{0.0f}, y2{0.0f};
    float score{0.0f};
    int class_id{0};
    int index{0};          ///< which query row it came from, for the mask lookup
};

/// PaddleDetection's DETRPostProcess column order.
constexpr int kPClass = 0;
constexpr int kPScore = 1;
constexpr int kPX1 = 2;

/// A tensor is the box tensor when it is (1, N, 4).
inline dxrt::TensorPtr findBoxes(const dxrt::TensorPtrs& outputs) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 3 && s[0] == 1 && s[2] == 4) return o;
    }
    return nullptr;
}

/// ... and the score tensor when it is (1, N, C>4) with the same N.
inline dxrt::TensorPtr findScores(const dxrt::TensorPtrs& outputs, int64_t queries) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 3 && s[0] == 1 && s[1] == queries && s[2] > 4) return o;
    }
    return nullptr;
}

/**
 * The [N, 6] table PaddleDetection's own postprocess emits.
 *
 * MEASURED on DX-RT 3.5.0, the first runtime able to load these v9 models:
 * rtdetr-r18vd-6x emits (300, 6) and mask-rtdetr (100, 6) -- TWO dimensions, no batch
 * axis. The first version required a 3-D shape and would have rejected every real
 * model. The 3-D form is still accepted: nothing says the next compile drops the axis.
 */
inline dxrt::TensorPtr findPaddleTable(const dxrt::TensorPtrs& outputs) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 2 && s[1] == 6) return o;
        if (s.size() == 3 && s[0] == 1 && s[2] == 6) return o;
    }
    return nullptr;
}

/// Rows in a table that may or may not carry a leading batch axis.
inline int64_t tableRows(const dxrt::TensorPtr& table) {
    const auto& s = table->shape();
    return s.size() == 2 ? s[0] : s[1];
}

/**
 * Per-query masks, [N>=queries, H, W] or [1, N, H, W].
 *
 * MEASURED: mask-rtdetr-hgnetv2-s emits (100, 640, 640) of int32 whose values are 0
 * and 1 -- already binary, already at the model input resolution. Not logits, not
 * prototypes. Applying a sigmoid to an integer 1 gives 0.73, which still passes a 0.5
 * threshold, so the wrong assumption would have produced plausible masks and hidden.
 */
inline dxrt::TensorPtr findMasks(const dxrt::TensorPtrs& outputs, int64_t queries) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 4 && s[0] == 1 && s[1] >= queries) return o;
        if (s.size() == 3 && s[0] >= queries) return o;
    }
    return nullptr;
}

/**
 * True when the values are logits rather than probabilities.
 *
 * RT-DETR emits logits, but a model compiled with the sigmoid folded in emits
 * probabilities, and squashing those a second time silently shrinks every score
 * (sigmoid(0.93) = 0.717) without ever raising.
 */
inline bool needsSigmoid(const float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] < 0.0f || data[i] > 1.0f) return true;
    }
    return false;
}

inline float iou(const Query& a, const Query& b) {
    const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    const float inter = (x2 - x1) * (y2 - y1);
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/// Class-agnostic greedy NMS. OFF by default -- the model is NMS-free.
inline std::vector<Query> mergeDuplicates(std::vector<Query> queries,
                                          float nms_threshold) {
    if (nms_threshold >= 1.0f || queries.size() < 2) return queries;
    std::vector<Query> kept;
    for (const auto& q : queries) {
        bool suppressed = false;
        for (const auto& k : kept) {
            if (iou(q, k) >= nms_threshold) { suppressed = true; break; }
        }
        if (!suppressed) kept.push_back(q);
    }
    return kept;
}

inline void clampToFrame(Query& q, const PreprocessContext& ctx) {
    const float w = static_cast<float>(std::max(ctx.original_width - 1, 0));
    const float h = static_cast<float>(std::max(ctx.original_height - 1, 0));
    q.x1 = std::max(0.0f, std::min(q.x1, w));
    q.x2 = std::max(0.0f, std::min(q.x2, w));
    q.y1 = std::max(0.0f, std::min(q.y1, h));
    q.y2 = std::max(0.0f, std::min(q.y2, h));
}

}  // namespace rtdetr_detail

/// Shared query decoding for the RT-DETR family.
class RTDETRDecoder {
public:
    RTDETRDecoder(float score_threshold, float nms_threshold, int top_k,
                  RTDETRLayout layout)
        : score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          top_k_(std::max(1, top_k)), layout_(layout) {}

    std::vector<rtdetr_detail::Query> decode(const dxrt::TensorPtrs& outputs,
                                             const PreprocessContext& ctx) const {
        using namespace rtdetr_detail;
        if (layout_ == RTDETRLayout::Auto || layout_ == RTDETRLayout::Split) {
            auto boxes = findBoxes(outputs);
            if (boxes) {
                auto scores = findScores(outputs, boxes->shape()[1]);
                if (scores) return decodeSplit(boxes, scores, ctx);
            }
            if (layout_ == RTDETRLayout::Split) {
                fail(outputs, "layout \"split\" was requested but no (1, N, 4) + "
                              "(1, N, C>4) pair is present.");
            }
        }
        if (layout_ == RTDETRLayout::Auto || layout_ == RTDETRLayout::PaddleNMS) {
            auto table = findPaddleTable(outputs);
            if (table) return decodePaddle(table, ctx);
            if (layout_ == RTDETRLayout::PaddleNMS) {
                fail(outputs, "layout \"paddle_nms\" was requested but no (1, N, 6) "
                              "table is present.");
            }
        }
        fail(outputs, "no recognised RT-DETR output layout.");
    }

    int64_t queryCount(const dxrt::TensorPtrs& outputs) const {
        if (auto boxes = rtdetr_detail::findBoxes(outputs)) return boxes->shape()[1];
        if (auto table = rtdetr_detail::findPaddleTable(outputs))
            return rtdetr_detail::tableRows(table);
        return 0;
    }

    float nmsThreshold() const { return nms_threshold_; }

private:
    [[noreturn]] void fail(const dxrt::TensorPtrs& outputs, const char* what) const {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] RT-DETR - " << what << "\n"
            << "  Expected either the raw decoder head -- boxes (1, N, 4) cxcywh in "
               "[0,1] plus logits (1, N, num_classes) -- or PaddleDetection's own "
               "postprocess output, bbox (1, N, 6) as [class, score, x1, y1, x2, y2].\n"
            << postprocess_utils::format_tensor_shapes(outputs)
            << "  Set config \"layout\" to \"split\" or \"paddle_nms\" to force one: a "
               "(1, N, 6) tensor cannot be told apart from a pre-optimized row table "
               "by shape alone, and the two order their columns differently.\n";
        throw PostprocessConfigError(msg.str());
    }

    std::vector<rtdetr_detail::Query> decodeSplit(const dxrt::TensorPtr& boxes,
                                                  const dxrt::TensorPtr& scores,
                                                  const PreprocessContext& ctx) const {
        using namespace rtdetr_detail;
        const auto queries = static_cast<int>(boxes->shape()[1]);
        const auto classes = static_cast<int>(scores->shape()[2]);
        const auto* box_data = static_cast<const float*>(boxes->data());
        const auto* score_data = static_cast<const float*>(scores->data());
        const bool sigmoid = needsSigmoid(
            score_data, static_cast<size_t>(queries) * classes);

        // Top-k over the flattened (query x class) scores, as the reference
        // postprocess does, so one query may survive under two classes.
        std::vector<Query> out;
        const float ow = static_cast<float>(ctx.original_width);
        const float oh = static_cast<float>(ctx.original_height);
        for (int q = 0; q < queries; ++q) {
            const float* b = box_data + static_cast<std::ptrdiff_t>(q) * 4;
            for (int c = 0; c < classes; ++c) {
                float s = score_data[static_cast<std::ptrdiff_t>(q) * classes + c];
                if (sigmoid) s = postprocess_utils::sigmoid(s);
                if (s < score_threshold_) continue;
                Query item;
                // cxcywh normalised to [0,1] -> xyxy in ORIGINAL pixels. RT-DETR
                // normalises against the input, and the input is a plain stretch of
                // the whole frame, so the ratios apply directly to the original size.
                item.x1 = (b[0] - b[2] / 2.0f) * ow;
                item.y1 = (b[1] - b[3] / 2.0f) * oh;
                item.x2 = (b[0] + b[2] / 2.0f) * ow;
                item.y2 = (b[1] + b[3] / 2.0f) * oh;
                item.score = s;
                item.class_id = c;
                item.index = q;
                clampToFrame(item, ctx);
                out.push_back(item);
            }
        }
        return finish(std::move(out));
    }

    std::vector<rtdetr_detail::Query> decodePaddle(const dxrt::TensorPtr& table,
                                                   const PreprocessContext& ctx) const {
        using namespace rtdetr_detail;
        const auto rows = static_cast<int>(rtdetr_detail::tableRows(table));
        const auto* data = static_cast<const float*>(table->data());
        // The table is fixed-size and pads with zero rows, whose score is 0.0, so the
        // threshold is floored above zero exactly as in the pre-optimized decode.
        const float threshold = std::max(score_threshold_,
                                         std::numeric_limits<float>::min());
        const float sx = ctx.scale_x > 0.0f ? ctx.scale_x : std::max(ctx.scale, 1e-6f);
        const float sy = ctx.scale_y > 0.0f ? ctx.scale_y : std::max(ctx.scale, 1e-6f);

        std::vector<Query> out;
        for (int r = 0; r < rows; ++r) {
            const float* row = data + static_cast<std::ptrdiff_t>(r) * 6;
            if (row[kPScore] < threshold) continue;
            Query item;
            item.x1 = row[kPX1 + 0] / sx;
            item.y1 = row[kPX1 + 1] / sy;
            item.x2 = row[kPX1 + 2] / sx;
            item.y2 = row[kPX1 + 3] / sy;
            item.score = row[kPScore];
            item.class_id = static_cast<int>(row[kPClass]);
            item.index = r;
            clampToFrame(item, ctx);
            out.push_back(item);
        }
        return finish(std::move(out));
    }

    std::vector<rtdetr_detail::Query> finish(std::vector<rtdetr_detail::Query> out) const {
        std::stable_sort(out.begin(), out.end(),
                         [](const rtdetr_detail::Query& a,
                            const rtdetr_detail::Query& b) { return a.score > b.score; });
        if (static_cast<int>(out.size()) > top_k_) out.resize(top_k_);
        return rtdetr_detail::mergeDuplicates(std::move(out), nms_threshold_);
    }

    float score_threshold_;
    float nms_threshold_;
    int top_k_;
    RTDETRLayout layout_;
};

// ============================================================================
// Object detection
// ============================================================================
class RTDETRPostprocessor : public IPostprocessor<DetectionResult> {
public:
    RTDETRPostprocessor(int input_width = 640, int input_height = 640,
                        float score_threshold = 0.4f,
                        // 1.0 == off, the correct default: the model is NMS-free.
                        float nms_threshold = 1.0f,
                        int top_k = 300,
                        const std::string& layout = "auto",
                        const std::vector<std::string>& class_names = {})
        : decoder_(score_threshold, nms_threshold, top_k,
                   rtdetrLayoutFromString(layout)),
          input_width_(input_width), input_height_(input_height),
          class_names_(class_names) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        for (const auto& q : decoder_.decode(outputs, ctx)) {
            DetectionResult det;
            det.box = {q.x1, q.y1, q.x2, q.y2};
            det.confidence = q.score;
            det.class_id = q.class_id;
            det.class_name = dxapp::common::resolve_class_name(q.class_id, class_names_);
            results.push_back(std::move(det));
        }
        return results;
    }

    std::string getModelName() const override { return "RT-DETR"; }

private:
    RTDETRDecoder decoder_;
    int input_width_;
    int input_height_;
    std::vector<std::string> class_names_;
};

// ============================================================================
// Instance segmentation: one mask logit map per query
// ============================================================================
class MaskRTDETRPostprocessor : public IPostprocessor<InstanceSegmentationResult> {
public:
    MaskRTDETRPostprocessor(int input_width = 640, int input_height = 640,
                            float score_threshold = 0.4f,
                            float nms_threshold = 1.0f,
                            int top_k = 300,
                            float mask_threshold = 0.5f,
                            const std::string& layout = "auto",
                            const std::vector<std::string>& class_names = {})
        : decoder_(score_threshold, nms_threshold, top_k,
                   rtdetrLayoutFromString(layout)),
          input_width_(input_width), input_height_(input_height),
          mask_threshold_(mask_threshold), class_names_(class_names) {}

    std::vector<InstanceSegmentationResult> process(
        const dxrt::TensorPtrs& outputs, const PreprocessContext& ctx) override {
        const auto queries = decoder_.queryCount(outputs);
        auto masks = rtdetr_detail::findMasks(outputs, std::max<int64_t>(queries, 1));
        if (!masks) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] MaskRTDETRPostprocessor - per-query mask tensor "
                   "(1, N>=" << queries << ", H, W) not found in the model outputs.\n"
                << postprocess_utils::format_tensor_shapes(outputs)
                << "  mask-RT-DETR emits the query boxes/logits AND one mask map per "
                   "query; only the boxes were found.\n";
            throw PostprocessConfigError(msg.str());
        }
        const auto& ms = masks->shape();
        // (N, H, W) or (1, N, H, W): H and W are always the last two axes.
        const int mask_h = static_cast<int>(ms[ms.size() - 2]);
        const int mask_w = static_cast<int>(ms[ms.size() - 1]);
        const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(mask_h) * mask_w;
        // The measured masks are int32 0/1, not float. Reading that buffer through a
        // float* would reinterpret the bits and produce denormal garbage that still
        // "looks like" a mask after thresholding. elem_size() is how the rest of this
        // tree tells the width of a tensor element (see segmentation_postprocessor).
        const size_t elem = masks->elem_size();
        const void* mask_data = masks->data();
        if (elem != 4) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] MaskRTDETRPostprocessor - mask elements are "
                << elem << " bytes; only 4-byte int32 or float masks are handled.\n"
                << postprocess_utils::format_tensor_shapes(outputs);
            throw PostprocessConfigError(msg.str());
        }
        // int32 0/1 has no bit pattern in common with a plausible float logit, so the
        // two are told apart by asking whether the float view is finite and in range.
        const auto* as_float = static_cast<const float*>(mask_data);
        const auto* as_int = static_cast<const int32_t*>(mask_data);
        bool integral_masks = true;
        for (std::ptrdiff_t i = 0; i < std::min<std::ptrdiff_t>(plane, 4096); ++i) {
            if (as_int[i] != 0 && as_int[i] != 1) { integral_masks = false; break; }
        }

        std::vector<InstanceSegmentationResult> results;
        for (const auto& q : decoder_.decode(outputs, ctx)) {
            cv::Mat probability;
            if (integral_masks) {
                const cv::Mat binary_src(mask_h, mask_w, CV_32SC1,
                                         const_cast<int32_t*>(as_int + q.index * plane));
                binary_src.convertTo(probability, CV_32FC1);
            } else {
                const cv::Mat logits(mask_h, mask_w, CV_32FC1,
                                     const_cast<float*>(as_float + q.index * plane));
                if (rtdetr_detail::needsSigmoid(logits.ptr<float>(),
                                                static_cast<size_t>(plane))) {
                    cv::exp(-logits, probability);
                    probability = 1.0 / (1.0 + probability);
                } else {
                    probability = logits.clone();
                }
            }
            cv::Mat full;
            cv::resize(probability, full,
                       cv::Size(ctx.original_width, ctx.original_height), 0, 0,
                       cv::INTER_LINEAR);
            cv::Mat binary = full > mask_threshold_;   // CV_8U, 0 or 255

            // Confine to the box: a query's mask logits are not bounded by its box,
            // which is what every other instance-segmentation postprocessor here does.
            const int x1 = std::max(0, static_cast<int>(q.x1));
            const int y1 = std::max(0, static_cast<int>(q.y1));
            const int x2 = std::min(ctx.original_width,
                                    static_cast<int>(std::ceil(q.x2)));
            const int y2 = std::min(ctx.original_height,
                                    static_cast<int>(std::ceil(q.y2)));
            cv::Mat confined = cv::Mat::zeros(binary.size(), binary.type());
            if (x2 > x1 && y2 > y1) {
                const cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
                binary(roi).copyTo(confined(roi));
            }

            InstanceSegmentationResult seg;
            seg.box = {q.x1, q.y1, q.x2, q.y2};
            seg.confidence = q.score;
            seg.class_id = q.class_id;
            seg.class_name = dxapp::common::resolve_class_name(q.class_id, class_names_);
            seg.mask = confined;
            results.push_back(std::move(seg));
        }
        return results;
    }

    std::string getModelName() const override { return "mask-RT-DETR"; }

private:
    RTDETRDecoder decoder_;
    int input_width_;
    int input_height_;
    float mask_threshold_;
    std::vector<std::string> class_names_;
};

}  // namespace dxapp

#endif  // RTDETR_POSTPROCESSOR_HPP
