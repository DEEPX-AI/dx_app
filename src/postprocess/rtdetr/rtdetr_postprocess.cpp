#include "rtdetr_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

// PaddleDetection's DETRPostProcess column order.
constexpr int kColClass = 0;
constexpr int kColScore = 1;
constexpr int kColX1 = 2;

std::string format_shapes(const dxrt::TensorPtrs& outputs) {
    std::ostringstream s;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        s << (i ? ", " : "") << "(";
        const auto& shape = outputs[i]->shape();
        for (std::size_t d = 0; d < shape.size(); ++d) {
            s << (d ? ", " : "") << shape[d];
        }
        s << ")";
    }
    return s.str();
}

dxrt::TensorPtr find_boxes(const dxrt::TensorPtrs& outputs) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 3 && s[0] == 1 && s[2] == 4) return o;
    }
    return nullptr;
}

dxrt::TensorPtr find_logits(const dxrt::TensorPtrs& outputs, int64_t queries) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 3 && s[0] == 1 && s[1] == queries && s[2] > 4) return o;
    }
    return nullptr;
}

dxrt::TensorPtr find_paddle_table(const dxrt::TensorPtrs& outputs) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 2 && s[1] == 6) return o;
        if (s.size() == 3 && s[0] == 1 && s[2] == 6) return o;
    }
    return nullptr;
}

const float* float_data(const dxrt::TensorPtr& tensor, const dxrt::TensorPtrs& all) {
    if (tensor->type() != dxrt::DataType::FLOAT) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] RTDETRPostProcess - expected float32 outputs, got " +
            format_shapes(all));
    }
    return static_cast<const float*>(tensor->data());
}

// True when the values are not yet probabilities. A model compiled with the
// sigmoid folded in emits [0, 1] scores, and squashing those again would halve
// every score without raising.
bool needs_sigmoid(const float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (data[i] < 0.0f || data[i] > 1.0f) return true;
    }
    return false;
}

// IoU of the two boxes clipped to the input frame: the Python postprocessor
// runs its NMS on boxes already clipped to the image, and the input frame is
// the closest stand-in for it that needs no preprocessing context.
float clipped_iou(const RTDETRResult& a, const RTDETRResult& b, float max_x, float max_y) {
    auto clip = [&](const std::vector<float>& box, float* out) {
        out[0] = std::min(std::max(box[0], 0.0f), max_x);
        out[1] = std::min(std::max(box[1], 0.0f), max_y);
        out[2] = std::min(std::max(box[2], 0.0f), max_x);
        out[3] = std::min(std::max(box[3], 0.0f), max_y);
    };
    float p[4], q[4];
    clip(a.box, p);
    clip(b.box, q);
    const float area_a = std::max(p[2] - p[0], 0.0f) * std::max(p[3] - p[1], 0.0f);
    const float area_b = std::max(q[2] - q[0], 0.0f) * std::max(q[3] - q[1], 0.0f);
    const float inter = std::max(std::min(p[2], q[2]) - std::max(p[0], q[0]), 0.0f) *
                        std::max(std::min(p[3], q[3]) - std::max(p[1], q[1]), 0.0f);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / std::max(uni, 1e-12f) : 0.0f;
}

[[noreturn]] void fail(const dxrt::TensorPtrs& outputs, const char* what) {
    std::ostringstream msg;
    msg << "[DXAPP] [ERROR] RTDETRPostProcess - " << what << "\n"
        << "  Got: " << format_shapes(outputs) << "\n"
        << "  Expected either the raw decoder head -- boxes (1, N, 4) cxcywh in [0,1] "
           "plus logits (1, N, num_classes) -- or PaddleDetection's own postprocess "
           "output, bbox (1, N, 6) as [class, score, x1, y1, x2, y2].\n"
        << "  Set layout to \"split\" or \"paddle_nms\" to force one; a (1, N, 6) "
           "tensor cannot be told apart from a pre-optimized row table by shape "
           "alone, and the two order their columns differently.";
    throw std::runtime_error(msg.str());
}

}  // namespace

RTDETRPostProcess::RTDETRPostProcess(int input_w, int input_h, float score_threshold,
                                     float nms_threshold, int top_k,
                                     const std::string& layout)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      top_k_(std::max(1, top_k)),
      layout_(layout.empty() ? "auto" : layout) {
    if (layout_ != "auto" && layout_ != "split" && layout_ != "paddle_nms") {
        throw std::invalid_argument(
            "[DXAPP] [ERROR] RTDETRPostProcess - unknown layout \"" + layout +
            "\"; expected \"auto\", \"split\" or \"paddle_nms\".");
    }
}

std::vector<RTDETRResult> RTDETRPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (layout_ == "auto" || layout_ == "split") {
        if (auto boxes = find_boxes(outputs)) {
            if (auto logits = find_logits(outputs, boxes->shape()[1])) {
                return decode_split(boxes, logits);
            }
        }
        if (layout_ == "split") {
            fail(outputs, "layout \"split\" was requested but no (1, N, 4) + "
                          "(1, N, C>4) pair is present.");
        }
    }
    if (layout_ == "auto" || layout_ == "paddle_nms") {
        if (auto table = find_paddle_table(outputs)) return decode_paddle(table);
        if (layout_ == "paddle_nms") {
            fail(outputs, "layout \"paddle_nms\" was requested but no (1, N, 6) "
                          "table is present.");
        }
    }
    fail(outputs, "no recognised RT-DETR output layout.");
}

std::vector<RTDETRResult> RTDETRPostProcess::decode_split(
    const dxrt::TensorPtr& boxes, const dxrt::TensorPtr& logits) const {
    const auto queries = static_cast<int>(boxes->shape()[1]);
    const auto classes = static_cast<int>(logits->shape()[2]);
    const float* box_data = float_data(boxes, {boxes, logits});
    const float* score_data = float_data(logits, {boxes, logits});
    const bool sigmoid = needs_sigmoid(score_data, static_cast<std::size_t>(queries) * classes);
    const float w_in = static_cast<float>(input_width_);
    const float h_in = static_cast<float>(input_height_);

    // Query-major, then class: the order numpy's stable argsort keeps for ties.
    std::vector<RTDETRResult> out;
    for (int q = 0; q < queries; ++q) {
        const float* b = box_data + static_cast<std::ptrdiff_t>(q) * 4;
        for (int c = 0; c < classes; ++c) {
            float s = score_data[static_cast<std::ptrdiff_t>(q) * classes + c];
            if (sigmoid) s = 1.0f / (1.0f + std::exp(-s));
            if (s < score_threshold_) continue;
            // cxcywh normalised to the input -> model-input pixels.
            out.emplace_back(std::vector<float>{(b[0] - b[2] / 2.0f) * w_in,
                                                (b[1] - b[3] / 2.0f) * h_in,
                                                (b[0] + b[2] / 2.0f) * w_in,
                                                (b[1] + b[3] / 2.0f) * h_in},
                             s, c, q);
        }
    }
    return finish(std::move(out));
}

std::vector<RTDETRResult> RTDETRPostProcess::decode_paddle(const dxrt::TensorPtr& table) const {
    const auto& shape = table->shape();
    const auto rows = static_cast<int>(shape.size() == 2 ? shape[0] : shape[1]);
    const float* data = float_data(table, {table});
    // The table is fixed-size and pads with zero rows, whose score is 0.0, so the
    // threshold is floored just above zero (numpy.finfo(float32).tiny).
    const float threshold = std::max(score_threshold_, std::numeric_limits<float>::min());

    std::vector<RTDETRResult> out;
    for (int r = 0; r < rows; ++r) {
        const float* row = data + static_cast<std::ptrdiff_t>(r) * 6;
        if (row[kColScore] < threshold) continue;
        out.emplace_back(std::vector<float>{row[kColX1], row[kColX1 + 1],
                                            row[kColX1 + 2], row[kColX1 + 3]},
                         row[kColScore], static_cast<int>(row[kColClass]), r);
    }
    return finish(std::move(out));
}

std::vector<RTDETRResult> RTDETRPostProcess::finish(std::vector<RTDETRResult> results) const {
    std::stable_sort(results.begin(), results.end(),
                     [](const RTDETRResult& a, const RTDETRResult& b) {
                         return a.confidence > b.confidence;
                     });
    if (static_cast<int>(results.size()) > top_k_) results.resize(top_k_);
    if (nms_threshold_ >= 1.0f || results.size() < 2) return results;

    // Class-agnostic greedy NMS, as the Python postprocessor's _merge_duplicates.
    const float max_x = static_cast<float>(input_width_ - 1);
    const float max_y = static_cast<float>(input_height_ - 1);
    std::vector<RTDETRResult> kept;
    for (auto& r : results) {
        bool suppressed = false;
        for (const auto& k : kept) {
            if (clipped_iou(r, k, max_x, max_y) >= nms_threshold_) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(std::move(r));
    }
    return kept;
}

// ---------------------------------------------------------------- mask-RT-DETR

MaskRTDETRPostProcess::MaskRTDETRPostProcess(int input_w, int input_h, float score_threshold,
                                             float nms_threshold, int top_k,
                                             const std::string& layout)
    : decoder_(input_w, input_h, score_threshold, nms_threshold, top_k, layout) {}

std::vector<MaskRTDETRResult> MaskRTDETRPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<RTDETRResult> detections = decoder_.postprocess(outputs);
    // Demand the mask tensor even for an empty table, so a wired-up-wrong model
    // is reported rather than looking like "nothing detected".
    int64_t needed = 1;
    for (const auto& d : detections) needed = std::max<int64_t>(needed, d.query + 1);

    dxrt::TensorPtr masks;
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if ((s.size() == 4 && s[0] == 1 && s[1] >= needed) || (s.size() == 3 && s[0] >= needed)) {
            masks = o;
            break;
        }
    }
    if (!masks) {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] MaskRTDETRPostProcess - per-query mask tensor (N>="
            << needed << ", H, W) not found in the model outputs.\n"
            << "  Got: " << format_shapes(outputs) << "\n"
            << "  mask-RT-DETR emits the query table AND one mask per query; only the "
               "table was found.";
        throw std::runtime_error(msg.str());
    }

    const auto& ms = masks->shape();
    const int mask_h = static_cast<int>(ms[ms.size() - 2]);
    const int mask_w = static_cast<int>(ms[ms.size() - 1]);
    const std::size_t plane = static_cast<std::size_t>(mask_h) * static_cast<std::size_t>(mask_w);
    const dxrt::DataType type = masks->type();
    const void* data = masks->data();

    std::vector<MaskRTDETRResult> results;
    results.reserve(detections.size());
    for (auto& det : detections) {
        MaskRTDETRResult item;
        item.mask.resize(plane);
        const std::size_t offset = static_cast<std::size_t>(det.query) * plane;
        if (type == dxrt::DataType::FLOAT) {
            const float* src = static_cast<const float*>(data) + offset;
            std::copy(src, src + plane, item.mask.begin());
        } else if (type == dxrt::DataType::INT32) {
            const int32_t* src = static_cast<const int32_t*>(data) + offset;
            for (std::size_t i = 0; i < plane; ++i) item.mask[i] = static_cast<float>(src[i]);
        } else if (type == dxrt::DataType::UINT8) {
            const uint8_t* src = static_cast<const uint8_t*>(data) + offset;
            for (std::size_t i = 0; i < plane; ++i) item.mask[i] = static_cast<float>(src[i]);
        } else {
            throw std::runtime_error(
                "[DXAPP] [ERROR] MaskRTDETRPostProcess - unsupported mask tensor type; "
                "expected float32, int32 or uint8. Got: " + format_shapes(outputs));
        }
        // An already-binary 0/1 mask needs no sigmoid; a float logit map does.
        if (needs_sigmoid(item.mask.data(), plane)) {
            for (float& v : item.mask) v = 1.0f / (1.0f + std::exp(-v));
        }
        item.mask_height = mask_h;
        item.mask_width = mask_w;
        item.detection = std::move(det);
        results.push_back(std::move(item));
    }
    return results;
}
