#include "preopt_topk_postprocess.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kColScore = 4;
constexpr int kColClass = 5;
constexpr int kColExtra = 6;

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

// The (1, K, C) table with K > C >= 6, identified by shape, never by position:
// the prototypes of a segmentation model are 4-D and a plain detection head
// would have its long axis last.
dxrt::TensorPtr find_row_table(const dxrt::TensorPtrs& outputs) {
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 3 && s[0] == 1 && s[1] > s[2] && s[2] >= kColExtra) return o;
    }
    std::ostringstream msg;
    msg << "[DXAPP] [ERROR] PreoptTopKPostProcess - pre-optimized row table (1, K, C>=6) "
           "not found in the model outputs. Got: " << format_shapes(outputs)
        << ". This postprocessor requires a model compiled with the pre-optimize "
           "(top-k) output configuration, and a DXRT built with ONNX Runtime so the "
           "model's CPU task actually runs.";
    throw std::runtime_error(msg.str());
}

// Same IoU as the Python select_rows(): clipped widths and heights, 0 when the
// union is empty.
float iou(const PreoptResult& a, const PreoptResult& b) {
    const float area_a = std::max(a.box[2] - a.box[0], 0.0f) * std::max(a.box[3] - a.box[1], 0.0f);
    const float area_b = std::max(b.box[2] - b.box[0], 0.0f) * std::max(b.box[3] - b.box[1], 0.0f);
    const float x1 = std::max(a.box[0], b.box[0]), y1 = std::max(a.box[1], b.box[1]);
    const float x2 = std::min(a.box[2], b.box[2]), y2 = std::min(a.box[3], b.box[3]);
    const float inter = std::max(x2 - x1, 0.0f) * std::max(y2 - y1, 0.0f);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / std::max(uni, 1e-12f) : 0.0f;
}

}  // namespace

PreoptTopKPostProcess::PreoptTopKPostProcess(int input_w, int input_h, float score_threshold,
                                             float nms_threshold, int top_k,
                                             int num_keypoints)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      top_k_(std::max(1, top_k)),
      num_keypoints_(std::max(0, num_keypoints)) {}

std::vector<PreoptResult> PreoptTopKPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    const dxrt::TensorPtr table = find_row_table(outputs);
    if (table->type() != dxrt::DataType::FLOAT) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] PreoptTopKPostProcess - expected a float32 row table, got " +
            format_shapes(outputs));
    }
    const auto& shape = table->shape();
    const int cols = static_cast<int>(shape[2]);
    const int min_cols = kColExtra + 3 * num_keypoints_;
    if (cols < min_cols) {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] PreoptTopKPostProcess - the row table has " << cols
            << " columns, expected at least " << min_cols << ". Got: " << format_shapes(outputs)
            << ".";
        throw std::runtime_error(msg.str());
    }
    // top_k bounds how many rows are considered, mirroring the model's own K.
    const int rows = std::min(static_cast<int>(shape[1]), top_k_);
    const float* data = static_cast<const float*>(table->data());
    // A fixed-size table pads with zero rows; never let a 0.0 threshold admit them.
    const float threshold = std::max(score_threshold_, std::numeric_limits<float>::min());

    std::vector<PreoptResult> kept;
    for (int r = 0; r < rows; ++r) {
        const float* row = data + static_cast<std::ptrdiff_t>(r) * cols;
        if (row[kColScore] < threshold) continue;
        PreoptResult item;
        item.box = {row[0], row[1], row[2], row[3]};
        item.confidence = row[kColScore];
        item.class_id = static_cast<int>(row[kColClass]);
        item.keypoints.assign(row + kColExtra, row + kColExtra + 3 * num_keypoints_);
        kept.push_back(std::move(item));
    }
    std::stable_sort(kept.begin(), kept.end(),
                     [](const PreoptResult& a, const PreoptResult& b) {
                         return a.confidence > b.confidence;
                     });
    if (nms_threshold_ >= 1.0f) return kept;

    std::vector<PreoptResult> merged;
    for (auto& r : kept) {
        bool suppressed = false;
        for (const auto& k : merged) {
            if (iou(r, k) >= nms_threshold_) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) merged.push_back(std::move(r));
    }
    return merged;
}
