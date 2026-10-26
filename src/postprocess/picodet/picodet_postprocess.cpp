#include "picodet_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

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

const float* float_data(const dxrt::TensorPtr& tensor, const dxrt::TensorPtrs& all) {
    if (tensor->type() != dxrt::DataType::FLOAT) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] PicoDetPostProcess - expected float32 outputs, got " +
            format_shapes(all));
    }
    return static_cast<const float*>(tensor->data());
}

// IoU of the two boxes clipped to the input frame: the Python postprocessor
// runs its NMS on boxes already clipped to the image, and the input frame is
// the closest stand-in for it that needs no preprocessing context.
float clipped_iou(const PicoDetResult& a, const PicoDetResult& b, float max_x, float max_y) {
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

}  // namespace

PicoDetPostProcess::PicoDetPostProcess(int input_w, int input_h, float score_threshold,
                                       float nms_threshold, int num_classes)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      num_classes_(std::max(1, num_classes)) {}

std::vector<PicoDetResult> PicoDetPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    // Pair the per-level tensors by anchor count: a box tensor has 4*(reg_max+1)
    // channels, anything else of shape (1, N, C) is a score tensor.
    std::map<int64_t, dxrt::TensorPtr> scores, boxes;
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() != 3 || s[0] != 1) continue;
        if (s[2] % 4 == 0 && s[2] >= 8) boxes[s[1]] = o;
        else scores[s[1]] = o;
    }

    // Largest grid first, as the Python postprocessor concatenates the levels.
    std::vector<int64_t> anchor_counts;
    for (auto it = scores.rbegin(); it != scores.rend(); ++it) {
        if (boxes.count(it->first)) anchor_counts.push_back(it->first);
    }

    std::vector<PicoDetResult> all;
    int levels = 0;
    for (int64_t anchors : anchor_counts) {
        const int grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(anchors))));
        if (static_cast<int64_t>(grid) * grid != anchors) continue;  // not a pyramid level
        ++levels;
        const int stride = input_width_ / grid;
        const dxrt::TensorPtr& score_t = scores[anchors];
        const dxrt::TensorPtr& box_t = boxes[anchors];
        const int classes = static_cast<int>(score_t->shape()[2]);
        const int bins = static_cast<int>(box_t->shape()[2]) / 4;  // reg_max + 1
        const float* score_data = float_data(score_t, outputs);
        const float* box_data = float_data(box_t, outputs);

        for (int64_t a = 0; a < anchors; ++a) {
            const float* sc = score_data + static_cast<std::ptrdiff_t>(a) * classes;
            int best_class = 0;
            float best = sc[0];
            for (int c = 1; c < classes; ++c) {
                if (sc[c] > best) {
                    best = sc[c];
                    best_class = c;
                }
            }
            if (best < score_threshold_) continue;

            // DFL: softmax over the bins of each side, then the expected bin index.
            const float* dist = box_data + static_cast<std::ptrdiff_t>(a) * 4 * bins;
            float distance[4];
            for (int side = 0; side < 4; ++side) {
                const float* d = dist + side * bins;
                float peak = d[0];
                for (int k = 1; k < bins; ++k) peak = std::max(peak, d[k]);
                float sum = 0.0f, expect = 0.0f;
                for (int k = 0; k < bins; ++k) {
                    const float e = std::exp(d[k] - peak);
                    sum += e;
                    expect += e * static_cast<float>(k);
                }
                distance[side] = expect / std::max(sum, 1e-12f) * static_cast<float>(stride);
            }
            const float cx = (static_cast<float>(a % grid) + 0.5f) * static_cast<float>(stride);
            const float cy = (static_cast<float>(a / grid) + 0.5f) * static_cast<float>(stride);
            all.emplace_back(std::vector<float>{cx - distance[0], cy - distance[1],
                                                cx + distance[2], cy + distance[3]},
                             best, best_class);
        }
    }
    if (levels == 0) {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] PicoDetPostProcess - no matching per-level score and box "
               "tensors found.\n  Got: " << format_shapes(outputs) << "\n"
            << "  Expected one (1, N, num_classes) and one (1, N, 4*(reg_max+1)) per pyramid "
               "level, with N a square anchor count. A SINGLE concatenated "
               "(1, N, num_classes + 4*(reg_max+1)) tensor is NanoDet's packaging -- use "
               "NanoDetPostProcess for that.";
        throw std::runtime_error(msg.str());
    }

    std::stable_sort(all.begin(), all.end(),
                     [](const PicoDetResult& a, const PicoDetResult& b) {
                         return a.confidence > b.confidence;
                     });
    if (nms_threshold_ >= 1.0f) return all;

    const float max_x = static_cast<float>(input_width_ - 1);
    const float max_y = static_cast<float>(input_height_ - 1);
    std::vector<PicoDetResult> kept;
    for (auto& r : all) {
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
