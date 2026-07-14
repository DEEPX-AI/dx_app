#include "mediapipe_hand_postprocess.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct ParsedTensor {
    size_t count;
    std::vector<float> values;

    ParsedTensor() : count(0), values() {}
};

struct BoxCandidate {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

std::vector<int64_t> squeeze_shape(const std::vector<int64_t>& shape) {
    std::vector<int64_t> squeezed;
    squeezed.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != 1) {
            squeezed.push_back(shape[i]);
        }
    }
    if (squeezed.empty()) {
        squeezed.push_back(1);
    }
    return squeezed;
}

float sigmoid_clamped(float x) {
    if (x < -100.0f) x = -100.0f;
    else if (x > 100.0f) x = 100.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

float clip01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

float box_iou(const BoxCandidate& lhs, const BoxCandidate& rhs) {
    const float inter_x1 = std::max(lhs.x1, rhs.x1);
    const float inter_y1 = std::max(lhs.y1, rhs.y1);
    const float inter_x2 = std::min(lhs.x2, rhs.x2);
    const float inter_y2 = std::min(lhs.y2, rhs.y2);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    const float lhs_area = std::max(0.0f, lhs.x2 - lhs.x1) * std::max(0.0f, lhs.y2 - lhs.y1);
    const float rhs_area = std::max(0.0f, rhs.x2 - rhs.x1) * std::max(0.0f, rhs.y2 - rhs.y1);
    const float union_area = lhs_area + rhs_area - inter_area;
    return union_area > 0.0f ? (inter_area / union_area) : 0.0f;
}

}  // namespace

MediaPipeHandPostProcess::MediaPipeHandPostProcess(int input_size, float score_threshold,
                                                   float nms_threshold)
    : input_size_(input_size),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      box_scale_(2.0f),
      box_shift_(0.3f) {
    generate_anchors();
}

MediaPipeHandPostProcess::MediaPipeHandPostProcess()
    : MediaPipeHandPostProcess(192, 0.5f, 0.3f) {}

void MediaPipeHandPostProcess::generate_anchors() {
    static const int kStrides[4] = {8, 16, 16, 16};
    static const int kNumLayers = 4;
    static const int kAnchorsPerCell = 2;
    static const float kAnchorOffset = 0.5f;

    anchors_.clear();
    anchors_.reserve(2016);

    int layer_id = 0;
    while (layer_id < kNumLayers) {
        int last_same = layer_id;
        int repeats = 0;
        while (last_same < kNumLayers && kStrides[last_same] == kStrides[layer_id]) {
            repeats += kAnchorsPerCell;
            ++last_same;
        }

        const int stride = kStrides[layer_id];
        const int feature_map = (input_size_ + stride - 1) / stride;
        for (int y = 0; y < feature_map; ++y) {
            for (int x = 0; x < feature_map; ++x) {
                const float cx = (static_cast<float>(x) + kAnchorOffset) /
                                 static_cast<float>(feature_map);
                const float cy = (static_cast<float>(y) + kAnchorOffset) /
                                 static_cast<float>(feature_map);
                for (int r = 0; r < repeats; ++r) {
                    anchors_.push_back(std::make_pair(cx, cy));
                }
            }
        }
        layer_id = last_same;
    }
}

std::vector<MediaPipeHandPostProcess::RawDet> MediaPipeHandPostProcess::decode_tensors(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<float> reg_flat;
    std::vector<float> score_flat;
    bool has_flat_reg = false;
    bool has_flat_score = false;

    for (size_t i = 0; i < outputs.size(); ++i) {
        const dxrt::TensorPtr& tensor = outputs[i];
        const float* data = static_cast<const float*>(tensor->data());
        const std::vector<int64_t> shape = squeeze_shape(tensor->shape());

        if (shape.size() == 2) {
            const int64_t rows = shape[0];
            const int64_t cols = shape[1];
            if (cols == 18 && !has_flat_reg) {
                reg_flat.assign(data, data + rows * cols);
                has_flat_reg = true;
            } else if (cols == 1 && !has_flat_score) {
                score_flat.assign(data, data + rows);
                has_flat_score = true;
            }
        } else if (shape.size() == 1 && !has_flat_score) {
            score_flat.assign(data, data + shape[0]);
            has_flat_score = true;
        }
    }

    if (!has_flat_reg || !has_flat_score) {
        reg_flat.clear();
        score_flat.clear();
        std::vector<ParsedTensor> regs;
        std::vector<ParsedTensor> clss;

        for (size_t i = 0; i < outputs.size(); ++i) {
            const dxrt::TensorPtr& tensor = outputs[i];
            const float* data = static_cast<const float*>(tensor->data());
            const std::vector<int64_t> shape = squeeze_shape(tensor->shape());

            if (shape.size() != 3) {
                continue;
            }

            const int64_t h = shape[0];
            const int64_t w = shape[1];
            const int64_t c = shape[2];
            if (c % 18 == 0) {
                ParsedTensor parsed;
                parsed.count = static_cast<size_t>(h * w * (c / 18));
                parsed.values.assign(data, data + parsed.count * 18);
                regs.push_back(parsed);
            } else if (c <= 8) {
                ParsedTensor parsed;
                parsed.count = static_cast<size_t>(h * w * c);
                parsed.values.assign(data, data + parsed.count);
                clss.push_back(parsed);
            }
        }

        if (regs.empty() || clss.empty()) {
            return std::vector<RawDet>();
        }

        std::sort(regs.begin(), regs.end(),
                  [](const ParsedTensor& lhs, const ParsedTensor& rhs) {
                      return lhs.count > rhs.count;
                  });
        std::sort(clss.begin(), clss.end(),
                  [](const ParsedTensor& lhs, const ParsedTensor& rhs) {
                      return lhs.count > rhs.count;
                  });

        size_t reg_total = 0;
        for (size_t i = 0; i < regs.size(); ++i) {
            reg_total += regs[i].values.size();
        }
        reg_flat.reserve(reg_total);
        for (size_t i = 0; i < regs.size(); ++i) {
            reg_flat.insert(reg_flat.end(), regs[i].values.begin(), regs[i].values.end());
        }

        size_t score_total = 0;
        for (size_t i = 0; i < clss.size(); ++i) {
            score_total += clss[i].values.size();
        }
        score_flat.reserve(score_total);
        for (size_t i = 0; i < clss.size(); ++i) {
            score_flat.insert(score_flat.end(), clss[i].values.begin(), clss[i].values.end());
        }
    }

    const size_t reg_count = reg_flat.size() / 18;
    const size_t score_count = score_flat.size();
    size_t num = anchors_.size();
    if (reg_count < num) num = reg_count;
    if (score_count < num) num = score_count;

    std::vector<RawDet> decoded;
    decoded.reserve(num);
    const float scale = static_cast<float>(input_size_);

    for (size_t i = 0; i < num; ++i) {
        const float* reg = &reg_flat[i * 18];
        const float score = sigmoid_clamped(score_flat[i]);
        if (score <= score_threshold_) {
            continue;
        }

        const float anchor_cx = anchors_[i].first;
        const float anchor_cy = anchors_[i].second;
        const float cx = reg[0] / scale + anchor_cx;
        const float cy = reg[1] / scale + anchor_cy;
        const float w = std::fabs(reg[2]) / scale;
        const float h = std::fabs(reg[3]) / scale;
        const float s = std::max(w, h);
        const float kx0 = reg[4] / scale + anchor_cx;
        const float ky0 = reg[5] / scale + anchor_cy;
        const float kx2 = reg[8] / scale + anchor_cx;
        const float ky2 = reg[9] / scale + anchor_cy;

        decoded.push_back(RawDet{cx, cy, s, kx0, ky0, kx2, ky2, score});
    }

    return decoded;
}

std::vector<MediaPipeHandDetection> MediaPipeHandPostProcess::apply_nms(
    std::vector<RawDet>& candidates) const {
    if (candidates.empty()) {
        return std::vector<MediaPipeHandDetection>();
    }

    std::vector<BoxCandidate> boxes;
    boxes.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        const RawDet& det = candidates[i];
        float dx = det.kx2 - det.kx0;
        float dy = det.ky2 - det.ky0;
        const float norm = std::sqrt(dx * dx + dy * dy) + 1e-9f;
        dx /= norm;
        dy /= norm;

        const float side = det.s * box_scale_;
        const float ccx = det.cx + box_shift_ * side * dx;
        const float ccy = det.cy + box_shift_ * side * dy;
        boxes.push_back(BoxCandidate{ccx - side * 0.5f, ccy - side * 0.5f, ccx + side * 0.5f,
                                     ccy + side * 0.5f, det.score});
    }

    std::vector<size_t> order(boxes.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&boxes](size_t lhs, size_t rhs) { return boxes[lhs].score > boxes[rhs].score; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<MediaPipeHandDetection> kept;
    kept.reserve(boxes.size());

    for (size_t oi = 0; oi < order.size(); ++oi) {
        const size_t idx = order[oi];
        if (suppressed[idx]) {
            continue;
        }

        const BoxCandidate& box = boxes[idx];
        kept.push_back(MediaPipeHandDetection(clip01(box.x1), clip01(box.y1), clip01(box.x2),
                                              clip01(box.y2), box.score));

        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const size_t other_idx = order[oj];
            if (suppressed[other_idx]) {
                continue;
            }
            if (box_iou(box, boxes[other_idx]) > nms_threshold_) {
                suppressed[other_idx] = true;
            }
        }
    }

    return kept;
}

std::vector<MediaPipeHandDetection> MediaPipeHandPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    std::vector<RawDet> candidates = decode_tensors(outputs);
    return apply_nms(candidates);
}
