#include "efficientdet_postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>

EfficientDetPostProcess::EfficientDetPostProcess(int input_w, int input_h,
                                                   float score_threshold,
                                                   float nms_threshold,
                                                   int num_classes,
                                                   bool has_background)
    : input_width_(input_w), input_height_(input_h),
      score_threshold_(score_threshold), nms_threshold_(nms_threshold),
      num_classes_(num_classes), has_background_(has_background) {}

EfficientDetPostProcess::EfficientDetPostProcess()
    : input_width_(512), input_height_(512),
      score_threshold_(0.3f), nms_threshold_(0.45f),
      num_classes_(90), has_background_(true) {}

static float iou(const std::vector<float>& a, const std::vector<float>& b) {
    float ix1 = std::max(a[0], b[0]);
    float iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]);
    float iy2 = std::min(a[3], b[3]);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<int> nms(const std::vector<EfficientDetResult>& dets, float threshold) {
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return dets[a].confidence > dets[b].confidence; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int> kept;
    for (int i : order) {
        if (suppressed[i]) continue;
        kept.push_back(i);
        for (size_t j = 0; j < order.size(); ++j) {
            int jj = order[j];
            if (suppressed[jj] || jj == i) continue;
            if (iou(dets[i].box, dets[jj].box) > threshold) {
                suppressed[jj] = true;
            }
        }
    }
    return kept;
}

std::vector<EfficientDetResult> EfficientDetPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    // Multi-output (BiFPN features + decoded tensors): filter to 2D tensors only
    if (outputs.size() > 4) {
        dxrt::TensorPtrs filtered;
        for (auto& t : outputs) {
            if (t->shape().size() == 2) {
                filtered.push_back(t);
            }
        }
        if (filtered.size() >= 2) {
            return process2Tensor(filtered);
        }
    }
    if (outputs.size() == 4) {
        return processTFFormat(outputs);
    }
    if (outputs.size() >= 2) {
        return process2Tensor(outputs);
    }
    return {};
}

std::vector<EfficientDetResult> EfficientDetPostProcess::processTFFormat(
    const dxrt::TensorPtrs& outputs) {
    std::vector<EfficientDetResult> results;

    // Identify tensors by last dimension
    const dxrt::TensorPtr* boxes_t = nullptr;
    const dxrt::TensorPtr* num_det_t = nullptr;
    std::vector<const dxrt::TensorPtr*> others;

    for (auto& t : outputs) {
        auto shape = t->shape();
        int64_t last_dim = shape.back();
        if (last_dim == 4 && shape.size() >= 2) {
            boxes_t = &t;
        } else if (shape.size() <= 2 && last_dim == 1) {
            num_det_t = &t;
        } else {
            others.push_back(&t);
        }
    }
    if (!boxes_t) return results;

    const float* boxes = static_cast<const float*>((*boxes_t)->data());
    auto boxes_shape = (*boxes_t)->shape();
    int N = static_cast<int>(boxes_shape.size() >= 2 ? boxes_shape[boxes_shape.size() - 2] : boxes_shape[0]);

    int num_det = N;
    if (num_det_t) {
        num_det = std::min(N, static_cast<int>(*static_cast<const float*>((*num_det_t)->data())));
    }

    // Remaining tensors: scores and classes
    // TF format outputs both as [1, N] floats — distinguish by value range:
    //   scores: probabilities in [0, 1]
    //   classes: integer IDs (can exceed 1.0)
    const float* scores = nullptr;
    const float* classes = nullptr;
    for (auto* ptr : others) {
        const float* data = static_cast<const float*>((*ptr)->data());
        auto shape = (*ptr)->shape();
        int len = static_cast<int>(shape.size() >= 2 ? shape[shape.size() - 2] : shape[0]);
        // Check if values exceed 1.0 (class IDs) or stay in [0,1] (scores)
        float max_val = 0.0f;
        for (int i = 0; i < std::min(len, num_det); ++i) {
            if (data[i] > max_val) max_val = data[i];
        }
        if (max_val > 1.0f) {
            classes = data;
        } else {
            scores = data;
        }
    }
    if (!scores) return results;

    for (int i = 0; i < num_det; ++i) {
        float score = scores[i];
        if (score < score_threshold_) continue;

        // TF format: [ymin, xmin, ymax, xmax] normalized
        float ymin = boxes[i * 4 + 0];
        float xmin = boxes[i * 4 + 1];
        float ymax = boxes[i * 4 + 2];
        float xmax = boxes[i * 4 + 3];

        float x1 = xmin * input_width_;
        float y1 = ymin * input_height_;
        float x2 = xmax * input_width_;
        float y2 = ymax * input_height_;

        int cls_id = classes ? static_cast<int>(classes[i]) - (has_background_ ? 1 : 0) : 0;

        results.emplace_back(
            std::vector<float>{x1, y1, x2, y2},
            score, cls_id);
    }

    auto kept = nms(results, nms_threshold_);
    std::vector<EfficientDetResult> final_results;
    for (int k : kept) {
        final_results.push_back(std::move(results[k]));
    }
    return final_results;
}

std::vector<EfficientDetResult> EfficientDetPostProcess::process2Tensor(
    const dxrt::TensorPtrs& outputs) {
    std::vector<EfficientDetResult> results;

    // Identify boxes (last dim=4) and scores tensor
    const dxrt::TensorPtr* boxes_t = nullptr;
    const dxrt::TensorPtr* scores_t = nullptr;

    for (auto& t : outputs) {
        auto shape = t->shape();
        int64_t last_dim = shape.back();
        if (last_dim == 4 && shape.size() >= 2) {
            boxes_t = &t;
        } else {
            scores_t = &t;
        }
    }
    if (!boxes_t || !scores_t) return results;

    auto boxes_shape = (*boxes_t)->shape();
    auto scores_shape = (*scores_t)->shape();

    int N = static_cast<int>(boxes_shape.size() >= 2 ? boxes_shape[boxes_shape.size() - 2] : boxes_shape[0]);
    const float* boxes = static_cast<const float*>((*boxes_t)->data());
    const float* scores_data = static_cast<const float*>((*scores_t)->data());

    int num_score_cols = 1;
    if (scores_shape.size() >= 2) {
        num_score_cols = static_cast<int>(scores_shape.back());
    }

    for (int i = 0; i < N; ++i) {
        float best_score = 0.0f;
        int best_cls = 0;

        if (num_score_cols > 1) {
            int start = has_background_ ? 1 : 0;
            for (int c = start; c < num_score_cols; ++c) {
                float s = scores_data[i * num_score_cols + c];
                if (s > best_score) {
                    best_score = s;
                    best_cls = c - (has_background_ ? 1 : 0);
                }
            }
        } else {
            best_score = scores_data[i];
        }

        if (best_score < score_threshold_) continue;

        // Boxes may be [ymin,xmin,ymax,xmax] normalized or [x1,y1,x2,y2] pixel
        float b0 = boxes[i * 4 + 0];
        float b1 = boxes[i * 4 + 1];
        float b2 = boxes[i * 4 + 2];
        float b3 = boxes[i * 4 + 3];

        float x1, y1, x2, y2;
        // If all values are in [0,1], assume normalized [ymin,xmin,ymax,xmax]
        if (b0 <= 1.0f && b1 <= 1.0f && b2 <= 1.0f && b3 <= 1.0f &&
            b0 >= 0.0f && b1 >= 0.0f) {
            x1 = b1 * input_width_;
            y1 = b0 * input_height_;
            x2 = b3 * input_width_;
            y2 = b2 * input_height_;
        } else {
            x1 = b0;
            y1 = b1;
            x2 = b2;
            y2 = b3;
        }

        results.emplace_back(
            std::vector<float>{x1, y1, x2, y2},
            best_score, best_cls);
    }

    auto kept = nms(results, nms_threshold_);
    std::vector<EfficientDetResult> final_results;
    for (int k : kept) {
        final_results.push_back(std::move(results[k]));
    }
    return final_results;
}
