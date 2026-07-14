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
    // Multi-output (BiFPN features + anchor regressions + class scores):
    // locate the box-regression tensor (last dim == 4) and the class-score
    // tensor (last dim == num_classes_) and run anchor-based decoding.
    if (outputs.size() > 4) {
        const dxrt::TensorPtr* boxes_t = nullptr;
        const dxrt::TensorPtr* scores_t = nullptr;
        for (auto& t : outputs) {
            const auto& shp = t->shape();
            if (shp.empty()) continue;
            const int64_t last = shp.back();
            if (last == 4 && boxes_t == nullptr) {
                boxes_t = &t;
            } else if (last == num_classes_ && scores_t == nullptr) {
                scores_t = &t;
            }
        }
        if (boxes_t != nullptr && scores_t != nullptr) {
            return processMultiOutputAnchors(*boxes_t, *scores_t);
        }
        // Fall back to 2D filtering for already-decoded multi-output models.
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

// Generate EfficientDet anchors for pyramid levels P3-P7. Ordering matches the
// golden Python implementation: level -> y -> x -> scale -> ratio.
void EfficientDetPostProcess::generate_anchors() {
    if (!anchors_.empty()) return;

    const int image_size = std::max(input_width_, input_height_);
    const int strides[5] = {8, 16, 32, 64, 128};
    const int anchor_sizes[5] = {32, 64, 128, 256, 512};
    const float scales[3] = {1.0f,
                             std::pow(2.0f, 1.0f / 3.0f),
                             std::pow(2.0f, 2.0f / 3.0f)};
    const float ratios[3][2] = {{1.0f, 1.0f}, {1.4f, 0.7f}, {0.7f, 1.4f}};

    for (int level = 0; level < 5; ++level) {
        const int stride = strides[level];
        const int feat = image_size / stride;
        const float base = static_cast<float>(anchor_sizes[level]);
        for (int y = 0; y < feat; ++y) {
            for (int x = 0; x < feat; ++x) {
                const float cx = (x + 0.5f) * stride;
                const float cy = (y + 0.5f) * stride;
                for (int si = 0; si < 3; ++si) {
                    for (int ri = 0; ri < 3; ++ri) {
                        const float w = base * scales[si] * ratios[ri][0];
                        const float h = base * scales[si] * ratios[ri][1];
                        anchors_.push_back({cx, cy, w, h});
                    }
                }
            }
        }
    }
}

std::vector<EfficientDetResult> EfficientDetPostProcess::processMultiOutputAnchors(
    const dxrt::TensorPtr& boxes_t, const dxrt::TensorPtr& scores_t) {
    generate_anchors();

    const auto& box_shape = boxes_t->shape();
    const auto& score_shape = scores_t->shape();
    const int num_anchors = static_cast<int>(box_shape[box_shape.size() - 2]);
    const int score_cols = static_cast<int>(score_shape.back());
    const float* box_data = static_cast<const float*>(boxes_t->data());
    const float* score_data = static_cast<const float*>(scores_t->data());

    const int n = std::min(num_anchors, static_cast<int>(anchors_.size()));
    // Foreground classes start at index 1 when a background class is present.
    const int cls_start = (has_background_ && score_cols > 1) ? 1 : 0;

    // Collect candidates above the score threshold (mirrors the golden order:
    // score-extract -> threshold -> top-K limit -> decode -> NMS).
    struct Cand {
        int anchor_idx;
        float score;
        int class_id;
    };
    std::vector<Cand> cands;
    for (int i = 0; i < n; ++i) {
        const float* sc = score_data + static_cast<size_t>(i) * score_cols;
        int best_cls = 0;
        float best = sc[cls_start];
        for (int c = cls_start + 1; c < score_cols; ++c) {
            if (sc[c] > best) {
                best = sc[c];
                best_cls = c - cls_start;
            }
        }
        if (best >= score_threshold_) {
            cands.push_back({i, best, best_cls});
        }
    }

    // Keep highest-scoring candidates (top max_nms_candidates_) before decode.
    if (max_nms_candidates_ > 0 &&
        static_cast<int>(cands.size()) > max_nms_candidates_) {
        std::partial_sort(
            cands.begin(), cands.begin() + max_nms_candidates_, cands.end(),
            [](const Cand& a, const Cand& b) { return a.score > b.score; });
        cands.resize(max_nms_candidates_);
    }

    const float image_size = static_cast<float>(std::max(input_width_, input_height_));
    std::vector<EfficientDetResult> dets;
    dets.reserve(cands.size());
    for (const auto& cd : cands) {
        const float* b = box_data + static_cast<size_t>(cd.anchor_idx) * 4;
        const auto& a = anchors_[cd.anchor_idx];
        // Regression format [dy, dx, dh, dw] relative to anchor [cx, cy, w, h].
        const float dy = b[0];
        const float dx = b[1];
        const float dh = std::max(-10.0f, std::min(10.0f, b[2]));
        const float dw = std::max(-10.0f, std::min(10.0f, b[3]));
        const float pcx = a[0] + dx * a[2];
        const float pcy = a[1] + dy * a[3];
        const float pw = a[2] * std::exp(dw);
        const float ph = a[3] * std::exp(dh);
        float x1 = std::max(0.0f, std::min(image_size, pcx - pw / 2.0f));
        float y1 = std::max(0.0f, std::min(image_size, pcy - ph / 2.0f));
        float x2 = std::max(0.0f, std::min(image_size, pcx + pw / 2.0f));
        float y2 = std::max(0.0f, std::min(image_size, pcy + ph / 2.0f));
        dets.emplace_back(std::vector<float>{x1, y1, x2, y2}, cd.score, cd.class_id);
    }

    // Global NMS (matches cv2.dnn.NMSBoxes over all classes).
    std::vector<int> keep = nms(dets, nms_threshold_);
    std::vector<EfficientDetResult> results;
    results.reserve(keep.size());
    for (int idx : keep) {
        results.push_back(dets[idx]);
    }
    return results;
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
