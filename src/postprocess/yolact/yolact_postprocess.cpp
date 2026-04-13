#include "yolact_postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>

YOLACTPostProcess::YOLACTPostProcess(int input_w, int input_h,
                                       float score_threshold,
                                       float nms_threshold,
                                       int num_classes,
                                       bool has_background)
    : input_width_(input_w), input_height_(input_h),
      score_threshold_(score_threshold), nms_threshold_(nms_threshold),
      num_classes_(num_classes), has_background_(has_background) {}

YOLACTPostProcess::YOLACTPostProcess()
    : input_width_(550), input_height_(550),
      score_threshold_(0.3f), nms_threshold_(0.5f),
      num_classes_(80), has_background_(true) {}

static float compute_iou(float ax1, float ay1, float ax2, float ay2,
                          float bx1, float by1, float bx2, float by2) {
    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

// Containment-aware overlap: max(IoU, 0.65 * IoMin).
// Catches a small box fully inside a large box (low IoU but high containment).
static float compute_overlap(float ax1, float ay1, float ax2, float ay2,
                              float bx1, float by1, float bx2, float by2) {
    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);
    float iou = inter / (area_a + area_b - inter + 1e-6f);
    float min_area = std::min(area_a, area_b);
    float iomin = (min_area > 0) ? inter / (min_area + 1e-6f) : 0.0f;
    return std::max(iou, 0.65f * iomin);
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ------------------------------------------------------------------
// SSD anchor generation matching YOLACT's original ordering
// ------------------------------------------------------------------

void YOLACTPostProcess::generateAnchors(int target_n) {
    static const int strides[] = {8, 16, 32, 64, 128};
    static const float base_sizes[] = {24.0f, 48.0f, 96.0f, 192.0f, 384.0f};
    static const float r13 = std::pow(2.0f, 1.0f / 3.0f);
    static const float r23 = std::pow(2.0f, 2.0f / 3.0f);

    // RetinaNet-style scales: [base, base*2^(1/3), base*2^(2/3)]
    // Ordering: scale (outer) → aspect_ratio (inner)
    struct AnchorConfig {
        std::vector<std::vector<float>> scales;
        std::vector<float> ars;
    };
    std::vector<AnchorConfig> configs = {
        // 3 RetinaNet scales × 3 ARs = 9/cell (primary)
        {{{base_sizes[0], base_sizes[0]*r13, base_sizes[0]*r23},
          {base_sizes[1], base_sizes[1]*r13, base_sizes[1]*r23},
          {base_sizes[2], base_sizes[2]*r13, base_sizes[2]*r23},
          {base_sizes[3], base_sizes[3]*r13, base_sizes[3]*r23},
          {base_sizes[4], base_sizes[4]*r13, base_sizes[4]*r23}},
         {1.0f, 0.5f, 2.0f}},
        // 1 scale × 3 ARs = 3/cell (YOLACT default)
        {{{base_sizes[0]}, {base_sizes[1]}, {base_sizes[2]}, {base_sizes[3]}, {base_sizes[4]}},
         {1.0f, 0.5f, 2.0f}},
    };

    int total_cells = 0;
    for (int s : strides) {
        total_cells += ((input_height_ + s - 1) / s) * ((input_width_ + s - 1) / s);
    }

    for (auto& cfg : configs) {
        int per_cell = static_cast<int>(cfg.scales[0].size() * cfg.ars.size());
        int expected = total_cells * per_cell;
        if (target_n > 0 && expected != target_n) continue;

        anchors_.clear();
        for (int fpn = 0; fpn < 5; ++fpn) {
            int stride = strides[fpn];
            int conv_h = (input_height_ + stride - 1) / stride;
            int conv_w = (input_width_ + stride - 1) / stride;
            for (int r = 0; r < conv_h; ++r) {
                for (int c = 0; c < conv_w; ++c) {
                    float cx = (c + 0.5f) * stride / static_cast<float>(input_width_);
                    float cy = (r + 0.5f) * stride / static_cast<float>(input_height_);
                    // Scale (outer) → AR (inner)
                    for (float sc : cfg.scales[fpn]) {
                        for (float ar : cfg.ars) {
                            float w = sc * std::sqrt(ar) / input_width_;
                            float h = sc / std::sqrt(ar) / input_height_;
                            anchors_.push_back({cx, cy, w, h});
                        }
                    }
                }
            }
        }
        return;
    }
    // fallback: retry without target constraint
    if (target_n > 0) generateAnchors(0);
}

void YOLACTPostProcess::decodeBoxes(
    const float* loc_data, int N,
    std::vector<std::array<float, 4>>& decoded) const {

    decoded.resize(N);

    // YOLACT always uses SSD-encoded deltas relative to anchors.
    // Previous heuristic auto-detection was unreliable; always use
    // SSD delta decoding (matching the Python postprocessor).
    if (!anchors_.empty()) {
        const float var0 = 0.1f, var1 = 0.2f;
        for (int i = 0; i < N; ++i) {
            float tx = loc_data[i * 4 + 0];
            float ty = loc_data[i * 4 + 1];
            float tw = loc_data[i * 4 + 2];
            float th = loc_data[i * 4 + 3];

            float a_cx = anchors_[i][0], a_cy = anchors_[i][1];
            float a_w  = anchors_[i][2], a_h  = anchors_[i][3];

            float cx = tx * var0 * a_w + a_cx;
            float cy = ty * var0 * a_h + a_cy;
            float w  = a_w * std::exp(std::min(tw * var1, 10.0f));
            float h  = a_h * std::exp(std::min(th * var1, 10.0f));

            float x1 = (cx - w * 0.5f) * input_width_;
            float y1 = (cy - h * 0.5f) * input_height_;
            float x2 = (cx + w * 0.5f) * input_width_;
            float y2 = (cy + h * 0.5f) * input_height_;

            decoded[i] = {
                std::max(0.0f, std::min(x1, static_cast<float>(input_width_))),
                std::max(0.0f, std::min(y1, static_cast<float>(input_height_))),
                std::max(0.0f, std::min(x2, static_cast<float>(input_width_))),
                std::max(0.0f, std::min(y2, static_cast<float>(input_height_)))
            };
        }
    } else {
        // Fallback: no anchors — assume normalized [0,1] coordinates
        for (int i = 0; i < N; ++i) {
            float x1 = loc_data[i * 4 + 0];
            float y1 = loc_data[i * 4 + 1];
            float x2 = loc_data[i * 4 + 2];
            float y2 = loc_data[i * 4 + 3];

            if (x1 <= 1.0f && y1 <= 1.0f && x2 <= 1.0f && y2 <= 1.0f &&
                x1 >= 0.0f && y1 >= 0.0f) {
                x1 *= input_width_;  y1 *= input_height_;
                x2 *= input_width_;  y2 *= input_height_;
            }
            decoded[i] = {
                std::max(0.0f, std::min(x1, static_cast<float>(input_width_))),
                std::max(0.0f, std::min(y1, static_cast<float>(input_height_))),
                std::max(0.0f, std::min(x2, static_cast<float>(input_width_))),
                std::max(0.0f, std::min(y2, static_cast<float>(input_height_)))
            };
        }
    }
}

void YOLACTPostProcess::identifyTensors(
    const dxrt::TensorPtrs& outputs,
    const dxrt::TensorPtr*& loc,
    const dxrt::TensorPtr*& conf,
    const dxrt::TensorPtr*& mask_coeff,
    const dxrt::TensorPtr*& proto) {

    loc = conf = mask_coeff = proto = nullptr;

    for (auto& t : outputs) {
        auto shape = t->shape();
        // Proto: 3D spatial tensor [H, W, C] or 4D [1, H, W, C]
        if (shape.size() == 4 && shape[1] > 4 && shape[2] > 4) {
            proto = &t;
            continue;
        }
        if (shape.size() == 3 && shape[0] > 4 && shape[1] > 4) {
            proto = &t;
            continue;
        }

        // Get last two dims for 2D/3D classification
        int64_t last_dim = shape.back();
        if (last_dim == 4) {
            loc = &t;
        } else if (last_dim == 32) {
            mask_coeff = &t;
        } else if (last_dim > 4 && last_dim != 32) {
            conf = &t;
        }
    }

    // Fallback: sort remaining 2D tensors by last dim
    if (!loc || !conf || !mask_coeff) {
        struct TInfo { const dxrt::TensorPtr* ptr; int64_t last_dim; };
        std::vector<TInfo> tensors_2d;
        for (auto& t : outputs) {
            if (&t == proto) continue;
            auto shape = t->shape();
            tensors_2d.push_back({&t, shape.back()});
        }
        std::sort(tensors_2d.begin(), tensors_2d.end(),
                  [](const TInfo& a, const TInfo& b) { return a.last_dim < b.last_dim; });
        if (tensors_2d.size() >= 3) {
            loc = tensors_2d[0].ptr;        // smallest last dim (4)
            mask_coeff = tensors_2d[1].ptr;  // medium (32)
            conf = tensors_2d[2].ptr;        // largest (num_classes)
        }
    }
}

std::vector<YOLACTResult> YOLACTPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    std::vector<YOLACTResult> results;
    if (outputs.size() < 4) return results;

    const dxrt::TensorPtr* loc_t = nullptr;
    const dxrt::TensorPtr* conf_t = nullptr;
    const dxrt::TensorPtr* mask_coeff_t = nullptr;
    const dxrt::TensorPtr* proto_t = nullptr;
    identifyTensors(outputs, loc_t, conf_t, mask_coeff_t, proto_t);

    if (!loc_t || !conf_t || !mask_coeff_t || !proto_t) return results;

    // Get dimensions
    auto conf_shape = (*conf_t)->shape();
    int N = static_cast<int>(conf_shape.size() == 3 ? conf_shape[1] : conf_shape[0]);
    int C = static_cast<int>(conf_shape.back());

    auto proto_shape = (*proto_t)->shape();
    int proto_h, proto_w, proto_c;
    if (proto_shape.size() == 4) {
        proto_h = static_cast<int>(proto_shape[1]);
        proto_w = static_cast<int>(proto_shape[2]);
        proto_c = static_cast<int>(proto_shape[3]);
    } else {
        proto_h = static_cast<int>(proto_shape[0]);
        proto_w = static_cast<int>(proto_shape[1]);
        proto_c = static_cast<int>(proto_shape[2]);
    }

    const float* loc_data = static_cast<const float*>((*loc_t)->data());
    const float* conf_data = static_cast<const float*>((*conf_t)->data());
    const float* coeff_data = static_cast<const float*>((*mask_coeff_t)->data());
    const float* proto_data = static_cast<const float*>((*proto_t)->data());

    // Generate SSD anchors (once) and decode boxes
    if (anchors_.empty() || static_cast<int>(anchors_.size()) != N) {
        generateAnchors(N);
    }
    std::vector<std::array<float, 4>> decoded_boxes;
    decodeBoxes(loc_data, N, decoded_boxes);

    // Collect detections: best class per anchor (argmax), matching Python postprocessor.
    // Previous code iterated all classes per anchor, causing multiple bounding boxes
    // per object when several classes exceeded the score threshold.
    std::vector<Detection> detections;
    int start_cls = has_background_ ? 1 : 0;

    for (int i = 0; i < N; ++i) {
        // Find best foreground class for this anchor
        float best_score = 0.0f;
        int best_cls = -1;
        for (int c = start_cls; c < C; ++c) {
            float score = conf_data[i * C + c];
            if (score > best_score) {
                best_score = score;
                best_cls = c;
            }
        }
        if (best_score < score_threshold_ || best_cls < 0) continue;

        float x1 = decoded_boxes[i][0];
        float y1 = decoded_boxes[i][1];
        float x2 = decoded_boxes[i][2];
        float y2 = decoded_boxes[i][3];

        if (x2 <= x1 || y2 <= y1) continue;

        // Reject near-full-image boxes (matching Python postprocessor threshold)
        float bw = x2 - x1, bh = y2 - y1;
        float img_area = static_cast<float>(input_width_ * input_height_);
        if ((bw * bh) / img_area > 0.65f) continue;

        int cls_id = best_cls - (has_background_ ? 1 : 0);
        detections.push_back({x1, y1, x2, y2, best_score, cls_id, i});
    }

    if (detections.empty()) return results;

    // NMS (per-class) + cross-class containment suppression
    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return detections[a].score > detections[b].score; });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<int> kept;
    for (int idx : order) {
        if (suppressed[idx]) continue;
        kept.push_back(idx);
        const auto& d = detections[idx];
        for (size_t j = 0; j < order.size(); ++j) {
            int jj = order[j];
            if (suppressed[jj] || jj == idx) continue;
            const auto& dj = detections[jj];
            // Containment-aware overlap: catches small box inside large box
            float overlap = compute_overlap(d.x1, d.y1, d.x2, d.y2,
                                             dj.x1, dj.y1, dj.x2, dj.y2);
            if (overlap > nms_threshold_) {
                suppressed[jj] = true;
            }
        }
    }

    // Generate masks for kept detections
    for (int k : kept) {
        const auto& d = detections[k];

        // Compute mask = sigmoid(coeff @ proto.T) for [proto_h, proto_w]
        std::vector<float> mask(proto_h * proto_w, 0.0f);
        const float* coeff = coeff_data + d.idx * proto_c;
        for (int ph = 0; ph < proto_h; ++ph) {
            for (int pw = 0; pw < proto_w; ++pw) {
                float val = 0.0f;
                for (int pc = 0; pc < proto_c; ++pc) {
                    val += coeff[pc] * proto_data[(ph * proto_w + pw) * proto_c + pc];
                }
                mask[ph * proto_w + pw] = sigmoid(val);
            }
        }

        // Crop mask to detection bounding box (zero out outside region)
        int bx1 = std::max(0, static_cast<int>(d.x1 * proto_w / input_width_));
        int by1 = std::max(0, static_cast<int>(d.y1 * proto_h / input_height_));
        int bx2 = std::min(proto_w, static_cast<int>(std::ceil(d.x2 * proto_w / input_width_)));
        int by2 = std::min(proto_h, static_cast<int>(std::ceil(d.y2 * proto_h / input_height_)));
        for (int ph = 0; ph < proto_h; ++ph) {
            for (int pw = 0; pw < proto_w; ++pw) {
                if (ph < by1 || ph >= by2 || pw < bx1 || pw >= bx2) {
                    mask[ph * proto_w + pw] = 0.0f;
                }
            }
        }

        YOLACTResult r;
        r.box = {d.x1, d.y1, d.x2, d.y2};
        r.confidence = d.score;
        r.class_id = d.class_id;
        r.mask = std::move(mask);
        r.mask_height = proto_h;
        r.mask_width = proto_w;
        results.push_back(std::move(r));
    }

    return results;
}
