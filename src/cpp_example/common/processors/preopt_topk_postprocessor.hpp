/**
 * @file preopt_topk_postprocessor.hpp
 * @brief Postprocessors for the pre-optimized YOLO26 models (top-k selected in the model)
 *
 * A pre-optimized .dxnn carries the whole tail of the YOLO26 end-to-end head:
 * the DFL integration runs on the NPU and a CPU task (executed by DXRT through
 * ONNX Runtime) applies sigmoid to the class logits, keeps the top-k (300)
 * anchors by their best class score, keeps the top-k (anchor, class) pairs of
 * those, decodes boxes and keypoints and gathers the mask coefficients. What the
 * application receives is a fixed-size table of K rows in input-pixel
 * (letterboxed) coordinates:
 *
 *   preopt_output [1, K, C]      C = 6 + extra
 *     [x1, y1, x2, y2, score, class_id, extra...]
 *     extra: detection    -> none                                     (C = 6)
 *            pose         -> 17 x (x, y, visibility)                  (C = 57)
 *            segmentation -> 32 mask coefficients                     (C = 38)
 *                            + prototypes output1 [1, 32, 160, 160]
 *
 * Rows are sorted by score, descending. The second top-k is taken over the
 * flattened (anchor x class) scores - as in Ultralytics' end-to-end
 * postprocess - so one anchor may appear several times with different classes
 * (e.g. the same box as "car" and "truck"). The model itself is NMS-free; the
 * optional class-agnostic NMS below only merges those duplicates.
 *
 * When DXRT runs without ONNX Runtime the CPU task is skipped and the
 * application receives the NPU tensors instead, NHWC, one set per stride
 * (8 / 16 / 32 -> 80x80 / 40x40 / 20x20 at 640x640):
 *
 *   preopt_bbox_tr_i [1, H, W, 4]    ltrb distances in grid units (DFL applied)
 *   preopt_cls_tr_i  [1, H, W, nc]   class logits (before sigmoid)
 *   preopt_kpt_tr_i  [1, H, W, 51]   17 x (dx, dy in grid units, visibility logit)
 *   preopt_mask_tr_i [1, H, W, 32]   mask coefficients
 *
 * PreoptTopKDecoder then performs the same selection on the CPU:
 *   box   = (cx - l, cy - t, cx + r, cy + b) * stride,  cx = gx + 0.5, cy = gy + 0.5
 *   kpt   = (dx + cx, dy + cy) * stride, visibility = sigmoid(v)
 *   score = sigmoid(logit)
 * Both paths yield the same rows (checked bit-exact against the model's own
 * preopt_output), so the applications behave the same with and without ORT.
 */

#ifndef PREOPT_TOPK_POSTPROCESSOR_HPP
#define PREOPT_TOPK_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/processors/postprocess_utils.hpp"
#include "common/processors/result_converters.hpp"
#include "common/processors/roi_instance_mask.hpp"
#include "common_util.hpp"

namespace dxapp {

/// Per-anchor side channel a pre-optimized model carries next to bbox / cls.
enum class PreoptExtra { None, Keypoints, MaskCoefs };

/// Column offsets of one pre-optimized row.
struct PreoptRow {
    static constexpr int kX1 = 0;
    static constexpr int kY1 = 1;
    static constexpr int kX2 = 2;
    static constexpr int kY2 = 3;
    static constexpr int kScore = 4;
    static constexpr int kClass = 5;
    static constexpr int kExtra = 6;
};

/**
 * @brief Produces the K x C row table of a pre-optimized model.
 *
 * With ORT the table is the model output and is used in place; without ORT it
 * is computed from the raw NPU tensors into the caller's scratch buffer.
 */
class PreoptTopKDecoder {
public:
    struct Rows {
        const float* data{nullptr};
        int count{0};
        int cols{0};
        const float* row(int i) const { return data + static_cast<std::ptrdiff_t>(i) * cols; }
    };

    PreoptTopKDecoder(int input_width, int input_height, int top_k,
                      PreoptExtra extra = PreoptExtra::None, int extra_channels = 0)
        : input_width_(input_width), input_height_(input_height),
          top_k_(std::max(1, top_k)), extra_(extra), extra_channels_(extra_channels) {}

    /// The [1, K, C] table produced by the model's CPU task, or nullptr.
    static dxrt::TensorPtr findRowTensor(const dxrt::TensorPtrs& outputs) {
        for (const auto& o : outputs) {
            const auto& s = o->shape();
            if (s.size() == 3 && s[0] == 1 && s[1] > s[2] && s[2] >= PreoptRow::kExtra) return o;
        }
        return nullptr;
    }

    /// The mask prototype tensor [1, nm, H, W] of a segmentation model, or nullptr.
    static dxrt::TensorPtr findProtoTensor(const dxrt::TensorPtrs& outputs, int num_mask_coefs) {
        for (const auto& o : outputs) {
            if (o->name() == "output1" && o->shape().size() == 4) return o;
        }
        for (const auto& o : outputs) {
            const auto& s = o->shape();
            if (s.size() == 4 && s[1] == num_mask_coefs && o->name().rfind("preopt_", 0) != 0) return o;
        }
        return nullptr;
    }

    /// Decode. `scratch` receives the table when it has to be computed here.
    Rows decode(const dxrt::TensorPtrs& outputs, std::vector<float>& scratch) const {
        if (auto table = findRowTensor(outputs)) {
            Rows rows;
            rows.data = static_cast<const float*>(table->data());
            rows.count = static_cast<int>(table->shape()[1]);
            rows.cols = static_cast<int>(table->shape()[2]);
            return rows;
        }
        return decodeRaw(outputs, scratch);
    }

    int cols() const { return PreoptRow::kExtra + extra_channels_; }
    int topK() const { return top_k_; }

private:
    // One stride level of the raw NPU output.
    struct Level {
        int h{0}, w{0}, stride{0}, anchor_offset{0};
        const float* bbox{nullptr};
        const float* cls{nullptr};
        const float* extra{nullptr};
        int nc{0};
    };

    static float sigmoid(float x) { return postprocess_utils::sigmoid(x); }

    static bool startsWith(const std::string& s, const char* prefix) {
        return s.rfind(prefix, 0) == 0;
    }

    [[noreturn]] void fail(const dxrt::TensorPtrs& outputs, const char* what) const {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] PreoptTopKDecoder - " << what << "\n"
            << "  Expected preopt_output (1, K, C) or the NPU tensors preopt_bbox_tr_i (1, H, W, 4) / "
               "preopt_cls_tr_i (1, H, W, nc)";
        if (extra_ == PreoptExtra::Keypoints) msg << " / preopt_kpt_tr_i (1, H, W, " << extra_channels_ << ")";
        if (extra_ == PreoptExtra::MaskCoefs) msg << " / preopt_mask_tr_i (1, H, W, " << extra_channels_ << ")";
        msg << "\n" << postprocess_utils::format_tensor_shapes(outputs)
            << "Please re-compile the model with the pre-optimize (top-k) output configuration.\n";
        throw PostprocessConfigError(msg.str());
    }

    // Group the raw NHWC tensors by grid size, largest grid (stride 8) first.
    std::vector<Level> collectLevels(const dxrt::TensorPtrs& outputs) const {
        std::vector<Level> levels;
        auto level_for = [&levels](int h, int w) -> Level& {
            for (auto& l : levels) if (l.h == h && l.w == w) return l;
            levels.push_back(Level{});
            levels.back().h = h;
            levels.back().w = w;
            return levels.back();
        };
        for (const auto& o : outputs) {
            const auto& s = o->shape();
            if (s.size() != 4 || s[0] != 1) continue;
            const std::string& name = o->name();
            // Prototypes (NCHW) are not a per-anchor level.
            if (name == "output1" || (extra_ == PreoptExtra::MaskCoefs && !startsWith(name, "preopt_") &&
                                      s[1] == extra_channels_)) continue;
            const int h = static_cast<int>(s[1]), w = static_cast<int>(s[2]), c = static_cast<int>(s[3]);
            const auto* data = static_cast<const float*>(o->data());
            Level& l = level_for(h, w);
            if (startsWith(name, "preopt_bbox") || (!startsWith(name, "preopt_") && c == 4)) {
                l.bbox = data;
            } else if (extra_ == PreoptExtra::Keypoints &&
                       (startsWith(name, "preopt_kpt") || (!startsWith(name, "preopt_") && c == extra_channels_))) {
                l.extra = data;
            } else if (extra_ == PreoptExtra::MaskCoefs &&
                       (startsWith(name, "preopt_mask") || (!startsWith(name, "preopt_") && c == extra_channels_))) {
                l.extra = data;
            } else if (startsWith(name, "preopt_cls") || !startsWith(name, "preopt_")) {
                l.cls = data;
                l.nc = c;
            }
        }
        if (levels.empty()) fail(outputs, "no pre-optimized output tensors found.");
        for (const auto& l : levels) {
            if (!l.bbox || !l.cls || (extra_ != PreoptExtra::None && !l.extra) || l.w <= 0)
                fail(outputs, "incomplete per-stride tensor set.");
            if (l.nc != levels.front().nc) fail(outputs, "class count differs between strides.");
        }
        std::sort(levels.begin(), levels.end(),
                  [](const Level& a, const Level& b) { return a.h * a.w > b.h * b.w; });
        int offset = 0;
        for (auto& l : levels) {
            l.stride = input_width_ / l.w;
            l.anchor_offset = offset;
            offset += l.h * l.w;
        }
        return levels;
    }

    // The CPU task of the model, done here: two-stage top-k over all anchors.
    Rows decodeRaw(const dxrt::TensorPtrs& outputs, std::vector<float>& scratch) const {
        const std::vector<Level> levels = collectLevels(outputs);
        const int nc = levels.front().nc;
        int num_anchors = 0;
        for (const auto& l : levels) num_anchors += l.h * l.w;

        // Stage 1: best class logit per anchor (sigmoid is monotonic, so the
        // logit orders anchors the same way the model's sigmoid scores do),
        // keep the top-k anchors. Ties resolve to the lower anchor index.
        std::vector<float> best(num_anchors);
        for (const auto& l : levels) {
            const int n = l.h * l.w;
            for (int sp = 0; sp < n; ++sp) {
                const float* c = l.cls + static_cast<std::ptrdiff_t>(sp) * nc;
                float m = c[0];
                for (int k = 1; k < nc; ++k) m = std::max(m, c[k]);
                best[l.anchor_offset + sp] = m;
            }
        }
        const int k1 = std::min(top_k_, num_anchors);
        std::vector<int> anchors(num_anchors);
        std::iota(anchors.begin(), anchors.end(), 0);
        std::partial_sort(anchors.begin(), anchors.begin() + k1, anchors.end(),
                          [&best](int a, int b) { return best[a] > best[b] || (best[a] == best[b] && a < b); });

        // Stage 2: scores of the k1 x nc (anchor, class) pairs, keep the top-k pairs.
        std::vector<float> scores(static_cast<size_t>(k1) * nc);
        for (int r = 0; r < k1; ++r) {
            const Level& l = levelOf(levels, anchors[r]);
            const float* c = l.cls + static_cast<std::ptrdiff_t>(anchors[r] - l.anchor_offset) * nc;
            for (int k = 0; k < nc; ++k) scores[static_cast<size_t>(r) * nc + k] = sigmoid(c[k]);
        }
        const int k2 = std::min(top_k_, k1 * nc);
        std::vector<int> pairs(scores.size());
        std::iota(pairs.begin(), pairs.end(), 0);
        std::partial_sort(pairs.begin(), pairs.begin() + k2, pairs.end(),
                          [&scores](int a, int b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });

        // Emit the rows: decode box / keypoints, copy mask coefficients.
        const int cols = this->cols();
        scratch.assign(static_cast<size_t>(k2) * cols, 0.0f);
        for (int i = 0; i < k2; ++i) {
            const int pair = pairs[i];
            const int anchor = anchors[pair / nc];
            const int cls = pair % nc;
            const Level& l = levelOf(levels, anchor);
            const int sp = anchor - l.anchor_offset;
            const float cx = static_cast<float>(sp % l.w) + 0.5f;
            const float cy = static_cast<float>(sp / l.w) + 0.5f;
            const float s = static_cast<float>(l.stride);
            const float* b = l.bbox + static_cast<std::ptrdiff_t>(sp) * 4;
            float* row = scratch.data() + static_cast<std::ptrdiff_t>(i) * cols;
            row[PreoptRow::kX1] = (cx - b[0]) * s;
            row[PreoptRow::kY1] = (cy - b[1]) * s;
            row[PreoptRow::kX2] = (cx + b[2]) * s;
            row[PreoptRow::kY2] = (cy + b[3]) * s;
            row[PreoptRow::kScore] = scores[pair];
            row[PreoptRow::kClass] = static_cast<float>(cls);
            if (extra_ == PreoptExtra::None) continue;
            const float* e = l.extra + static_cast<std::ptrdiff_t>(sp) * extra_channels_;
            float* out = row + PreoptRow::kExtra;
            if (extra_ == PreoptExtra::Keypoints) {
                for (int k = 0; k + 2 < extra_channels_; k += 3) {
                    out[k]     = (e[k] + cx) * s;
                    out[k + 1] = (e[k + 1] + cy) * s;
                    out[k + 2] = sigmoid(e[k + 2]);
                }
            } else {
                std::copy(e, e + extra_channels_, out);
            }
        }
        Rows rows;
        rows.data = scratch.data();
        rows.count = k2;
        rows.cols = cols;
        return rows;
    }

    static const Level& levelOf(const std::vector<Level>& levels, int anchor) {
        for (size_t i = levels.size(); i-- > 1;) {
            if (anchor >= levels[i].anchor_offset) return levels[i];
        }
        return levels.front();
    }

    int input_width_;
    int input_height_;
    int top_k_;
    PreoptExtra extra_;
    int extra_channels_;
};

namespace preopt_detail {

inline float iou(const float* a, const float* b) {
    const float x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    const float inter = (x2 - x1) * (y2 - y1);
    const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/**
 * Rows at or above the score threshold, best first. With nms_threshold < 1 a
 * class-agnostic greedy NMS drops rows whose IoU with a kept row is >= the
 * threshold - this is what removes the same-box / other-class duplicates.
 */
inline std::vector<int> selectRows(const PreoptTopKDecoder::Rows& rows,
                                   float score_threshold, float nms_threshold) {
    std::vector<int> candidates;
    for (int i = 0; i < rows.count; ++i) {
        if (rows.row(i)[PreoptRow::kScore] >= score_threshold) candidates.push_back(i);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&rows](int a, int b) {
        return rows.row(a)[PreoptRow::kScore] > rows.row(b)[PreoptRow::kScore];
    });
    if (nms_threshold >= 1.0f) return candidates;

    std::vector<int> keep;
    for (int idx : candidates) {
        bool suppressed = false;
        for (int kept : keep) {
            if (iou(rows.row(idx), rows.row(kept)) >= nms_threshold) { suppressed = true; break; }
        }
        if (!suppressed) keep.push_back(idx);
    }
    return keep;
}

inline std::vector<float> boxOf(const float* row) {
    return {row[PreoptRow::kX1], row[PreoptRow::kY1], row[PreoptRow::kX2], row[PreoptRow::kY2]};
}

}  // namespace preopt_detail

// ============================================================================
// Object detection: rows [x1, y1, x2, y2, score, class_id]
// ============================================================================
class PreoptDetectionPostprocessor : public IPostprocessor<DetectionResult> {
public:
    PreoptDetectionPostprocessor(int input_width = 640, int input_height = 640,
                                 float score_threshold = 0.3f, float nms_threshold = 0.45f,
                                 int top_k = 300,
                                 const std::vector<std::string>& class_names = {})
        : decoder_(input_width, input_height, top_k),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          class_names_(class_names) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<float> scratch;
        const auto rows = decoder_.decode(outputs, scratch);
        std::vector<DetectionResult> results;
        for (int i : preopt_detail::selectRows(rows, score_threshold_, nms_threshold_)) {
            const float* row = rows.row(i);
            DetectionResult det;
            det.box = preopt_detail::boxOf(row);
            det.confidence = row[PreoptRow::kScore];
            det.class_id = static_cast<int>(row[PreoptRow::kClass]);
            det.class_name = dxapp::common::resolve_class_name(det.class_id, class_names_);
            detail::scaleBox(det.box, ctx);
            results.push_back(std::move(det));
        }
        return results;
    }

    std::string getModelName() const override { return "YOLO26-preopt"; }

private:
    PreoptTopKDecoder decoder_;
    float score_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
};

// ============================================================================
// Pose: rows [x1, y1, x2, y2, score, class_id, (x, y, visibility) x 17]
// ============================================================================
class PreoptPosePostprocessor : public IPostprocessor<PoseResult> {
public:
    PreoptPosePostprocessor(int input_width = 640, int input_height = 640,
                            float score_threshold = 0.3f, float nms_threshold = 0.45f,
                            int top_k = 300, int num_keypoints = 17)
        : decoder_(input_width, input_height, top_k, PreoptExtra::Keypoints, 3 * num_keypoints),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          num_keypoints_(num_keypoints) {}

    std::vector<PoseResult> process(const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) override {
        std::vector<float> scratch;
        const auto rows = decoder_.decode(outputs, scratch);
        if (rows.cols < PreoptRow::kExtra + 3 * num_keypoints_) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] PreoptPosePostprocessor - rows have " << rows.cols
                << " columns, expected " << PreoptRow::kExtra + 3 * num_keypoints_ << ".\n"
                << postprocess_utils::format_tensor_shapes(outputs);
            throw PostprocessConfigError(msg.str());
        }
        const auto w = static_cast<float>(ctx.original_width);
        const auto h = static_cast<float>(ctx.original_height);

        std::vector<PoseResult> results;
        for (int i : preopt_detail::selectRows(rows, score_threshold_, nms_threshold_)) {
            const float* row = rows.row(i);
            PoseResult pose;
            pose.box = preopt_detail::boxOf(row);
            pose.confidence = row[PreoptRow::kScore];
            detail::scaleBox(pose.box, ctx);
            pose.keypoints.reserve(num_keypoints_);
            const float* kp = row + PreoptRow::kExtra;
            for (int k = 0; k < num_keypoints_; ++k, kp += 3) {
                Keypoint point(kp[0], kp[1], kp[2]);
                detail::scaleKeypoint(point, ctx);
                point.x = std::max(0.0f, std::min(point.x, w));
                point.y = std::max(0.0f, std::min(point.y, h));
                pose.keypoints.push_back(point);
            }
            results.push_back(std::move(pose));
        }
        return results;
    }

    std::string getModelName() const override { return "YOLO26-pose-preopt"; }

private:
    PreoptTopKDecoder decoder_;
    float score_threshold_;
    float nms_threshold_;
    int num_keypoints_;
};

// ============================================================================
// Instance segmentation: rows [x1, y1, x2, y2, score, class_id, coef x 32]
//                        + prototypes output1 [1, 32, H, W]
// ============================================================================
class PreoptSegPostprocessor : public IPostprocessor<InstanceSegmentationResult> {
public:
    PreoptSegPostprocessor(int input_width = 640, int input_height = 640,
                           float score_threshold = 0.3f, float nms_threshold = 0.45f,
                           int top_k = 300, int num_mask_coefs = 32,
                           const std::vector<std::string>& class_names = {})
        : decoder_(input_width, input_height, top_k, PreoptExtra::MaskCoefs, num_mask_coefs),
          input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          num_mask_coefs_(num_mask_coefs), class_names_(class_names) {}

    std::vector<InstanceSegmentationResult> process(const dxrt::TensorPtrs& outputs,
                                                    const PreprocessContext& ctx) override {
        const auto proto = PreoptTopKDecoder::findProtoTensor(outputs, num_mask_coefs_);
        if (!proto) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] PreoptSegPostprocessor - mask prototype tensor output1 (1, "
                << num_mask_coefs_ << ", H, W) not found.\n"
                << postprocess_utils::format_tensor_shapes(outputs);
            throw PostprocessConfigError(msg.str());
        }
        std::vector<float> scratch;
        const auto rows = decoder_.decode(outputs, scratch);
        if (rows.cols < PreoptRow::kExtra + num_mask_coefs_) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] PreoptSegPostprocessor - rows have " << rows.cols
                << " columns, expected " << PreoptRow::kExtra + num_mask_coefs_ << ".\n"
                << postprocess_utils::format_tensor_shapes(outputs);
            throw PostprocessConfigError(msg.str());
        }

        const auto& ps = proto->shape();
        const int proto_c = static_cast<int>(ps[1]);
        const int proto_h = static_cast<int>(ps[2]);
        const int proto_w = static_cast<int>(ps[3]);
        const auto* proto_data = static_cast<const float*>(proto->data());

        std::vector<InstanceSegmentationResult> results;
        std::vector<float> coefs(num_mask_coefs_);
        for (int i : preopt_detail::selectRows(rows, score_threshold_, nms_threshold_)) {
            const float* row = rows.row(i);
            std::copy(row + PreoptRow::kExtra, row + PreoptRow::kExtra + num_mask_coefs_, coefs.begin());

            InstanceSegmentationResult seg;
            // The ROI mask is built from the box in model-input coordinates and
            // lands directly at the original image resolution.
            seg.mask = roiInstanceMask(coefs, proto_data, proto_c, proto_h, proto_w,
                                       row[PreoptRow::kX1], row[PreoptRow::kY1],
                                       row[PreoptRow::kX2], row[PreoptRow::kY2],
                                       input_width_, input_height_, ctx);
            seg.box = preopt_detail::boxOf(row);
            detail::scaleBox(seg.box, ctx);
            seg.confidence = row[PreoptRow::kScore];
            seg.class_id = static_cast<int>(row[PreoptRow::kClass]);
            seg.class_name = dxapp::common::resolve_class_name(seg.class_id, class_names_);
            results.push_back(std::move(seg));
        }
        return results;
    }

    std::string getModelName() const override { return "YOLO26-seg-preopt"; }

private:
    PreoptTopKDecoder decoder_;
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    int num_mask_coefs_;
    std::vector<std::string> class_names_;
};

}  // namespace dxapp

#endif  // PREOPT_TOPK_POSTPROCESSOR_HPP
