/**
 * @file mediapipe_hand_postprocessor.hpp
 * @brief MediaPipe Palm/Hand Detector postprocessor
 *
 * Ported from Python mediapipe_hand_postprocessor.py (dx-modelzoo custom_ops.py based).
 *
 * Model outputs (dxrt provides pre-flattened):
 *   - regression: [1, 2016, 18]  (cx_off, cy_off, w, h, 7 keypoints × 2)
 *   - scores:     [1, 2016, 1]   (sigmoid logit)
 *
 * Anchor generation: MediaPipe SsdAnchorsCalculator
 *   strides=[8,16,16,16], 2 anchors per cell.
 *   stride-8 (1 layer):  24×24×2 = 1152 anchors
 *   stride-16 (3 layers): 12×12×6 = 864 anchors
 *   Total: 2016
 *
 * Decoding: MediaPipe TensorsToDetectionsCalculator
 * Box expansion: DetectionsToRects (box_scale=2.0, box_shift=0.3)
 */

#ifndef MEDIAPIPE_HAND_POSTPROCESSOR_HPP
#define MEDIAPIPE_HAND_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

class MediaPipeHandPostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    MediaPipeHandPostprocessor(int input_size = 192,
                               float score_threshold = 0.5f,
                               float nms_threshold   = 0.3f,
                               float box_scale       = 2.0f,
                               float box_shift       = 0.3f)
        : input_size_(input_size),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold),
          box_scale_(box_scale),
          box_shift_(box_shift) {
        generateAnchors();
    }

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                              const PreprocessContext& ctx) override {
        std::vector<FaceDetectionResult> results;

        // ── Parse outputs → reg [N,18] and scores [N] ──────────────────────
        const float* reg_data   = nullptr;
        const float* score_data = nullptr;
        int N_reg = 0;

        for (auto& t : outputs) {
            auto sh = t->shape();
            int last = static_cast<int>(sh.back());
            // flat [1, N, 18] or [N, 18]
            if (last == 18) {
                reg_data = static_cast<const float*>(t->data());
                N_reg = static_cast<int>(sh.size() >= 2 ? sh[sh.size()-2] : sh[0]);
            } else if (last == 1) {
                score_data = static_cast<const float*>(t->data());
            }
        }

        // Fallback: NHWC feature-map format [1, H, W, C]
        if (!reg_data || !score_data) {
            return parseNHWCFallback(outputs, ctx);
        }

        int N = std::min(N_reg, static_cast<int>(anchors_.size()));
        float scale = static_cast<float>(input_size_);

        std::vector<float> x1v, y1v, x2v, y2v, sv;
        x1v.reserve(N); y1v.reserve(N); x2v.reserve(N); y2v.reserve(N); sv.reserve(N);

        for (int i = 0; i < N; ++i) {
            float raw_score = score_data[i];
            float score = 1.0f / (1.0f + std::exp(-std::max(-100.f, std::min(100.f, raw_score))));
            if (score < score_threshold_) continue;

            const float* r = reg_data + i * 18;
            float acx = anchors_[i][0];
            float acy = anchors_[i][1];

            float cx = r[0] / scale + acx;
            float cy = r[1] / scale + acy;
            float w  = std::abs(r[2]) / scale;
            float h  = std::abs(r[3]) / scale;
            float s  = std::max(w, h);

            // Wrist kp=0, Middle-finger MCP kp=2
            float kx0 = r[4 + 0] / scale + acx;
            float ky0 = r[4 + 1] / scale + acy;
            float kx2 = r[4 + 4] / scale + acx;
            float ky2 = r[4 + 5] / scale + acy;
            float dx = kx2 - kx0, dy = ky2 - ky0;
            float norm = std::sqrt(dx*dx + dy*dy) + 1e-9f;
            dx /= norm; dy /= norm;

            // Palm → full-hand box expansion
            float side = s * box_scale_;
            float ccx  = cx + box_shift_ * side * dx;
            float ccy  = cy + box_shift_ * side * dy;

            x1v.push_back(ccx - side * 0.5f);
            y1v.push_back(ccy - side * 0.5f);
            x2v.push_back(ccx + side * 0.5f);
            y2v.push_back(ccy + side * 0.5f);
            sv.push_back(score);
        }

        return applyNMSAndScale(x1v, y1v, x2v, y2v, sv, ctx);
    }

    std::string getModelName() const override { return "MediaPipeHandDetector"; }

private:
    // ── Anchor generation (MediaPipe SsdAnchorsCalculator) ────────────────

    void generateAnchors() {
        // strides=[8,16,16,16], 2 anchors per cell per layer.
        // Consecutive layers with same stride are merged (stride-16 has 3 layers → 6 anchors/cell).
        static const int strides[]     = {8, 16, 16, 16};
        static const int num_layers    = 4;
        static const int anchors_per_cell = 2;

        int layer_id = 0;
        while (layer_id < num_layers) {
            int last_same = layer_id;
            int repeats   = 0;
            while (last_same < num_layers && strides[last_same] == strides[layer_id]) {
                repeats += anchors_per_cell;
                ++last_same;
            }
            int stride     = strides[layer_id];
            int feature_map = static_cast<int>(std::ceil(static_cast<float>(input_size_) / stride));
            for (int y = 0; y < feature_map; ++y) {
                for (int x = 0; x < feature_map; ++x) {
                    float cx = (x + 0.5f) / feature_map;
                    float cy = (y + 0.5f) / feature_map;
                    for (int k = 0; k < repeats; ++k) {
                        anchors_.push_back({cx, cy});
                    }
                }
            }
            layer_id = last_same;
        }
    }

    // ── NHWC fallback ──────────────────────────────────────────────────────

    std::vector<FaceDetectionResult> parseNHWCFallback(const dxrt::TensorPtrs& outputs,
                                                        const PreprocessContext& ctx) {
        std::vector<float> reg_flat, score_flat;

        for (auto& t : outputs) {
            auto sh = t->shape();
            if (sh.size() != 4) continue;
            int H = static_cast<int>(sh[1]);
            int W = static_cast<int>(sh[2]);
            int C = static_cast<int>(sh[3]);
            const float* data = static_cast<const float*>(t->data());
            if (C % 18 == 0) {
                // regression: [1,H,W,A*18] → [H*W*A, 18]
                int A = C / 18;
                reg_flat.reserve(reg_flat.size() + H * W * A * 18);
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        for (int a = 0; a < A; ++a)
                            for (int v = 0; v < 18; ++v)
                                reg_flat.push_back(data[(y*W+x)*C + a*18 + v]);
            } else if (C <= 8) {
                // scores: [1,H,W,A] → [H*W*A]
                score_flat.insert(score_flat.end(), data, data + H * W * C);
            }
        }

        if (reg_flat.empty() || score_flat.empty()) return {};

        // Reconstruct as flat tensors and recurse (or just decode inline)
        int N = static_cast<int>(std::min(score_flat.size(),
                                           reg_flat.size() / 18));
        N = std::min(N, static_cast<int>(anchors_.size()));
        float scale = static_cast<float>(input_size_);

        std::vector<float> x1v, y1v, x2v, y2v, sv;
        for (int i = 0; i < N; ++i) {
            float raw_score = score_flat[i];
            float score = 1.0f / (1.0f + std::exp(-std::max(-100.f, std::min(100.f, raw_score))));
            if (score < score_threshold_) continue;

            const float* r = reg_flat.data() + i * 18;
            float acx = anchors_[i][0], acy = anchors_[i][1];

            float cx = r[0] / scale + acx, cy = r[1] / scale + acy;
            float w  = std::abs(r[2]) / scale, h = std::abs(r[3]) / scale;
            float s  = std::max(w, h);

            float kx0 = r[4]   / scale + acx, ky0 = r[5]   / scale + acy;
            float kx2 = r[4+4] / scale + acx, ky2 = r[4+5] / scale + acy;
            float dx = kx2 - kx0, dy = ky2 - ky0;
            float norm = std::sqrt(dx*dx + dy*dy) + 1e-9f;
            dx /= norm; dy /= norm;

            float side = s * box_scale_;
            float ccx = cx + box_shift_ * side * dx;
            float ccy = cy + box_shift_ * side * dy;

            x1v.push_back(ccx - side * 0.5f);
            y1v.push_back(ccy - side * 0.5f);
            x2v.push_back(ccx + side * 0.5f);
            y2v.push_back(ccy + side * 0.5f);
            sv.push_back(score);
        }

        return applyNMSAndScale(x1v, y1v, x2v, y2v, sv, ctx);
    }

    // ── NMS + coordinate mapping ───────────────────────────────────────────

    std::vector<FaceDetectionResult> applyNMSAndScale(
            const std::vector<float>& x1v, const std::vector<float>& y1v,
            const std::vector<float>& x2v, const std::vector<float>& y2v,
            const std::vector<float>& sv,
            const PreprocessContext& ctx) const {
        if (sv.empty()) return {};

        // NMS in normalized coords
        std::vector<cv::Rect2d> nms_boxes;
        nms_boxes.reserve(sv.size());
        for (size_t i = 0; i < sv.size(); ++i) {
            nms_boxes.emplace_back(
                static_cast<double>(x1v[i]),
                static_cast<double>(y1v[i]),
                static_cast<double>(x2v[i] - x1v[i]),
                static_cast<double>(y2v[i] - y1v[i]));
        }
        std::vector<int> kept;
        cv::dnn::NMSBoxes(nms_boxes, const_cast<std::vector<float>&>(sv),
                          score_threshold_, nms_threshold_, kept);

        float ow = static_cast<float>(ctx.original_width);
        float oh = static_cast<float>(ctx.original_height);

        std::vector<FaceDetectionResult> results;
        results.reserve(kept.size());
        for (int k : kept) {
            float bx1 = std::max(0.f, std::min(x1v[k] * ow, ow - 1.f));
            float by1 = std::max(0.f, std::min(y1v[k] * oh, oh - 1.f));
            float bx2 = std::max(0.f, std::min(x2v[k] * ow, ow - 1.f));
            float by2 = std::max(0.f, std::min(y2v[k] * oh, oh - 1.f));
            FaceDetectionResult face;
            face.box        = {bx1, by1, bx2, by2};
            face.confidence = sv[k];
            results.push_back(face);
        }
        return results;
    }

    int   input_size_;
    float score_threshold_;
    float nms_threshold_;
    float box_scale_;
    float box_shift_;
    std::vector<std::array<float, 2>> anchors_;  // cx, cy (normalized)
};

}  // namespace dxapp

#endif  // MEDIAPIPE_HAND_POSTPROCESSOR_HPP
