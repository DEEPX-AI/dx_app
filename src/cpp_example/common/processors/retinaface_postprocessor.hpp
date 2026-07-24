/**
 * @file retinaface_postprocessor.hpp
 * @brief RetinaFace face detection postprocessor with 5-point landmarks
 * 
 * Ported from Python retinaface_postprocessor.py.
 * 
 * Anchor-based face detection with multi-scale feature maps.
 * Output: 3 tensors (auto-sorted by last dimension):
 *   - scores:    [1, N, 2]   (bg/face softmax, last_dim=2)
 *   - boxes:     [1, N, 4]   (anchor offsets, last_dim=4)
 *   - landmarks: [1, N, 10]  (5 keypoints × 2, last_dim=10)
 * 
 * Prior boxes generated from strides [8, 16, 32].
 * Variance = [0.1, 0.2] for decoding.
 */

/**
 * @file retinaface_postprocessor.hpp
 * @brief RetinaFace face detection postprocessor with 5-point landmarks
 * 
 * Ported from Python retinaface_postprocessor.py.
 * 
 * Supports two output formats automatically:
 *   1. NHWC feature-map format (9 tensors: 3 strides × bbox/cls/lmk):
 *        [1, H, W, A*4], [1, H, W, A*2], [1, H, W, A*10]  × 3 strides
 *   2. Flattened format (3 tensors):
 *        [1, N, 4], [1, N, 2], [1, N, 10]
 * 
 * Anchors: 2 per location, strides [8, 16, 32], min_sizes [[16,32],[64,128],[256,512]].
 * Variance = [0.1, 0.2] for decoding.
 */

#ifndef RETINAFACE_POSTPROCESSOR_HPP
#define RETINAFACE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace dxapp {

class RetinaFacePostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    static constexpr int NUM_ANCHORS = 2;

    RetinaFacePostprocessor(int input_width = 640, int input_height = 640,
                            float score_threshold = 0.5f,
                            float nms_threshold = 0.4f)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold) {
        generatePriorBoxes();
    }

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                              const PreprocessContext& ctx) override {
        std::vector<FaceDetectionResult> results;
        if (outputs.empty()) return results;

        // Flatten outputs to [N, 4], [N, 2], [N, 10]
        std::vector<float> boxes_flat, scores_flat, lmks_flat;
        bool ok = isNHWCFormat(outputs)
            ? parseNHWC(outputs, boxes_flat, scores_flat, lmks_flat)
            : parseFlat(outputs, boxes_flat, scores_flat, lmks_flat);
        if (!ok || scores_flat.empty()) return results;

        int N = static_cast<int>(scores_flat.size() / 2);
        N = std::min(N, static_cast<int>(priors_.size()));

        const float* boxes_data  = boxes_flat.data();
        const float* scores_data = scores_flat.data();
        const float* lmks_data   = lmks_flat.empty() ? nullptr : lmks_flat.data();

        std::vector<cv::Rect2d> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_indices;
        collectCandidates(N, scores_data, boxes_data, nms_boxes, nms_scores, nms_indices);

        if (nms_boxes.empty()) return results;

        std::vector<int> kept;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, kept);

        for (int k : kept) {
            int i = nms_indices[k];

            float cx, cy, bw, bh;
            decodePriorBox(i, boxes_data, cx, cy, bw, bh);

            float x1 = (cx - bw * 0.5f) * input_width_;
            float y1 = (cy - bh * 0.5f) * input_height_;
            float x2 = (cx + bw * 0.5f) * input_width_;
            float y2 = (cy + bh * 0.5f) * input_height_;

            float pcx = priors_[i][0], pcy = priors_[i][1];
            float pw  = priors_[i][2], ph  = priors_[i][3];
            std::vector<Keypoint> landmarks(5);
            if (lmks_data) {
                for (int j = 0; j < 5; ++j) {
                    float lx = (pcx + lmks_data[i * 10 + j * 2]     * variance_[0] * pw) * input_width_;
                    float ly = (pcy + lmks_data[i * 10 + j * 2 + 1] * variance_[0] * ph) * input_height_;
                    landmarks[j] = Keypoint(lx, ly, 1.0f);
                }
            }

            scaleResultCoords(ctx, x1, y1, x2, y2, landmarks);

            FaceDetectionResult face;
            face.box = {
                std::max(0.0f, std::min(x1, static_cast<float>(ctx.original_width))),
                std::max(0.0f, std::min(y1, static_cast<float>(ctx.original_height))),
                std::max(0.0f, std::min(x2, static_cast<float>(ctx.original_width))),
                std::max(0.0f, std::min(y2, static_cast<float>(ctx.original_height)))
            };
            face.confidence = nms_scores[k];
            face.landmarks = std::move(landmarks);
            results.push_back(face);
        }
        return results;
    }

    std::string getModelName() const override { return "RetinaFace"; }

private:
    // ── Format detection ─────────────────────────────────────────────────────

    bool isNHWCFormat(const dxrt::TensorPtrs& outputs) const {
        for (auto& t : outputs) {
            if (t->shape().size() == 4) return true;
        }
        return false;
    }

    // ── NHWC parsing ──────────────────────────────────────────────────────────
    // Groups 9 feature-map tensors by (H,W), identifies bbox/cls/lmk
    // by last_dim / NUM_ANCHORS (4=bbox, 2=cls, 10=lmk).
    // Reshapes [H,W,A*k] → [H*W*A, k] and concatenates across strides.

    bool parseNHWC(const dxrt::TensorPtrs& outputs,
                   std::vector<float>& boxes_out,
                   std::vector<float>& scores_out,
                   std::vector<float>& lmks_out) const {
        // key=(H,W), value=list of (tensor_data, H, W, C)
        using MapKey = std::pair<int,int>;
        std::map<MapKey, std::vector<std::tuple<const float*, int, int, int>>> groups;

        for (auto& t : outputs) {
            auto sh = t->shape();
            if (sh.size() != 4) continue;
            int H = static_cast<int>(sh[1]);
            int W = static_cast<int>(sh[2]);
            int C = static_cast<int>(sh[3]);
            const float* data = static_cast<const float*>(t->data());
            groups[{H, W}].emplace_back(data, H, W, C);
        }

        if (groups.empty()) return false;

        // Process strides from large spatial to small (stride 8 first)
        std::vector<MapKey> keys;
        for (auto& kv : groups) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end(),
                  [](const MapKey& a, const MapKey& b){ return a.first > b.first; });

        for (auto& key : keys) {
            const float* bbox_d = nullptr, *cls_d = nullptr, *lmk_d = nullptr;
            int H = 0, W = 0;

            for (auto& entry : groups[key]) {
                const float* data = std::get<0>(entry);
                int h = std::get<1>(entry);
                int w = std::get<2>(entry);
                int c = std::get<3>(entry);
                H = h; W = w;
                int per_anchor = c / NUM_ANCHORS;
                if (per_anchor == 4)  bbox_d = data;
                else if (per_anchor == 2)  cls_d  = data;
                else if (per_anchor == 10) lmk_d  = data;
            }
            if (!bbox_d || !cls_d) continue;

            int n = H * W * NUM_ANCHORS;

            // Reshape [H, W, A*k] → [H*W*A, k]: for each spatial cell (y,x),
            // then for each anchor: store k values contiguously.
            auto flatten = [&](const float* src, int k, std::vector<float>& dst) {
                dst.reserve(dst.size() + n * k);
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        for (int a = 0; a < NUM_ANCHORS; ++a)
                            for (int v = 0; v < k; ++v)
                                dst.push_back(src[(y * W + x) * NUM_ANCHORS * k + a * k + v]);
            };

            flatten(bbox_d, 4,  boxes_out);
            flatten(cls_d,  2,  scores_out);
            if (lmk_d) flatten(lmk_d, 10, lmks_out);
        }
        return !boxes_out.empty();
    }

    // ── Flat parsing ─────────────────────────────────────────────────────────
    // Handles 3 tensors [1,N,4], [1,N,2], [1,N,10].

    bool parseFlat(const dxrt::TensorPtrs& outputs,
                   std::vector<float>& boxes_out,
                   std::vector<float>& scores_out,
                   std::vector<float>& lmks_out) const {
        const dxrt::TensorPtr* scores_t = nullptr;
        const dxrt::TensorPtr* boxes_t  = nullptr;
        const dxrt::TensorPtr* lmks_t   = nullptr;
        for (auto& t : outputs) {
            int last = static_cast<int>(t->shape().back());
            if (last == 2 && !scores_t) scores_t = &t;
            else if (last == 4 && !boxes_t)  boxes_t  = &t;
            else if (last == 10 && !lmks_t)  lmks_t   = &t;
        }
        if (!scores_t || !boxes_t) return false;

        auto sh = (*boxes_t)->shape();
        int N = static_cast<int>(sh.size() >= 2 ? sh[sh.size()-2] : sh[0]);

        const float* bd = static_cast<const float*>((*boxes_t)->data());
        const float* sd = static_cast<const float*>((*scores_t)->data());

        boxes_out.assign(bd, bd + N * 4);
        scores_out.assign(sd, sd + N * 2);
        if (lmks_t) {
            const float* ld = static_cast<const float*>((*lmks_t)->data());
            lmks_out.assign(ld, ld + N * 10);
        }
        return true;
    }

    // ── Decoding helpers ─────────────────────────────────────────────────────

    void decodePriorBox(int i, const float* boxes_data,
                        float& cx, float& cy, float& bw, float& bh) const {
        float pcx = priors_[i][0], pcy = priors_[i][1];
        float pw  = priors_[i][2], ph  = priors_[i][3];
        cx = pcx + boxes_data[i * 4 + 0] * variance_[0] * pw;
        cy = pcy + boxes_data[i * 4 + 1] * variance_[0] * ph;
        bw = pw  * std::exp(boxes_data[i * 4 + 2] * variance_[1]);
        bh = ph  * std::exp(boxes_data[i * 4 + 3] * variance_[1]);
    }

    void collectCandidates(int N, const float* scores_data, const float* boxes_data,
                           std::vector<cv::Rect2d>& nms_boxes,
                           std::vector<float>& nms_scores,
                           std::vector<int>& nms_indices) const {
        for (int i = 0; i < N; ++i) {
            // Softmax-like: face score is index 1 of [bg, face]
            float bg   = scores_data[i * 2 + 0];
            float face = scores_data[i * 2 + 1];
            float denom = std::exp(bg - face) + 1.0f;
            float face_score = 1.0f / denom;
            if (face_score < score_threshold_) continue;

            float cx, cy, bw, bh;
            decodePriorBox(i, boxes_data, cx, cy, bw, bh);

            float x1 = (cx - bw * 0.5f) * input_width_;
            float y1 = (cy - bh * 0.5f) * input_height_;
            float x2 = (cx + bw * 0.5f) * input_width_;
            float y2 = (cy + bh * 0.5f) * input_height_;

            nms_indices.push_back(i);
            nms_scores.push_back(face_score);
            nms_boxes.emplace_back(
                static_cast<double>(x1), static_cast<double>(y1),
                static_cast<double>(x2 - x1), static_cast<double>(y2 - y1));
        }
    }

    void scaleResultCoords(const PreprocessContext& ctx,
                           float& x1, float& y1, float& x2, float& y2,
                           std::vector<Keypoint>& landmarks) const {
        // Coordinates are in model-input (letterboxed) space. Remove padding,
        // then divide by the resize scale. Letterbox uses a UNIFORM scale
        // (ctx.scale) — using per-axis original/input ratios here breaks the
        // padded axis (boxes misplaced, as if padding were ignored). Stretch
        // resize (no padding) sets per-axis scale_x/scale_y instead.
        float sx = (ctx.scale_x > 0.f) ? (1.0f / ctx.scale_x)
                                       : (ctx.scale > 0.f ? 1.0f / ctx.scale : 1.0f);
        float sy = (ctx.scale_y > 0.f) ? (1.0f / ctx.scale_y)
                                       : (ctx.scale > 0.f ? 1.0f / ctx.scale : 1.0f);
        float px = static_cast<float>(ctx.pad_x);
        float py = static_cast<float>(ctx.pad_y);

        x1 = (x1 - px) * sx; y1 = (y1 - py) * sy;
        x2 = (x2 - px) * sx; y2 = (y2 - py) * sy;
        for (auto& kp : landmarks) {
            kp.x = (kp.x - px) * sx;
            kp.y = (kp.y - py) * sy;
        }
    }

    void generatePriorBoxes() {
        int strides[]     = {8, 16, 32};
        int min_sizes[][2] = {{16, 32}, {64, 128}, {256, 512}};

        for (int s = 0; s < 3; ++s) {
            int stride = strides[s];
            int fh = (input_height_ + stride - 1) / stride;
            int fw = (input_width_  + stride - 1) / stride;
            for (int y = 0; y < fh; ++y) {
                for (int x = 0; x < fw; ++x) {
                    for (int k = 0; k < 2; ++k) {
                        float cx = (x + 0.5f) * stride / input_width_;
                        float cy = (y + 0.5f) * stride / input_height_;
                        float pw = static_cast<float>(min_sizes[s][k]) / input_width_;
                        float ph = static_cast<float>(min_sizes[s][k]) / input_height_;
                        priors_.push_back({cx, cy, pw, ph});
                    }
                }
            }
        }
    }

    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    float variance_[2] = {0.1f, 0.2f};
    std::vector<std::array<float, 4>> priors_;
};

}  // namespace dxapp

#endif  // RETINAFACE_POSTPROCESSOR_HPP
