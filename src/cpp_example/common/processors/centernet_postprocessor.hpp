/**
 * @file centernet_postprocessor.hpp
 * @brief CenterNet (Objects as Points) detection postprocessor
 * 
 * Ported from Python centernet_postprocessor.py.
 * 
 * CenterNet outputs 3 heatmap-based tensors:
 *   - [1, C, H/4, W/4]   center heatmap (class-wise)
 *   - [1, 2, H/4, W/4]   size regression (w, h)
 *   - [1, 2, H/4, W/4]   center offset (dx, dy)
 * 
 * Algorithm:
 *   1. Identify tensors by channel count (C channels → heatmap, 2 ch → size/offset)
 *   2. Pseudo-NMS on heatmap (3x3 max-pool, keep peaks only)
 *   3. Find top-K scoring centers across all classes
 *   4. Decode: cx = (x + offset_dx) * stride, cy = (y + offset_dy) * stride
 *   5. Size: w = size_w * stride, h = size_h * stride
 *   6. NMS with cv::dnn::NMSBoxes
 *   7. Scale to original coordinates
 */

#ifndef CENTERNET_POSTPROCESSOR_HPP
#define CENTERNET_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dxapp {

class CenterNetPostprocessor : public IPostprocessor<DetectionResult> {
public:
    CenterNetPostprocessor(int input_width = 512, int input_height = 512,
                           float score_threshold = 0.3f,
                           float nms_threshold = 0.5f,
                           int num_classes = 80,
                           int stride = 4,
                           int top_k = 100)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          num_classes_(num_classes), stride_(stride), top_k_(top_k) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        if (outputs.size() < 3) return results;

        const dxrt::TensorPtr* heatmap_t = nullptr;
        const dxrt::TensorPtr* size_t_ptr = nullptr;
        const dxrt::TensorPtr* offset_t = nullptr;
        if (!identifyTensors_(outputs, heatmap_t, size_t_ptr, offset_t)) return results;

        auto shape = (*heatmap_t)->shape();
        int C = static_cast<int>(shape[1]);
        int H = static_cast<int>(shape[2]);
        int W = static_cast<int>(shape[3]);

        const float* heatmap = static_cast<const float*>((*heatmap_t)->data());
        const float* sizes   = static_cast<const float*>((*size_t_ptr)->data());
        const float* offsets = static_cast<const float*>((*offset_t)->data());

        auto heatmap_nms = computePseudoNms_(heatmap, C, H, W);
        auto peaks       = extractTopKPeaks_(heatmap_nms, C, H, W);

        // Decode boxes
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;
        std::vector<std::array<float, 4>> float_boxes;

        for (auto& p : peaks) {
            float off_x = offsets[0 * H * W + p.y * W + p.x];
            float off_y = offsets[1 * H * W + p.y * W + p.x];
            float w = sizes[0 * H * W + p.y * W + p.x];
            float h = sizes[1 * H * W + p.y * W + p.x];

            float cx = (p.x + off_x) * stride_;
            float cy = (p.y + off_y) * stride_;
            float bw = w * stride_;
            float bh = h * stride_;

            float x1 = cx - bw * 0.5f;
            float y1 = cy - bh * 0.5f;
            float x2 = cx + bw * 0.5f;
            float y2 = cy + bh * 0.5f;

            nms_boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            float_boxes.push_back({x1, y1, x2, y2});
            nms_scores.push_back(p.score);
            nms_class_ids.push_back(p.cls);
        }

        if (nms_boxes.empty()) return results;

        std::vector<int> indices;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, indices);

        for (int idx : indices) {
            auto coords = scaleBoxCoords_(
                float_boxes[idx][0], float_boxes[idx][1],
                float_boxes[idx][2], float_boxes[idx][3], ctx);
            DetectionResult det;
            det.box = {coords[0], coords[1], coords[2], coords[3]};
            det.confidence = nms_scores[idx];
            det.class_id = nms_class_ids[idx];
            det.class_name = dxapp::common::get_coco_class_name(det.class_id);
            results.push_back(det);
        }

        return results;
    }

    std::string getModelName() const override { return "CenterNet"; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    int num_classes_;
    int stride_;
    int top_k_;

    struct Peak_ { float score; int cls; int y; int x; };

    // Check whether the pixel at (c, y, x) is a strict local maximum in a 3×3
    // neighbourhood of the C×H×W heatmap (zero-padded at borders).
    static bool isLocalMax3x3_(const float* heatmap, int c, int y, int x, int H, int W) {
        float center = heatmap[c * H * W + y * W + x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dy == 0 && dx == 0) continue;
                int ny = y + dy;
                int nx = x + dx;
                if (ny >= 0 && ny < H && nx >= 0 && nx < W &&
                    heatmap[c * H * W + ny * W + nx] > center) return false;
            }
        }
        return true;
    }

    // Categorise outputs into heatmap / size / offset tensors.
    // Returns false when mandatory tensors are missing.
    bool identifyTensors_(const dxrt::TensorPtrs& outputs,
                          const dxrt::TensorPtr*& heatmap_t,
                          const dxrt::TensorPtr*& size_t_ptr,
                          const dxrt::TensorPtr*& offset_t) const {
        std::vector<const dxrt::TensorPtr*> two_ch;
        for (auto& t : outputs) {
            auto sh = t->shape();
            if (sh.size() < 3) continue;
            int ch = static_cast<int>(sh.size() == 4 ? sh[1] : sh[0]);
            if (ch > 2) heatmap_t = &t;
            else if (ch == 2) two_ch.push_back(&t);
        }
        if (!heatmap_t || two_ch.size() < 2) return false;

        auto sh = (*heatmap_t)->shape();
        int H = static_cast<int>(sh[2]), W = static_cast<int>(sh[3]);
        const float* d0 = static_cast<const float*>((*two_ch[0])->data());
        const float* d1 = static_cast<const float*>((*two_ch[1])->data());
        float s0 = 0, s1 = 0;
        int n = std::min(2 * H * W, 1000);
        for (int i = 0; i < n; ++i) { s0 += std::abs(d0[i]); s1 += std::abs(d1[i]); }
        if (s0 >= s1) { size_t_ptr = two_ch[0]; offset_t = two_ch[1]; }
        else           { size_t_ptr = two_ch[1]; offset_t = two_ch[0]; }
        return true;
    }

    // 3×3 max-pool pseudo-NMS on a flat C×H×W heatmap.
    std::vector<float> computePseudoNms_(const float* heatmap, int C, int H, int W) const {
        std::vector<float> nms(C * H * W, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int idx = c * H * W + y * W + x;
                    nms[idx] = isLocalMax3x3_(heatmap, c, y, x, H, W) ? heatmap[idx] : 0.0f;
                }
            }
        }
        return nms;
    }

    // Collect above-threshold peaks and return the top-K sorted by score.
    std::vector<Peak_> extractTopKPeaks_(const std::vector<float>& nms,
                                          int C, int H, int W) const {
        std::vector<Peak_> peaks;
        peaks.reserve(C * H * W / 4);
        int total = C * H * W;
        for (int i = 0; i < total; ++i) {
            float s = nms[i];
            if (s <= score_threshold_) continue;
            int c = i / (H * W);
            int rem = i % (H * W);
            int y = rem / W;
            int x = rem % W;
            peaks.push_back({s, c, y, x});
        }
        std::sort(peaks.begin(), peaks.end(),
                  [](const Peak_& a, const Peak_& b) { return a.score > b.score; });
        if ((int)peaks.size() > top_k_) peaks.resize(top_k_);
        return peaks;
    }

    // Scale and clamp a single box to original-image coordinates.
    std::array<float, 4> scaleBoxCoords_(float x1, float y1, float x2, float y2,
                                          const PreprocessContext& ctx) const {
        float ow = static_cast<float>(ctx.original_width);
        float oh = static_cast<float>(ctx.original_height);
        if (ctx.pad_x == 0 && ctx.pad_y == 0) {
            float sx = ow / input_width_, sy = oh / input_height_;
            return { std::max(0.0f, std::min(x1 * sx, ow)),
                     std::max(0.0f, std::min(y1 * sy, oh)),
                     std::max(0.0f, std::min(x2 * sx, ow)),
                     std::max(0.0f, std::min(y2 * sy, oh)) };
        }
        return { std::max(0.0f, std::min((x1 - ctx.pad_x) / ctx.scale, ow)),
                 std::max(0.0f, std::min((y1 - ctx.pad_y) / ctx.scale, oh)),
                 std::max(0.0f, std::min((x2 - ctx.pad_x) / ctx.scale, ow)),
                 std::max(0.0f, std::min((y2 - ctx.pad_y) / ctx.scale, oh)) };
    }
};

}  // namespace dxapp

#endif  // CENTERNET_POSTPROCESSOR_HPP
