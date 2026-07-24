#include "superpoint_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

const int kCellSize = 8;
const int kSemiChannels = 65;
const int kDescChannels = 256;
const float kEpsilon = 1e-6f;

struct TensorView {
    const float* data;
    int channels;
    int height;
    int width;

    TensorView() : data(nullptr), channels(0), height(0), width(0) {}
};

TensorView make_tensor_view(const dxrt::TensorPtr& tensor) {
    const std::vector<int64_t>& shape = tensor->shape();

    TensorView view;
    if (shape.size() == 4) {
        view.channels = static_cast<int>(shape[1]);
        view.height = static_cast<int>(shape[2]);
        view.width = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        view.channels = static_cast<int>(shape[0]);
        view.height = static_cast<int>(shape[1]);
        view.width = static_cast<int>(shape[2]);
    } else {
        throw std::runtime_error("SuperPointPostProcess: unexpected output shape");
    }

    view.data = static_cast<const float*>(tensor->data());
    return view;
}

/**
 * Fast approximate NMS on keypoint candidates.
 *
 * Candidates must be sorted by score descending before calling.
 * Suppresses all points within an (2*dist+1)×(2*dist+1) neighbourhood
 * (infinity-norm) around each kept point.
 *
 * Returns indices into `candidates` of surviving points (already in
 * descending score order).
 */
std::vector<size_t> nms_fast(const std::vector<SuperPointKeypoint>& candidates,
                              int H, int W, int dist_thresh) {
    // grid: 0=empty, 1=pending, -1=kept
    std::vector<int> grid(H * W, 0);
    std::vector<int> inds_grid(H * W, -1);

    // Mark pending positions (first-come = highest score since sorted)
    for (size_t i = 0; i < candidates.size(); ++i) {
        int rx = static_cast<int>(std::round(candidates[i].x));
        int ry = static_cast<int>(std::round(candidates[i].y));
        rx = std::max(0, std::min(rx, W - 1));
        ry = std::max(0, std::min(ry, H - 1));
        if (grid[ry * W + rx] == 0) {
            grid[ry * W + rx] = 1;
            inds_grid[ry * W + rx] = static_cast<int>(i);
        }
    }

    const int pad = dist_thresh;
    const int pH = H + 2 * pad;
    const int pW = W + 2 * pad;
    std::vector<int> pgrid(pH * pW, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            pgrid[(y + pad) * pW + (x + pad)] = grid[y * W + x];

    std::vector<size_t> kept;
    for (size_t i = 0; i < candidates.size(); ++i) {
        int rx = static_cast<int>(std::round(candidates[i].x));
        int ry = static_cast<int>(std::round(candidates[i].y));
        rx = std::max(0, std::min(rx, W - 1));
        ry = std::max(0, std::min(ry, H - 1));
        const int px = rx + pad;
        const int py = ry + pad;
        if (pgrid[py * pW + px] == 1) {
            for (int dy = -pad; dy <= pad; ++dy)
                for (int dx = -pad; dx <= pad; ++dx)
                    pgrid[(py + dy) * pW + (px + dx)] = 0;
            pgrid[py * pW + px] = -1;
            kept.push_back(i);
        }
    }
    return kept;
}

/**
 * Bilinear interpolation of the coarse descriptor map at a pixel location.
 * Equivalent to torch.nn.functional.grid_sample (align_corners=False convention).
 *
 * px, py: keypoint position in full-resolution pixel space
 * H, W:   full image dimensions
 */
std::vector<float> sample_desc_bilinear(const TensorView& desc,
                                         float px, float py,
                                         int H, int W) {
    // Normalise pixel coords to [-1, 1]
    const float sx = px / (W * 0.5f) - 1.0f;
    const float sy = py / (H * 0.5f) - 1.0f;

    // Map to descriptor-map coordinates
    const float xd = (sx + 1.0f) * 0.5f * (desc.width - 1);
    const float yd = (sy + 1.0f) * 0.5f * (desc.height - 1);

    const int x0 = std::max(0, std::min(static_cast<int>(xd), desc.width - 1));
    const int y0 = std::max(0, std::min(static_cast<int>(yd), desc.height - 1));
    const int x1 = std::min(x0 + 1, desc.width - 1);
    const int y1 = std::min(y0 + 1, desc.height - 1);

    const float wa = (x1 - xd) * (y1 - yd);
    const float wb = (x1 - xd) * (yd - y0);
    const float wc = (xd - x0) * (y1 - yd);
    const float wd = (xd - x0) * (yd - y0);

    const int plane = desc.height * desc.width;
    std::vector<float> d(kDescChannels);
    for (int c = 0; c < kDescChannels; ++c) {
        const float* p = desc.data + c * plane;
        d[c] = wa * p[y0 * desc.width + x0]
             + wb * p[y1 * desc.width + x0]
             + wc * p[y0 * desc.width + x1]
             + wd * p[y1 * desc.width + x1];
    }
    return d;
}

}  // namespace

SuperPointPostProcess::SuperPointPostProcess()
    : input_width_(0), input_height_(0), conf_threshold_(0.015f), top_k_(500),
      nms_dist_(4), border_remove_(4) {}

SuperPointPostProcess::SuperPointPostProcess(
    int input_w, int input_h, float conf_threshold, int top_k,
    int nms_dist, int border_remove)
    : input_width_(input_w),
      input_height_(input_h),
      conf_threshold_(conf_threshold),
      top_k_(top_k),
      nms_dist_(nms_dist),
      border_remove_(border_remove) {}

SuperPointResult SuperPointPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    TensorView semi_tensor;
    TensorView desc_tensor;
    bool has_semi = false;
    bool has_desc = false;

    for (size_t i = 0; i < outputs.size(); ++i) {
        TensorView view = make_tensor_view(outputs[i]);
        if (!has_semi && view.channels == kSemiChannels) {
            semi_tensor = view;
            has_semi = true;
        } else if (!has_desc && view.channels == kDescChannels) {
            desc_tensor = view;
            has_desc = true;
        }
    }

    SuperPointResult result;
    if (!has_semi || !has_desc) {
        return result;
    }

    const int hc = semi_tensor.height;
    const int wc = semi_tensor.width;
    const int heatmap_h = hc * kCellSize;
    const int heatmap_w = wc * kCellSize;
    const int cell_area = kCellSize * kCellSize;
    const int semi_plane = hc * wc;

    // --- Build heatmap via softmax (dustbin excluded, max-stabilised) ---
    std::vector<float> heatmap(heatmap_h * heatmap_w, 0.0f);

    for (int y = 0; y < hc; ++y) {
        for (int x = 0; x < wc; ++x) {
            const int offset = y * wc + x;
            float max_logit = semi_tensor.data[offset];
            for (int c = 1; c < cell_area; ++c) {
                const float logit = semi_tensor.data[c * semi_plane + offset];
                if (logit > max_logit) max_logit = logit;
            }

            float sum_exp = 0.0f;
            std::array<float, kCellSize * kCellSize> probs;
            for (int c = 0; c < cell_area; ++c) {
                const float p = std::exp(semi_tensor.data[c * semi_plane + offset] - max_logit);
                probs[c] = p;
                sum_exp += p;
            }

            const float denom = sum_exp + kEpsilon;
            for (int c = 0; c < cell_area; ++c) {
                const int ry = c / kCellSize;
                const int rx = c % kCellSize;
                heatmap[(y * kCellSize + ry) * heatmap_w + (x * kCellSize + rx)] = probs[c] / denom;
            }
        }
    }

    // --- Threshold → candidate list ---
    std::vector<SuperPointKeypoint> candidates;
    candidates.reserve(512);
    for (int hy = 0; hy < heatmap_h; ++hy) {
        for (int hx = 0; hx < heatmap_w; ++hx) {
            const float score = heatmap[hy * heatmap_w + hx];
            if (score >= conf_threshold_) {
                candidates.emplace_back(static_cast<float>(hx),
                                        static_cast<float>(hy),
                                        score);
            }
        }
    }

    if (candidates.empty()) return result;

    // --- Sort by score descending ---
    std::sort(candidates.begin(), candidates.end(),
              [](const SuperPointKeypoint& a, const SuperPointKeypoint& b) {
                  return a.score > b.score;
              });

    // --- NMS ---
    std::vector<size_t> kept = nms_fast(candidates, heatmap_h, heatmap_w, nms_dist_);

    // --- Border removal ---
    const int bord = border_remove_;
    std::vector<size_t> final_inds;
    final_inds.reserve(kept.size());
    for (size_t idx : kept) {
        const float px = candidates[idx].x;
        const float py = candidates[idx].y;
        if (px >= bord && px < heatmap_w - bord &&
            py >= bord && py < heatmap_h - bord) {
            final_inds.push_back(idx);
        }
    }

    // --- Top-K ---
    const size_t keep_count = (top_k_ < 0)
        ? final_inds.size()
        : static_cast<size_t>(top_k_);
    if (final_inds.size() > keep_count) {
        final_inds.resize(keep_count);
    }

    result.keypoints.reserve(final_inds.size());
    result.descriptors.reserve(final_inds.size());

    // --- Bilinear descriptor interpolation + L2 normalisation ---
    for (size_t idx : final_inds) {
        const SuperPointKeypoint& kp = candidates[idx];

        std::vector<float> descriptor = sample_desc_bilinear(
            desc_tensor, kp.x, kp.y, heatmap_h, heatmap_w);

        float norm_sq = 0.0f;
        for (int d = 0; d < kDescChannels; ++d) norm_sq += descriptor[d] * descriptor[d];
        const float inv_norm = 1.0f / (std::sqrt(norm_sq) + kEpsilon);
        for (int d = 0; d < kDescChannels; ++d) descriptor[d] *= inv_norm;

        result.keypoints.push_back(kp);
        result.descriptors.push_back(std::move(descriptor));
    }

    return result;
}
