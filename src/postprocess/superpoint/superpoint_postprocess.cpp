#include "superpoint_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
 * Suppresses all points within an (2*dist+1)x(2*dist+1) neighbourhood
 * (infinity-norm) around each kept point.
 *
 * Returns indices into `candidates` of surviving points (already in
 * descending score order).
 *
 * `top_k` / `border_remove` let the scan stop early: the caller applies border
 * removal then top-k straight after this call, and both this scan and the
 * output are ordered by descending score, so once `top_k` border-passing points
 * are kept no later candidate can still reach the output. Pass top_k <= 0 to
 * scan every candidate. `n_inside` reports how many border-passing points were
 * found, so the caller can detect a truncated candidate list and retry.
 */
std::vector<size_t> nms_fast(const std::vector<SuperPointKeypoint>& candidates,
                              int H, int W, int dist_thresh,
                              int top_k, int border_remove, int* n_inside) {
    const int pad = dist_thresh;
    const int pW = W + 2 * pad;
    const int span = 2 * pad + 1;

    // Single pre-padded occupancy grid: 0 = empty/suppressed, 1 = alive,
    // -1 = kept. Padding removes the per-point clamp on the suppression box,
    // and signed char keeps the (2*dist+1)^2 clear cheap.
    std::vector<signed char> grid(static_cast<size_t>(H + 2 * pad) * pW, 0);
    for (size_t i = 0; i < candidates.size(); ++i) {
        int rx = std::max(0, std::min(static_cast<int>(std::round(candidates[i].x)), W - 1));
        int ry = std::max(0, std::min(static_cast<int>(std::round(candidates[i].y)), H - 1));
        signed char& cell = grid[static_cast<size_t>(ry + pad) * pW + (rx + pad)];
        if (cell == 0) cell = 1;
    }

    std::vector<size_t> kept;
    kept.reserve(top_k > 0 ? static_cast<size_t>(top_k) + 64 : 4096);

    const int hi_x = W - border_remove;
    const int hi_y = H - border_remove;
    int inside = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        int rx = std::max(0, std::min(static_cast<int>(std::round(candidates[i].x)), W - 1));
        int ry = std::max(0, std::min(static_cast<int>(std::round(candidates[i].y)), H - 1));
        const int px = rx + pad;
        const int py = ry + pad;
        if (grid[static_cast<size_t>(py) * pW + px] == 1) {
            for (int dy = -pad; dy <= pad; ++dy) {
                std::memset(&grid[static_cast<size_t>(py + dy) * pW + (px - pad)], 0, span);
            }
            grid[static_cast<size_t>(py) * pW + px] = -1;
            kept.push_back(i);
            if (rx >= border_remove && rx < hi_x && ry >= border_remove && ry < hi_y) {
                ++inside;
                if (top_k > 0 && inside >= top_k) break;
            }
        }
    }
    if (n_inside) *n_inside = inside;
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
      nms_dist_(4), border_remove_(kCellSize) {}

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
    // Walked one cell-row at a time so every channel read is contiguous in x.
    // The previous cell-major order gathered 64 values a full plane apart,
    // touching a separate cache line per channel. Numerically identical.
    std::vector<float> heatmap(heatmap_h * heatmap_w, 0.0f);
    {
        std::vector<float> tile(static_cast<size_t>(cell_area) * wc);
        std::vector<float> max_logit(wc);
        std::vector<float> sum_exp(wc);

        for (int y = 0; y < hc; ++y) {
            const float* base = semi_tensor.data + y * wc;

            for (int x = 0; x < wc; ++x) max_logit[x] = base[x];
            for (int c = 1; c < cell_area; ++c) {
                const float* row = base + static_cast<size_t>(c) * semi_plane;
                for (int x = 0; x < wc; ++x) {
                    if (row[x] > max_logit[x]) max_logit[x] = row[x];
                }
            }

            for (int x = 0; x < wc; ++x) sum_exp[x] = 0.0f;
            for (int c = 0; c < cell_area; ++c) {
                const float* row = base + static_cast<size_t>(c) * semi_plane;
                float* t = &tile[static_cast<size_t>(c) * wc];
                for (int x = 0; x < wc; ++x) {
                    const float p = std::exp(row[x] - max_logit[x]);
                    t[x] = p;
                    sum_exp[x] += p;
                }
            }

            for (int c = 0; c < cell_area; ++c) {
                const float* t = &tile[static_cast<size_t>(c) * wc];
                float* dst = &heatmap[static_cast<size_t>(y * kCellSize + c / kCellSize) * heatmap_w
                                      + (c % kCellSize)];
                for (int x = 0; x < wc; ++x) {
                    dst[static_cast<size_t>(x) * kCellSize] = t[x] / (sum_exp[x] + kEpsilon);
                }
            }
        }
    }

    // --- Threshold → candidate list ---
    // The default threshold (0.015) sits at the mean of a 64-way softmax, so
    // roughly a fifth of all pixels qualify -- reserve for that, not for 512,
    // which used to force ~7 reallocations of a 60k-element vector.
    std::vector<SuperPointKeypoint> candidates;
    candidates.reserve(static_cast<size_t>(heatmap_h) * heatmap_w / 3);
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

    // Descending score, ties broken by ascending y then x. Candidates are built
    // in row-major order, so this is the order a stable sort would give -- and
    // it is the same rule the Python postprocessor uses, which keeps the two
    // implementations byte-comparable. Without an explicit tie-break the result
    // would depend on std::sort's (unspecified) handling of equal elements,
    // which NPU quantisation makes common: identical scores cluster.
    struct ByScoreThenPosition {
        bool operator()(const SuperPointKeypoint& a, const SuperPointKeypoint& b) const {
            if (a.score != b.score) return a.score > b.score;
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        }
    };
    const ByScoreThenPosition cmp;

    // Only the highest-scoring candidates can survive NMS into the top-k, so
    // rank just the head of the list instead of sorting all ~60k. Measured
    // depth needed for top_k=500 is under 900; kHeadCandidates leaves headroom,
    // and the fallback below re-runs on the full list if a dense scene ever
    // exhausts the head.
    //
    // INVARIANT: whichever branch runs, `kept` ends up holding indices into
    // `candidates`, so everything downstream can read candidates[kept[i]]
    // without knowing which path was taken. The two branches differ only in
    // what `candidates` contains by then -- see below.
    const size_t kHeadCandidates = 2048;
    std::vector<size_t> kept;
    int n_inside = 0;
    bool head_sufficed = false;

    if (top_k_ > 0 && candidates.size() > kHeadCandidates
        && kHeadCandidates > static_cast<size_t>(top_k_)) {
        // nth_element only permutes, so candidates keeps every point for the
        // fallback; the copy below is just the 2048-element head.
        std::nth_element(candidates.begin(), candidates.begin() + kHeadCandidates,
                         candidates.end(), cmp);
        std::vector<SuperPointKeypoint> head(candidates.begin(),
                                             candidates.begin() + kHeadCandidates);
        std::sort(head.begin(), head.end(), cmp);

        kept = nms_fast(head, heatmap_h, heatmap_w, nms_dist_, top_k_, border_remove_, &n_inside);
        if (n_inside >= top_k_) {
            // The head held every point that could reach the output, so the
            // rest of the list is dead weight: swap it in as `candidates` and
            // let the full list die with `head` at the end of this block. That
            // re-points kept's indices at candidates, restoring the invariant.
            candidates.swap(head);
            head_sufficed = true;
        }
        // Otherwise `candidates` still holds every point (permuted, not
        // truncated, by nth_element) and the stale `kept` from this attempt is
        // overwritten below -- never appended to.
    }

    if (!head_sufficed) {
        std::sort(candidates.begin(), candidates.end(), cmp);
        kept = nms_fast(candidates, heatmap_h, heatmap_w, nms_dist_,
                        top_k_, border_remove_, &n_inside);
    }

    // --- Border removal ---
    // Defaults to one full cell (kCellSize), not the reference implementation's
    // 4 px. The model collapses each border cell's 64-way softmax onto its first
    // sub-pixel, so the last cell row/column carries a ridge ~7x the image mean
    // while the 7 sub-rows behind it are dead. A 4 px margin trims only the dead
    // part and leaves the ridge, which was filling 15-22% of the top-k with
    // points along the image edge rather than real corners.
    //
    // The trailing/right edges lose nothing real: behind the ridge the model
    // emits no candidates at all. The leading/top-left rows 1..7 do carry
    // signal, but recovering them is a net loss because top_k is a fixed
    // budget -- measured at top_k=500, border_remove of 1/2/4 admitted 72-93
    // edge points and evicted the same number of better interior ones, taking
    // corner-response quality from 10.8x a random-position control down to
    // 9.1x and 5.0x on two scenes. The cost of keeping 8 is a 24x18 px band
    // (5.8% of a 1920x1080 frame); that is accepted for visualisation. Genuine
    // edge coverage needs overlapping tiles, not a smaller margin.
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
