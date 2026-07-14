#include "dope_postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

constexpr int DOPEPostProcess::NUM_BELIEFS;

DOPEPostProcess::DOPEPostProcess()
    : input_width_(640), input_height_(480) {}

DOPEPostProcess::DOPEPostProcess(int input_w, int input_h)
    : input_width_(input_w), input_height_(input_h) {}

// 3x3 belief-weighted sub-pixel centroid refinement around (iy, ix).
// Mirrors DopeDecode._subpixel() from dx-modelzoo dope/custom_ops.py.
static std::pair<float, float> subpixel_refine(const float* hm, int h, int w, int iy, int ix) {
    const int y0 = std::max(iy - 1, 0), y1 = std::min(iy + 2, h);
    const int x0 = std::max(ix - 1, 0), x1 = std::min(ix + 2, w);
    float s = 0.0f, cx = 0.0f, cy = 0.0f;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const float v = std::max(0.0f, hm[y * w + x]);
            s  += v;
            cx += v * static_cast<float>(x);
            cy += v * static_cast<float>(y);
        }
    }
    if (s <= 1e-9f) return {static_cast<float>(ix), static_cast<float>(iy)};
    return {cx / s, cy / s};
}

DopeResult DOPEPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error("DOPEPostProcess: no output tensors");
    }

    // dxrt returns a single merged tensor [1, 25, H, W]:
    //   channels  0– 8: 9 belief maps (8 vertices + centroid) — conv2d_84
    //   channels  9–24: 16 affinity fields                    — conv2d_91 (unused)
    const auto& t0    = outputs[0];
    const auto& shape = t0->shape();

    int total_ch, hm_h, hm_w;
    if (shape.size() == 4) {
        total_ch = static_cast<int>(shape[1]);
        hm_h     = static_cast<int>(shape[2]);
        hm_w     = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        total_ch = static_cast<int>(shape[0]);
        hm_h     = static_cast<int>(shape[1]);
        hm_w     = static_cast<int>(shape[2]);
    } else {
        throw std::runtime_error("DOPEPostProcess: unexpected tensor rank");
    }

    if (total_ch < NUM_BELIEFS) {
        throw std::runtime_error("DOPEPostProcess: output has fewer than 9 channels");
    }

    const float* data  = static_cast<const float*>(t0->data());
    const int    plane = hm_h * hm_w;

    DopeResult result;
    result.peaks.reserve(NUM_BELIEFS);

    for (int k = 0; k < NUM_BELIEFS; ++k) {
        const float* hm  = data + k * plane;
        const int flat_idx = static_cast<int>(std::max_element(hm, hm + plane) - hm);
        const int iy = flat_idx / hm_w;
        const int ix = flat_idx % hm_w;
        // Sub-pixel refinement: 3x3 belief-weighted centroid
        auto [rfx, rfy] = subpixel_refine(hm, hm_h, hm_w, iy, ix);
        // +0.5 pixel-center offset → heatmap-space coords (not yet image-space)
        result.peaks.emplace_back(rfx + 0.5f, rfy + 0.5f, hm[flat_idx]);
    }

    return result;
}
