#include "depth_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>

DepthPostProcess::DepthPostProcess(int input_w, int input_h)
    : input_width_(input_w), input_height_(input_h) {}

DepthPostProcess::DepthPostProcess()
    : input_width_(224), input_height_(224) {}

DepthResult DepthPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        return DepthResult();
    }

    const auto& shape = outputs[0]->shape();
    const float* data = static_cast<const float*>(outputs[0]->data());

    // Determine H, W from shape
    // Supports [1, 1, H, W], [1, H, W], [H, W]
    int H, W;
    if (shape.size() == 4) {
        H = static_cast<int>(shape[2]);
        W = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        H = static_cast<int>(shape[1]);
        W = static_cast<int>(shape[2]);
    } else if (shape.size() == 2) {
        H = static_cast<int>(shape[0]);
        W = static_cast<int>(shape[1]);
    } else {
        std::cerr << "DepthPostProcess: unsupported tensor rank " << shape.size() << std::endl;
        return DepthResult();
    }

    int total = H * W;

    // Find min/max for normalization
    float d_min = data[0];
    float d_max = data[0];
    for (int i = 1; i < total; ++i) {
        if (data[i] < d_min) d_min = data[i];
        if (data[i] > d_max) d_max = data[i];
    }

    // Normalize to 0-255
    std::vector<uint8_t> depth_map(total);
    float range = d_max - d_min;
    if (range > 1e-6f) {
        for (int i = 0; i < total; ++i) {
            float normalized = (data[i] - d_min) / range * 255.0f;
            depth_map[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, normalized)));
        }
    } else {
        std::fill(depth_map.begin(), depth_map.end(), 0);
    }

    return DepthResult(depth_map, W, H);
}
