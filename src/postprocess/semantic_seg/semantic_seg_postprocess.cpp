#include "semantic_seg_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

SemanticSegPostProcess::SemanticSegPostProcess(int input_w, int input_h, int num_classes)
    : input_width_(input_w), input_height_(input_h), num_classes_(num_classes) {}

SemanticSegPostProcess::SemanticSegPostProcess()
    : input_width_(640), input_height_(640), num_classes_(0) {}

SemanticSegResult SemanticSegPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        return SemanticSegResult();
    }

    const auto& shape = outputs[0]->shape();
    const float* data = static_cast<const float*>(outputs[0]->data());

    // Determine layout and dimensions
    // NCHW: [1, C, H, W] or [C, H, W]
    // NHWC: [1, H, W, C] or [H, W, C]
    int C, H, W;
    bool is_nchw;

    if (shape.size() == 4) {
        // [1, ?, ?, ?] — determine NCHW vs NHWC
        int dim1 = static_cast<int>(shape[1]);
        int dim2 = static_cast<int>(shape[2]);
        int dim3 = static_cast<int>(shape[3]);

        // Heuristic: if dim1 is small (< 256) and dim2/dim3 are larger, it's NCHW
        // If dim3 is small (< 256) and dim1/dim2 are larger, it's NHWC
        if (dim1 <= dim3 && dim1 < 256) {
            // NCHW: [1, C, H, W]
            C = dim1;
            H = dim2;
            W = dim3;
            is_nchw = true;
        } else {
            // NHWC: [1, H, W, C]
            H = dim1;
            W = dim2;
            C = dim3;
            is_nchw = false;
        }
    } else if (shape.size() == 3) {
        // [C, H, W] or [H, W, C]
        int dim0 = static_cast<int>(shape[0]);
        int dim1 = static_cast<int>(shape[1]);
        int dim2 = static_cast<int>(shape[2]);

        if (dim0 <= dim2 && dim0 < 256) {
            C = dim0;
            H = dim1;
            W = dim2;
            is_nchw = true;
        } else {
            H = dim0;
            W = dim1;
            C = dim2;
            is_nchw = false;
        }
    } else {
        std::cerr << "SemanticSegPostProcess: unsupported tensor rank " << shape.size() << std::endl;
        return SemanticSegResult();
    }

    if (num_classes_ > 0 && C != num_classes_) {
        std::cerr << "SemanticSegPostProcess: expected " << num_classes_
                  << " classes but got " << C << std::endl;
    }

    std::vector<int> class_map;
    if (is_nchw) {
        class_map = apply_argmax_nchw(data, C, H, W);
    } else {
        class_map = apply_argmax_nhwc(data, H, W, C);
    }

    SemanticSegResult result(class_map, W, H);
    result.num_classes = C;
    return result;
}

std::vector<int> SemanticSegPostProcess::apply_argmax_nchw(
    const float* data, int C, int H, int W) const {
    std::vector<int> class_map(H * W);

    auto argmax_nchw = [&](int pixel_idx) {
        int best = 0;
        float best_val = data[pixel_idx];
        for (int c = 1; c < C; ++c) {
            float v = data[c * H * W + pixel_idx];
            if (v > best_val) { best_val = v; best = c; }
        }
        return best;
    };

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            class_map[h * W + w] = argmax_nchw(h * W + w);
        }
    }
    return class_map;
}

std::vector<int> SemanticSegPostProcess::apply_argmax_nhwc(
    const float* data, int H, int W, int C) const {
    std::vector<int> class_map(H * W);

    auto argmax_nhwc = [&](const float* pixel) {
        int best = 0;
        float best_val = pixel[0];
        for (int c = 1; c < C; ++c) {
            if (pixel[c] > best_val) { best_val = pixel[c]; best = c; }
        }
        return best;
    };

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            class_map[h * W + w] = argmax_nhwc(data + (h * W + w) * C);
        }
    }
    return class_map;
}
