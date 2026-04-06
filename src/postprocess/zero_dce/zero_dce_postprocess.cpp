#include "zero_dce_postprocess.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

ZeroDCEPostProcess::ZeroDCEPostProcess(int input_w, int input_h)
    : input_width_(input_w), input_height_(input_h) {}

ZeroDCEPostProcess::ZeroDCEPostProcess()
    : input_width_(256), input_height_(256) {}

ZeroDCEResult ZeroDCEPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "[DXAPP] [ER] ZeroDCEPostProcess::postprocess - No output tensors provided.");
    }

    const auto& output = outputs[0];
    const auto& shape = output->shape();

    // Expected shape: [1, 3, H, W] for RGB enhanced image
    int c = 3, h, w;
    if (shape.size() == 4) {
        c = static_cast<int>(shape[1]);
        h = static_cast<int>(shape[2]);
        w = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        c = static_cast<int>(shape[0]);
        h = static_cast<int>(shape[1]);
        w = static_cast<int>(shape[2]);
    } else {
        h = input_height_;
        w = input_width_;
    }

    const float* data = static_cast<const float*>(output->data());
    int total = c * h * w;

    std::vector<float> result_image(total);
    for (int i = 0; i < total; ++i) {
        result_image[i] = std::max(0.0f, std::min(1.0f, data[i]));
    }

    return ZeroDCEResult(std::move(result_image), h, w, c);
}
