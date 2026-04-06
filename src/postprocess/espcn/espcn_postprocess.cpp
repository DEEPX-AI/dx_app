#include "espcn_postprocess.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

ESPCNPostProcess::ESPCNPostProcess(int input_w, int input_h, int scale_factor)
    : input_width_(input_w), input_height_(input_h), scale_factor_(scale_factor) {}

ESPCNPostProcess::ESPCNPostProcess()
    : input_width_(0), input_height_(0), scale_factor_(2) {}

ESPCNResult ESPCNPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "[DXAPP] [ER] ESPCNPostProcess::postprocess - No output tensors provided.");
    }

    const auto& output = outputs[0];
    const auto& shape = output->shape();

    // Expected shape: [1, C, H, W] where C=1 for Y-channel
    int c = 1, h, w;
    if (shape.size() == 4) {
        c = static_cast<int>(shape[1]);
        h = static_cast<int>(shape[2]);
        w = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        c = static_cast<int>(shape[0]);
        h = static_cast<int>(shape[1]);
        w = static_cast<int>(shape[2]);
    } else if (shape.size() == 2) {
        h = static_cast<int>(shape[0]);
        w = static_cast<int>(shape[1]);
    } else {
        h = input_height_ * scale_factor_;
        w = input_width_ * scale_factor_;
    }

    const float* data = static_cast<const float*>(output->data());
    int total = c * h * w;

    std::vector<float> result_image(total);
    for (int i = 0; i < total; ++i) {
        result_image[i] = std::max(0.0f, std::min(1.0f, data[i]));
    }

    return ESPCNResult(std::move(result_image), h, w, c);
}
