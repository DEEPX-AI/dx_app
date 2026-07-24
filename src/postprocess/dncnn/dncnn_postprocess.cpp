#include "dncnn_postprocess.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

// Constructor
DnCNNPostProcess::DnCNNPostProcess(int input_w, int input_h)
    : input_width_(input_w), input_height_(input_h) {}

// Default constructor
DnCNNPostProcess::DnCNNPostProcess()
    : input_width_(50), input_height_(50) {}

// Process model outputs
DnCNNResult DnCNNPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] DnCNNPostProcess::postprocess - No output tensors provided.");
    }

    const auto& output = outputs[0];
    const auto& shape = output->shape();

    // Expected shape: [1, C, H, W] where C is typically 1 for grayscale
    if (shape.size() < 3) {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] DnCNNPostProcess::postprocess - Unexpected output shape: (";
        for (size_t i = 0; i < shape.size(); ++i) {
            msg << shape[i];
            if (i != shape.size() - 1) msg << ", ";
        }
        msg << "). Expected at least 3 dimensions [C, H, W] or [1, C, H, W].";
        throw std::runtime_error(msg.str());
    }

    // Determine C, H and W from shape. Output layout is [1, C, H, W] (NCHW).
    int c, h, w;
    if (shape.size() == 4) {
        c = static_cast<int>(shape[1]);
        h = static_cast<int>(shape[2]);
        w = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        c = static_cast<int>(shape[0]);
        h = static_cast<int>(shape[1]);
        w = static_cast<int>(shape[2]);
    } else {
        c = 1;
        h = input_height_;
        w = input_width_;
    }

    const float* data = static_cast<const float*>(output->data());
    int total = c * h * w;

    // Clip every channel to [0, 1]. Multi-channel (e.g. color DnCNN) output is
    // preserved in CHW order so the caller can reconstruct the color image.
    std::vector<float> result_image(total);
    for (int i = 0; i < total; ++i) {
        result_image[i] = std::max(0.0f, std::min(1.0f, data[i]));
    }

    return DnCNNResult(std::move(result_image), h, w, c);
}
