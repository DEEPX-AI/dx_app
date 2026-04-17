#include "face3d_postprocess.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

Face3DPostProcess::Face3DPostProcess(int input_w, int input_h)
    : input_width_(input_w), input_height_(input_h) {}

Face3DPostProcess::Face3DPostProcess()
    : input_width_(120), input_height_(120) {}

Face3DResult Face3DPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "[DXAPP] [ER] Face3DPostProcess::postprocess - No output tensors provided.");
    }

    const auto& output = outputs[0];
    const auto& shape = output->shape();

    // Expected shape: [1, 62] for 3DMM parameters (62 = 12 pose + 40 shape + 10 expression)
    int num_params = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i == 0 && shape.size() > 1) continue; // skip batch
        num_params *= static_cast<int>(shape[i]);
    }

    const float* data = static_cast<const float*>(output->data());
    std::vector<float> params(data, data + num_params);

    return Face3DResult(std::move(params), num_params);
}
