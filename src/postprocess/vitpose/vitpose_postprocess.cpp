#include "vitpose_postprocess.h"

#include <algorithm>
#include <stdexcept>

VitPosePostProcess::VitPosePostProcess()
    : input_width_(192), input_height_(256) {}

VitPosePostProcess::VitPosePostProcess(int input_w, int input_h)
    : input_width_(input_w), input_height_(input_h) {}

VitPoseResult VitPosePostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error("VitPosePostProcess: no output tensors");
    }

    const auto& tensor = outputs[0];
    const auto& shape = tensor->shape();

    int num_kp, hm_h, hm_w;
    if (shape.size() == 4) {
        num_kp = static_cast<int>(shape[1]);
        hm_h   = static_cast<int>(shape[2]);
        hm_w   = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        num_kp = static_cast<int>(shape[0]);
        hm_h   = static_cast<int>(shape[1]);
        hm_w   = static_cast<int>(shape[2]);
    } else {
        throw std::runtime_error("VitPosePostProcess: unexpected output shape");
    }

    const float* data = static_cast<const float*>(tensor->data());
    const int plane = hm_h * hm_w;

    VitPoseResult result;
    result.keypoints.reserve(num_kp);

    for (int k = 0; k < num_kp; ++k) {
        const float* hm = data + k * plane;
        int flat_idx = static_cast<int>(
            std::max_element(hm, hm + plane) - hm);
        int y_hm = flat_idx / hm_w;
        int x_hm = flat_idx % hm_w;
        float conf = hm[flat_idx];
        result.keypoints.emplace_back(
            static_cast<float>(x_hm),
            static_cast<float>(y_hm),
            conf);
    }

    return result;
}
