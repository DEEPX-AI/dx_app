#include "realesrgan_postprocess.h"

#include <algorithm>
#include <stdexcept>

RealESRGANPostProcess::RealESRGANPostProcess()
    : input_width_(0), input_height_(0), scale_factor_(4) {}

RealESRGANPostProcess::RealESRGANPostProcess(int input_w, int input_h, int scale_factor)
    : input_width_(input_w), input_height_(input_h), scale_factor_(scale_factor) {}

RealESRGANResult RealESRGANPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error("RealESRGANPostProcess: no output tensors");
    }

    const auto& tensor = outputs[0];
    const auto& shape = tensor->shape();

    int out_c = 3;
    int out_h = input_height_ * scale_factor_;
    int out_w = input_width_ * scale_factor_;

    if (shape.size() == 4) {
        out_c = static_cast<int>(shape[1]);
        out_h = static_cast<int>(shape[2]);
        out_w = static_cast<int>(shape[3]);
    } else if (shape.size() == 3) {
        out_c = static_cast<int>(shape[0]);
        out_h = static_cast<int>(shape[1]);
        out_w = static_cast<int>(shape[2]);
    } else {
        throw std::runtime_error("RealESRGANPostProcess: unexpected output shape");
    }

    const float* data = static_cast<const float*>(tensor->data());
    const int plane_size = out_h * out_w;
    std::vector<float> image(out_h * out_w * out_c);

    for (int c = 0; c < out_c; ++c) {
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                float value = data[c * plane_size + h * out_w + w];
                value = std::max(0.0f, std::min(1.0f, value));
                image[(h * out_w + w) * out_c + c] = value;
            }
        }
    }

    return RealESRGANResult(std::move(image), out_h, out_w, out_c);
}
