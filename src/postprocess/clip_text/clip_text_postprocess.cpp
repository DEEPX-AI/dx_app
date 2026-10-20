#include "clip_text_postprocess.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace {

std::string format_shapes(const dxrt::TensorPtrs& outputs) {
    std::ostringstream s;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        s << (i ? ", " : "") << "(";
        const auto& shape = outputs[i]->shape();
        for (std::size_t d = 0; d < shape.size(); ++d) {
            s << (d ? ", " : "") << shape[d];
        }
        s << ")";
    }
    return s.str();
}

std::size_t element_count(const dxrt::TensorPtr& tensor) {
    std::size_t n = 1;
    for (int64_t d : tensor->shape()) n *= static_cast<std::size_t>(d);
    return n;
}

// Index of the largest token id, whatever integer or float type it is stored as.
std::size_t argmax_token(const dxrt::TensorPtr& ids) {
    const std::size_t n = element_count(ids);
    std::size_t best = 0;
    double best_value = 0.0;
    auto scan = [&](auto* data) {
        best_value = static_cast<double>(data[0]);
        for (std::size_t i = 1; i < n; ++i) {
            if (static_cast<double>(data[i]) > best_value) {
                best_value = static_cast<double>(data[i]);
                best = i;
            }
        }
    };
    switch (ids->type()) {
        case dxrt::DataType::FLOAT: scan(static_cast<const float*>(ids->data())); break;
        case dxrt::DataType::INT32: scan(static_cast<const int32_t*>(ids->data())); break;
        case dxrt::DataType::INT64: scan(static_cast<const int64_t*>(ids->data())); break;
        case dxrt::DataType::UINT8: scan(static_cast<const uint8_t*>(ids->data())); break;
        default:
            throw std::runtime_error(
                "[DXAPP] [ERROR] CLIPTextPostProcess - unsupported token id tensor type");
    }
    return best;
}

}  // namespace

CLIPTextPostProcess::CLIPTextPostProcess(bool normalize) : normalize_(normalize) {}

std::vector<float> CLIPTextPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostProcess - no output tensors");
    }
    const dxrt::TensorPtr& output = outputs[0];
    if (output->type() != dxrt::DataType::FLOAT) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] CLIPTextPostProcess - expected a float32 feature tensor, got " +
            format_shapes(outputs));
    }
    const auto& shape = output->shape();
    const float* data = static_cast<const float*>(output->data());

    std::vector<float> embedding;
    if (shape.size() == 3) {
        const std::size_t seq = static_cast<std::size_t>(shape[1]);
        const std::size_t dim = static_cast<std::size_t>(shape[2]);
        std::size_t eos = seq - 1;
        if (outputs.size() > 1 && outputs[1]->shape().size() >= 2) {
            // Token ids available: EOS has the highest id in CLIP's vocabulary.
            eos = std::min(argmax_token(outputs[1]), seq - 1);
        } else {
            // CLIP pads with zeros after EOS: the last non-zero row is EOS.
            for (std::size_t p = seq; p-- > 0;) {
                const float* row = data + p * dim;
                double sq = 0.0;
                for (std::size_t i = 0; i < dim; ++i) sq += static_cast<double>(row[i]) * row[i];
                if (std::sqrt(sq) > 1e-6) {
                    eos = p;
                    break;
                }
            }
        }
        embedding.assign(data + eos * dim, data + (eos + 1) * dim);
    } else {
        embedding.assign(data, data + element_count(output));
    }

    if (normalize_) {
        double sq = 0.0;
        for (float v : embedding) sq += static_cast<double>(v) * v;
        const float norm = static_cast<float>(std::sqrt(sq));
        if (norm > 0.0f) {
            for (float& v : embedding) v /= norm;
        }
    }
    return embedding;
}
