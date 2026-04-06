#include "embedding_postprocess.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

EmbeddingPostProcess::EmbeddingPostProcess(bool l2_normalize)
    : l2_normalize_(l2_normalize) {}

EmbeddingResult EmbeddingPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "[DXAPP] [ER] EmbeddingPostProcess::postprocess - No output tensors provided.");
    }

    const auto& output = outputs[0];
    const auto& shape = output->shape();

    // Determine embedding dimension from last axis
    int dim = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i == 0 && shape.size() > 1) continue; // skip batch
        dim *= static_cast<int>(shape[i]);
    }

    const float* data = static_cast<const float*>(output->data());
    std::vector<float> embedding(data, data + dim);

    // L2 normalize
    if (l2_normalize_) {
        float norm = 0.0f;
        for (int i = 0; i < dim; ++i) {
            norm += embedding[i] * embedding[i];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (int i = 0; i < dim; ++i) {
                embedding[i] /= norm;
            }
        }
    }

    return EmbeddingResult(std::move(embedding), dim);
}
