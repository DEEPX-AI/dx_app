/**
 * @file embedding_postprocessor.hpp
 * @brief Embedding/feature extraction postprocessors (v3-native)
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Supports ArcFace, CLIP, ReID, and similar embedding models.
 */

#ifndef EMBEDDING_POSTPROCESSOR_HPP
#define EMBEDDING_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

/**
 * @brief Generic embedding postprocessor
 * 
 * Input: Single tensor [1, D] or [D] (feature vector output).
 * Output: EmbeddingResult with extracted feature vector.
 * 
 * The output tensor is treated as a 1D feature vector.
 * Optional L2 normalization is applied.
 */
class GenericEmbeddingPostprocessor : public IPostprocessor<EmbeddingResult> {
public:
    GenericEmbeddingPostprocessor(int input_width, int input_height, bool normalize = true)
        : input_width_(input_width), input_height_(input_height), normalize_(normalize) {}

    std::vector<EmbeddingResult> process(const dxrt::TensorPtrs& outputs,
                                          const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};

        auto output = outputs[0];
        auto shape = output->shape();
        const float* data = static_cast<const float*>(output->data());
        if (!data) return {};

        // Determine embedding dimension from shape
        int dimension = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i == 0 && shape.size() > 1 && shape[0] == 1) continue; // skip batch dim
            dimension *= static_cast<int>(shape[i]);
        }

        // Extract feature vector
        std::vector<float> embedding(data, data + dimension);

        // Optional L2 normalization
        if (normalize_) {
            float norm = 0.0f;
            for (float v : embedding) {
                norm += v * v;
            }
            norm = std::sqrt(norm);
            if (norm > 1e-6f) {
                for (float& v : embedding) {
                    v /= norm;
                }
            }
        }

        EmbeddingResult result;
        result.embedding = std::move(embedding);
        result.dimension = dimension;

        return { result };
    }

    std::string getModelName() const override { return "GenericEmbedding"; }

private:
    int input_width_;
    int input_height_;
    bool normalize_;
};

}  // namespace dxapp

#endif  // EMBEDDING_POSTPROCESSOR_HPP
