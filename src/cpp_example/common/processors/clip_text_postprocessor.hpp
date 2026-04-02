/**
 * @file clip_text_postprocessor.hpp
 * @brief CLIP Text Encoder postprocessor (EOS token extraction + L2 normalize)
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Ported from Python clip_text_postprocessor.py.
 * 
 * Output: [1, seq_len, D] or [1, D]
 * For 3D output: extract features at the EOS token position (argmax of input_ids)
 * Then L2 normalize the extracted embedding.
 */

#ifndef CLIP_TEXT_POSTPROCESSOR_HPP
#define CLIP_TEXT_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>

namespace dxapp {

class CLIPTextPostprocessor : public IPostprocessor<EmbeddingResult> {
public:
    CLIPTextPostprocessor(int input_width = 77, int input_height = 512,
                          bool normalize = true)
        : input_width_(input_width), input_height_(input_height),
          normalize_(normalize) {}

    std::vector<EmbeddingResult> process(const dxrt::TensorPtrs& outputs,
                                          const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};

        auto output = outputs[0];
        auto shape = output->shape();
        const float* data = static_cast<const float*>(output->data());
        if (!data) return {};

        std::vector<float> embedding;

        if (shape.size() == 3) {
            // [1, seq_len, D] → extract EOS token (last non-padding token)
            int seq_len = static_cast<int>(shape[1]);
            int dim = static_cast<int>(shape[2]);
            
            // Use the last token position as EOS approximation
            // (In actual CLIP, EOS position = argmax(input_ids), but without
            //  access to input_ids, we use the last valid position)
            int eos_pos = seq_len - 1;
            const float* eos_features = data + eos_pos * dim;
            embedding.assign(eos_features, eos_features + dim);
        } else {
            // [1, D] or [D] → use directly
            int dim = 1;
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i == 0 && shape.size() > 1 && shape[0] == 1) continue;
                dim *= static_cast<int>(shape[i]);
            }
            embedding.assign(data, data + dim);
        }

        // L2 normalization
        if (normalize_) {
            float norm = 0.0f;
            for (float v : embedding) norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 1e-6f) {
                for (float& v : embedding) v /= norm;
            }
        }

        EmbeddingResult result;
        result.dimension = static_cast<int>(embedding.size());
        result.embedding = std::move(embedding);
        return { result };
    }

    std::string getModelName() const override { return "CLIPText"; }

private:
    int input_width_;
    int input_height_;
    bool normalize_;
};

}  // namespace dxapp

#endif  // CLIP_TEXT_POSTPROCESSOR_HPP
