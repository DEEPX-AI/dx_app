/**
 * @file classification_postprocessor.hpp
 * @brief Unified Classification Postprocessors for v3 interface
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Groups all classification postprocessors:
 *   - EfficientNet (argmax-based, no legacy postprocess lib)
 */

#ifndef CLASSIFICATION_POSTPROCESSOR_HPP
#define CLASSIFICATION_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace dxapp {

// ============================================================================
// EfficientNet Classification Postprocessor
// No legacy postprocess library exists — implements argmax inline.
// Output tensor: float[num_classes] (e.g., 1000 for ImageNet)
// ============================================================================
class EfficientNetPostprocessor : public IPostprocessor<ClassificationResult> {
public:
    EfficientNetPostprocessor(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    std::vector<ClassificationResult> process(const dxrt::TensorPtrs& outputs,
                                              const PreprocessContext& ctx) override {
        std::vector<ClassificationResult> results;
        if (outputs.empty()) return results;

        const auto& output_tensor = outputs.front();
        
        if (output_tensor->type() == dxrt::DataType::FLOAT) {
            const float* data = static_cast<const float*>(output_tensor->data());
            
            // Create index array and sort by confidence (descending)
            std::vector<int> indices(num_classes_);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(),
                              indices.begin() + std::min(top_k_, num_classes_),
                              indices.end(),
                              [data](int a, int b) { return data[a] > data[b]; });
            
            int k = std::min(top_k_, num_classes_);
            results.reserve(k);
            for (int i = 0; i < k; ++i) {
                ClassificationResult cr;
                cr.class_id = indices[i];
                cr.confidence = data[indices[i]];
                results.push_back(cr);
            }
        } else {
            // Non-float output: first value is class ID directly
            const uint16_t* data = static_cast<const uint16_t*>(output_tensor->data());
            ClassificationResult cr;
            cr.class_id = static_cast<int>(data[0]);
            cr.confidence = 1.0f;
            results.push_back(cr);
        }
        
        return results;
    }

    std::string getModelName() const override { return "EfficientNet"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace dxapp

#endif  // CLASSIFICATION_POSTPROCESSOR_HPP
