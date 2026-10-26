/**
 * @file classification_postprocessor.hpp
 * @brief Unified Classification Postprocessors for v3 interface
 *
 * Groups all classification postprocessors:
 *   - EfficientNet (argmax-based, no legacy postprocess lib)
 */

#ifndef CLASSIFICATION_POSTPROCESSOR_HPP
#define CLASSIFICATION_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/labels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace dxapp {

namespace detail {

// A model whose network already ends in softmax hands us a distribution,
// not raw logits. Re-applying softmax to that would flatten it towards
// uniform (double softmax) - detect it (non-negative, sums to ~1 over the
// full class count) and pass it through unchanged instead.
inline bool IsAlreadyDistribution(const float* data, int size) {
    if (size <= 0) return false;
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        if (data[i] < -1e-4f) return false;
        sum += data[i];
    }
    return std::fabs(sum - 1.0) <= 1e-3;
}

/// Elements in `tensor`, from its shape (0 for an empty or non-positive
/// shape). The postprocessor never reads past this, whatever num_classes
/// the factory configured.
inline int64_t TensorElementCount(const dxrt::Tensor& tensor) {
    const std::vector<int64_t>& shape = tensor.shape();
    if (shape.empty()) return 0;
    int64_t count = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) return 0;
        count *= shape[i];
    }
    return count;
}

/// Reads the first element of an integer output tensor at its own width.
/// Returns false for an empty tensor or a non-integer type. The element
/// count comes from the shape (dxrt leaves elem_size() 0 on tensors built
/// from a shape and a pointer, so size_in_bytes() is not used).
inline bool ReadIntegerElement0(const dxrt::Tensor& tensor, int* value) {
    const void* data = tensor.data();
    if (data == nullptr || TensorElementCount(tensor) <= 0) return false;
    switch (tensor.type()) {
        case dxrt::DataType::UINT8:  *value = *static_cast<const uint8_t*>(data); break;
        case dxrt::DataType::INT8:   *value = *static_cast<const int8_t*>(data); break;
        case dxrt::DataType::UINT16: *value = *static_cast<const uint16_t*>(data); break;
        case dxrt::DataType::INT16:  *value = *static_cast<const int16_t*>(data); break;
        case dxrt::DataType::UINT32:
            *value = static_cast<int>(*static_cast<const uint32_t*>(data)); break;
        case dxrt::DataType::INT32:  *value = *static_cast<const int32_t*>(data); break;
        case dxrt::DataType::UINT64:
            *value = static_cast<int>(*static_cast<const uint64_t*>(data)); break;
        case dxrt::DataType::INT64:
            *value = static_cast<int>(*static_cast<const int64_t*>(data)); break;
        default: return false;
    }
    return true;
}

}  // namespace detail

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
                                              const PreprocessContext& /*ctx*/) override {
        std::vector<ClassificationResult> results;
        if (outputs.empty()) return results;

        const auto& output_tensor = outputs.front();
        if (!output_tensor || output_tensor->data() == nullptr) return results;
        // Never read past the tensor: an output shorter than the configured
        // class count is ranked over the elements it has, as the Python
        // postprocessor ranks the array it is given.
        const int64_t elements = detail::TensorElementCount(*output_tensor);
        if (elements <= 0) return results;
        const int classes =
            static_cast<int>(std::min<int64_t>(num_classes_, elements));
        if (classes <= 0) return results;

        if (output_tensor->type() == dxrt::DataType::FLOAT) {
            const float* data = static_cast<const float*>(output_tensor->data());

            // Numerically stable probabilities: a distribution the network
            // already produced is returned unchanged (guards against double
            // softmax); raw logits are converted with dxapp::softmax.
            std::vector<float> probabilities;
            if (detail::IsAlreadyDistribution(data, classes)) {
                probabilities.assign(data, data + classes);
            } else {
                probabilities = dxapp::softmax(std::vector<float>(data, data + classes));
            }

            // Create index array and sort by probability (descending). This
            // is the same order as sorting the raw logits - softmax is
            // strictly monotonic, so argmax/ranking is unchanged.
            std::vector<int> indices(classes);
            std::iota(indices.begin(), indices.end(), 0);
            const int k = std::max(0, std::min(top_k_, classes));
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                              [&probabilities](int a, int b) {
                                  return probabilities[a] > probabilities[b];
                              });

            std::vector<std::pair<int, float>> top_k_list;
            top_k_list.reserve(k);
            for (int i = 0; i < k; ++i) {
                top_k_list.push_back(std::make_pair(indices[i], probabilities[indices[i]]));
            }

            results.reserve(k);
            for (int i = 0; i < k; ++i) {
                ClassificationResult cr;
                cr.class_id = indices[i];
                cr.confidence = probabilities[indices[i]];
                cr.class_name = ClassName(cr.class_id);
                if (i == 0) cr.top_k = top_k_list;
                results.push_back(cr);
            }
        } else {
            // Non-float output: first value is class ID directly, read at the
            // tensor's own element width so a 1-byte output is not over-read.
            int class_id = 0;
            if (!detail::ReadIntegerElement0(*output_tensor, &class_id)) return results;
            ClassificationResult cr;
            cr.class_id = class_id;
            cr.confidence = 1.0f;
            cr.class_name = ClassName(cr.class_id);
            results.push_back(cr);
        }

        return results;
    }

    std::string getModelName() const override { return "EfficientNet"; }

private:
    /// ImageNet's name for a 1000-class head; no name for any other class count.
    std::string ClassName(int class_id) const {
        return num_classes_ == 1000 ? getImageNetClassName(class_id) : std::string();
    }

    int num_classes_;
    int top_k_;
};

}  // namespace dxapp

#endif  // CLASSIFICATION_POSTPROCESSOR_HPP
