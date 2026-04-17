#include "classification_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

ClassificationPostProcess::ClassificationPostProcess(int top_k)
    : top_k_(top_k) {}

std::vector<float> ClassificationPostProcess::apply_softmax(
    const float* data, int num_classes) const {
    std::vector<float> probs(num_classes);

    // Find max for numerical stability
    float max_val = data[0];
    for (int i = 1; i < num_classes; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        probs[i] = std::exp(data[i] - max_val);
        sum += probs[i];
    }

    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < num_classes; ++i) {
            probs[i] /= sum;
        }
    }

    return probs;
}

std::vector<ClassificationResult> ClassificationPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        return {};
    }

    const auto& shape = outputs[0]->shape();
    const float* data = static_cast<const float*>(outputs[0]->data());

    // Compute total number of classes from shape
    int num_classes = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        num_classes *= static_cast<int>(shape[i]);
    }

    if (num_classes <= 0) {
        return {};
    }

    // Handle single-value output (argmax already applied in model)
    if (num_classes == 1) {
        int class_id = static_cast<int>(data[0]);
        return {ClassificationResult(class_id, 1.0f)};
    }

    // Apply softmax
    std::vector<float> probabilities = apply_softmax(data, num_classes);

    // Get top-K indices
    std::vector<int> indices(num_classes);
    std::iota(indices.begin(), indices.end(), 0);

    int k = std::min(top_k_, num_classes);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&probabilities](int a, int b) {
                          return probabilities[a] > probabilities[b];
                      });

    std::vector<ClassificationResult> results;
    results.reserve(k);
    for (int i = 0; i < k; ++i) {
        results.emplace_back(indices[i], probabilities[indices[i]]);
    }

    return results;
}
