#include "classification_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace {

// True when the network already ends in softmax: every value >= -1e-4 and the
// sum within 1e-3 of 1. Same tolerances as detail::IsAlreadyDistribution() in
// the C++ example and _is_already_distribution() in the Python postprocessor,
// so all three report the same confidences.
bool is_already_distribution(const float* data, int size) {
    if (size <= 0) return false;
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        if (data[i] < -1e-4f) return false;
        sum += data[i];
    }
    return std::fabs(sum - 1.0) <= 1e-3;
}

}  // namespace

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

    // Apply softmax unless the network already did
    std::vector<float> probabilities =
        is_already_distribution(data, num_classes)
            ? std::vector<float>(data, data + num_classes)
            : apply_softmax(data, num_classes);

    // Get top-K indices
    std::vector<int> indices(num_classes);
    std::iota(indices.begin(), indices.end(), 0);

    // Ties go to the lower class id, like the Python postprocessor's stable sort.
    int k = std::min(top_k_, num_classes);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&probabilities](int a, int b) {
                          if (probabilities[a] != probabilities[b]) {
                              return probabilities[a] > probabilities[b];
                          }
                          return a < b;
                      });

    std::vector<ClassificationResult> results;
    results.reserve(k);
    for (int i = 0; i < k; ++i) {
        results.emplace_back(indices[i], probabilities[indices[i]]);
    }

    return results;
}
