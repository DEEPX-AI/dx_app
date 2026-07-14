#include "attribute_postprocess.h"

#include <algorithm>
#include <cmath>

AttributePostProcess::AttributePostProcess(float threshold, bool softmax_pairs)
    : threshold_(threshold), softmax_pairs_(softmax_pairs) {}

std::vector<AttributeResult> AttributePostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    std::vector<AttributeResult> results;
    if (outputs.empty()) {
        return results;
    }

    const float* data = static_cast<const float*>(outputs[0]->data());
    const auto& shape = outputs[0]->shape();

    if (softmax_pairs_) {
        // CelebA layout: [1, N, 2] (or [N, 2]). Softmax over the last pair,
        // take the positive-class probability.
        int num_attrs;
        if (shape.size() >= 2) {
            // last dim is the pair size (2); the attribute count is the
            // second-to-last dim.
            num_attrs = static_cast<int>(shape[shape.size() - 2]);
        } else {
            num_attrs = 0;
        }
        for (int i = 0; i < num_attrs; ++i) {
            float l0 = data[i * 2 + 0];
            float l1 = data[i * 2 + 1];
            float m = std::max(l0, l1);
            float e0 = std::exp(l0 - m);
            float e1 = std::exp(l1 - m);
            float prob = e1 / (e0 + e1);
            if (prob > threshold_) {
                results.emplace_back(i, prob);
            }
        }
    } else {
        // DeepMAR layout: [1, N] (or [N]). Sigmoid per attribute.
        int num_attrs = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            int dim = static_cast<int>(shape[d]);
            if (dim > num_attrs) num_attrs = dim;
        }
        for (int i = 0; i < num_attrs; ++i) {
            float prob = 1.0f / (1.0f + std::exp(-data[i]));
            if (prob > threshold_) {
                results.emplace_back(i, prob);
            }
        }
    }

    std::sort(results.begin(), results.end(),
              [](const AttributeResult& a, const AttributeResult& b) {
                  return a.confidence > b.confidence;
              });
    return results;
}
