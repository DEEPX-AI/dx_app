#ifndef ATTRIBUTE_POSTPROCESS_H
#define ATTRIBUTE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <vector>

/**
 * @brief Single recognized attribute (index + probability).
 */
struct AttributeResult {
    int class_id{0};
    float confidence{0.0f};

    AttributeResult() = default;
    AttributeResult(int id, float conf) : class_id(id), confidence(conf) {}
};

/**
 * @brief Multi-label attribute-recognition post-processing.
 *
 * Supports two output layouts:
 *   - Sigmoid mode (DeepMAR):  [1, N]    logits -> sigmoid per attribute.
 *   - Softmax-pair mode (CelebA): [1, N, 2] logits -> softmax per attribute,
 *     positive-class probability.
 *
 * Attributes whose probability exceeds the threshold are returned, sorted by
 * descending confidence (matching the Python AttributePostprocessor).
 */
class AttributePostProcess {
   private:
    float threshold_{0.5f};
    bool softmax_pairs_{false};

   public:
    /**
     * @param threshold     Probability threshold (strictly greater-than).
     * @param softmax_pairs  False -> sigmoid mode ([1, N]); True -> softmax
     *                       pair mode ([1, N, 2]).
     */
    explicit AttributePostProcess(float threshold = 0.5f,
                                  bool softmax_pairs = false);
    ~AttributePostProcess() = default;

    std::vector<AttributeResult> postprocess(const dxrt::TensorPtrs& outputs);

    float get_threshold() const { return threshold_; }
    bool get_softmax_pairs() const { return softmax_pairs_; }
};

#endif  // ATTRIBUTE_POSTPROCESS_H
