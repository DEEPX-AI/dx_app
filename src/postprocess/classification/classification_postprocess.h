#ifndef CLASSIFICATION_POSTPROCESS_H
#define CLASSIFICATION_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief Classification result structure
 */
struct ClassificationResult {
    int class_id{-1};
    float confidence{0.0f};

    ClassificationResult() = default;

    ClassificationResult(int cls_id, float conf)
        : class_id(cls_id), confidence(conf) {}

    ~ClassificationResult() = default;
};

/**
 * @brief Generic classification post-processing class
 *
 * Handles softmax + top-K for classification models.
 * Input tensor: [1, num_classes] or [1, 1, num_classes] (raw logits or probabilities)
 */
class ClassificationPostProcess {
   private:
    int top_k_{5};

    std::vector<float> apply_softmax(const float* data, int num_classes) const;

   public:
    /**
     * @brief Constructor
     * @param top_k Number of top predictions to return (default: 5)
     */
    explicit ClassificationPostProcess(int top_k = 5);
    ~ClassificationPostProcess() = default;

    std::vector<ClassificationResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_top_k() const { return top_k_; }
};

#endif  // CLASSIFICATION_POSTPROCESS_H
