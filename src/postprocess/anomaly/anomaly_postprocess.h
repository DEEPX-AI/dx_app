#ifndef ANOMALY_POSTPROCESS_H
#define ANOMALY_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief Anomaly-detection result at the feature-map resolution
 *
 * The map is not resized to the input image: this library has no OpenCV, so the
 * caller resizes it and clips it to 0..1 (the Python examples use cv2.resize with
 * INTER_LINEAR).
 */
struct AnomalyMapResult {
    std::vector<float> anomaly_map;  // H x W row-major, frame-relative
    int width{0};
    int height{0};
    float score{0.0f};
    int channels{0};

    AnomalyMapResult() = default;
    ~AnomalyMapResult() = default;
};

/**
 * @brief EfficientAD anomaly map from its teacher, student and autoencoder
 *
 * Input: three float feature maps, (1, C, H, W) or (1, H, W, C) - a 2C-channel
 * student plus a C-channel teacher and autoencoder. `roles` names them in arrival
 * order; when it does not name all three, the widest map is the student and the
 * other two are the teacher and the autoencoder as they arrive.
 *
 *   st    = mean_c((teacher - student[:C])^2)
 *   ae    = mean_c((autoencoder - student[C:])^2)
 *   map   = st_weight * norm(st) + ae_weight * norm(ae)   (norm: per-frame min-max)
 *   score = percentile(st_weight * st + ae_weight * ae, score_percentile)
 *
 * Same computation as EfficientADPostprocessor in the Python and C++ examples.
 * The score is frame-comparable for this model set only: the published EfficientAD
 * score divides by training-set quantiles that a .dxnn does not carry.
 */
class EfficientADPostProcess {
   private:
    std::vector<std::string> roles_;
    float st_weight_{0.5f};
    float ae_weight_{0.5f};
    float score_percentile_{99.0f};

   public:
    /**
     * @brief Constructor
     * @param roles Output roles in arrival order ("student", "teacher", "autoencoder")
     * @param st_weight Weight of the student-teacher branch
     * @param ae_weight Weight of the autoencoder branch
     * @param score_percentile Percentile of the raw combination reported as score,
     *        in [0, 100]; anything else throws std::invalid_argument
     */
    explicit EfficientADPostProcess(std::vector<std::string> roles = {},
                                    float st_weight = 0.5f, float ae_weight = 0.5f,
                                    float score_percentile = 99.0f);
    ~EfficientADPostProcess() = default;

    AnomalyMapResult postprocess(const dxrt::TensorPtrs& outputs);

    const std::vector<std::string>& get_roles() const { return roles_; }
    float get_score_percentile() const { return score_percentile_; }
};

/**
 * @brief Feature-response map of a single anomaly network (PatchCore backbone)
 *
 * Input: the first float (1, C, H, W) or (1, H, W, C) map in the outputs.
 * map = per-pixel L2 norm across channels, min-max normalised to 0..1.
 * score = score_percentile of that map at feature resolution. It is a relative
 * severity, not a PatchCore distance, which needs a memory bank of training
 * features. The Python example recomputes it on the map resized to the image.
 */
class AnomalyFeaturePostProcess {
   private:
    float score_percentile_{99.0f};

   public:
    /**
     * @brief Constructor
     * @param score_percentile Percentile of the normalised map reported as score,
     *        in [0, 100]; anything else throws std::invalid_argument
     */
    explicit AnomalyFeaturePostProcess(float score_percentile = 99.0f);
    ~AnomalyFeaturePostProcess() = default;

    AnomalyMapResult postprocess(const dxrt::TensorPtrs& outputs);

    float get_score_percentile() const { return score_percentile_; }
};

#endif  // ANOMALY_POSTPROCESS_H
