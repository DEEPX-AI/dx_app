#ifndef PREOPT_TOPK_POSTPROCESS_H
#define PREOPT_TOPK_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <vector>

/**
 * @brief One row of a pre-optimized YOLO model's top-k table, in model-input
 *        (letterboxed) pixels
 *
 * keypoints holds num_keypoints x (x, y, visibility) for a pose model and is
 * empty for a detection model. The caller undoes the letterbox and clips to
 * the image, as the Python examples do through the runner.
 */
struct PreoptResult {
    std::vector<float> box{};        // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};
    std::vector<float> keypoints{};  // x, y, visibility triplets

    PreoptResult() = default;
    ~PreoptResult() = default;
};

/**
 * @brief Pre-optimized (top-k in the model) YOLO detection / pose post-processing
 *
 * A pre-optimized .dxnn carries the end-to-end head: what arrives is one table
 * (1, K, C), C = 6 + extra, rows [x1, y1, x2, y2, score, class_id, extra...]
 * sorted by score, where extra is 17 x (x, y, visibility) for pose. Rows at or
 * above score_threshold (floored just above 0, because an unfilled table pads
 * with zero rows) are kept, best first, and the class-agnostic greedy NMS only
 * merges the duplicates the flattened (anchor x class) top-k produces; set
 * nms_threshold >= 1.0 to keep every row. Same decode as the Python
 * PreoptDetectionPostprocessor / PreoptPosePostprocessor.
 */
class PreoptTopKPostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    float score_threshold_{0.3f};
    float nms_threshold_{0.45f};
    int top_k_{300};
    int num_keypoints_{0};

   public:
    /**
     * @brief Constructor
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Minimum score (default 0.3)
     * @param nms_threshold Greedy NMS IoU; >= 1.0 disables it (default 0.45)
     * @param top_k Table rows considered, mirroring the model's own K (default 300)
     * @param num_keypoints Keypoints per row for a pose model; 0 for detection
     */
    PreoptTopKPostProcess(int input_w, int input_h, float score_threshold = 0.3f,
                          float nms_threshold = 0.45f, int top_k = 300,
                          int num_keypoints = 0);
    ~PreoptTopKPostProcess() = default;

    std::vector<PreoptResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    int get_top_k() const { return top_k_; }
    int get_num_keypoints() const { return num_keypoints_; }
};

#endif  // PREOPT_TOPK_POSTPROCESS_H
