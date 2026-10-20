#ifndef PICODET_POSTPROCESS_H
#define PICODET_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <vector>

/**
 * @brief One PP-PicoDet detection, in model-input (letterboxed) pixels
 */
struct PicoDetResult {
    std::vector<float> box{};  // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};

    PicoDetResult() = default;
    PicoDetResult(std::vector<float> box_val, float conf, int cls_id)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id) {}
    ~PicoDetResult() = default;
};

/**
 * @brief PP-PicoDet (PP-ShiTu mainbody detector) post-processing
 *
 * The GFL head arrives as one score tensor (1, N, num_classes) and one box
 * distribution tensor (1, N, 4 * (reg_max + 1)) per pyramid level, paired by
 * anchor count N (a square grid), never by position. Scores are already
 * sigmoid; the distributions are softmaxed and integrated here (DFL). Boxes
 * are decoded in model-input pixels, sorted by score and merged with a greedy
 * class-agnostic NMS whose IoU is taken on boxes clipped to the input frame.
 * Same decode as the Python PicoDetPostprocessor, which clips to the image
 * (inside the letterbox) before its NMS; a box reaching into the padding can
 * be merged differently.
 */
class PicoDetPostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    float score_threshold_{0.3f};
    float nms_threshold_{0.5f};
    int num_classes_{1};

   public:
    /**
     * @brief Constructor
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Minimum best-class score (default 0.3)
     * @param nms_threshold Greedy NMS IoU; >= 1.0 disables it (default 0.5)
     * @param num_classes Number of classes (default 1, mainbody)
     */
    PicoDetPostProcess(int input_w, int input_h, float score_threshold = 0.3f,
                       float nms_threshold = 0.5f, int num_classes = 1);
    ~PicoDetPostProcess() = default;

    std::vector<PicoDetResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    int get_num_classes() const { return num_classes_; }
};

#endif  // PICODET_POSTPROCESS_H
