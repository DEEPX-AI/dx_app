#ifndef RTDETR_POSTPROCESS_H
#define RTDETR_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief One RT-DETR detection, in model-input pixels
 *
 * The caller maps the box to the original image (the Python examples divide by
 * the per-axis resize scale and clip to the frame).
 */
struct RTDETRResult {
    std::vector<float> box{};  // x1, y1, x2, y2 in model-input pixels
    float confidence{0.0f};
    int class_id{0};
    int query{0};              // decoder query row the box came from

    RTDETRResult() = default;
    RTDETRResult(std::vector<float> box_val, float conf, int cls_id, int query_row)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), query(query_row) {}
    ~RTDETRResult() = default;
};

/**
 * @brief RT-DETR / RT-DETRv2 / RT-DETRv3 post-processing (NMS-free)
 *
 * Accepts either output layout and infers it from the shapes, as the Python
 * RTDETRPostprocessor does:
 *
 *   split       boxes (1, N, 4) cxcywh in [0, 1] + logits (1, N, C > 4)
 *   paddle_nms  bbox (N, 6) or (1, N, 6) as [class, score, x1, y1, x2, y2]
 *               in model-input pixels (what the compiled models emit)
 *
 * Split logits get a sigmoid unless they already are probabilities, scores are
 * taken over the flattened (query x class) grid, rows below score_threshold are
 * dropped, the rest are sorted by score (stable) and cut to top_k. nms_threshold
 * 1.0 (the default) keeps the model NMS-free; below 1.0 a class-agnostic greedy
 * NMS runs, with the IoU taken on boxes clipped to the input frame (the Python
 * postprocessor clips to the image first; the two agree when the input is a
 * plain stretch of the whole frame).
 */
class RTDETRPostProcess {
   private:
    int input_width_{640};
    int input_height_{640};
    float score_threshold_{0.4f};
    float nms_threshold_{1.0f};  // 1.0 == off
    int top_k_{300};
    std::string layout_{"auto"};

    std::vector<RTDETRResult> decode_split(const dxrt::TensorPtr& boxes,
                                           const dxrt::TensorPtr& logits) const;
    std::vector<RTDETRResult> decode_paddle(const dxrt::TensorPtr& table) const;
    std::vector<RTDETRResult> finish(std::vector<RTDETRResult> results) const;

   public:
    /**
     * @brief Constructor
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Minimum score (default 0.4)
     * @param nms_threshold Greedy NMS IoU; 1.0 disables it (default)
     * @param top_k Maximum detections kept (default 300)
     * @param layout "auto", "split" or "paddle_nms"; anything else throws
     *        std::invalid_argument
     */
    RTDETRPostProcess(int input_w, int input_h, float score_threshold = 0.4f,
                      float nms_threshold = 1.0f, int top_k = 300,
                      const std::string& layout = "auto");
    ~RTDETRPostProcess() = default;

    std::vector<RTDETRResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    int get_top_k() const { return top_k_; }
    const std::string& get_layout() const { return layout_; }
};

/**
 * @brief One mask-RT-DETR instance: the detection plus its mask probabilities
 *
 * The mask stays at the resolution the model emits it in (mask_height x
 * mask_width, row-major). The caller resizes it to the image, thresholds it and
 * crops it to the box, as the Python MaskRTDETRPostprocessor does.
 */
struct MaskRTDETRResult {
    RTDETRResult detection{};
    std::vector<float> mask{};  // probabilities in [0, 1]
    int mask_height{0};
    int mask_width{0};

    MaskRTDETRResult() = default;
    ~MaskRTDETRResult() = default;
};

/**
 * @brief mask-RT-DETR instance segmentation
 *
 * Decodes the query table like RTDETRPostProcess, then gathers each kept
 * query's mask from the per-query mask tensor, (N, H, W) or (1, N, H, W). An
 * int32/uint8 mask (the compiled models emit 0/1) is converted as is; a float
 * logit map gets a sigmoid.
 */
class MaskRTDETRPostProcess {
   private:
    RTDETRPostProcess decoder_;

   public:
    MaskRTDETRPostProcess(int input_w, int input_h, float score_threshold = 0.4f,
                          float nms_threshold = 1.0f, int top_k = 300,
                          const std::string& layout = "auto");
    ~MaskRTDETRPostProcess() = default;

    std::vector<MaskRTDETRResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return decoder_.get_input_width(); }
    int get_input_height() const { return decoder_.get_input_height(); }
    float get_score_threshold() const { return decoder_.get_score_threshold(); }
    float get_nms_threshold() const { return decoder_.get_nms_threshold(); }
    int get_top_k() const { return decoder_.get_top_k(); }
    const std::string& get_layout() const { return decoder_.get_layout(); }
};

#endif  // RTDETR_POSTPROCESS_H
