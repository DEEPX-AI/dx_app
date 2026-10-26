/**
 * @file yolo26-m-pose_640x640_pre-optimized_factory.hpp
 * @brief YOLO-pose-preopt factory -- pre-optimized models, whose head runs inside the .dxnn
 *
 * Hand-written rather than carried over from a donor: no other family in this tree
 * consumes a pre-optimized model, so there is no existing factory whose behaviour
 * would be right. The postprocessing contract it targets was measured on an M1
 * (DXRT v3.4.2) against dx_yolo26's three real pre-optimized models -- see
 * common/processors/preopt_topk_postprocessor.hpp.
 */

#ifndef YOLO26_M_POSE_640X640_PRE_OPTIMIZED_FACTORY_HPP
#define YOLO26_M_POSE_640X640_PRE_OPTIMIZED_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/preopt_topk_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>
#include <vector>

namespace dxapp {
namespace v_yolo26_m_pose_640x640_pre_optimized {

class YoloPreoptPoseFactory : public IPoseFactory {
public:

    YoloPreoptPoseFactory(float score_threshold = 0.3f,
                          float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // A pre-optimized model carries a CPU task that DXRT runs through ONNX
        // Runtime; with ORT the postprocessor receives the ready-made top-k table and
        // without it the raw per-stride tensors. The decoder handles both, so the flag
        // changes nothing here.
        (void)is_ort_configured;
        return std::make_unique<PreoptPosePostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            top_k_, num_keypoints_);
    }

    VisualizerPtr<PoseResult> createVisualizer() override {
        return std::make_unique<PoseVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        top_k_ = config.get<int>("top_k", top_k_);
        num_keypoints_ = config.get<int>("num_keypoints", num_keypoints_);
    }

    std::string getModelName() const override {
        return "yolo26-m-pose_640x640_pre-optimized";
    }
    std::string getTaskType() const override { return "pose_estimation"; }

private:
    float score_threshold_;
    float nms_threshold_;
    // The model itself selects this many rows; the decoder never needs more.
    int top_k_{300};
    int num_keypoints_{17};
};

}  // namespace v_yolo26_m_pose_640x640_pre_optimized
}  // namespace dxapp

#endif  // YOLO26_M_POSE_640X640_PRE_OPTIMIZED_FACTORY_HPP
