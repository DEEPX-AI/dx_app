/**
 * @file centerpose-fpn_regnet-x1.6gf_512x512_factory.hpp
 * @brief Centerpose_regnetx_1_6gf_fpn Abstract Factory implementation
 * 
 * Uses CenterPose heatmap-based postprocessor (6-tensor, stride 4).
 */

#ifndef CENTERPOSE_FPN_REGNET_X1_6GF_512X512_FACTORY_HPP
#define CENTERPOSE_FPN_REGNET_X1_6GF_512X512_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/centerpose_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_centerpose_fpn_regnet_x1_6gf_512x512 {

class CenterposeFactory : public IPoseFactory {
public:

    CenterposeFactory(float score_threshold = 0.3f,
                      float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<CenterPosePostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            17  // COCO body pose: 17 keypoints
        );
    }

    VisualizerPtr<PoseResult> createVisualizer() override {
        return std::make_unique<PoseVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "centerpose-fpn_regnet-x1.6gf_512x512";
    }
    std::string getTaskType() const override { return "pose_estimation"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace v_centerpose_fpn_regnet_x1_6gf_512x512
}  // namespace dxapp

#endif  // CENTERPOSE_FPN_REGNET_X1_6GF_512X512_FACTORY_HPP
