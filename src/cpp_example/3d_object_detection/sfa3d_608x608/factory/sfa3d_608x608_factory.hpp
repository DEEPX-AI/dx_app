/**
 * @file sfa3d_608x608_factory.hpp
 * @brief SFA3D 608x608 factory
 */

#ifndef SFA3D_608X608_FACTORY_HPP
#define SFA3D_608X608_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/sfa3d_bev_preprocessor.hpp"
#include "common/processors/sfa3d_postprocessor.hpp"
#include "common/visualizers/sfa3d_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class SFA3D608x608Factory : public I3DDetectionFactory {
public:
    explicit SFA3D608x608Factory(
        float score_threshold = 0.3f,
        float nms_threshold = 0.2f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SFA3DBEVPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<Detection3DResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<SFA3DPostprocessor>(
            input_width, input_height, score_threshold_, nms_threshold_);
    }

    VisualizerPtr<Detection3DResult> createVisualizer() override {
        return std::make_unique<SFA3DVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "sfa3d_608x608"; }
    std::string getTaskType() const override { return "3d_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // SFA3D_608X608_FACTORY_HPP
