/**
 * @file vitpose-s_256x192_factory.hpp
 * @brief VitposeFactory Abstract Factory implementation
 */

#ifndef VITPOSE_S_256X192_FACTORY_HPP
#define VITPOSE_S_256X192_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/pose_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_vitpose_s_256x192 {

class VitposeFactory : public IPoseFactory {
public:

    VitposeFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool /*is_ort_configured*/ = false) override {
        return std::make_unique<VitPosePostprocessor>(input_width, input_height);
    }

    VisualizerPtr<PoseResult> createVisualizer() override {
        return std::make_unique<PoseVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& /*config*/) override {}

    std::string getModelName() const override {
        return "vitpose-s_256x192";
    }
    std::string getTaskType() const override { return "pose_estimation"; }
};

}  // namespace v_vitpose_s_256x192
}  // namespace dxapp

#endif  // VITPOSE_S_256X192_FACTORY_HPP
