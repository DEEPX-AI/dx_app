/**
 * @file dark-hrnet-w32_256x192_factory.hpp
 * @brief DarkHrnet Abstract Factory implementation
 */

#ifndef DARK_HRNET_W32_256X192_FACTORY_HPP
#define DARK_HRNET_W32_256X192_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/pose_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

// Carried over from pose_estimation/vitpose: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what dark_hrnet needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_dark_hrnet_w32_256x192 {

class DarkHrnetFactory : public IPoseFactory {
public:

    DarkHrnetFactory() = default;

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
        return "dark-hrnet-w32_256x192";
    }
    std::string getTaskType() const override { return "pose_estimation"; }
};

}  // namespace v_dark_hrnet_w32_256x192
}  // namespace dxapp

#endif  // DARK_HRNET_W32_256X192_FACTORY_HPP
