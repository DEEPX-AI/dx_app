/**
 * @file ppmatting-hrnet-w48-distinctions_512x512_factory.hpp
 * @brief PP-Matting factory
 *
 * Hand-written, not carried over from a donor: no existing family in this tree
 * consumes this model's output, so there is no factory whose behaviour would be right.
 * A matting model emits a continuous alpha, so the result carries both
 * the thresholded class map (which the visualizer renders) and the matte.
 */

#ifndef PPMATTING_HRNET_W48_DISTINCTIONS_512X512_FACTORY_HPP
#define PPMATTING_HRNET_W48_DISTINCTIONS_512X512_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/matting_postprocessor.hpp"
#include "common/visualizers/matting_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>
#include <vector>

namespace dxapp {
namespace v_ppmatting_hrnet_w48_distinctions_512x512 {

class PpmattingFactory : public ISegmentationFactory {
public:

    PpmattingFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // Illumination gain, gated on frame brightness -- see the preprocessor's
        // own comment for the measurement. Defaults come from config.json so the
        // two trees are configured from one file.
        return std::make_unique<SimpleResizePreprocessor>(
            input_width, input_height, cv::COLOR_BGR2RGB, false,
            std::array<float, 3>{0.f, 0.f, 0.f}, false, false,
            mean_target_, mean_target_above_);
    }

    // ISegmentationFactory's createPostprocessor takes no is_ort_configured flag,
    // unlike the detection and instance-segmentation interfaces.
    PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<PPMattingPostprocessor>(
            input_width, input_height, alpha_threshold_);
    }

    VisualizerPtr<SegmentationResult> createVisualizer() override {
        // NOT SemanticSegmentationVisualizer: that reads `mask` as class IDs and
        // paints a 19-colour Cityscapes palette, which renders a continuous
        // matte as a two-tone segmentation and discards `alpha` entirely.
        return std::make_unique<MattingVisualizer>(checker_size_);
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        alpha_threshold_ = config.get<float>("alpha_threshold", alpha_threshold_);
        checker_size_ = config.get<int>("checker_size", checker_size_);
        mean_target_ = config.get<float>("mean_target", mean_target_);
        mean_target_above_ = config.get<float>("mean_target_above", mean_target_above_);
    }

    std::string getModelName() const override {
        return "ppmatting-hrnet-w48-distinctions_512x512";
    }
    std::string getTaskType() const override { return "image_matting"; }

private:
    float alpha_threshold_{0.5f};
    int checker_size_{16};
    float mean_target_{110.f};
    float mean_target_above_{195.f};
};

}  // namespace v_ppmatting_hrnet_w48_distinctions_512x512
}  // namespace dxapp

#endif  // PPMATTING_HRNET_W48_DISTINCTIONS_512X512_FACTORY_HPP
