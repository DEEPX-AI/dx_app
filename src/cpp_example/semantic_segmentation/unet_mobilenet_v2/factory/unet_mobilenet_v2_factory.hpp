/**
 * @file unet_mobilenet_v2_factory.hpp
 * @brief Unet_mobilenet_v2 Abstract Factory implementation for semantic segmentation
 */

#ifndef UNET_MOBILENET_V2_FACTORY_HPP
#define UNET_MOBILENET_V2_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

namespace dxapp {

class Unet_mobilenet_v2Factory : public ISegmentationFactory {
public:
    Unet_mobilenet_v2Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<DeepLabv3Postprocessor>(
            input_width, input_height
        );
    }

    VisualizerPtr<SegmentationResult> createVisualizer() override {
        // UNet-MobileNetV2: 3-class segmentation (pet / background / boundary)
        // Model class mapping: 0=foreground(pet), 1=background, 2=boundary
        std::vector<cv::Vec3b> palette = {
            {0,   200, 255},   // 0: foreground/pet (warm yellow in BGR)
            {0,     0,   0},   // 1: background    (black = skip)
            {0,   128, 255},   // 2: boundary      (orange in BGR)
        };
        return std::make_unique<SemanticSegmentationVisualizer>(std::move(palette), true);
    }

    std::string getModelName() const override { return "Unet_mobilenet_v2"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }
};

}  // namespace dxapp

#endif  // UNET_MOBILENET_V2_FACTORY_HPP
