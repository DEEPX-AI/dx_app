/**
 * @file dncnn_color_blind_factory.hpp
 * @brief DnCNN_color_blind Abstract Factory implementation for image restoration
 * 
 * Uses v3-native DnCNN postprocessor with grayscale preprocessor.
 */

#ifndef DNCNN_COLOR_BLIND_FACTORY_HPP
#define DNCNN_COLOR_BLIND_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/restoration_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class DnCNN_color_blindFactory : public IRestorationFactory {
public:
    DnCNN_color_blindFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // Color model (3-channel) — use simple resize, not grayscale
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<DnCNNPostprocessor>(
            input_width, input_height
        );
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override { return "DnCNN_color_blind"; }
    std::string getTaskType() const override { return "image_denoising"; }
};

}  // namespace dxapp

#endif  // DNCNN_COLOR_BLIND_FACTORY_HPP
