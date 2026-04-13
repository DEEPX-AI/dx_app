/**
 * @file dncnn3_factory.hpp
 * @brief DnCNN3 Abstract Factory implementation for image restoration
 * 
 * Uses v3-native DnCNN postprocessor with grayscale preprocessor.
 */

#ifndef DNCNN3_FACTORY_HPP
#define DNCNN3_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/grayscale_preprocessor.hpp"
#include "common/processors/restoration_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class DnCNN3Factory : public IRestorationFactory {
public:
    DnCNN3Factory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<GrayscaleResizePreprocessor>(input_width, input_height);
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

    std::string getModelName() const override { return "DnCNN3"; }
    std::string getTaskType() const override { return "image_denoising"; }
};

}  // namespace dxapp

#endif  // DNCNN3_FACTORY_HPP
