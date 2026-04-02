/**
 * @file dncnn_25_factory.hpp
 * @brief DnCNN_25 Abstract Factory implementation for image restoration
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses v3-native DnCNN postprocessor with grayscale preprocessor.
 */

#ifndef DNCNN_25_FACTORY_HPP
#define DNCNN_25_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/grayscale_preprocessor.hpp"
#include "common/processors/restoration_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class DnCNN_25Factory : public IRestorationFactory {
public:
    DnCNN_25Factory() = default;

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

    std::string getModelName() const override { return "DnCNN_25"; }
    std::string getTaskType() const override { return "image_denoising"; }
};

}  // namespace dxapp

#endif  // DNCNN_25_FACTORY_HPP
