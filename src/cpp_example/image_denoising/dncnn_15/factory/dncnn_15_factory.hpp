/**
 * @file dncnn_15_factory.hpp
 * @brief DnCNN_15 Abstract Factory implementation for image restoration
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Uses v3-native DnCNN postprocessor with grayscale preprocessor.
 */

#ifndef DNCNN_15_FACTORY_HPP
#define DNCNN_15_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/grayscale_preprocessor.hpp"
#include "common/processors/restoration_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class DnCNN_15Factory : public IRestorationFactory {
public:
    DnCNN_15Factory() = default;

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

    std::string getModelName() const override { return "DnCNN_15"; }
    std::string getTaskType() const override { return "image_denoising"; }
};

}  // namespace dxapp

#endif  // DNCNN_15_FACTORY_HPP
