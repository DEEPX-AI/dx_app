/**
 * @file dncnn-15_512x512_factory.hpp
 * @brief DnCNN_15 Abstract Factory implementation for image restoration
 * 
 * Uses v3-native DnCNN postprocessor with grayscale preprocessor.
 */

#ifndef DNCNN_15_512X512_FACTORY_HPP
#define DNCNN_15_512X512_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/grayscale_preprocessor.hpp"
#include "common/processors/restoration_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/simple_resize_preprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_dncnn_15_512x512 {

class DncnnFactory : public IRestorationFactory {
public:

    DncnnFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // default: dncnn-15_512x512
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

    std::string getModelName() const override {
        return "dncnn-15_512x512";
    }
    std::string getTaskType() const override { return "image_denoising"; }

private:
};

}  // namespace v_dncnn_15_512x512
}  // namespace dxapp

#endif  // DNCNN_15_512X512_FACTORY_HPP
