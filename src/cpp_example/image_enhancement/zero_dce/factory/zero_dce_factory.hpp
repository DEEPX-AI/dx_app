/**
 * @file zero_dce_factory.hpp
 * @brief Zero-DCE Abstract Factory implementation for image enhancement
 * 
 * Uses Zero-DCE specific postprocessor with iterative LE curve enhancement.
 * SimpleResizePreprocessor stores normalized RGB input for LE curve application.
 */

#ifndef ZERO_DCE_FACTORY_HPP
#define ZERO_DCE_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/zero_dce_postprocessor.hpp"
#include "common/visualizers/restoration_visualizer.hpp"

namespace dxapp {

class ZeroDCEFactory : public IRestorationFactory {
public:
    ZeroDCEFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // store_source=true: store resized RGB for LE curve in postprocessor
        return std::make_unique<SimpleResizePreprocessor>(
            input_width, input_height, cv::COLOR_BGR2RGB, true);
    }

    PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<ZeroDCEPostprocessor>(
            input_width, input_height, 8  // 8 iterations
        );
    }

    VisualizerPtr<RestorationResult> createVisualizer() override {
        return std::make_unique<RestorationVisualizer>();
    }

    std::string getModelName() const override { return "Zero-DCE"; }
    std::string getTaskType() const override { return "image_enhancement"; }
};

}  // namespace dxapp

#endif  // ZERO_DCE_FACTORY_HPP
