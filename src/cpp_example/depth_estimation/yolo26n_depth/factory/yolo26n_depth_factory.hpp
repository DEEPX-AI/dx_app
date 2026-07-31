/**
 * @file yolo26n_depth_factory.hpp
 * @brief Yolo26nDepthFactory Abstract Factory implementation
 *
 * YOLO26n-Depth monocular depth estimation (Ultralytics PR #25065, branch
 * `depth_anything`). ONNX `output0` [1,1,640,640] dense depth map, compiled to
 * yolo26n-depth_640x640.dxnn (INT8, DX-M1).
 */

#ifndef YOLO26N_DEPTH_FACTORY_HPP
#define YOLO26N_DEPTH_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/depth_postprocessor.hpp"
#include "common/visualizers/depth_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Yolo26nDepthFactory : public IDepthEstimationFactory {
public:
    Yolo26nDepthFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DepthResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<FastDepthPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<DepthResult> createVisualizer() override {
        return std::make_unique<DepthVisualizer>();
    }

    std::string getModelName() const override { return "YOLO26n-Depth"; }
    std::string getTaskType() const override { return "depth_estimation"; }

    // YOLO26-Depth expects float32 input, YOLO-normalized (pixel/255 only, NO
    // ImageNet mean/std) — matching the INT8 calibration domain. apply_mean_std
    // = true selects the float32 /255 path; mean=0, std=1 leaves it at pixel/255.
    // (Without float32 the runner would feed a uint8 buffer 1/4 the size the
    // model reads -> out-of-bounds read / segfault.)
    InputNormalizationParams getInputNormalization() const override {
        return {true, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    }
};

}  // namespace dxapp

#endif  // YOLO26N_DEPTH_FACTORY_HPP
