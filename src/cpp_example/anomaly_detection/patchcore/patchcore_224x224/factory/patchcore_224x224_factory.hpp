/**
 * @file patchcore_224x224_factory.hpp
 * @brief PatchCore factory
 *
 * Hand-written, not carried over from a donor: no existing family in this tree
 * consumes this model's output, so there is no factory whose behaviour would be right.
 *
 * ONE network, on purpose: PatchCore's real metric needs a memory bank of training
 * features -- a fit step, not a model -- so this declares no companions and the
 * response is a FEATURE MAGNITUDE, not the published anomaly score. See the
 * postprocessor header. It shares the anomaly runner and visualizer with EfficientAD,
 * which is what keeps the render identical to the Python example's.
 */

#ifndef PATCHCORE_224X224_FACTORY_HPP
#define PATCHCORE_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/anomaly_postprocessor.hpp"
#include "common/visualizers/anomaly_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>
#include <vector>

namespace dxapp {
namespace v_patchcore_224x224 {

class PatchcoreFactory : public IAnomalyDetectionFactory {
public:

    PatchcoreFactory() = default;

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<AnomalyResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<AnomalyFeaturePostprocessor>(
            input_width, input_height);
    }

    VisualizerPtr<AnomalyResult> createVisualizer() override {
        return std::make_unique<AnomalyVisualizer>();
    }

    std::string getModelName() const override {
        return "patchcore_224x224";
    }
    std::string getTaskType() const override { return "anomaly_detection"; }

private:
};

}  // namespace v_patchcore_224x224
}  // namespace dxapp

#endif  // PATCHCORE_224X224_FACTORY_HPP
