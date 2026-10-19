/**
 * @file ssd-mobilenetv2-lite_300x300_factory.hpp
 * @brief SSD MobileNet V1 Abstract Factory implementation
 * 
 * Uses v3-native SSD postprocessor (no legacy postprocess lib).
 */

#ifndef SSD_MOBILENETV2_LITE_300X300_FACTORY_HPP
#define SSD_MOBILENETV2_LITE_300X300_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/ssd_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_ssd_mobilenetv2_lite_300x300 {

class SsdFactory : public IDetectionFactory {
public:

    SsdFactory(float score_threshold = 0.3f,
                  float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<SSDPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            num_classes_,
            true, "voc", class_names_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        class_names_ = config.get_string_list("class_names");
        num_classes_ = config.get<int>("num_classes", num_classes_);
    }

    std::string getModelName() const override {
        return "ssd-mobilenetv2-lite_300x300";
    }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_threshold_;
    float nms_threshold_;
    int num_classes_{20};
    std::vector<std::string> class_names_;
};

}  // namespace v_ssd_mobilenetv2_lite_300x300
}  // namespace dxapp

#endif  // SSD_MOBILENETV2_LITE_300X300_FACTORY_HPP
