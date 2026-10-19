/**
 * @file pp-shituv1-mainbody-detection_640x640_factory.hpp
 * @brief PpShitu Abstract Factory implementation
 *
 * PP-PicoDet mainbody detector (PP-ShiTu's first stage), one class: per-level GFL
 * decoding with PicoDetPostprocessor. NanoDetPostprocessor, the family this was first
 * carried over from, reads ONE concatenated tensor; this model emits a score and a box
 * tensor per pyramid level.
 */

#ifndef PP_SHITUV1_MAINBODY_DETECTION_640X640_FACTORY_HPP
#define PP_SHITUV1_MAINBODY_DETECTION_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/picodet_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>
#include <vector>

namespace dxapp {
namespace v_pp_shituv1_mainbody_detection_640x640 {

class PpShituFactory : public IDetectionFactory {
public:
    PpShituFactory(float conf_threshold = 0.4f, float nms_threshold = 0.5f)
        : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        (void)is_ort_configured;
        return std::make_unique<PicoDetPostprocessor>(
            input_width, input_height, conf_threshold_, nms_threshold_, num_classes_, class_names_);
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        // config.json names it conf_threshold, as the Python example reads it; score_threshold still works.
        conf_threshold_ = config.get<float>("score_threshold", conf_threshold_);
        conf_threshold_ = config.get<float>("conf_threshold", conf_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        num_classes_ = config.get<int>("num_classes", num_classes_);
        const std::vector<std::string> names = config.get_string_list("class_names");
        if (!names.empty()) class_names_ = names;
    }

    std::string getModelName() const override {
        return "pp-shituv1-mainbody-detection_640x640";
    }
    std::string getTaskType() const override { return "mainbody_detection"; }

private:
    float conf_threshold_;
    float nms_threshold_;
    int num_classes_{1};
    std::vector<std::string> class_names_{"mainbody"};  // one class: the image's main object
};

}  // namespace v_pp_shituv1_mainbody_detection_640x640
}  // namespace dxapp

#endif  // PP_SHITUV1_MAINBODY_DETECTION_640X640_FACTORY_HPP
