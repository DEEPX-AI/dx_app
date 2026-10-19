/**
 * @file yolov12-cls-m_224x224_factory.hpp
 * @brief Yolov12Cls Abstract Factory implementation for classification
 */

#ifndef YOLOV12_CLS_M_224X224_FACTORY_HPP
#define YOLOV12_CLS_M_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

// Carried over from image_classification/yolo26_cls: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what yolov12_cls needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_yolov12_cls_m_224x224 {

class Yolov12ClsFactory : public IClassificationFactory {
public:

    Yolov12ClsFactory(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<ClassificationResult> createPostprocessor(
        int /*input_width*/, int /*input_height*/) override {
        return std::make_unique<EfficientNetPostprocessor>(num_classes_, top_k_);
    }

    VisualizerPtr<ClassificationResult> createVisualizer() override {
        return std::make_unique<ClassificationResultVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        num_classes_ = config.get<int>("num_classes", num_classes_);
        top_k_ = config.get<int>("top_k", top_k_);
    }

    std::string getModelName() const override {
        return "yolov12-cls-m_224x224";
    }
    std::string getTaskType() const override { return "image_classification"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace v_yolov12_cls_m_224x224
}  // namespace dxapp

#endif  // YOLOV12_CLS_M_224X224_FACTORY_HPP
