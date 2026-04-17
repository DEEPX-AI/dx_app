#include "yolov8_ppu_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

#include "common_util.hpp"

// YOLOv8PPUResult methods implementation
float YOLOv8PPUResult::iou(const YOLOv8PPUResult& other) const {
    float x_left = std::max(box[0], other.box[0]);
    float y_top = std::max(box[1], other.box[1]);
    float x_right = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;

    return intersection_area / union_area;
}

bool YOLOv8PPUResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv8PPUPostProcess::YOLOv8PPUPostProcess(const int input_w, const int input_h,
                                             const float score_threshold,
                                             const float nms_threshold) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;

    ppu_output_names_ = {"BBOX"};
}

// Default constructor
YOLOv8PPUPostProcess::YOLOv8PPUPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.4;
    nms_threshold_ = 0.5;

    ppu_output_names_ = {"BBOX"};
}

// Process model outputs
std::vector<YOLOv8PPUResult> YOLOv8PPUPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<YOLOv8PPUResult> detections;

    if (outputs.front()->type() != dxrt::DataType::BBOX) {
        int i = 0;
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv8 PPU PostProcess - Tensor output type must be "
               "dxrt::DataType::BBOX.\n"
            << "  Unexpected Tensors\n";
        for (auto& o : outputs) {
            msg << "    Output shape [" << i++ << "]: (";
            for (size_t j = 0; j < o->shape().size(); ++j) {
                msg << o->shape()[j];
                if (j != o->shape().size() - 1) msg << ", ";
            }
            msg << "), Type = " << outputs.front()->type() << "\n";
        }
        msg << ", Expected (x, ), Type = dxrt::DataType::BBOX.\n"
            << "Please re-compile the model with the correct output configuration.\n";

        throw std::runtime_error(msg.str());
    }

    detections = decoding_ppu_outputs(outputs);
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results (anchor-free)
std::vector<YOLOv8PPUResult> YOLOv8PPUPostProcess::decoding_ppu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv8PPUResult> detections;

    if (outputs.empty() || outputs[0]->shape().size() < 2) {
        throw std::runtime_error("[DXAPP] [ER] YOLOv8 PPU decoding - Invalid output shape");
    }

    auto num_elements = outputs[0]->shape()[1];
    auto* raw_data = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());
    for (int i = 0; i < num_elements; i++) {
        auto& bbox_data = raw_data[i];

        if (bbox_data.score < score_threshold_) continue;

        float x = bbox_data.x;
        float y = bbox_data.y;
        float w = bbox_data.w;
        float h = bbox_data.h;

        YOLOv8PPUResult result;
        result.confidence = bbox_data.score;
        result.class_id = bbox_data.label;
        result.class_name = dxapp::common::get_coco_class_name(result.class_id);
        result.box = {x - w / 2, y - h / 2, x + w / 2, y + h / 2};

        detections.push_back(std::move(result));
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOv8PPUResult> YOLOv8PPUPostProcess::apply_nms(
    const std::vector<YOLOv8PPUResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    std::vector<YOLOv8PPUResult> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const YOLOv8PPUResult& a, const YOLOv8PPUResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv8PPUResult> result;

    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(sorted_detections[i]);

        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (!suppressed[j] && sorted_detections[i].iou(sorted_detections[j]) > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// Set thresholds
void YOLOv8PPUPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string YOLOv8PPUPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv8 PPU PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  NMS threshold: " << nms_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n";

    for (auto& name : ppu_output_names_) {
        oss << "  PPU output name: " << name << "\n";
    }

    return oss.str();
}
