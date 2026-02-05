#include "yolov26_postprocess.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <sstream>

#include "common_util.hpp"

// YOLOv26Result methods implementation
float YOLOv26Result::iou(const YOLOv26Result& other) const {
    // Calculate intersection coordinates
    float x_left = std::max(box[0], other.box[0]);
    float y_top = std::max(box[1], other.box[1]);
    float x_right = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    // Check if there is intersection
    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;

    return intersection_area / union_area;
}

bool YOLOv26Result::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv26PostProcess::YOLOv26PostProcess(const int input_w, const int input_h,
                                     const float score_threshold,
                                     const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for YOLOv26
    cpu_output_names_ = {"output0"};
}

// Default constructor
YOLOv26PostProcess::YOLOv26PostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.45;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for YOLOv26
    cpu_output_names_ = {"output0"};
}

// Process model outputs
std::vector<YOLOv26Result> YOLOv26PostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (!is_ort_configured_) {
        throw std::runtime_error("YOLOv26PostProcess currently supports only ORT inference mode.");
    }

    std::vector<YOLOv26Result> detections = decoding_cpu_outputs(outputs);

    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv26Result> YOLOv26PostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv26Result> detections;

    // Assume the first tensor has shape [1, 300, 6] with layout:
    // [x1, y1, x2, y2, score, class_id], same as Python yolov26_sync.
    if (outputs.empty()) {
        return detections;
    }

    const auto& tensor = outputs[0];
    const float* data = static_cast<const float*>(tensor->data());
    const auto& shape = tensor->shape();

    if (shape.size() != 3) {
        return detections;
    }

    const int num_dets = static_cast<int>(shape[1]);  // e.g., 300
    const int stride = static_cast<int>(shape[2]);     // e.g., 6

    for (int i = 0; i < num_dets; ++i) {
        const float* det = data + i * stride;

        float score = det[4];
        if (score < score_threshold_) {
            continue;
        }

        int class_id = static_cast<int>(det[5]);
        if (class_id < 0 || class_id >= num_classes_) {
            continue;
        }

        YOLOv26Result result;
        result.confidence = score;
        result.class_id = class_id;
        result.class_name = dxapp::common::get_coco_class_name(class_id);
        result.box = {det[0], det[1], det[2], det[3]};

        detections.push_back(result);
    }
    return detections;
}

// Set thresholds
void YOLOv26PostProcess::set_thresholds(float score_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
}

// Get configuration information
std::string YOLOv26PostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv26 PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n"
        << "  Is Ort Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";

    for (auto& cpu_output_name : cpu_output_names_) {
        oss << "  CPU output name: " << cpu_output_name << "\n";
    }

    return oss.str();
}
