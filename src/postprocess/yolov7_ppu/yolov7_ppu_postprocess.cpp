#include "yolov7_ppu_postprocess.h"

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

#include "common_util.hpp"

// YOLOv7PPUResult methods implementation
float YOLOv7PPUResult::iou(const YOLOv7PPUResult& other) const {
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

bool YOLOv7PPUResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv7PPUPostProcess::YOLOv7PPUPostProcess(const int input_w, const int input_h,
                                             const float obj_threshold, const float score_threshold,
                                             const float nms_threshold) {
    input_width_ = input_w;
    input_height_ = input_h;
    object_threshold_ = obj_threshold;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;

    // Initialize model-specific parameters for YOLOv7
    ppu_output_names_ = {"BBOX"};
    anchors_by_strides_ = {{8, {{12, 16}, {19, 36}, {40, 28}}},
                           {16, {{36, 75}, {76, 55}, {72, 146}}},
                           {32, {{142, 110}, {192, 243}, {459, 401}}}};
}

// Default constructor
YOLOv7PPUPostProcess::YOLOv7PPUPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    object_threshold_ = 0.3;
    score_threshold_ = 0.4;
    nms_threshold_ = 0.5;

    // Initialize model-specific parameters for YOLOv7
    ppu_output_names_ = {"BBOX"};
    anchors_by_strides_ = {{8, {{12, 16}, {19, 36}, {40, 28}}},
                           {16, {{36, 75}, {76, 55}, {72, 146}}},
                           {32, {{142, 110}, {192, 243}, {459, 401}}}};
}

// Process model outputs
std::vector<YOLOv7PPUResult> YOLOv7PPUPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<YOLOv7PPUResult> detections;

    if (outputs.front()->type() != dxrt::DataType::BBOX) {
        int i = 0;
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv7 PPU PostProcess - Tensor output type must be "
               "dxrt::DataType::BBOX.\n"
            << "  Unexpected Tensors\n";
        for (auto& o : outputs) {
            msg << "    Output shape [" << i++ << "]: (";
            for (size_t i = 0; i < o->shape().size(); ++i) {
                msg << o->shape()[i];
                if (i != o->shape().size() - 1) msg << ", ";
            }
            msg << "), Type = " << outputs.front()->type() << "\n";
        }
        msg << ", Expected (1, x ,x, x), Type = dxrt::DataType::BBOX.\n"
            << "Please re-compile the model with the correct output configuration.\n";

        throw std::runtime_error(msg.str());  // 안전한 종료: 상위로 에러 전달
    }

    detections = decoding_ppu_outputs(outputs);

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv7PPUResult> YOLOv7PPUPostProcess::decoding_ppu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv7PPUResult> detections;
    // YOLOv7 PPU
    // "name":"BBOX", "shape":[1, num_emelements]
    auto num_elements = outputs[0]->shape()[1];
    auto* raw_data = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());
    for (int i = 0; i < num_elements; i++) {
        YOLOv7PPUResult result;
        auto bbox_data = raw_data[i];

        if (bbox_data.score < score_threshold_) continue;

        int gX = bbox_data.grid_x;
        int gY = bbox_data.grid_y;

        auto stride = std::next(anchors_by_strides_.begin(), bbox_data.layer_idx)->first;
        const auto& anchors = anchors_by_strides_.at(stride);

        float x, y, w, h;

        x = (bbox_data.x * 2. - 0.5 + gX) * stride;
        y = (bbox_data.y * 2. - 0.5 + gY) * stride;
        w = (bbox_data.w * bbox_data.w * 4.) * anchors[bbox_data.box_idx].first;
        h = (bbox_data.h * bbox_data.h * 4.) * anchors[bbox_data.box_idx].second;

        result.confidence = bbox_data.score;
        result.class_id = bbox_data.label;
        result.class_name = dxapp::common::get_coco_class_name(result.class_id);
        result.box.emplace_back(x - w / 2);
        result.box.emplace_back(y - h / 2);
        result.box.emplace_back(x + w / 2);
        result.box.emplace_back(y + h / 2);

        detections.push_back(result);
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOv7PPUResult> YOLOv7PPUPostProcess::apply_nms(
    const std::vector<YOLOv7PPUResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<YOLOv7PPUResult> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const YOLOv7PPUResult& a, const YOLOv7PPUResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv7PPUResult> result;

    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        result.push_back(sorted_detections[i]);

        // Suppress overlapping detections
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (!suppressed[j]) {
                float iou_value = sorted_detections[i].iou(sorted_detections[j]);
                if (iou_value > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

// Set thresholds
void YOLOv7PPUPostProcess::set_thresholds(float obj_threshold, float score_threshold,
                                           float nms_threshold) {
    if (obj_threshold >= 0.0f && obj_threshold <= 1.0f) {
        object_threshold_ = obj_threshold;
    }
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string YOLOv7PPUPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv7 PPU PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Object threshold: " << object_threshold_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  NMS threshold: " << nms_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n";

    for (auto& as : anchors_by_strides_) {
        oss << "  Stride: " << as.first << " Anchors: ";
        for (auto& a : as.second) {
            oss << a.first << ", " << a.second << " | ";
        }
        oss << "\n";
    }
    for (auto& ppu_output_name : ppu_output_names_) {
        oss << "  PPU output name: " << ppu_output_name << "\n";
    }

    return oss.str();
}
