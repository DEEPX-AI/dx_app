#include "scrfd_ppu_postprocess.h"

#include <dxrt/datatype.h>

#include <cmath>
#include <iostream>
#include <iterator>
#include <sstream>

// SCRFDPPUResult methods implementation
float SCRFDPPUResult::iou(const SCRFDPPUResult& other) const {
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

// Constructor
SCRFDPPUPostProcess::SCRFDPPUPostProcess(const int input_w, const int input_h,
                                           const float score_threshold, const float nms_threshold) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;

    // Initialize model-specific parameters for SCRFD
    ppu_output_names_ = {"FACE"};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Default constructor
SCRFDPPUPostProcess::SCRFDPPUPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.6f;
    nms_threshold_ = 0.45f;

    // Initialize model-specific parameters for SCRFD
    ppu_output_names_ = {"FACE"};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Process model outputs
std::vector<SCRFDPPUResult> SCRFDPPUPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<SCRFDPPUResult> detections;
    if (outputs.front()->type() != dxrt::DataType::FACE) {
        int i = 0;
        std::ostringstream msg;
        msg << "[DXAPP] [ER] SCRFD PPU PostProcess - Tensor output type must be "
               "dxrt::DataType::FACE.\n"
            << "  Unexpected Tensors\n";
        for (auto& o : outputs) {
            msg << "    Output shape [" << i++ << "]: (";
            for (size_t i = 0; i < o->shape().size(); ++i) {
                msg << o->shape()[i];
                if (i != o->shape().size() - 1) msg << ", ";
            }
            msg << "), Type = " << outputs.front()->type() << "\n";
        }
        msg << ", Expected (1, x ,x, x), Type = dxrt::DataType::FACE.\n"
            << "Please re-compile the model with the correct output configuration.\n";

        throw std::runtime_error(msg.str());  // 안전한 종료: 상위로 에러 전달
    }

    detections = decoding_ppu_outputs(outputs);

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results
std::vector<SCRFDPPUResult> SCRFDPPUPostProcess::decoding_ppu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<SCRFDPPUResult> detections;
    // SCRFD PPU
    // "name":"FACE", "shape":[1, num_emelements]
    auto num_elements = outputs[0]->shape()[1];
    auto* raw_data = static_cast<dxrt::DeviceFace_t*>(outputs[0]->data());
    for (int i = 0; i < num_elements; i++) {
        auto face_data = raw_data[i];

        if (face_data.score > score_threshold_) {
            SCRFDPPUResult result;
            result.confidence = face_data.score;
            auto stride = std::next(anchors_by_strides_.begin(), face_data.layer_idx)->first;
            int grid_x = face_data.grid_x;
            int grid_y = face_data.grid_y;
            int x1 = (grid_x - face_data.x) * stride;
            int y1 = (grid_y - face_data.y) * stride;
            int x2 = (grid_x + face_data.w) * stride;
            int y2 = (grid_y + face_data.h) * stride;

            result.box.emplace_back(x1);
            result.box.emplace_back(y1);
            result.box.emplace_back(x2);
            result.box.emplace_back(y2);

            for (int kpt = 0; kpt < num_landmarks_; ++kpt) {
                float lx = (grid_x + face_data.kpts[kpt][0]) * stride;
                float ly = (grid_y + face_data.kpts[kpt][1]) * stride;
                result.landmarks.emplace_back(lx);
                result.landmarks.emplace_back(ly);
            }

            detections.push_back(result);
        }
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<SCRFDPPUResult> SCRFDPPUPostProcess::apply_nms(
    const std::vector<SCRFDPPUResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<SCRFDPPUResult> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const SCRFDPPUResult& a, const SCRFDPPUResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<SCRFDPPUResult> result;

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
void SCRFDPPUPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string SCRFDPPUPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "SCRFD PPU PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
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
