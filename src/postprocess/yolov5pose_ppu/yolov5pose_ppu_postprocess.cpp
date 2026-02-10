#include "yolov5pose_ppu_postprocess.h"

#include <dxrt/datatype.h>

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

// YOLOv5PosePPUResult methods implementation
float YOLOv5PosePPUResult::iou(const YOLOv5PosePPUResult& other) const {
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

bool YOLOv5PosePPUResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv5PosePPUPostProcess::YOLOv5PosePPUPostProcess(const int input_w, const int input_h,
                                                     const float score_threshold,
                                                     const float nms_threshold) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;

    // Initialize model-specific parameters for YOLOv5Pose
    ppu_output_names_ = {"POSE"};
    anchors_by_strides_ = {{8, {{19, 27}, {44, 40}, {38, 94}}},
                           {16, {{96, 68}, {86, 152}, {180, 137}}},
                           {32, {{140, 301}, {303, 264}, {238, 542}}},
                           {64, {{436, 615}, {739, 380}, {925, 792}}}};
}

// Default constructor
YOLOv5PosePPUPostProcess::YOLOv5PosePPUPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.5f;
    nms_threshold_ = 0.45f;

    // Initialize model-specific parameters for YOLOv5Pose
    ppu_output_names_ = {"POSE"};
    anchors_by_strides_ = {{8, {{19, 27}, {44, 40}, {38, 94}}},
                           {16, {{96, 68}, {86, 152}, {180, 137}}},
                           {32, {{140, 301}, {303, 264}, {238, 542}}},
                           {64, {{436, 615}, {739, 380}, {925, 792}}}};
}

// Process model outputs
std::vector<YOLOv5PosePPUResult> YOLOv5PosePPUPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    std::vector<YOLOv5PosePPUResult> detections;

    if (outputs.front()->type() != dxrt::DataType::POSE) {
        int i = 0;
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv5Pose PPU PostProcess - Tensor output type must be "
               "dxrt::DataType::POSE.\n"
            << "  Unexpected Tensors\n";
        for (auto& o : outputs) {
            msg << "    Output shape [" << i++ << "]: (";
            for (size_t i = 0; i < o->shape().size(); ++i) {
                msg << o->shape()[i];
                if (i != o->shape().size() - 1) msg << ", ";
            }
            msg << "), Type = " << outputs.front()->type() << "\n";
        }
        msg << ", Expected (1, x ,x, x), Type = dxrt::DataType::POSE.\n"
            << "Please re-compile the model with the correct output configuration.\n";

        throw std::runtime_error(msg.str());  // 안전한 종료: 상위로 에러 전달
    }

    detections = decoding_ppu_outputs(outputs);
    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv5PosePPUResult> YOLOv5PosePPUPostProcess::decoding_ppu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5PosePPUResult> detections;
    // YOLOv5Pose PPU
    // "name":"POSE", "shape":[1, num_emelements]
    auto num_elements = outputs[0]->shape()[1];
    auto* raw_data = static_cast<dxrt::DevicePose_t*>(outputs[0]->data());
    for (int i = 0; i < num_elements; i++) {
        auto pose_data = raw_data[i];

        if (pose_data.score < score_threshold_) continue;

        YOLOv5PosePPUResult result;
        int gX = pose_data.grid_x;
        int gY = pose_data.grid_y;

        auto stride = std::next(anchors_by_strides_.begin(), pose_data.layer_idx)->first;
        const auto& anchors = anchors_by_strides_.at(stride);

        float x, y, w, h;

        x = (pose_data.x * 2. - 0.5 + gX) * stride;
        y = (pose_data.y * 2. - 0.5 + gY) * stride;
        w = (pose_data.w * pose_data.w * 4.) * anchors[pose_data.box_idx].first;
        h = (pose_data.h * pose_data.h * 4.) * anchors[pose_data.box_idx].second;

        result.confidence = pose_data.score;
        result.box.emplace_back(x - w / 2);
        result.box.emplace_back(y - h / 2);
        result.box.emplace_back(x + w / 2);
        result.box.emplace_back(y + h / 2);

        result.confidence = pose_data.score;

        for (int kpt = 0; kpt < num_landmarks_; ++kpt) {
            float lx = (pose_data.grid_x - 0.5 + pose_data.kpts[kpt][0] * 2) * stride;
            float ly = (pose_data.grid_y - 0.5 + pose_data.kpts[kpt][1] * 2) * stride;
            float ls = pose_data.kpts[kpt][2];

            result.landmarks.emplace_back(lx);
            result.landmarks.emplace_back(ly);
            result.landmarks.emplace_back(ls);
        }
        detections.push_back(result);
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOv5PosePPUResult> YOLOv5PosePPUPostProcess::apply_nms(
    const std::vector<YOLOv5PosePPUResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<YOLOv5PosePPUResult> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const YOLOv5PosePPUResult& a, const YOLOv5PosePPUResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv5PosePPUResult> result;

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
void YOLOv5PosePPUPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string YOLOv5PosePPUPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOV5Pose PostProcess Configuration:\n"
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
