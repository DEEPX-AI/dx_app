#include "yolov5face_postprocess.h"

#include <cmath>
#include <iterator>
#include <sstream>

// YOLOv5FaceResult methods implementation
float YOLOv5FaceResult::iou(const YOLOv5FaceResult& other) const {
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
YOLOv5FacePostProcess::YOLOv5FacePostProcess(const int input_w, const int input_h,
                                             const float obj_threshold, const float score_threshold,
                                             const float nms_threshold,
                                             const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    object_threshold_ = obj_threshold;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for YOLOv5Face
    cpu_output_names_ = {"704"};
    npu_output_names_ = {"/model.23/m.0/Conv_output_0", "/model.23/m.1/Conv_output_0",
                         "/model.23/m.2/Conv_output_0"};
    anchors_by_strides_ = {{8, {{4, 5}, {8, 10}, {13, 16}}},
                           {16, {{23, 29}, {43, 55}, {73, 105}}},
                           {32, {{146, 217}, {231, 300}, {335, 433}}}};
}

// Default constructor
YOLOv5FacePostProcess::YOLOv5FacePostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    object_threshold_ = 0.5f;
    score_threshold_ = 0.6f;
    nms_threshold_ = 0.45f;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for YOLOv5Face
    cpu_output_names_ = {"704"};
    npu_output_names_ = {"/model.23/m.0/Conv_output_0", "/model.23/m.1/Conv_output_0",
                         "/model.23/m.2/Conv_output_0"};
    anchors_by_strides_ = {{8, {{4, 5}, {8, 10}, {13, 16}}},
                           {16, {{23, 29}, {43, 55}, {73, 105}}},
                           {32, {{146, 217}, {231, 300}, {335, 433}}}};
}

// Process model outputs
std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    auto aligned_outputs = align_tensors(outputs);

    std::vector<YOLOv5FaceResult> detections;
    if (is_ort_configured_) {
        detections = decoding_cpu_outputs(aligned_outputs);
    } else {
        detections = decoding_npu_outputs(aligned_outputs);
    }

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

dxrt::TensorPtrs YOLOv5FacePostProcess::align_tensors(const dxrt::TensorPtrs& outputs) const {
    dxrt::TensorPtrs aligned;
    if (is_ort_configured_) {
        for (const auto& output : outputs) {
            if (output->shape().size() == 3) {
                aligned.push_back(output);
                break;
            }
        }
        return aligned;  // ORT inference does not require reordering
    } else {
        // Align outputs based on anchors_by_strides
        for (const auto& as : anchors_by_strides_) {
            for (const auto& output : outputs) {
                if (output->shape().size() == 4 && output->shape()[2] == input_width_ / as.first &&
                    output->shape()[3] == input_height_ / as.first &&
                    output->shape()[1] == static_cast<int64_t>(as.second.size() * 16)) {
                    aligned.push_back(output);
                    break;
                }
            }
        }
    }
    if (aligned.empty()) {
        std::cerr << "[DXAPP] [ERROR] Failed to align output tensors based on "
                     "anchors_by_strides."
                  << std::endl;
        return outputs;  // Fallback to original outputs if alignment fails
    }
    return aligned;
}

// Decode model outputs to detection results
std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5FaceResult> detections;

    // YOLOFaceV5 typically has 3 output tensors for different scales
    // Each output contains: [batch, anchors * 16, grid_y, grid_x]
    // Where 16 = [x, y, w, h, obj_conf, cls_conf, 5 + num_landmarks*2]

    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const float* output = static_cast<const float*>(outputs[output_idx]->data());
        auto stride = std::next(anchors_by_strides_.begin(), output_idx)->first;
        const auto& anchors = anchors_by_strides_.at(stride);
        int grid_x_size = input_width_ / stride;
        int grid_y_size = input_height_ / stride;
        // Process each grid cell
        for (int anchor = 0; anchor < static_cast<int>(anchors.size()); ++anchor) {
            int anchor_width = anchors[anchor].first;
            int anchor_height = anchors[anchor].second;
            for (int grid_y = 0; grid_y < grid_y_size; ++grid_y) {
                for (int grid_x = 0; grid_x < grid_x_size; ++grid_x) {
                    int objectness_idx =
                        ((anchor * (num_classes_ + 15)) + 15) * grid_x_size * grid_y_size +
                        grid_y * grid_x_size + grid_x;
                    auto objectness_score = sigmoid(output[objectness_idx]);
                    auto cls_conf_idx = ((anchor * 16) + 4) * grid_x_size * grid_y_size +
                                        grid_y * grid_x_size + grid_x;
                    auto cls_conf = objectness_score * sigmoid(output[cls_conf_idx]);
                    if (cls_conf < score_threshold_) {
                        continue;
                    }
                    YOLOv5FaceResult result;
                    std::vector<float> box_temp{0.f, 0.f, 0.f, 0.f};
                    result.confidence = cls_conf;
                    for (int i = 0; i < 4; i++) {
                        int box_idx = ((anchor * 16) + i) * grid_x_size * grid_y_size +
                                      grid_y * grid_x_size + grid_x;
                        box_temp[i] = output[box_idx];
                    }
                    box_temp[0] = (sigmoid(box_temp[0]) * 2.0f - 0.5 + grid_x) * stride;
                    box_temp[1] = (sigmoid(box_temp[1]) * 2.0f - 0.5 + grid_y) * stride;
                    box_temp[2] = pow(sigmoid(box_temp[2]) * 2.0f, 2.0f) * anchor_width;
                    box_temp[3] = pow(sigmoid(box_temp[3]) * 2.0f, 2.0f) * anchor_height;
                    result.box.emplace_back(box_temp[0] - box_temp[2] / 2.0f);
                    result.box.emplace_back(box_temp[1] - box_temp[3] / 2.0f);
                    result.box.emplace_back(box_temp[0] + box_temp[2] / 2.0f);
                    result.box.emplace_back(box_temp[1] + box_temp[3] / 2.0f);

                    for (int kpt = 0; kpt < num_landmarks_; ++kpt) {
                        int kpt_idx = ((anchor * 16) + 5 + (kpt * 2)) * grid_x_size * grid_y_size +
                                      grid_y * grid_x_size + grid_x;
                        result.landmarks.emplace_back(output[kpt_idx] * anchor_width +
                                                      (grid_x * stride));
                        kpt_idx = ((anchor * 16) + 6 + (kpt * 2)) * grid_x_size * grid_y_size +
                                  grid_y * grid_x_size + grid_x;
                        result.landmarks.emplace_back(output[kpt_idx] * anchor_height +
                                                      (grid_y * stride));
                    }
                    detections.push_back(result);
                }
            }
        }
    }
    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5FaceResult> detections;

    // YOLOFaceV5 typically has 1 output tensor
    // output tensor contains: [batch, number of detections, 16]
    // Where 16 = [x, y, w, h, obj_conf, num_landmarks*2, cls_conf]

    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const float* output = static_cast<const float*>(outputs[output_idx]->data());
        auto num_dets = outputs[output_idx]->shape()[1];
        auto attribute_size = outputs[output_idx]->shape()[2];
        for (int i = 0; i < num_dets; ++i) {
            const float* det = output + i * attribute_size;
            auto objectness_score = det[4];
            if (objectness_score < object_threshold_) {
                continue;
            }
            auto cls_conf = det[15];
            auto conf = objectness_score * cls_conf;
            if (conf < score_threshold_) {
                continue;
            }
            YOLOv5FaceResult result;
            std::vector<float> box_temp{0.f, 0.f, 0.f, 0.f};
            result.confidence = conf;
            result.box.emplace_back(det[0] - det[2] / 2.0f);
            result.box.emplace_back(det[1] - det[3] / 2.0f);
            result.box.emplace_back(det[0] + det[2] / 2.0f);
            result.box.emplace_back(det[1] + det[3] / 2.0f);

            for (int kpt = 0; kpt < num_landmarks_; ++kpt) {
                int kpt_idx = kpt * 2 + 5;
                result.landmarks.emplace_back(det[kpt_idx + 0]);
                result.landmarks.emplace_back(det[kpt_idx + 1]);
            }
            detections.push_back(result);
        }
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::apply_nms(
    const std::vector<YOLOv5FaceResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<YOLOv5FaceResult> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const YOLOv5FaceResult& a, const YOLOv5FaceResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv5FaceResult> result;

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
void YOLOv5FacePostProcess::set_thresholds(float obj_threshold, float score_threshold,
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
std::string YOLOv5FacePostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv5Face PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Object threshold: " << object_threshold_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  NMS threshold: " << nms_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n"
        << "  Is Ort Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";

    for (auto& as : anchors_by_strides_) {
        oss << "  Stride: " << as.first << " Anchors: ";
        for (auto& a : as.second) {
            oss << a.first << ", " << a.second << " | ";
        }
        oss << "\n";
    }
    for (auto& cpu_output_name : cpu_output_names_) {
        oss << "  CPU output name: " << cpu_output_name << "\n";
    }
    for (auto& npu_output_name : npu_output_names_) {
        oss << "  NPU output name: " << npu_output_name << "\n";
    }

    return oss.str();
}
