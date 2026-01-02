#include "yolov5pose_postprocess.h"

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

// YOLOv5PoseResult methods implementation
float YOLOv5PoseResult::iou(const YOLOv5PoseResult& other) const {
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

bool YOLOv5PoseResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv5PosePostProcess::YOLOv5PosePostProcess(const int input_w, const int input_h,
                                             const float obj_threshold, const float score_threshold,
                                             const float nms_threshold,
                                             const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    object_threshold_ = obj_threshold;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for YOLOv5Pose
    cpu_output_names_ = {"detections"};
    npu_output_names_ = {};

    anchors_ = {};
    strides_ = {};

    if (!is_ort_configured_) {
        throw std::invalid_argument(
            "ORT-OFF output postprocessing is not supported for YOLOV5Pose\n"
            "please dxrt build with USE_ORT=ON");
    }
}

// Default constructor
YOLOv5PosePostProcess::YOLOv5PosePostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    object_threshold_ = 0.5;
    score_threshold_ = 0.5;
    nms_threshold_ = 0.45;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for YOLOv5Pose
    cpu_output_names_ = {"detections"};
    npu_output_names_ = {};
    anchors_ = {};
    strides_ = {};
}

// Process model outputs
std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<YOLOv5PoseResult> detections;
    if (is_ort_configured_) {
        detections = decoding_cpu_outputs(outputs);
    } else {
        throw std::invalid_argument(
            "NPU output postprocessing is not supported for YOLOV5Pose\n"
            "please dxrt build with USE_ORT=ON");
    }

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5PoseResult> detections;
    (void)outputs;
    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5PoseResult> detections;

    // YOLOV5Pose typically has 1 output tensor
    // output tensor contains: [batch, number of detections, 57]
    // Where 57 = [x, y, w, h, obj_conf, cls_conf, num_landmarks*3]

    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const float* output = static_cast<const float*>(outputs[output_idx]->data());
        auto num_dets = outputs[output_idx]->shape()[1];
        for (int i = 0; i < num_dets; ++i) {
            const float* det = output + i * 57;
            auto objectness_score = det[4];
            if (objectness_score < object_threshold_) {
                continue;
            }
            auto cls_conf = det[5];
            auto conf = objectness_score * cls_conf;
            if (conf < score_threshold_) {
                continue;
            }
            YOLOv5PoseResult result;
            std::vector<float> box_temp{0.f, 0.f, 0.f, 0.f};
            result.confidence = conf;
            result.box.emplace_back(det[0] - det[2] / 2.0f);
            result.box.emplace_back(det[1] - det[3] / 2.0f);
            result.box.emplace_back(det[0] + det[2] / 2.0f);
            result.box.emplace_back(det[1] + det[3] / 2.0f);

            for (int kpt = 0; kpt < num_landmarks_; ++kpt) {
                int kpt_idx = kpt * 3 + 6;
                result.landmarks.emplace_back(det[kpt_idx + 0]);
                result.landmarks.emplace_back(det[kpt_idx + 1]);
                result.landmarks.emplace_back(det[kpt_idx + 2]);
            }
            detections.push_back(result);
        }
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::apply_nms(
    const std::vector<YOLOv5PoseResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<YOLOv5PoseResult> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const YOLOv5PoseResult& a, const YOLOv5PoseResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv5PoseResult> result;

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
void YOLOv5PosePostProcess::set_thresholds(float obj_threshold, float score_threshold,
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
std::string YOLOv5PosePostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOV5Pose PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Object threshold: " << object_threshold_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  NMS threshold: " << nms_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n"
        << "  Is Ort Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";

    for (auto& anchor : anchors_) {
        oss << "  Anchor: ";
        for (auto& a : anchor) {
            oss << a << " ";
        }
        oss << "\n";
    }
    for (auto& stride : strides_) {
        oss << "  Stride: " << stride << "\n";
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
