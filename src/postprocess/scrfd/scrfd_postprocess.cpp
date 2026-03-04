#include "scrfd_postprocess.h"

#include <cmath>
#include <iterator>
#include <sstream>

// SCRFDResult methods implementation
float SCRFDResult::iou(const SCRFDResult& other) const {
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
SCRFDPostProcess::SCRFDPostProcess(const int input_w, const int input_h,
                                   const float score_threshold, const float nms_threshold,
                                   const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for SCRFD
    cpu_output_names_ = {"score_8", "score_16", "score_32", "bbox_8", "bbox_16",
                         "bbox_32", "kps_8",    "kps_16",   "kps_32"};
    npu_output_names_ = {};
    // anchors_by_strides_ 변수 선언 및 초기화 (std::map<int,
    // std::vector<std::pair<int, int>>>)
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
    if (!is_ort_configured_) {
        throw std::invalid_argument(
            "ORT-OFF output postprocessing is not supported for SCRFD\n"
            "please dxrt build with USE_ORT=ON");
    }
}

// Default constructor
SCRFDPostProcess::SCRFDPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.6f;
    nms_threshold_ = 0.45f;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for SCRFD
    cpu_output_names_ = {"score_8", "score_16", "score_32", "bbox_8", "bbox_16",
                         "bbox_32", "kps_8",    "kps_16",   "kps_32"};
    npu_output_names_ = {};
    // anchors_by_strides_ 변수 선언 및 초기화 (std::map<int,
    // std::vector<std::pair<int, int>>>)
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Process model outputs
std::vector<SCRFDResult> SCRFDPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<SCRFDResult> detections;
    if (is_ort_configured_) {
        detections = decoding_cpu_outputs(outputs);
    } else {
        throw std::invalid_argument(
            "NPU output postprocessing is not supported for SCRFD\n"
            "please dxrt build with USE_ORT=ON");
    }

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results
std::vector<SCRFDResult> SCRFDPostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<SCRFDResult> detections;
    (void)outputs;
    return detections;
}

// Decode model outputs to detection results
std::vector<SCRFDResult> SCRFDPostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<SCRFDResult> detections;

    // SCRFD typically has 9 output tensor
    // "name":"score_8", "shape":[1, 12800, 1]
    // "name":"score_16", "shape":[1, 3200, 1]
    // "name":"score_32", "shape":[1, 800, 1]
    // "name":"bbox_8", "shape":[1, 12800, 4]
    // "name":"bbox_16", "shape":[1, 3200, 4]
    // "name":"bbox_32", "shape":[1, 800, 4]
    // "name":"kps_8", "shape":[1, 12800, 10]
    // "name":"kps_16", "shape":[1, 3200, 10]
    // "name":"kps_32", "shape":[1, 800, 10]

    auto* score_8 = static_cast<float*>(outputs[0]->data());
    auto* score_16 = static_cast<float*>(outputs[1]->data());
    auto* score_32 = static_cast<float*>(outputs[2]->data());
    std::vector<float*> score_list = {score_8, score_16, score_32};
    auto* bbox_8 = static_cast<float*>(outputs[3]->data());
    auto* bbox_16 = static_cast<float*>(outputs[4]->data());
    auto* bbox_32 = static_cast<float*>(outputs[5]->data());
    std::vector<float*> bbox_list = {bbox_8, bbox_16, bbox_32};
    auto* kpt_8 = static_cast<float*>(outputs[6]->data());
    auto* kpt_16 = static_cast<float*>(outputs[7]->data());
    auto* kpt_32 = static_cast<float*>(outputs[8]->data());
    std::vector<float*> kpt_list = {kpt_8, kpt_16, kpt_32};
    for (int i = 0; i < static_cast<int>(anchors_by_strides_.size()); i++) {
        int stride = std::next(anchors_by_strides_.begin(), i)->first;
        int num_detections = std::pow(input_width_ / stride, 2) * 2;
        for (int j = 0; j < num_detections; j++) {
            int box_idx = j * 4;
            int kpt_idx = j * 10;
            auto* score_ptr = score_list[i];
            auto* bbox_ptr = bbox_list[i];
            auto* kpt_ptr = kpt_list[i];
            if (score_ptr[j] > score_threshold_) {
                SCRFDResult result;
                result.confidence = score_ptr[j];

                int grid_x = (j / 2) % (input_width_ / stride);
                int grid_y = (j / 2) / (input_height_ / stride);

                float cx = static_cast<float>(grid_x * stride);
                float cy = static_cast<float>(grid_y * stride);

                result.box.emplace_back(cx - (bbox_ptr[box_idx + 0] * stride));
                result.box.emplace_back(cy - (bbox_ptr[box_idx + 1] * stride));
                result.box.emplace_back(cx + (bbox_ptr[box_idx + 2] * stride));
                result.box.emplace_back(cy + (bbox_ptr[box_idx + 3] * stride));

                for (int kpt = 0; kpt < num_landmarks_; ++kpt) {
                    result.landmarks.emplace_back(cx + (kpt_ptr[kpt_idx + (kpt * 2) + 0] * stride));
                    result.landmarks.emplace_back(cy + (kpt_ptr[kpt_idx + (kpt * 2) + 1] * stride));
                }
                detections.push_back(result);
            }
        }
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<SCRFDResult> SCRFDPostProcess::apply_nms(
    const std::vector<SCRFDResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<SCRFDResult> sorted_detections = detections;
    std::sort(
        sorted_detections.begin(), sorted_detections.end(),
        [](const SCRFDResult& a, const SCRFDResult& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<SCRFDResult> result;

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
void SCRFDPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string SCRFDPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "SCRFD PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
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
