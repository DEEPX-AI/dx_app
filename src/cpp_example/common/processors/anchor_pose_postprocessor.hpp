/**
 * @file anchor_pose_postprocessor.hpp
 * @brief YOLOv5Pose estimation postprocessor with anchor-based decoding
 */
#ifndef ANCHOR_POSE_POSTPROCESSOR_HPP
#define ANCHOR_POSE_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

#include "postprocess_utils.hpp"

/**
 * @brief YOLOV5Pose detection result structure
 * Contains bounding box coordinates, confidence scores, and pose landmarks
 */
struct YOLOv5PoseResult {
    // Detection data
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    std::vector<float>
        landmarks{};  // Pose landmarks (17 points * 3 coordinates)

    // Default constructor
    YOLOv5PoseResult() = default;

    // Parameterized constructor
    YOLOv5PoseResult(std::vector<float> box_val, const float conf,
                     std::vector<float> land_marks)
        : box(std::move(box_val)),
          confidence(conf),
          landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    YOLOv5PoseResult(const std::vector<float>& box_val, const float conf,
                     const std::vector<float>& land_marks);

    // Rule of Zero: compiler-generated copy/move are sufficient
    ~YOLOv5PoseResult() = default;
    YOLOv5PoseResult(const YOLOv5PoseResult&) = default;
    YOLOv5PoseResult& operator=(const YOLOv5PoseResult&) = default;
    YOLOv5PoseResult(YOLOv5PoseResult&&) = default;
    YOLOv5PoseResult& operator=(YOLOv5PoseResult&&) = default;

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Calculate intersection over union with another detection
    float iou(const YOLOv5PoseResult& other) const;

    // Validation methods
    bool is_valid() const;
    bool is_invalid(int image_width, int image_height) const;
    bool has_landmarks() const;
};

/**
 * @brief YOLOV5Pose post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class YOLOv5PosePostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (default YOLO size)
    int input_height_{640};  // Model input height (default YOLO size)
    int image_width_{0};     // Original image width
    int image_height_{0};    // Original image height

    // Detection thresholds - using const for better performance
    float object_threshold_{0.5f};  // Object confidence threshold
    float score_threshold_{0.5f};   // Class confidence threshold
    float nms_threshold_{0.45f};    // NMS IoU threshold

    // Model configuration - using const where appropriate
    enum { num_classes_ = 1 };        // Number of classes (pose detection)
    enum { num_landmarks_ = 17 };     // Number of facial landmarks
    enum { landmarks_coords_ = 34 };  // Total landmark coordinates (17*2)

    bool is_ort_configured_{false};  // Whether ORT is configured

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> cpu_output_names_;  // CPU output tensor names
    std::vector<std::string>
        npu_output_names_;  // NPU output tensor names (stride 8,16,32)
    std::vector<std::vector<int>>
        anchors_;               // Anchor points (3 pairs per scale)
    std::vector<int> strides_;  // Feature map strides

    // Private helper methods - const correctness
    std::vector<YOLOv5PoseResult> decoding_cpu_outputs(
        const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv5PoseResult> decoding_npu_outputs(
        const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv5PoseResult> apply_nms(
        const std::vector<YOLOv5PoseResult>& detections) const;

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param obj_threshold Object confidence threshold
     * @param score_threshold Class confidence threshold
     * @param nms_threshold NMS IoU threshold
     * @param is_ort_configured Whether ORT is configured (default: false)
     * @note num_classes and num_landmarks are fixed constants for face
     * detection
     */

    YOLOv5PosePostProcess(const int input_w, const int input_h,
                          const float obj_threshold,
                          const float score_threshold,
                          const float nms_threshold,
                          const bool is_ort_configured = false);

    YOLOv5PosePostProcess();

    /**
     * @brief Destructor
     */
    ~YOLOv5PosePostProcess() = default;

    /**
     * @brief Process YOLOV5Face model outputs
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<YOLOv5PoseResult> postprocess(const dxrt::TensorPtrs& outputs);

    /**
     * @brief Set new thresholds
     * @param obj_threshold New object confidence threshold
     * @param score_threshold New class confidence threshold
     * @param nms_threshold New NMS IoU threshold
     */
    void set_thresholds(const float obj_threshold, const float score_threshold,
                        const float nms_threshold);

    /**
     * @brief Get current configuration
     * @return String representation of current configuration
     */
    std::string get_config_info() const;

    // Getters for current configuration - const correctness
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_object_threshold() const { return object_threshold_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }

    // Static configuration getters
    static int get_num_classes() { return num_classes_; }
    static int get_num_landmarks() { return num_landmarks_; }
    static int get_landmarks_coords() { return landmarks_coords_; }

    // Model configuration getters
    const std::vector<std::string>& get_cpu_output_names() const {
        return cpu_output_names_;
    }
    const std::vector<std::string>& get_npu_output_names() const {
        return npu_output_names_;
    }
    const std::vector<std::vector<int>>& get_anchors() const {
        return anchors_;
    }
    const std::vector<int>& get_strides() const { return strides_; }
};

// ============================================================================
// Implementation (merged from .cpp - all definitions are inline)
// ============================================================================

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

// YOLOv5PoseResult methods implementation
inline float YOLOv5PoseResult::iou(const YOLOv5PoseResult& other) const {
    return postprocess_utils::compute_iou(box, other.box);
}

inline bool YOLOv5PoseResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

inline YOLOv5PosePostProcess::YOLOv5PosePostProcess(const int input_w, const int input_h,
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
inline YOLOv5PosePostProcess::YOLOv5PosePostProcess() {
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
inline std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
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
inline std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5PoseResult> detections;
    (void)outputs;
    return detections;
}

// Decode model outputs to detection results
inline std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::decoding_cpu_outputs(
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
inline std::vector<YOLOv5PoseResult> YOLOv5PosePostProcess::apply_nms(
    const std::vector<YOLOv5PoseResult>& detections) const {
    return postprocess_utils::apply_nms(detections, nms_threshold_);
}

// Set thresholds
inline void YOLOv5PosePostProcess::set_thresholds(float obj_threshold, float score_threshold,
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
inline std::string YOLOv5PosePostProcess::get_config_info() const {
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

#endif  // ANCHOR_POSE_POSTPROCESSOR_HPP
