/**
 * @file scrfd_face_postprocessor.hpp
 * @brief SCRFD face detection postprocessor
 */
#ifndef SCRFD_FACE_POSTPROCESSOR_HPP
#define SCRFD_FACE_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

#include "postprocess_utils.hpp"

/**
 * @brief SCRFD detection result structure
 * Contains bounding box coordinates, confidence scores, and facial landmarks
 */
struct SCRFDResult {
    // Detection data
    std::vector<float> box{};        // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};          // Detection confidence score
    std::vector<float> landmarks{};  // Facial landmarks (5 points * 2 coordinates)

    // Default constructor
    SCRFDResult() = default;

    // Parameterized constructor
    SCRFDResult(std::vector<float> box_val, const float conf, std::vector<float> land_marks)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    SCRFDResult(const std::vector<float>& box_val, const float conf,
                const std::vector<float>& land_marks);

    // Rule of Zero: compiler-generated copy/move are sufficient
    ~SCRFDResult() = default;
    SCRFDResult(const SCRFDResult&) = default;
    SCRFDResult& operator=(const SCRFDResult&) = default;
    SCRFDResult(SCRFDResult&&) = default;
    SCRFDResult& operator=(SCRFDResult&&) = default;

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Calculate intersection over union with another detection
    float iou(const SCRFDResult& other) const;

    // Validation methods
    bool is_valid() const;
    bool has_landmarks() const;
};

/**
 * @brief SCRFD post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class SCRFDPostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (default YOLO size)
    int input_height_{640};  // Model input height (default YOLO size)

    // Detection thresholds - using const for better performance
    float score_threshold_{0.5f};  // Class confidence threshold
    float nms_threshold_{0.45f};   // NMS IoU threshold

    // Model configuration - using const where appropriate
    enum { num_classes_ = 1 };        // Number of classes (face detection)
    enum { num_landmarks_ = 5 };      // Number of facial landmarks
    enum { landmarks_coords_ = 10 };  // Total landmark coordinates (5*2)

    bool is_ort_configured_{false};  // Is ONNX Runtime configured

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> cpu_output_names_;  // CPU output tensor names
    std::vector<std::string> npu_output_names_;  // NPU output tensor names (stride 8, 16, 32)
    std::map<int, std::vector<std::pair<int, int>>>
        anchors_by_strides_;  // Anchors organized by stride

    // (Moved to protected) helper method declarations

   protected:
    // Helper methods - allow subclass override of NPU decoding
    std::vector<SCRFDResult> decoding_cpu_outputs(const dxrt::TensorPtrs& outputs) const;
    virtual std::vector<SCRFDResult> decoding_npu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<SCRFDResult> apply_nms(const std::vector<SCRFDResult>& detections) const;

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Class confidence threshold
     * @param nms_threshold NMS IoU threshold
     * @param is_ort_configured Whether ONNX Runtime is configured (default:
     * false)
     * @note num_classes and num_landmarks are fixed constants for face
     * detection
     */

    SCRFDPostProcess(const int input_w, const int input_h, const float score_threshold,
                     const float nms_threshold, const bool is_ort_configured = false);

    SCRFDPostProcess();

    /**
     * @brief Destructor
     */
    virtual ~SCRFDPostProcess() = default;

    /**
     * @brief Process SCRFD model outputs
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<SCRFDResult> postprocess(const dxrt::TensorPtrs& outputs);

    /**
     * @brief Set new thresholds
     * @param score_threshold New class confidence threshold
     * @param nms_threshold New NMS IoU threshold
     */
    void set_thresholds(const float score_threshold, const float nms_threshold);

    /**
     * @brief Get current configuration
     * @return String representation of current configuration
     */
    std::string get_config_info() const;

    // Getters for current configuration - const correctness
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }

    // Static configuration getters
    static int get_num_classes() { return num_classes_; }
    static int get_num_landmarks() { return num_landmarks_; }
    static int get_landmarks_coords() { return landmarks_coords_; }

    // Model configuration getters
    const std::vector<std::string>& get_cpu_output_names() const { return cpu_output_names_; }
    const std::vector<std::string>& get_npu_output_names() const { return npu_output_names_; }
};

// ============================================================================
// Implementation (merged from .cpp - all definitions are inline)
// ============================================================================

#include <cmath>
#include <iterator>
#include <sstream>

// SCRFDResult methods implementation
inline float SCRFDResult::iou(const SCRFDResult& other) const {
    return postprocess_utils::compute_iou(box, other.box);
}

inline SCRFDPostProcess::SCRFDPostProcess(const int input_w, const int input_h,
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
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
    if (!is_ort_configured_) {
        throw std::invalid_argument(
            "ORT-OFF output postprocessing is not supported for SCRFD\n"
            "please dxrt build with USE_ORT=ON");
    }
}

// Default constructor
inline SCRFDPostProcess::SCRFDPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.6;
    nms_threshold_ = 0.45;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for SCRFD
    cpu_output_names_ = {"score_8", "score_16", "score_32", "bbox_8", "bbox_16",
                         "bbox_32", "kps_8",    "kps_16",   "kps_32"};
    npu_output_names_ = {};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Process model outputs
inline std::vector<SCRFDResult> SCRFDPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
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
inline std::vector<SCRFDResult> SCRFDPostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<SCRFDResult> detections;
    (void)outputs;
    return detections;
}

// Decode model outputs to detection results
inline std::vector<SCRFDResult> SCRFDPostProcess::decoding_cpu_outputs(
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
            if (score_ptr[j] <= score_threshold_) continue;

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
    return detections;
}

// Apply Non-Maximum Suppression
inline std::vector<SCRFDResult> SCRFDPostProcess::apply_nms(
    const std::vector<SCRFDResult>& detections) const {
    return postprocess_utils::apply_nms(detections, nms_threshold_);
}

// Set thresholds
inline void SCRFDPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
inline std::string SCRFDPostProcess::get_config_info() const {
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

#endif  // SCRFD_FACE_POSTPROCESSOR_HPP
