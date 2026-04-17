/**
 * @file scrfd_face_ppu_postprocessor.hpp
 * @brief SCRFD PPU face detection postprocessor
 */
#ifndef SCRFD_FACE_PPU_POSTPROCESSOR_HPP
#define SCRFD_FACE_PPU_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

#include "postprocess_utils.hpp"

/**
 * @brief SCRFD PPU detection result structure
 * Contains bounding box coordinates, confidence scores, and facial landmarks
 */
struct SCRFDPPUResult {
    // Detection data
    std::vector<float> box{};        // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};          // Detection confidence score
    std::vector<float> landmarks{};  // Facial landmarks (5 points * 2 coordinates)

    // Default constructor
    SCRFDPPUResult() = default;

    // Parameterized constructor
    SCRFDPPUResult(std::vector<float> box_val, const float conf, std::vector<float> land_marks)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    SCRFDPPUResult(const std::vector<float>& box_val, const float conf,
                    const std::vector<float>& land_marks);

    // Rule of Zero: compiler-generated copy/move are sufficient
    ~SCRFDPPUResult() = default;
    SCRFDPPUResult(const SCRFDPPUResult&) = default;
    SCRFDPPUResult& operator=(const SCRFDPPUResult&) = default;
    SCRFDPPUResult(SCRFDPPUResult&&) = default;
    SCRFDPPUResult& operator=(SCRFDPPUResult&&) = default;

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Calculate intersection over union with another detection
    float iou(const SCRFDPPUResult& other) const;

    // Validation methods
    bool is_valid() const;
    bool has_landmarks() const;
};

/**
 * @brief SCRFD PPU post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class SCRFDPPUPostProcess {
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

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> ppu_output_names_;  // PPU(NPU) output tensor names
    std::map<int, std::vector<std::pair<int, int>>>
        anchors_by_strides_;  // Anchors organized by stride

    // (Moved to protected) helper method declarations

   protected:
    // Helper methods - allow subclass override of NPU decoding
    std::vector<SCRFDPPUResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<SCRFDPPUResult> apply_nms(const std::vector<SCRFDPPUResult>& detections) const;

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Class confidence threshold
     * @param nms_threshold NMS IoU threshold
     * false)
     * @note num_classes and num_landmarks are fixed constants for face
     * detection
     */

    SCRFDPPUPostProcess(const int input_w, const int input_h, const float score_threshold,
                         const float nms_threshold);

    SCRFDPPUPostProcess();

    /**
     * @brief Destructor
     */
    ~SCRFDPPUPostProcess() = default;

    /**
     * @brief Process SCRFD model outputs
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<SCRFDPPUResult> postprocess(const dxrt::TensorPtrs& outputs);

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
    const std::vector<std::string>& get_ppu_output_names() const { return ppu_output_names_; }
};

// ============================================================================
// Implementation (merged from .cpp - all definitions are inline)
// ============================================================================

#include <dxrt/datatype.h>

#include <cmath>
#include <iostream>
#include <iterator>
#include <sstream>

// SCRFDPPUResult methods implementation
inline float SCRFDPPUResult::iou(const SCRFDPPUResult& other) const {
    return postprocess_utils::compute_iou(box, other.box);
}

inline SCRFDPPUPostProcess::SCRFDPPUPostProcess(const int input_w, const int input_h,
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
inline SCRFDPPUPostProcess::SCRFDPPUPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.6;
    nms_threshold_ = 0.45;

    // Initialize model-specific parameters for SCRFD
    ppu_output_names_ = {"FACE"};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Process model outputs
inline std::vector<SCRFDPPUResult> SCRFDPPUPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<SCRFDPPUResult> detections;
    if (outputs.front()->type() != dxrt::DataType::FACE) {
        std::ostringstream msg;
        msg << "[DXAPP] [ER] SCRFD PPU PostProcess - Tensor output type must be "
               "dxrt::DataType::FACE.\n"
            << "  Unexpected Tensors\n";
        msg << postprocess_utils::format_tensor_shapes_with_type(outputs);
        msg << ", Expected (1, x ,x, x), Type = dxrt::DataType::FACE.\n"
            << "Please re-compile the model with the correct output configuration.\n";

        throw PostprocessConfigError(msg.str());  // Safe termination: propagate error to caller
    }

    detections = decoding_ppu_outputs(outputs);

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results
inline std::vector<SCRFDPPUResult> SCRFDPPUPostProcess::decoding_ppu_outputs(
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
inline std::vector<SCRFDPPUResult> SCRFDPPUPostProcess::apply_nms(
    const std::vector<SCRFDPPUResult>& detections) const {
    return postprocess_utils::apply_nms(detections, nms_threshold_);
}

// Set thresholds
inline void SCRFDPPUPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
inline std::string SCRFDPPUPostProcess::get_config_info() const {
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

#endif  // SCRFD_FACE_PPU_POSTPROCESSOR_HPP
