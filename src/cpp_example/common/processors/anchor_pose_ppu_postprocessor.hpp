/**
 * @file anchor_pose_ppu_postprocessor.hpp
 * @brief YOLOv5Pose PPU estimation postprocessor
 *
 * Part of DX-APP v3.0.0 refactoring.
 */
#ifndef ANCHOR_POSE_PPU_POSTPROCESSOR_HPP
#define ANCHOR_POSE_PPU_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

#include "postprocess_utils.hpp"

/**
 * @brief YOLOV5Pose PPU detection result structure
 * Contains bounding box coordinates, confidence scores, and pose landmarks
 */
struct YOLOv5PosePPUResult {
    // Detection data
    std::vector<float> box{};        // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};          // Detection confidence score
    std::vector<float> landmarks{};  // Pose landmarks (17 points * 3 coordinates)

    // Default constructor
    YOLOv5PosePPUResult() = default;

    // Parameterized constructor
    YOLOv5PosePPUResult(std::vector<float> box_val, const float conf,
                         std::vector<float> land_marks)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    YOLOv5PosePPUResult(const std::vector<float>& box_val, const float conf,
                         const std::vector<float>& land_marks);

    // Rule of Zero: compiler-generated copy/move are sufficient
    ~YOLOv5PosePPUResult() = default;
    YOLOv5PosePPUResult(const YOLOv5PosePPUResult&) = default;
    YOLOv5PosePPUResult& operator=(const YOLOv5PosePPUResult&) = default;
    YOLOv5PosePPUResult(YOLOv5PosePPUResult&&) = default;
    YOLOv5PosePPUResult& operator=(YOLOv5PosePPUResult&&) = default;

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Calculate intersection over union with another detection
    float iou(const YOLOv5PosePPUResult& other) const;

    // Validation methods
    bool is_valid() const;
    bool is_invalid(int image_width, int image_height) const;
    bool has_landmarks() const;
};

/**
 * @brief YOLOV5Pose PPU post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class YOLOv5PosePPUPostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (default YOLO size)
    int input_height_{640};  // Model input height (default YOLO size)
    int image_width_{0};     // Original image width
    int image_height_{0};    // Original image height

    // Detection thresholds - using const for better performance
    float score_threshold_{0.5f};  // Class confidence threshold
    float nms_threshold_{0.45f};   // NMS IoU threshold

    // Model configuration - using const where appropriate
    enum { num_classes_ = 1 };        // Number of classes (pose detection)
    enum { num_landmarks_ = 17 };     // Number of facial landmarks
    enum { landmarks_coords_ = 34 };  // Total landmark coordinates (17*2)

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> ppu_output_names_;  // PPU output tensor names
    std::map<int, std::vector<std::pair<int, int>>>
        anchors_by_strides_;  // Anchors organized by stride

    // Private helper methods - const correctness
    std::vector<YOLOv5PosePPUResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv5PosePPUResult> apply_nms(
        const std::vector<YOLOv5PosePPUResult>& detections) const;

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Class confidence threshold
     * @param nms_threshold NMS IoU threshold
     * @note num_classes and num_landmarks are fixed constants for face
     * detection
     */

    YOLOv5PosePPUPostProcess(const int input_w, const int input_h, const float score_threshold,
                              const float nms_threshold);

    YOLOv5PosePPUPostProcess();

    /**
     * @brief Destructor
     */
    ~YOLOv5PosePPUPostProcess() = default;

    /**
     * @brief Process YOLOV5Pose model outputs
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<YOLOv5PosePPUResult> postprocess(const dxrt::TensorPtrs& outputs);

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

    const std::map<int, std::vector<std::pair<int, int>>>& get_anchors_by_strides() const {
        return anchors_by_strides_;
    }

    // Model configuration getters
    const std::vector<std::string>& get_ppu_output_names() const { return ppu_output_names_; }
};

// ============================================================================
// Implementation (merged from .cpp - all definitions are inline)
// ============================================================================

#include <dxrt/datatype.h>

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

// YOLOv5PosePPUResult methods implementation
inline float YOLOv5PosePPUResult::iou(const YOLOv5PosePPUResult& other) const {
    return postprocess_utils::compute_iou(box, other.box);
}

inline bool YOLOv5PosePPUResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

inline YOLOv5PosePPUPostProcess::YOLOv5PosePPUPostProcess(const int input_w, const int input_h,
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
inline YOLOv5PosePPUPostProcess::YOLOv5PosePPUPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.5;
    nms_threshold_ = 0.45;

    // Initialize model-specific parameters for YOLOv5Pose
    ppu_output_names_ = {"POSE"};
    anchors_by_strides_ = {{8, {{19, 27}, {44, 40}, {38, 94}}},
                           {16, {{96, 68}, {86, 152}, {180, 137}}},
                           {32, {{140, 301}, {303, 264}, {238, 542}}},
                           {64, {{436, 615}, {739, 380}, {925, 792}}}};
}

// Process model outputs
inline std::vector<YOLOv5PosePPUResult> YOLOv5PosePPUPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    std::vector<YOLOv5PosePPUResult> detections;

    if (outputs.front()->type() != dxrt::DataType::POSE) {
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv5Pose PPU PostProcess - Tensor output type must be "
               "dxrt::DataType::POSE.\n"
            << "  Unexpected Tensors\n";
        msg << postprocess_utils::format_tensor_shapes_with_type(outputs);
        msg << ", Expected (1, x ,x, x), Type = dxrt::DataType::POSE.\n"
            << "Please re-compile the model with the correct output configuration.\n";

        throw PostprocessConfigError(msg.str());  // Safe termination: propagate error to caller
    }

    detections = decoding_ppu_outputs(outputs);
    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Decode model outputs to detection results
inline std::vector<YOLOv5PosePPUResult> YOLOv5PosePPUPostProcess::decoding_ppu_outputs(
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
inline std::vector<YOLOv5PosePPUResult> YOLOv5PosePPUPostProcess::apply_nms(
    const std::vector<YOLOv5PosePPUResult>& detections) const {
    return postprocess_utils::apply_nms(detections, nms_threshold_);
}

// Set thresholds
inline void YOLOv5PosePPUPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
inline std::string YOLOv5PosePPUPostProcess::get_config_info() const {
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

#endif  // ANCHOR_POSE_PPU_POSTPROCESSOR_HPP
