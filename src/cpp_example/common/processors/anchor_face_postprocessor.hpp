/**
 * @file anchor_face_postprocessor.hpp
 * @brief YOLOv5Face detection postprocessor with anchor-based decoding
 */
#ifndef ANCHOR_FACE_POSTPROCESSOR_HPP
#define ANCHOR_FACE_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

#include "postprocess_utils.hpp"

/**
 * @brief YOLOV5Face detection result structure
 * Contains bounding box coordinates, confidence scores, and facial landmarks
 */
struct YOLOv5FaceResult {
    // Detection data
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    std::vector<float>
        landmarks{};  // Facial landmarks (5 points * 2 coordinates)

    // Default constructor
    YOLOv5FaceResult() = default;

    // Parameterized constructor
    YOLOv5FaceResult(std::vector<float> box_val, const float conf, std::vector<float> land_marks)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    YOLOv5FaceResult(const std::vector<float>& box_val, const float conf,
                     const std::vector<float>& land_marks);

    // Rule of Zero: compiler-generated copy/move are sufficient
    ~YOLOv5FaceResult() = default;
    YOLOv5FaceResult(const YOLOv5FaceResult&) = default;
    YOLOv5FaceResult& operator=(const YOLOv5FaceResult&) = default;
    YOLOv5FaceResult(YOLOv5FaceResult&&) = default;
    YOLOv5FaceResult& operator=(YOLOv5FaceResult&&) = default;

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Calculate intersection over union with another detection
    float iou(const YOLOv5FaceResult& other) const;

    // Validation methods
    bool is_valid() const;
    bool has_landmarks() const;
};

/**
 * @brief YOLOV5Face post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class YOLOv5FacePostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (default YOLO size)
    int input_height_{640};  // Model input height (default YOLO size)

    // Detection thresholds - using const for better performance
    float object_threshold_{0.5f};  // Object confidence threshold
    float score_threshold_{0.5f};   // Class confidence threshold
    float nms_threshold_{0.45f};    // NMS IoU threshold

    // Model configuration - using const where appropriate
    enum { num_classes_ = 1 };        // Number of classes (face detection)
    enum { num_landmarks_ = 5 };      // Number of facial landmarks
    enum { landmarks_coords_ = 10 };  // Total landmark coordinates (5*2)

    bool is_ort_configured_{false};  // Whether ORT inference is configured

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> cpu_output_names_;  // CPU output tensor names
    std::vector<std::string> npu_output_names_;  // NPU output tensor names (stride 8,16,32)
    std::map<int, std::vector<std::pair<int, int>>>
        anchors_by_strides_;  // Anchors organized by stride

    // Private helper methods - const correctness
    std::vector<YOLOv5FaceResult> decoding_cpu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv5FaceResult> decoding_npu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv5FaceResult> apply_nms(const std::vector<YOLOv5FaceResult>& detections) const;

    // Helper: decode YOLOv7Face 21-col format detections from a single tensor
    void decode_v7face_dets_(const float* output, int num_dets, int cols,
                             std::vector<YOLOv5FaceResult>& detections) const;
    // Helper: decode YOLOv5Face 16-col format detections from a single tensor
    void decode_v5face_dets_(const float* output, int num_dets,
                             std::vector<YOLOv5FaceResult>& detections) const;

    // Helper: decode one grid cell for face NPU output.
    // Populates `result` and returns true when confidence passes threshold.
    bool decode_face_npu_cell(const float* output, int anchor, int grid_x, int grid_y,
                              int grid_x_size, int grid_y_size,
                              int anchor_width, int anchor_height,
                              int stride, YOLOv5FaceResult& result) const;

    static float sigmoid(float x) { return postprocess_utils::sigmoid(x); }

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param obj_threshold Object confidence threshold
     * @param score_threshold Class confidence threshold
     * @param nms_threshold NMS IoU threshold
     * @param is_ort_configured Whether ORT inference is configured (default:
     * false)
     * @note num_classes and num_landmarks are fixed constants for face
     * detection
     */

    YOLOv5FacePostProcess(const int input_w, const int input_h, const float obj_threshold,
                          const float score_threshold, const float nms_threshold,
                          const bool is_ort_configured = false);

    YOLOv5FacePostProcess();

    /**
     * @brief Destructor
     */
    ~YOLOv5FacePostProcess() = default;

    /**
     * @brief Process YOLOV5Face model outputs
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<YOLOv5FaceResult> postprocess(const dxrt::TensorPtrs& outputs);

    /**
     * @brief Align output tensors based on model configuration
     * @param outputs Vector of raw output tensors from the model
     * @return Vector of aligned tensors ready for postprocessing
     */
    dxrt::TensorPtrs align_tensors(const dxrt::TensorPtrs& outputs) const;

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
    bool get_is_ort_configured() const { return is_ort_configured_; }

    // Static configuration getters
    static int get_num_classes() { return num_classes_; }
    static int get_num_landmarks() { return num_landmarks_; }
    static int get_landmarks_coords() { return landmarks_coords_; }

    const std::map<int, std::vector<std::pair<int, int>>>& get_anchors_by_strides() const {
        return anchors_by_strides_;
    }

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

// YOLOv5FaceResult methods implementation
inline float YOLOv5FaceResult::iou(const YOLOv5FaceResult& other) const {
    return postprocess_utils::compute_iou(box, other.box);
}

inline YOLOv5FacePostProcess::YOLOv5FacePostProcess(const int input_w, const int input_h,
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
inline YOLOv5FacePostProcess::YOLOv5FacePostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    object_threshold_ = 0.5;
    score_threshold_ = 0.6;
    nms_threshold_ = 0.45;
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
inline std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    static bool debug_logged = false;
    if (!debug_logged && getenv("DXAPP_DEBUG")) {
        debug_logged = true;
        fprintf(stderr, "[DEBUG] is_ort_configured=%d, raw_outputs=%zu\n",
                is_ort_configured_, outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            fprintf(stderr, "[DEBUG]   output[%zu] shape:", i);
            for (auto s : outputs[i]->shape()) fprintf(stderr, " %ld", s);
            fprintf(stderr, "\n");
        }
    }

    auto aligned_outputs = align_tensors(outputs);

    std::vector<YOLOv5FaceResult> detections;
    if (is_ort_configured_) {
        detections = decoding_cpu_outputs(aligned_outputs);
    } else {
        detections = decoding_npu_outputs(aligned_outputs);
    }

    static const bool s_debug = (std::getenv("DXAPP_DEBUG") != nullptr);
    if (s_debug) {
        fprintf(stderr, "[DEBUG] pre-NMS=%zu", detections.size());
    }

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    if (getenv("DXAPP_DEBUG")) {
        fprintf(stderr, " post-NMS=%zu", detections.size());
        for (size_t d = 0; d < detections.size() && d < 5; ++d) {
            auto& b = detections[d].box;
            fprintf(stderr, " [%.1f,%.1f,%.1f,%.1f c=%.3f]",
                    b[0], b[1], b[2], b[3], detections[d].confidence);
        }
        fprintf(stderr, "\n");
    }

    return detections;
}

inline dxrt::TensorPtrs YOLOv5FacePostProcess::align_tensors(const dxrt::TensorPtrs& outputs) const {
    dxrt::TensorPtrs aligned;
    if (is_ort_configured_) {
        for (const auto& output : outputs) {
            if (output->shape().size() == 3) {
                aligned.push_back(output);
                break;
            }
        }
        return aligned;  // ORT inference does not require reordering
    }
    // Align outputs based on anchors_by_strides
    for (const auto& as : anchors_by_strides_) {
        for (const auto& output : outputs) {
            if (output->shape().size() != 4) continue;
            if (output->shape()[2] != input_width_ / as.first) continue;
            if (output->shape()[3] != input_height_ / as.first) continue;
            if (output->shape()[1] != static_cast<int64_t>(as.second.size() * 16)) continue;
            aligned.push_back(output);
            break;
        }
    }
    if (aligned.empty()) {
        throw std::runtime_error("[DXAPP] Failed to align output tensors based on "
                                "anchors_by_strides.");
    }
    return aligned;
}

// Decode model outputs to detection results
inline std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5FaceResult> detections;

    auto decode_stride_npu = [&](const float* output, int stride,
                                 const auto& anchors,
                                 int grid_x_size, int grid_y_size) {
        for (int anchor = 0; anchor < static_cast<int>(anchors.size()); ++anchor) {
            int anchor_width  = anchors[anchor].first;
            int anchor_height = anchors[anchor].second;
            for (int grid_y = 0; grid_y < grid_y_size; ++grid_y) {
                for (int grid_x = 0; grid_x < grid_x_size; ++grid_x) {
                    YOLOv5FaceResult result;
                    if (decode_face_npu_cell(output, anchor, grid_x, grid_y,
                                            grid_x_size, grid_y_size,
                                            anchor_width, anchor_height,
                                            stride, result))
                        detections.push_back(result);
                }
            }
        }
    };

    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        auto output = static_cast<const float*>(outputs[output_idx]->data());
        auto stride = std::next(anchors_by_strides_.begin(), output_idx)->first;
        const auto& anchors = anchors_by_strides_.at(stride);
        int grid_x_size = input_width_ / stride;
        int grid_y_size = input_height_ / stride;
        decode_stride_npu(output, stride, anchors, grid_x_size, grid_y_size);
    }
    return detections;
}

// Helper implementation: decode one face NPU grid cell
inline bool YOLOv5FacePostProcess::decode_face_npu_cell(
    const float* output, int anchor, int grid_x, int grid_y,
    int grid_x_size, int grid_y_size,
    int anchor_width, int anchor_height,
    int stride, YOLOv5FaceResult& result) const {
    int objectness_idx =
        ((anchor * (num_classes_ + 15)) + 15) * grid_x_size * grid_y_size +
        grid_y * grid_x_size + grid_x;
    auto objectness_score = sigmoid(output[objectness_idx]);
    auto cls_conf_idx = ((anchor * 16) + 4) * grid_x_size * grid_y_size +
                        grid_y * grid_x_size + grid_x;
    auto cls_conf = objectness_score * sigmoid(output[cls_conf_idx]);
    if (cls_conf < score_threshold_) return false;

    result.confidence = cls_conf;
    std::vector<float> box_temp{0.f, 0.f, 0.f, 0.f};
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
        result.landmarks.emplace_back(output[kpt_idx] * anchor_width + (grid_x * stride));
        kpt_idx = ((anchor * 16) + 6 + (kpt * 2)) * grid_x_size * grid_y_size +
                  grid_y * grid_x_size + grid_x;
        result.landmarks.emplace_back(output[kpt_idx] * anchor_height + (grid_y * stride));
    }
    return true;
}

// Decode model outputs to detection results
inline std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5FaceResult> detections;

    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const auto& shape = outputs[output_idx]->shape();
        auto output = static_cast<const float*>(outputs[output_idx]->data());
        auto num_dets = shape[1];
        // Detect output format: 16 cols = YOLOv5Face, >= 21 cols = YOLOv7Face
        int cols = (shape.size() >= 3) ? static_cast<int>(shape[2]) : 16;

        if (cols >= 21) {
            decode_v7face_dets_(output, static_cast<int>(num_dets), cols, detections);
        } else {
            decode_v5face_dets_(output, static_cast<int>(num_dets), detections);
        }
    }
    return detections;
}

// Helper: decode YOLOv7Face 21-col format detections
inline void YOLOv5FacePostProcess::decode_v7face_dets_(
    const float* output, int num_dets, int cols,
    std::vector<YOLOv5FaceResult>& detections) const {
    for (int i = 0; i < num_dets; ++i) {
        const float* det = output + i * cols;
        float objectness_score = det[4];
        if (objectness_score < object_threshold_) continue;

        float cls_conf = sigmoid(det[20]);  // raw logit → sigmoid
        float conf = objectness_score * cls_conf;
        if (conf < score_threshold_) continue;

        YOLOv5FaceResult result;
        result.confidence = conf;
        result.box.emplace_back(det[0] - det[2] / 2.0f);
        result.box.emplace_back(det[1] - det[3] / 2.0f);
        result.box.emplace_back(det[0] + det[2] / 2.0f);
        result.box.emplace_back(det[1] + det[3] / 2.0f);

        // Keypoints as (conf, x, y) triplets starting at index 5
        for (int kpt = 0; kpt < num_landmarks_; ++kpt) {
            int base = 5 + kpt * 3;
            result.landmarks.emplace_back(det[base + 1]);
            result.landmarks.emplace_back(det[base + 2]);
        }
        detections.push_back(result);
    }
}

// Helper: decode YOLOv5Face 16-col format detections
inline void YOLOv5FacePostProcess::decode_v5face_dets_(
    const float* output, int num_dets,
    std::vector<YOLOv5FaceResult>& detections) const {
    for (int i = 0; i < num_dets; ++i) {
        const float* det = output + i * 16;
        float objectness_score = det[4];
        if (objectness_score < object_threshold_) continue;

        float cls_conf = det[15];
        float conf = objectness_score * cls_conf;
        if (conf < score_threshold_) continue;

        YOLOv5FaceResult result;
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

// Apply Non-Maximum Suppression
inline std::vector<YOLOv5FaceResult> YOLOv5FacePostProcess::apply_nms(
    const std::vector<YOLOv5FaceResult>& detections) const {
    return postprocess_utils::apply_nms(detections, nms_threshold_);
}

// Set thresholds
inline void YOLOv5FacePostProcess::set_thresholds(float obj_threshold, float score_threshold,
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
inline std::string YOLOv5FacePostProcess::get_config_info() const {
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

#endif  // ANCHOR_FACE_POSTPROCESSOR_HPP
