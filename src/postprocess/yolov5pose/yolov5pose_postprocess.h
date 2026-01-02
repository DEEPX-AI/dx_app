#ifndef YOLOV5POSE_POSTPROCESS_H
#define YOLOV5POSE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOV5Pose detection result structure
 * Contains bounding box coordinates, confidence scores, and pose landmarks
 */
struct YOLOv5PoseResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    std::vector<float>
        landmarks{};  // Pose landmarks (17 points * 3 coordinates)

    // Default constructor with explicit initialization
    YOLOv5PoseResult() {}

    // Parameterized constructor with move semantics for better performance
    YOLOv5PoseResult(std::vector<float> box_val, const float conf,
                     std::vector<float> land_marks)
        : box(std::move(box_val)),
          confidence(conf),
          landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    YOLOv5PoseResult(const std::vector<float>& box_val, const float conf,
                     const std::vector<float>& land_marks);

    // Destructor
    ~YOLOv5PoseResult() {}

    // Copy and move constructors/operators
    YOLOv5PoseResult(const YOLOv5PoseResult& other)
        : box(other.box),
          confidence(other.confidence),
          landmarks(other.landmarks) {}
    YOLOv5PoseResult& operator=(const YOLOv5PoseResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            landmarks = other.landmarks;
        }
        return *this;
    }
    YOLOv5PoseResult(YOLOv5PoseResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          landmarks(std::move(other.landmarks)) {}
    YOLOv5PoseResult& operator=(YOLOv5PoseResult&& other) {
        if (this != &other) {
            box = std::move(other.box);
            confidence = other.confidence;
            landmarks = std::move(other.landmarks);
        }
        return *this;
    }

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
    ~YOLOv5PosePostProcess() {}

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

#endif  // YOLOV5POSE_POSTPROCESS_H
