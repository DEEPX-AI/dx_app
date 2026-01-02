#ifndef YOLOV5FACE_POSTPROCESS_H
#define YOLOV5FACE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOV5Face detection result structure
 * Contains bounding box coordinates, confidence scores, and facial landmarks
 */
struct YOLOv5FaceResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    std::vector<float>
        landmarks{};  // Facial landmarks (5 points * 2 coordinates)

    // Default constructor with explicit initialization
    YOLOv5FaceResult() {}

    // Parameterized constructor with move semantics for better performance
    YOLOv5FaceResult(std::vector<float> box_val, const float conf, std::vector<float> land_marks)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    YOLOv5FaceResult(const std::vector<float>& box_val, const float conf,
                     const std::vector<float>& land_marks);

    // Destructor
    ~YOLOv5FaceResult() {}

    // Copy and move constructors/operators
    YOLOv5FaceResult(const YOLOv5FaceResult& other)
        : box(other.box), confidence(other.confidence), landmarks(other.landmarks) {}
    YOLOv5FaceResult& operator=(const YOLOv5FaceResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            landmarks = other.landmarks;
        }
        return *this;
    }
    YOLOv5FaceResult(YOLOv5FaceResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          landmarks(std::move(other.landmarks)) {}
    YOLOv5FaceResult& operator=(YOLOv5FaceResult&& other) {
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

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

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
    ~YOLOv5FacePostProcess() {}

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

#endif  // YOLOV5FACE_POSTPROCESS_H
