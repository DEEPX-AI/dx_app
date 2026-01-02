#ifndef SCRFD_POSTPROCESS_H
#define SCRFD_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief SCRFD detection result structure
 * Contains bounding box coordinates, confidence scores, and facial landmarks
 */
struct SCRFDResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};        // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};          // Detection confidence score
    std::vector<float> landmarks{};  // Facial landmarks (5 points * 2 coordinates)

    // Default constructor with explicit initialization
    SCRFDResult() {}

    // Parameterized constructor with move semantics for better performance
    SCRFDResult(std::vector<float> box_val, const float conf, std::vector<float> land_marks)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    SCRFDResult(const std::vector<float>& box_val, const float conf,
                const std::vector<float>& land_marks);

    // Destructor
    ~SCRFDResult() {}

    // Copy and move constructors/operators
    SCRFDResult(const SCRFDResult& other)
        : box(other.box), confidence(other.confidence), landmarks(other.landmarks) {}
    SCRFDResult& operator=(const SCRFDResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            landmarks = other.landmarks;
        }
        return *this;
    }
    SCRFDResult(SCRFDResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          landmarks(std::move(other.landmarks)) {}
    SCRFDResult& operator=(SCRFDResult&& other) {
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
    ~SCRFDPostProcess() {}

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

#endif  // SCRFD_POSTPROCESS_H
