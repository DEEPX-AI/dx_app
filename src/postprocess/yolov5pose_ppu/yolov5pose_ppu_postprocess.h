#ifndef YOLOV5POSE_PPU_POSTPROCESS_H
#define YOLOV5POSE_PPU_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOV5Pose PPU detection result structure
 * Contains bounding box coordinates, confidence scores, and pose landmarks
 */
struct YOLOv5PosePPUResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};        // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};          // Detection confidence score
    std::vector<float> landmarks{};  // Pose landmarks (17 points * 3 coordinates)

    // Default constructor with explicit initialization
    YOLOv5PosePPUResult() {}

    // Parameterized constructor with move semantics for better performance
    YOLOv5PosePPUResult(std::vector<float> box_val, const float conf,
                         std::vector<float> land_marks)
        : box(std::move(box_val)), confidence(conf), landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    YOLOv5PosePPUResult(const std::vector<float>& box_val, const float conf,
                         const std::vector<float>& land_marks);

    // Destructor
    ~YOLOv5PosePPUResult() {}

    // Copy and move constructors/operators
    YOLOv5PosePPUResult(const YOLOv5PosePPUResult& other)
        : box(other.box), confidence(other.confidence), landmarks(other.landmarks) {}
    YOLOv5PosePPUResult& operator=(const YOLOv5PosePPUResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            landmarks = other.landmarks;
        }
        return *this;
    }
    YOLOv5PosePPUResult(YOLOv5PosePPUResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          landmarks(std::move(other.landmarks)) {}
    YOLOv5PosePPUResult& operator=(YOLOv5PosePPUResult&& other) {
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
    ~YOLOv5PosePPUPostProcess() {}

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

#endif  // YOLOV5POSE_PPU_POSTPROCESS_H
