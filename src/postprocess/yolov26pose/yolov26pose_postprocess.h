#ifndef YOLOV26POSE_POSTPROCESS_H
#define YOLOV26POSE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOV26Pose detection result structure
 * Contains bounding box coordinates, confidence scores, and pose landmarks
 */
struct YOLOv26PoseResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    std::vector<float>
        landmarks{};  // Pose landmarks (17 points * 3 coordinates)

    // Default constructor with explicit initialization
    YOLOv26PoseResult() {}

    // Parameterized constructor with move semantics for better performance
    YOLOv26PoseResult(std::vector<float> box_val, const float conf,
                     std::vector<float> land_marks)
        : box(std::move(box_val)),
          confidence(conf),
          landmarks(std::move(land_marks)) {}

    // Legacy constructor for backward compatibility
    YOLOv26PoseResult(const std::vector<float>& box_val, const float conf,
                     const std::vector<float>& land_marks);

    // Destructor
    ~YOLOv26PoseResult() {}

    // Copy and move constructors/operators
    YOLOv26PoseResult(const YOLOv26PoseResult& other)
        : box(other.box),
          confidence(other.confidence),
          landmarks(other.landmarks) {}
    YOLOv26PoseResult& operator=(const YOLOv26PoseResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            landmarks = other.landmarks;
        }
        return *this;
    }
    YOLOv26PoseResult(YOLOv26PoseResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          landmarks(std::move(other.landmarks)) {}
    YOLOv26PoseResult& operator=(YOLOv26PoseResult&& other) {
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
    float iou(const YOLOv26PoseResult& other) const;

    // Validation methods
    bool is_valid() const;
    bool is_invalid(int image_width, int image_height) const;
    bool has_landmarks() const;
};

/**
 * @brief YOLOV26Pose post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class YOLOv26PosePostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (default YOLO size)
    int input_height_{640};  // Model input height (default YOLO size)
    int image_width_{0};     // Original image width
    int image_height_{0};    // Original image height

    // Detection thresholds
    float score_threshold_{0.5f};   // Score threshold (obj score)

    // Model configuration - using const where appropriate
    enum { num_classes_ = 1 };        // Number of classes (pose detection)
    enum { num_landmarks_ = 17 };     // Number of facial landmarks
    enum { landmarks_coords_ = 34 };  // Total landmark coordinates (17*2)

    bool is_ort_configured_{false};  // Whether ORT is configured

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> cpu_output_names_;  // CPU output tensor names

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Class confidence threshold
     * @param is_ort_configured Whether ORT is configured (default: false)
     * @note num_classes and num_landmarks are fixed constants for face
     * detection
     */

    YOLOv26PosePostProcess(const int input_w, const int input_h,
                          const float score_threshold,
                          const bool is_ort_configured = false);

    YOLOv26PosePostProcess();

    /**
     * @brief Destructor
     */
    ~YOLOv26PosePostProcess() {}

    /**
     * @brief Process YOLOV26Pose model outputs (NMS-free)
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<YOLOv26PoseResult> postprocess(const dxrt::TensorPtrs& outputs);

    /**
     * @brief Set new thresholds
     * @param score_threshold New class confidence threshold
     */
    void set_thresholds(const float score_threshold);
    /**
     * @brief Get current configuration
     * @return String representation of current configuration
     */
    std::string get_config_info() const;

    // Getters for current configuration - const correctness
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }

    // Static configuration getters
    static int get_num_classes() { return num_classes_; }
    static int get_num_landmarks() { return num_landmarks_; }
    static int get_landmarks_coords() { return landmarks_coords_; }

    // Model configuration getters
    const std::vector<std::string>& get_cpu_output_names() const {
        return cpu_output_names_;
    }
};

#endif  // YOLOV26POSE_POSTPROCESS_H
