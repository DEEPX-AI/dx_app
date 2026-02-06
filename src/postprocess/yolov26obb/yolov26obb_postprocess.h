#ifndef YOLOV26OBB_POSTPROCESS_H
#define YOLOV26OBB_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOv26Obb detection result structure
 * Contains bounding box coordinates, confidence scores, and class information
 */
struct YOLOv26ObbResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    int class_id{0};           // Object class ID (0-79 for COCO classes)
    std::string class_name{};  // Object class name

    // Default constructor with explicit initialization
    YOLOv26ObbResult() {}

    // Parameterized constructor with move semantics for better performance
    YOLOv26ObbResult(std::vector<float> box_val, const float conf, const int cls_id,
                 const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    // Legacy constructor for backward compatibility
    YOLOv26ObbResult(const std::vector<float>& box_val, const float conf, const int cls_id,
                 const std::string& cls_name);

    // Destructor
    ~YOLOv26ObbResult() {}

    // Copy and move constructors/operators
    YOLOv26ObbResult(const YOLOv26ObbResult& other)
        : box(other.box),
          confidence(other.confidence),
          class_id(other.class_id),
          class_name(other.class_name) {}
    YOLOv26ObbResult& operator=(const YOLOv26ObbResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            class_id = other.class_id;
            class_name = other.class_name;
        }
        return *this;
    }
    YOLOv26ObbResult(YOLOv26ObbResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          class_id(other.class_id),
          class_name(std::move(other.class_name)) {}
    YOLOv26ObbResult& operator=(YOLOv26ObbResult&& other) {
        if (this != &other) {
            box = std::move(other.box);
            confidence = other.confidence;
            class_id = other.class_id;
            class_name = std::move(other.class_name);
        }
        return *this;
    }

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Calculate intersection over union with another detection
    float iou(const YOLOv26ObbResult& other) const;

    // Validation methods
    bool is_invalid(int image_width, int image_height) const;
};

/**
 * @brief YOLOv26Obb post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class YOLOv26ObbPostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (default YOLO size)
    int input_height_{640};  // Model input height (default YOLO size)

    // Detection thresholds - using const for better performance
    float score_threshold_{0.5f};  // Class confidence threshold

    // Model configuration - using const where appropriate
    enum { num_classes_ = 80 };  // Number of classes (COCO dataset)

    bool is_ort_configured_{false};  // Whether ORT inference is configured

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> cpu_output_names_;  // CPU output tensor names

    // Private helper methods - const correctness
    std::vector<YOLOv26ObbResult> decoding_cpu_outputs(const dxrt::TensorPtrs& outputs) const;

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Class confidence threshold
     * @param is_ort_configured Whether ORT inference is configured (default:
     * false)
     * @note num_classes is fixed constant for COCO object detection
     */

    YOLOv26ObbPostProcess(const int input_w, const int input_h, const float score_threshold, const bool is_ort_configured = false);

    YOLOv26ObbPostProcess();

    /**
     * @brief Destructor
     */
    ~YOLOv26ObbPostProcess() {}

    /**
     * @brief Process YOLOv26Obb model outputs
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<YOLOv26ObbResult> postprocess(const dxrt::TensorPtrs& outputs);

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
    bool get_is_ort_configured() const { return is_ort_configured_; }

    // Static configuration getters
    static int get_num_classes() { return num_classes_; }

    // Model configuration getters
    const std::vector<std::string>& get_cpu_output_names() const { return cpu_output_names_; }
};

#endif  // YOLOV26OBB_POSTPROCESS_H
