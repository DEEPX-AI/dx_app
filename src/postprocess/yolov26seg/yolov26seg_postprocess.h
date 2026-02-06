#ifndef YOLOV26SEG_POSTPROCESS_H
#define YOLOV26SEG_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOv26n detection result structure
 * Contains bounding box coordinates, confidence scores, and class information
 */
struct YOLOv26SegResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    int class_id{0};           // Object class ID (0-79 for COCO classes)
    std::string class_name{};  // Object class name

    // Segmentation data
    std::vector<float> seg_mask_coef{};  // Segmentation mask coefficients (32 values)
    std::vector<float> mask{};           // Binary segmentation mask (flattened H*W)
    int mask_height{0};                  // Height of the segmentation mask
    int mask_width{0};                   // Width of the segmentation mask

    // Default constructor with explicit initialization
    YOLOv26SegResult() {}

    // Parameterized constructor with move semantics for better performance
    YOLOv26SegResult(std::vector<float> box_val, const float conf, const int cls_id,
                     const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    // Legacy constructor for backward compatibility
    YOLOv26SegResult(const std::vector<float>& box_val, const float conf, const int cls_id,
                     const std::string& cls_name);

    // Destructor
    ~YOLOv26SegResult() {}

    // Copy and move constructors/operators
    YOLOv26SegResult(const YOLOv26SegResult& other)
        : box(other.box),
          confidence(other.confidence),
          class_id(other.class_id),
          class_name(other.class_name),
          seg_mask_coef(other.seg_mask_coef),
          mask(other.mask),
          mask_height(other.mask_height),
          mask_width(other.mask_width) {}
    YOLOv26SegResult& operator=(const YOLOv26SegResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            class_id = other.class_id;
            class_name = other.class_name;
            seg_mask_coef = other.seg_mask_coef;
            mask = other.mask;
            mask_height = other.mask_height;
            mask_width = other.mask_width;
        }
        return *this;
    }
    YOLOv26SegResult(YOLOv26SegResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          class_id(other.class_id),
          class_name(std::move(other.class_name)),
          seg_mask_coef(std::move(other.seg_mask_coef)),
          mask(std::move(other.mask)),
          mask_height(other.mask_height),
          mask_width(other.mask_width) {}
    YOLOv26SegResult& operator=(YOLOv26SegResult&& other) {
        if (this != &other) {
            box = std::move(other.box);
            confidence = other.confidence;
            class_id = other.class_id;
            class_name = std::move(other.class_name);
            seg_mask_coef = std::move(other.seg_mask_coef);
            mask = std::move(other.mask);
            mask_height = other.mask_height;
            mask_width = other.mask_width;
        }
        return *this;
    }

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Validation methods
    bool is_invalid(int image_width, int image_height) const;
};

/**
 * @brief YOLOv26n post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class YOLOv26SegPostProcess {
   private:
    // Image dimensions
    int input_width_{640};   // Model input width
    int input_height_{640};  // Model input height

    // Detection threshold (NMS-free, score-based filtering only)
    float score_threshold_{0.5f};

    bool is_ort_configured_{false};  // Whether ORT inference is configured

    // Optional output names for debugging / configuration
    std::vector<std::string> cpu_output_names_{};  // detection, mask outputs

    // Private helper methods
    std::vector<YOLOv26SegResult> decoding_cpu_outputs(const dxrt::TensorPtrs& outputs) const;
    void decoding_mask_cpu_outputs(const dxrt::TensorPtrs& outputs,
                                   std::vector<YOLOv26SegResult>& detections);

    // Segmentation helper
    std::vector<std::vector<float>> process_segmentation_masks(
        const float* mask_output, const std::vector<YOLOv26SegResult>& detections, int mask_height,
        int mask_width) const;

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

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

    YOLOv26SegPostProcess(const int input_w, const int input_h, const float score_threshold, const bool is_ort_configured = false);

    YOLOv26SegPostProcess();

    /**
     * @brief Destructor
     */
    ~YOLOv26SegPostProcess() {}

    /**
     * @brief Process YOLOv26seg model outputs (NMS-free)
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection + mask results
     */
    std::vector<YOLOv26SegResult> postprocess(const dxrt::TensorPtrs& outputs);

    /**
     * @brief Set new thresholds (score only; NMS parameter is ignored)
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
};

#endif  // YOLOV26SEG_POSTPROCESS_H