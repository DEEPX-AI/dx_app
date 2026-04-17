#ifndef YOLOV8_PPU_POSTPROCESS_H
#define YOLOV8_PPU_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOv8 PPU detection result structure
 * Contains bounding box coordinates, confidence scores, and class information
 */
struct YOLOv8PPUResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    int class_id{0};           // Object class ID (0-79 for COCO classes)
    std::string class_name{};  // Object class name

    // Default constructor with explicit initialization
    YOLOv8PPUResult() = default;

    // Parameterized constructor with move semantics for better performance
    YOLOv8PPUResult(std::vector<float> box_val, const float conf, const int cls_id,
                     const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    // Destructor
    ~YOLOv8PPUResult() = default;
    YOLOv8PPUResult(const YOLOv8PPUResult&) = default;
    YOLOv8PPUResult& operator=(const YOLOv8PPUResult&) = default;
    YOLOv8PPUResult(YOLOv8PPUResult&&) noexcept = default;
    YOLOv8PPUResult& operator=(YOLOv8PPUResult&&) noexcept = default;

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Calculate intersection over union with another detection
    float iou(const YOLOv8PPUResult& other) const;

    // Validation methods
    bool is_invalid(int image_width, int image_height) const;
};

/**
 * @brief YOLOv8 PPU post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 *
 * Unlike YOLOv5/v7 PPU which use anchor-based decoding, YOLOv8 PPU uses
 * anchor-free decoding where BBOX outputs contain direct x, y, w, h values.
 */
class YOLOv8PPUPostProcess {
   private:
    // Image dimensions
    int input_width_{640};
    int input_height_{640};

    // Detection thresholds
    float score_threshold_{0.4f};
    float nms_threshold_{0.5f};

    // Model configuration
    enum { num_classes_ = 80 };

    // Model-specific configuration parameters
    std::vector<std::string> ppu_output_names_;

    // Private helper methods
    std::vector<YOLOv8PPUResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv8PPUResult> apply_nms(const std::vector<YOLOv8PPUResult>& detections) const;

   public:
    YOLOv8PPUPostProcess(const int input_w, const int input_h, const float score_threshold,
                          const float nms_threshold);

    YOLOv8PPUPostProcess();

    ~YOLOv8PPUPostProcess() = default;

    std::vector<YOLOv8PPUResult> postprocess(const dxrt::TensorPtrs& outputs);

    dxrt::TensorPtrs align_tensors(const dxrt::TensorPtrs& outputs) const;

    void set_thresholds(const float score_threshold, const float nms_threshold);

    std::string get_config_info() const;

    // Getters
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    static int get_num_classes() { return num_classes_; }
    const std::vector<std::string>& get_ppu_output_names() const { return ppu_output_names_; }
};

#endif  // YOLOV8_PPU_POSTPROCESS_H
