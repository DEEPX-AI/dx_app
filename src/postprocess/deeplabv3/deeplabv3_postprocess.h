#ifndef DEEPLABV3_POSTPROCESS_H
#define DEEPLABV3_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <map>
#include <string>
#include <vector>

/**
 * @brief DeepLabv3 segmentation result structure
 * Contains segmentation mask, class predictions, and confidence information
 */
struct DeepLabv3Result {
    // Segmentation mask data (flattened H*W arrays)
    std::vector<int> segmentation_mask;    // Segmentation mask with class IDs (H*W)
    std::vector<int> class_ids;            // List of unique class IDs present in the mask
    std::vector<std::string> class_names;  // Corresponding class names

    // Image dimensions
    int width{0};
    int height{0};

    // Statistics
    int num_classes{0};  // Number of classes in the segmentation

    // Default constructor
    DeepLabv3Result() {}

    // Parameterized constructor
    DeepLabv3Result(const std::vector<int>& seg_mask, int w, int h) : width(w), height(h) {
        segmentation_mask.assign(seg_mask.begin(), seg_mask.end());
    }

    // Destructor
    ~DeepLabv3Result() {}

    // Copy constructor and assignment operator
    DeepLabv3Result(const DeepLabv3Result& other)
        : segmentation_mask(other.segmentation_mask),
          class_ids(other.class_ids),
          class_names(other.class_names),
          width(other.width),
          height(other.height),
          num_classes(other.num_classes) {}

    DeepLabv3Result& operator=(const DeepLabv3Result& other) {
        if (this != &other) {
            segmentation_mask = other.segmentation_mask;
            class_ids = other.class_ids;
            class_names = other.class_names;
            width = other.width;
            height = other.height;
            num_classes = other.num_classes;
        }
        return *this;
    }

    // Move constructor and assignment operator
    DeepLabv3Result(DeepLabv3Result&& other) noexcept
        : segmentation_mask(std::move(other.segmentation_mask)),
          class_ids(std::move(other.class_ids)),
          class_names(std::move(other.class_names)),
          width(other.width),
          height(other.height),
          num_classes(other.num_classes) {}

    DeepLabv3Result& operator=(DeepLabv3Result&& other) noexcept {
        if (this != &other) {
            segmentation_mask = std::move(other.segmentation_mask);
            class_ids = std::move(other.class_ids);
            class_names = std::move(other.class_names);
            width = other.width;
            height = other.height;
            num_classes = other.num_classes;
        }
        return *this;
    }

    // Utility methods
    std::vector<std::vector<std::pair<int, int>>> get_class_pixels(int class_id) const;
    float get_class_area_ratio(int class_id) const;
};

/**
 * @brief DeepLabv3 post-processing class
 * Handles semantic segmentation results processing, softmax, and argmax operations
 */
class DeepLabv3PostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (DeepLabV3PlusMobileNetV2_2.dxnn)
    int input_height_{640};  // Model input height (DeepLabV3PlusMobileNetV2_2.dxnn)

    // Model configuration - using const where appropriate
    enum { num_classes_ = 19 };  // Number of classes (Cityscapes dataset: 19 urban scene classes)

    // Model-specific configuration parameters
    std::vector<std::string> cpu_output_names_;  // CPU output tensor names
    std::vector<std::string> npu_output_names_;  // NPU output tensor names

    // Private helper methods - const correctness
    DeepLabv3Result decode_segmentation_output(const dxrt::TensorPtrs& outputs) const;
    std::vector<int> apply_argmax(const float* softmax_output, int height, int width,
                                  int num_classes) const;

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @note This postprocessor is specifically designed for DeepLabV3PlusMobileNetV2_2.dxnn
     * (Cityscapes) Expected model specs: Input[1,640,640,3], Output[1,19,640,640], and Trained on
     * Cityscapes urban scene segmentation dataset
     */
    DeepLabv3PostProcess(const int input_w, const int input_h);

    DeepLabv3PostProcess();

    /**
     * @brief Destructor
     */
    ~DeepLabv3PostProcess() {}

    /**
     * @brief Process DeepLabv3 model outputs
     * @param outputs Vector of output tensors from the model
     * @return Processed segmentation result
     */
    DeepLabv3Result postprocess(const dxrt::TensorPtrs& outputs);

    /**
     * @brief Get current configuration
     * @return String representation of current configuration
     */
    std::string get_config_info() const;

    // Getters for current configuration - const correctness
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }

    // Static configuration getters
    static int get_num_classes() { return num_classes_; }
};

#endif  // DEEPLABV3_POSTPROCESS_H
