#ifndef ARGMAX_SEMANTIC_SEG_POSTPROCESSOR_HPP
#define ARGMAX_SEMANTIC_SEG_POSTPROCESSOR_HPP

/**
 * @file argmax_semantic_seg_postprocessor.hpp
 * @brief DEPRECATED — Legacy DeepLabv3 postprocessor
 * 
 * This file contains the legacy DeepLabv3PostProcess class with hardcoded
 * 19-class Cityscapes config and tensor name "1063".
 * 
 * For new code, use the v3-native DeepLabv3Postprocessor from
 * segmentation_postprocessor.hpp, which supports:
 *   - Any number of classes (auto-detected from tensor shape)
 *   - Both NCHW and NHWC layouts
 *   - Both int16 (DX compiler output) and float tensors
 *   - No hardcoded tensor names
 * 
 * Retained only for backward compatibility with legacy result converters.
 */

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

    // Segmentation statistics
    int num_classes{0};  // Number of classes in the segmentation

    // Default constructor
    DeepLabv3Result() = default;

    // Parameterized constructor
    DeepLabv3Result(const std::vector<int>& seg_mask, int w, int h) : width(w), height(h) {
        segmentation_mask.assign(seg_mask.begin(), seg_mask.end());
    }

    // Rule of Zero: compiler-generated copy/move are sufficient
    ~DeepLabv3Result() = default;
    DeepLabv3Result(const DeepLabv3Result&) = default;
    DeepLabv3Result& operator=(const DeepLabv3Result&) = default;
    DeepLabv3Result(DeepLabv3Result&&) noexcept = default;
    DeepLabv3Result& operator=(DeepLabv3Result&&) noexcept = default;

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
    ~DeepLabv3PostProcess() = default;

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

// ============================================================================
// Implementation (merged from .cpp - all definitions are inline)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

// Custom exception for model output shape mismatch
class ModelShapeMismatchException : public std::exception {
   private:
    std::string message_;

   public:
    explicit ModelShapeMismatchException(const std::string& message) : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

inline std::vector<std::vector<std::pair<int, int>>> DeepLabv3Result::get_class_pixels(
    int class_id) const {
    std::vector<std::vector<std::pair<int, int>>> class_pixels;

    for (int y = 0; y < height; ++y) {
        std::vector<std::pair<int, int>> row_pixels;
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (idx < static_cast<int>(segmentation_mask.size()) &&
                segmentation_mask[idx] == class_id) {
                row_pixels.push_back({x, y});
            }
        }
        if (!row_pixels.empty()) {
            class_pixels.push_back(row_pixels);
        }
    }

    return class_pixels;
}

inline float DeepLabv3Result::get_class_area_ratio(int class_id) const {
    int count = 0;
    for (int class_val : segmentation_mask) {
        if (class_val == class_id) {
            count++;
        }
    }

    int total_pixels = width * height;
    return (total_pixels > 0) ? static_cast<float>(count) / static_cast<float>(total_pixels) : 0.0f;
}

inline DeepLabv3PostProcess::DeepLabv3PostProcess(const int input_w, const int input_h) {
    input_width_ = input_w;
    input_height_ = input_h;

    /**
     * @brief Initialize model-specific parameters for DeepLabV3PlusMobileNetV2_2.dxnn
     *
     * Compatible Model Specification:
     * - Input: input.1, UINT8, [1, 640, 640, 3]
     * - Output: 1063, FLOAT, [1, 19, 640, 640]
     * - Target: NPU execution
     */
    cpu_output_names_ = {""};
    npu_output_names_ = {"1063"};
}

// Default constructor
inline DeepLabv3PostProcess::DeepLabv3PostProcess() {
    input_width_ = 640;
    input_height_ = 640;

    // Initialize model-specific parameters for DeepLabV3PlusMobileNetV2_2.dxnn
    cpu_output_names_ = {""};
    npu_output_names_ = {"1063"};
}

// Process model outputs
inline DeepLabv3Result DeepLabv3PostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    return decode_segmentation_output(outputs);
}

// Decode segmentation output from model
inline DeepLabv3Result DeepLabv3PostProcess::decode_segmentation_output(
    const dxrt::TensorPtrs& outputs) const {
    if (outputs.empty()) {
        return DeepLabv3Result();
    }

    /**
     * @brief Decode segmentation logits to class predictions
     *
     * Expected format: [1, 19, 640, 640] FLOAT logits from tensor "1063"
     *
     * @note Model-specific compatibility: DeepLabV3PlusMobileNetV2_2.dxnn only
     */
    const float* output_data = static_cast<const float*>(outputs[0]->data());

    // Get tensor dimensions
    const auto& shape = outputs[0]->shape();
    const int num_classes = static_cast<int>(shape[1]);
    const int height = static_cast<int>(shape[2]);
    const int width = static_cast<int>(shape[3]);

    // Validate model compatibility with expected DeepLabV3PlusMobileNetV2_2.dxnn specs
    if (num_classes != num_classes_ || height != input_height_ || width != input_width_) {
        std::cerr << "Model output shape mismatch! Expected [1," << num_classes_ << ","
                  << input_height_ << "," << input_width_ << "] but got [1," << num_classes << ","
                  << height << "," << width
                  << "]. This postprocessor is specifically designed for "
                     "DeepLabV3PlusMobileNetV2_2.dxnn"
                  << std::endl;
    }

    // Apply argmax to get class predictions
    std::vector<int> class_predictions = apply_argmax(output_data, height, width, num_classes);

    // Create result
    DeepLabv3Result result(class_predictions, width, height);

    return result;
}

// Apply argmax to get class predictions
inline std::vector<int> DeepLabv3PostProcess::apply_argmax(const float* npu_outputs, int height, int width,
                                                    int num_classes) const {
    std::vector<int> class_predictions(height * width);

    auto argmax_channel = [&](int pixel_idx) {
        int best = 0;
        float best_val = npu_outputs[pixel_idx];
        for (int c = 1; c < num_classes; ++c) {
            float v = npu_outputs[c * height * width + pixel_idx];
            if (v > best_val) { best_val = v; best = c; }
        }
        return best;
    };

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int pixel_idx = h * width + w;
            // Only assign non-background class if confidence is above threshold
            class_predictions[pixel_idx] = argmax_channel(pixel_idx);
        }
    }

    return class_predictions;
}

// Get configuration information
inline std::string DeepLabv3PostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "DeepLabv3 PostProcess Configuration:\n"
        << "  Model: DeepLabV3PlusMobileNetV2_2.dxnn (Cityscapes dataset)\n"
        << "  Expected Input: [1, 640, 640, 3] (UINT8)\n"
        << "  Expected Output: [1, 19, 640, 640] (FLOAT) from tensor '1063'\n"
        << "  Dataset: Cityscapes urban scene segmentation (19 classes)\n"
        << "  Current Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Number of classes: " << num_classes_ << " (Cityscapes)\n";

    for (const auto& cpu_output_name : cpu_output_names_) {
        oss << "  CPU output name: " << cpu_output_name << "\n";
    }
    for (const auto& npu_output_name : npu_output_names_) {
        oss << "  NPU output name: " << npu_output_name << "\n";
    }

    oss << "\n  WARNING: This postprocessor is ONLY compatible with "
           "DeepLabV3PlusMobileNetV2_2.dxnn\n";
    oss << "  For other DeepLabv3 models, you may need to adjust class count and tensor "
           "names.\n";

    return oss.str();
}

#endif  // ARGMAX_SEMANTIC_SEG_POSTPROCESSOR_HPP
