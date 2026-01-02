#include "deeplabv3_postprocess.h"

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

std::vector<std::vector<std::pair<int, int>>> DeepLabv3Result::get_class_pixels(
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

float DeepLabv3Result::get_class_area_ratio(int class_id) const {
    int count = 0;
    for (int class_val : segmentation_mask) {
        if (class_val == class_id) {
            count++;
        }
    }

    int total_pixels = width * height;
    return (total_pixels > 0) ? static_cast<float>(count) / static_cast<float>(total_pixels) : 0.0f;
}

// Constructor
DeepLabv3PostProcess::DeepLabv3PostProcess(const int input_w, const int input_h) {
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
DeepLabv3PostProcess::DeepLabv3PostProcess() {
    input_width_ = 640;
    input_height_ = 640;

    // Initialize model-specific parameters for DeepLabV3PlusMobileNetV2_2.dxnn
    cpu_output_names_ = {""};
    npu_output_names_ = {"1063"};
}

// Process model outputs
DeepLabv3Result DeepLabv3PostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    return decode_segmentation_output(outputs);
}

// Decode segmentation output from model
DeepLabv3Result DeepLabv3PostProcess::decode_segmentation_output(
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
std::vector<int> DeepLabv3PostProcess::apply_argmax(const float* npu_outputs, int height, int width,
                                                    int num_classes) const {
    std::vector<int> class_predictions(height * width);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int pixel_idx = h * width + w;

            int best_class = -1;
            float best_prob = -1.0f;

            for (int c = 0; c < num_classes; ++c) {
                int prob_idx = c * height * width + pixel_idx;
                if (npu_outputs[prob_idx] > best_prob) {
                    best_prob = npu_outputs[prob_idx];
                    best_class = c;
                }
            }
            // Only assign non-background class if confidence is above threshold
            class_predictions[pixel_idx] = best_class;
        }
    }

    return class_predictions;
}

// Get configuration information
std::string DeepLabv3PostProcess::get_config_info() const {
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
