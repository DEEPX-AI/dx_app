# DeepLabV3 Semantic Segmentation Post-Processing

This module provides post-processing for DeepLabv3 semantic segmentation models. It processes model outputs to generate segmentation masks with class predictions and confidence scores for each pixel.

## Overview

⚠️ **MODEL-SPECIFIC POSTPROCESSOR**: This library is specifically designed for `DeepLabV3PlusMobileNetV2_2.dxnn` trained on Cityscapes dataset and is only compatible with models having the exact same specifications.

**Required Model Specifications:**
- **Input**: `[1, 640, 640, 3]` UINT8 tensor named `input.1`
- **Output**: `[1, 19, 640, 640]` FLOAT tensor named `1063`
- **Classes**: Exactly 19 Cityscapes urban scene classes
- **Dataset**: Cityscapes urban scene segmentation

The DeepLabv3 post-processing library handles:
- Softmax activation on model logits
- Argmax operation for class prediction  
- Confidence threshold filtering
- Model compatibility validation
- Segmentation mask generation

## Features

- **Urban Scene Segmentation**: Pixel-wise classification with 19 Cityscapes classes for autonomous driving scenarios
- **Model Validation**: Automatic verification of tensor shapes and dimensions
- **Confidence Filtering**: Configurable confidence threshold for predictions
- **Multi-format Support**: Both CPU (ORT) and NPU model formats
- **Efficient Processing**: Optimized softmax and argmax operations
- **Memory Safe**: Proper RAII and move semantics
- **Error Handling**: Clear error messages for model incompatibility

## Files

- **deeplabv3_postprocess.h**: Header file with class and structure definitions
- **deeplabv3_postprocess.cpp**: Implementation of DeepLabv3 post-processing logic
- **CMakeLists.txt**: Build configuration for the shared library

## Data Structures

### DeepLabv3Result
Contains segmentation results:
```cpp
struct DeepLabv3Result {
    std::vector<int> segmentation_mask;      // Class IDs for each pixel (H*W)
    std::vector<float> confidence_mask;      // Confidence values (H*W)
    std::vector<int> class_ids;              // Unique classes present
    std::vector<std::string> class_names;    // Class names
    int width, height;                       // Image dimensions
    int num_classes;                         // Number of classes found
    float mean_confidence;                   // Average confidence
};
```

### DeepLabv3PostProcess
Main processing class:
```cpp
class DeepLabv3PostProcess {
public:
    DeepLabv3PostProcess(int input_w, int input_h, float confidence_threshold, bool is_ort_configured = false);
    DeepLabv3Result postprocess(const dxrt::TensorPtrs& outputs);
    void set_confidence_threshold(float threshold);
    // ... other methods
};
```

## Usage

### Basic Usage
```cpp
#include "deeplabv3_postprocess.h"

// Create post-processor
DeepLabv3PostProcess postprocessor(513, 513, 0.5f, false);

// Process model outputs
dxrt::TensorPtrs outputs = model.inference(input);
DeepLabv3Result result = postprocessor.postprocess(outputs);

// Access results
for (int y = 0; y < result.height; ++y) {
    for (int x = 0; x < result.width; ++x) {
        int class_id = result.get_pixel_class(x, y);
        float confidence = result.get_pixel_confidence(x, y);
        // Process pixel-wise results
    }
}
```

### Configuration
```cpp
// Create with custom settings
DeepLabv3PostProcess postprocessor(
    513,    // input width
    513,    // input height  
    0.7f,   // confidence threshold
    false   // is ORT configured
);

// Update threshold at runtime
postprocessor.set_confidence_threshold(0.6f);

// Get configuration info
std::string config = postprocessor.get_config_info();
std::cout << config << std::endl;
```

### Result Analysis
```cpp
DeepLabv3Result result = postprocessor.postprocess(outputs);

// Analyze segmentation
std::cout << "Found " << result.num_classes << " classes" << std::endl;
std::cout << "Mean confidence: " << result.mean_confidence << std::endl;

// Get class-specific information
for (int class_id : result.class_ids) {
    float area_ratio = result.get_class_area_ratio(class_id);
    std::cout << "Class " << class_id << " covers " 
              << (area_ratio * 100) << "% of image" << std::endl;
}
```

## Model Specification (DeepLabV3PlusMobileNetV2_2.dxnn)

### Required Model Input
- **Tensor Name**: `input.1`
- **Shape**: `[1, 640, 640, 3]`
- **Type**: UINT8
- **Format**: RGB image (Height x Width x Channels)

### Required Model Output  
- **Tensor Name**: `1063`
- **Shape**: `[1, 19, 640, 640]`
- **Type**: FLOAT
- **Format**: Logits (Batch x Classes x Height x Width)

### Cityscapes Classes (19 total)
The library supports the following 19 Cityscapes urban scene classes:
0. road, 1. sidewalk, 2. building, 3. wall, 4. fence, 5. pole, 6. traffic_light,
7. traffic_sign, 8. vegetation, 9. terrain, 10. sky, 11. person, 12. rider, 13. car,
14. truck, 15. bus, 16. train, 17. motorcycle, 18. bicycle

These classes are specifically designed for autonomous driving and urban scene understanding scenarios.

⚠️ **Compatibility Warning**: This class mapping is specific to Cityscapes dataset. Other segmentation datasets may have different class counts and mappings.

## Building

The library is built as part of the main project:
```bash
cd /path/to/dx_app
./build.sh
```

The shared library `libdxapp_deeplabv3_postprocess.so` will be generated in the `lib/` directory.

## Dependencies

- **dxrt**: DEEPX runtime library
- **C++17**: Standard library with filesystem support
- **CMake 3.16+**: Build system

## Performance Considerations

- Uses optimized softmax with numerical stability (max subtraction)
- Efficient memory layout for pixel-wise operations  
- Configurable confidence thresholding to reduce false positives
- RAII and move semantics for efficient memory management
