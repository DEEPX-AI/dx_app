#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dxrt/dxrt_api.h>
#include "nms.h"

#define sigmoid(x) (1 / (1 + std::exp(-x)))

enum class PostProcType
{
    OD = 0,
    POSE,
    FACE,
    YOLOV8
};

struct YoloLayerParam
{
    // Layer name identifier
    std::string name;
    
    // Grid dimensions
    int32_t numGridX;    // Number of grid cells in X direction
    int32_t numGridY;    // Number of grid cells in Y direction
    
    // Number of anchor boxes per grid cell
    int32_t numBoxes;
    
    // Anchor box dimensions
    std::vector<float> anchorWidth;   // Width of anchor boxes
    std::vector<float> anchorHeight;  // Height of anchor boxes
    
    // Tensor indices for layer outputs
    std::vector<int32_t> tensorIdx;
    
    // Scale factors for x,y coordinates
    float scaleX{0.0f};  // X coordinate scale factor
    float scaleY{0.0f};  // Y coordinate scale factor

    // Default constructor
    YoloLayerParam() = default;
    ~YoloLayerParam() = default;

    // Constructor with parameters
    YoloLayerParam(std::string _name, int _gx, int _gy, int _numB, 
                   const std::vector<float> &_vAnchorW, const std::vector<float> &_vAnchorH, const std::vector<int> &_vTensorIdx, 
                   float _sx = 0.f, float _sy = 0.f)
    :name(_name), numGridX(_gx), numGridY(_gy), numBoxes(_numB), 
     anchorWidth(_vAnchorW), anchorHeight(_vAnchorH),
     tensorIdx(_vTensorIdx), scaleX(_sx), scaleY(_sy)
    {}

    // Copy constructor
    YoloLayerParam(const YoloLayerParam& other)
    :name(other.name), numGridX(other.numGridX), numGridY(other.numGridY), numBoxes(other.numBoxes),
     anchorWidth(other.anchorWidth), anchorHeight(other.anchorHeight),
     tensorIdx(other.tensorIdx), scaleX(other.scaleX), scaleY(other.scaleY)
    {}

    void Show();
};
struct YoloParam
{
    // Image dimension variables
    int32_t height{0};
    int32_t width{0};

    // Threshold variables
    float confThreshold{0.0f};  // Confidence threshold
    float scoreThreshold{0.0f}; // Score threshold  
    float iouThreshold{0.0f};   // IoU threshold

    // Object detection variables
    uint32_t numBoxes{0};      // Number of bounding boxes
    uint32_t numClasses{0};    // Number of classes

    // onnx output name
    std::string onnxOutputName = "";

    // Layer and class information
    std::vector<YoloLayerParam> layers{};     // YOLO layer parameters
    std::vector<std::string> classNames{};    // Class name list
    
    // Post-processing type
    PostProcType postproc_type{PostProcType::OD};
    
    // Default constructor
    YoloParam() = default;

    // Display configuration info
    void Show();
};

class Yolo
{
private:
    // Configuration
    YoloParam cfg;
    
    // Core data structures
    std::vector<BoundingBox> Result;
    std::vector<float> Boxes;
    std::vector<float> Keypoints;
    std::vector<std::vector<std::pair<float, int>>> ScoreIndices;

    int anchorSize = 0;
    bool is_onnx_output = false;
    std::vector<int32_t> onnxOutputIdx={};

public:
    // Constructors/Destructor
    Yolo();
    Yolo(YoloParam &_cfg);
    ~Yolo();

    // Core processing functions
    bool LayerReorder(dxrt::Tensors output_info);
    
    // PostProc variants
    std::vector<BoundingBox> PostProc(dxrt::TensorPtrs& dataSrc);

    void onnx_post_processing(dxrt::TensorPtrs &outputs, int64_t num_elements);
    void raw_post_processing(dxrt::TensorPtrs &outputs);

    // Utility functions
    void ShowResult(void) {
        std::cout << "  Detected " << std::dec << Result.size() << " boxes." << std::endl;
        for(int i=0; i<(int)Result.size(); i++) {
            Result[i].Show();
        }
    }
};
