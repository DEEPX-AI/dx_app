#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "dxrt/dxrt_api.h"
#include "nms.h"
#define sigmoid(x) (1 / (1 + std::exp(-x)))

struct AnchorBoxParam
{
    int num_grid_x;
    int num_grid_y;
    std::vector<float> width;
    std::vector<float> height;
    int num_boxes;
    void Show();
};
struct YoloParam
{
    int image_size;
    float conf_threshold;
    float score_threshold;
    float iou_threshold;
    int num_classes;
    int num_layers;
    std::vector<AnchorBoxParam> anchorBoxes;
    std::vector<std::string> class_names;
    void Show();
};

class Yolo
{
private:
    YoloParam cfg;
    std::vector<dxrt::Tensor> TensorData;
    float *Boxes = nullptr;
    float *Scores = nullptr;
    std::vector< std::vector<std::pair<float, int>> > ScoreIndices;
    std::vector< BoundingBox > Result;
    std::vector< std::string > ClassNames;
    // PriorBoxGenerator pb;
    uint32_t numClasses;
    uint32_t numBoxes;
    uint32_t numLayers;
    struct OutputLayer
    {
        int     boxes;
        int     gridX;
        int     gridY;
        int     stride;
        int     dataAlign;
        int     dataOffset;
		vector<float> anchorWidth;
		vector<float> anchorHeight;
        void Show() {
            cout << dec << "  OutputLayer: " << gridX << "x" << gridY << ", "
                << boxes << " boxes, data align " <<  dataAlign
                << ", data offset 0x" << hex << dataOffset << dec << std::endl;
        }
    };
    std::vector<OutputLayer> OutputLayers;
    std::vector<shared_ptr<dxrt::Tensor>> outputs;
public:
    ~Yolo();
    Yolo();
    Yolo(YoloParam &_cfg, std::vector<dxrt::Tensor> &_tensorData);
    void SetOutputLayer();
    std::vector< BoundingBox > PostProc(float *data);
    std::vector< BoundingBox > PostProc(std::vector<shared_ptr<dxrt::Tensor>> outputs_, void* saveTo=nullptr);
    void FilterWithSort(float *data, std::vector<shared_ptr<dxrt::Tensor>> outputs_);
    void ShowResult(void){
        std::cout << "  Detected " << dec << Result.size() << " boxes." << std::endl;
        for(int i=0;i<(int)Result.size();i++)
        {
            Result[i].Show();
        }
    }
};
