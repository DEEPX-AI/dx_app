#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "dxrt/dxrt_api.h"
#include "nms.h"
#define sigmoid(x) (1 / (1 + std::exp(-x)))

struct YoloLayerParam
{
    int numGridX;
    int numGridY;
    int numBoxes;
    std::vector<float> anchorWidth;
    std::vector<float> anchorHeight;
    std::vector<int> tensorIdx;
    float scaleX=0;
    float scaleY=0;
    void Show();
};
struct YoloParam
{
    int height;
    int width;
    float confThreshold;
    float scoreThreshold;
    float iouThreshold;
    int numBoxes;
    int numClasses;
    std::vector<YoloLayerParam> layers;
    std::vector<std::string> classNames;
    void Show();
};

class Yolo
{
private:
    YoloParam cfg;
    std::vector<float> Boxes;
    std::vector< std::vector<std::pair<float, int>> > ScoreIndices;
    std::vector< BoundingBox > Result;
    std::vector< std::string > ClassNames;
    bool concatedTensors = false;
    bool hasAnchors;
    uint32_t numClasses;
    uint32_t numBoxes;
    uint32_t numLayers;
    std::vector<shared_ptr<dxrt::Tensor>> outputs;
public:
    ~Yolo();
    Yolo();
    Yolo(YoloParam &_cfg);
    /* for concated tensor */
    std::vector< BoundingBox > PostProc(float *data);
    /* for separate tensors */
    std::vector< BoundingBox > PostProc(std::vector<shared_ptr<dxrt::Tensor>> outputs_, void* saveTo=nullptr);
    void FilterWithSort(std::vector<shared_ptr<dxrt::Tensor>> outputs_);
    void FilterWithSort(float *data);
    void ShowResult(void){
        std::cout << "  Detected " << dec << Result.size() << " boxes." << std::endl;
        for(int i=0;i<(int)Result.size();i++)
        {
            Result[i].Show();
        }
    }
};
