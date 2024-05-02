#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "dxrt/dxrt_api.h"
#include "nms.h"
#define sigmoid(x) (1 / (1 + std::exp(-x)))

struct PriorBoxDim
{
    int num_grid_x;
    int num_grid_y;
    int num_boxes;
};
struct PriorBoxParam
{
    int num_layers;
    float min_scale;
    float max_scale;
    float center_variance;
    float size_variance;
    std::vector<PriorBoxDim> dim;
    std::string data_file;
    void Show();
};
struct SsdParam
{
    int image_size;
    bool use_softmax;
    float score_threshold;
    float iou_threshold;
    int num_classes;
    std::vector<std::string> class_names;
    PriorBoxParam priorBoxes;
    void Show();
};

class Ssd
{
private:
    SsdParam cfg;
    float *PriorBoxes;
    float *Boxes;
    float *Scores;
    std::vector< std::vector<std::pair<float, int>> > ScoreIndices;
    std::vector< BoundingBox > Result;
    std::vector< std::string > ClassNames;
    uint32_t numClasses;
    uint32_t numBoxes;
    uint32_t numLayers;
    struct OutputLayer
    {
        unsigned int locAlign;
        unsigned int locOffset;
        unsigned int scoreOffset;
        unsigned int scoreAlign;
        int     boxes;
        int     gridX;
        int     gridY;
        void Show() {
            cout << "OutputLayer: " << gridX << "x" << gridY << ", "
                << boxes << " boxes, loc at " << hex << locOffset << 
                ", score at " << scoreOffset << ", loc align " << hex << locAlign << 
                ", score align " << scoreAlign;
            cout << dec << endl;
        }
    };
    std::vector<OutputLayer> OutputLayers;
    std::vector<shared_ptr<dxrt::Tensor>> outputs;
public:
    ~Ssd();
    Ssd();
    Ssd(SsdParam &_cfg);
    void SetOutputLayer();
    void CreatePriorBoxes(const std::string &f); /* Import Eyenix-specific prior box bin. data */
    // void CreatePriorBoxes(PriorBoxGenerator &pb);
    std::vector< BoundingBox > PostProc(float *data);
    std::vector< BoundingBox > PostProc(std::vector<shared_ptr<dxrt::Tensor>> outputs_, void* saveTo=nullptr);
    void FilterWithSoftmax(vector<shared_ptr<dxrt::Tensor>> outputs_);
    void FilterWithSigmoid(vector<shared_ptr<dxrt::Tensor>> outputs_);
    float* GetPriorBoxes(){ return PriorBoxes; }
    float* GetBoxes(){ return Boxes; }
    float* GetScores(){ return Scores; }
    void ShowResult(void){
        std::cout << "  Detected " << dec << Result.size() << " boxes." << std::endl;
        for(int i=0;i<(int)Result.size();i++)
        {
            Result[i].Show();
        }
    }
};