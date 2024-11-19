#include <algorithm>
#include "yolo.h"
#include "dxrt/util.h"

using namespace std;

// #define DUMP_DATA

void YoloLayerParam::Show()
{
    cout << "    - LayerParam: [ " << numGridX << " x " << numGridY << " x " << numBoxes << "boxes" << "], anchorWidth [";
    for(auto &w : anchorWidth) cout << w << ", ";
    cout << "], anchorHeight [";
    for(auto &h : anchorHeight) cout << h << ", ";
    cout << "], tensor index [";
    for(auto &t : tensorIdx) cout << t << ", ";
    cout << "]" << endl;
}
void YoloParam::Show()
{
    cout << "  YoloParam: " << endl << "    - conf_threshold: " << confThreshold << ", "
        << "score_threshold: " << scoreThreshold << ", "
        << "iou_threshold: " << iouThreshold << ", "
        << "num_classes: " << numClasses << ", "
        << "num_layers: " << layers.size() << endl;
    for(auto &layer:layers) layer.Show();
    cout << "    - classes: [";
    for(auto &c : classNames) cout << c << ", ";
    cout << "]" << endl;
}
Yolo::Yolo() { }
Yolo::~Yolo() { }
Yolo::Yolo(YoloParam &_cfg) :cfg(_cfg)
{
    //setup number of boxes, classes, layers    
    numClasses = cfg.numClasses;
    if(cfg.numBoxes<=0)
    {
        numBoxes = 0;
        for(auto &layer:cfg.layers)
        {
            numBoxes += layer.numGridX*layer.numGridY*layer.numBoxes;
        }
    }
    else
    {
        numBoxes = cfg.numBoxes;
    }
    if(cfg.layers.empty())
    {
        concatedTensors = true;
        hasAnchors = false;        
    }
    else
    {
        hasAnchors = cfg.layers[0].anchorWidth.size()>0?true:false;
    }
    numLayers = cfg.layers.size();
    ClassNames = cfg.classNames;
    //allocate memory
    Boxes = vector<float>(numBoxes*4);
    cout << "YOLO created : " << numBoxes << " boxes, " << numClasses << " classes, " 
        << numLayers << " layers." << endl;
    cfg.Show();
    //prepare score indices
    for(size_t i=0; i<numClasses; i++)
    {
        vector<pair<float, int>> v;
        ScoreIndices.emplace_back(v);
    }
}

void Yolo::LayerReorder(dxrt::Tensors output_info)
{
    std::vector<YoloLayerParam> temp;
    for(size_t i=0;i<output_info.size();i++)
    {
        for(size_t j=0;j<output_info.size();j++)
        {
            if(output_info[i].name() == cfg.layers[j].name)
            {
                temp.emplace_back(cfg.layers[j]);
                break;
            }
        }
    }
    cfg.layers.clear();
    cfg.layers = temp;
}

void Yolo::LayerInverse(int mode)
{
    std::sort(cfg.layers.begin(), cfg.layers.end(), 
                [&](const YoloLayerParam &a, const YoloLayerParam &b)
                {
                    if(mode > 0)
                        return a.numGridX < b.numGridX;
                    return a.numGridX > b.numGridX;
                });
}

static bool scoreComapre(const std::pair<float, int> &a, const std::pair<float, int> &b)
{
    if(a.first > b.first)
        return true;
    else
        return false;
};

void Yolo::FilterWithSort(float *org)
{
    int x = 0, y = 1, w = 2, h = 3;
    float ScoreThreshold = cfg.scoreThreshold;
    float conf_threshold = cfg.confThreshold;
    float score, score1;
    for(int boxIdx=0;boxIdx<(int)numBoxes;boxIdx++)
    {
        bool boxDecoded = false;
        float *data = org + (4+1+numClasses)*boxIdx;
        score1 = data[4];
        if(data[4]>conf_threshold)
        {
            for(int cls=0; cls<(int)numClasses; cls++)
            {
                score = score1 * data[5+cls];
                if(score > ScoreThreshold)
                {
                    ScoreIndices[cls].emplace_back(score, boxIdx);
                    if(!boxDecoded)
                    {
                        Boxes[boxIdx*4+0] = data[x] - data[w] / 2.; /*x1*/
                        Boxes[boxIdx*4+1] = data[y] - data[h] / 2.; /*y1*/
                        Boxes[boxIdx*4+2] = data[x] + data[w] / 2.; /*x2*/
                        Boxes[boxIdx*4+3] = data[y] + data[h] / 2.; /*y2*/
                        boxDecoded = true;
                    }
                }
            }
        }
    }
    for(int cls=0;cls<(int)numClasses;cls++)
    {
        sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), scoreComapre);
    }
}
void Yolo::FilterWithSort(vector<shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
    int x = 0, y = 1, w = 2, h = 3;
    float ScoreThreshold = cfg.scoreThreshold;
    float conf_threshold = cfg.confThreshold;
    float rawThreshold = log(conf_threshold/(1-conf_threshold));
    float score, score1, box_temp[4];
    float *boxLocation, *boxScore, *classScore, *data;
    if(hasAnchors)
    {
        for(auto &layer:cfg.layers)        
        {
            int strideX = cfg.width / layer.numGridX;
            int strideY = cfg.height / layer.numGridY;
            int numGridX = layer.numGridX;
            int numGridY = layer.numGridY;
            int tensorIdx = layer.tensorIdx[0];
            float scale_x_y = layer.scaleX;
            for(int gY=0; gY<numGridY; gY++)
            {
                for(int gX=0; gX<numGridX; gX++)
                {
                    for(int box=0; box<layer.numBoxes; box++)
                    { 
                        bool boxDecoded = false;
                        data = (float*)(outputs_[tensorIdx]->data(gY, gX, box*(4+1+numClasses)));
                        // cout << boxIdx << ": " << hex << data << ", " << dec << gX << " x " << gY << ", " << dec << data[4] << ", " << sigmoid(data[4]) << endl;
                        if(data[4]>rawThreshold)
                        {
                            score1 = sigmoid(data[4]);
                            /* Step1 - obj_conf > CONF_THRESHOLD */
                            if(score1 > conf_threshold)
                            {
                                for(int cls=0; cls<(int)numClasses;cls++)
                                {
                                    score = score1 * sigmoid(data[5+cls]); /*conf = obj_conf * cls_conf*/
                                    /* Step2 - obj_conf * cls_conf > CONF_THRESHOLD */
                                    if (score > ScoreThreshold)
                                    {
                                        /* cout << boxIdx << ": " << gX << ", " << gY << ", " << cls << ", " << \
                                            score1 << ", " << data[5+cls] << ", " << x << ", " << y << ", " << w << ", " << h << " . " << endl; */
                                        ScoreIndices[cls].emplace_back(score, boxIdx);
                                        if(!boxDecoded)
                                        {
                                            if(scale_x_y==0)
                                            {
                                                box_temp[x] = ( sigmoid(data[x]) * 2. - 0.5 + gX ) * strideX;
                                                box_temp[y] = ( sigmoid(data[y]) * 2. - 0.5 + gY ) * strideY;
                                            }
                                            else
                                            {
                                                box_temp[x] = (sigmoid(data[x] * scale_x_y  - 0.5 * (scale_x_y - 1)) + gX) * strideX;
                                                box_temp[y] = (sigmoid(data[y] * scale_x_y  - 0.5 * (scale_x_y - 1)) + gY) * strideY;
                                            }
                                            box_temp[w] = pow((sigmoid(data[w]) * 2.), 2) * layer.anchorWidth[box];
                                            box_temp[h] = pow((sigmoid(data[h]) * 2.), 2) * layer.anchorHeight[box];
                                            Boxes[boxIdx*4+0] = box_temp[x] - box_temp[w] / 2.; /*x1*/
                                            Boxes[boxIdx*4+1] = box_temp[y] - box_temp[h] / 2.; /*y1*/
                                            Boxes[boxIdx*4+2] = box_temp[x] + box_temp[w] / 2.; /*x2*/
                                            Boxes[boxIdx*4+3] = box_temp[y] + box_temp[h] / 2.; /*y2*/
                                            boxDecoded = true;
                                        }
                                    }
                                }
                            }
                        }
                        // cout << dec << boxIdx << " : " << hex << data << " : +0x" << data-org << dec << endl;
                        // LOG_VALUE(boxIdx);
                        boxIdx++;
                    }
                }
            }
        }
    }
    else
    {
        for(auto &layer:cfg.layers)        
        {
            int strideX = cfg.width / layer.numGridX;
            int strideY = cfg.height / layer.numGridY;
            int numGridX = layer.numGridX;
            int numGridY = layer.numGridY;
            int locationTensorIdx = layer.tensorIdx[0];
            int boxScoreTensorIdx = layer.tensorIdx[1];
            int clsScoreTensorIdx = layer.tensorIdx[2];
            for(int gY=0; gY<numGridY; gY++)
            {
                for(int gX=0; gX<numGridX; gX++)
                {
                    for(int box=0; box<layer.numBoxes; box++)
                    { 
                        bool boxDecoded = false;
                        boxScore = (float*)(outputs_[boxScoreTensorIdx]->data(gY, gX, 0));
                        if(boxScore[0]>conf_threshold)
                        {
                            boxLocation = (float*)(outputs_[locationTensorIdx]->data(gY, gX, 0));                        
                            classScore = (float*)(outputs_[clsScoreTensorIdx]->data(gY, gX, 0));
                            for(int cls=0; cls<(int)numClasses;cls++)
                            {
                                score = boxScore[0]*classScore[cls];
                                if (score > ScoreThreshold)
                                {
                                    float x = (boxLocation[0] + gX ) * strideX;
                                    float y = (boxLocation[1] + gY ) * strideY;
                                    float w = exp(boxLocation[2]) * strideX;
                                    float h = exp(boxLocation[3]) * strideY;
                                    ScoreIndices[cls].emplace_back(score, boxIdx);
                                    if(!boxDecoded)
                                    {
                                        Boxes[boxIdx*4+0] = x - w / 2.; /*x1*/
                                        Boxes[boxIdx*4+1] = y - h / 2.; /*y1*/
                                        Boxes[boxIdx*4+2] = x + w / 2.; /*x2*/
                                        Boxes[boxIdx*4+3] = y + h / 2.; /*y2*/
                                        boxDecoded = true;
                                    }
                                }
                            }
                        }
                        boxIdx++;
                    }
                }
            }
        }
    }
    for(int cls=0;cls<(int)numClasses;cls++)
    {
        sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), scoreComapre);
    }
}
vector< BoundingBox > Yolo::PostProc(float *data)
{
    for(int cls=0;cls<(int)numClasses;cls++)
    {
        ScoreIndices[cls].clear();
    }
    Result.clear();
    FilterWithSort(data);
    Nms(
        numClasses,
        0,
        ClassNames, 
        ScoreIndices, Boxes.data(), cfg.iouThreshold,
        Result,
        0
    );
    return Result;
}
vector< BoundingBox > Yolo::PostProc(vector<shared_ptr<dxrt::Tensor>> outputs_, void *saveTo)
{
    vector< BoundingBox > result;
    if(outputs_.front()->type()==dxrt::DataType::BBOX)
    {
        int boxIdx = 0;
        float x, y, w, h;
        int numElements = outputs_.front()->shape().front();
        dxrt::DeviceBoundingBox_t *dataSrc = (dxrt::DeviceBoundingBox_t *)outputs_.front()->data();
        for(uint32_t label=0 ; label<numClasses ; label++)
        {
            ScoreIndices[label].clear();
        }
        for(int i=0 ; i<numElements ; i++)
        {
            dxrt::DeviceBoundingBox_t *data = dataSrc + i;            
            auto layer = cfg.layers[data->layer_idx];
            int strideX = cfg.width / layer.numGridX;
            int strideY = cfg.height / layer.numGridY;
            int gX = data->grid_x;
            int gY = data->grid_y;
            float scale_x_y = layer.scaleX;            

            ScoreIndices[data->label].emplace_back(data->score, boxIdx);
            if(scale_x_y==0)
            {
                x = ( data->x * 2. - 0.5 + gX ) * strideX;
                y = ( data->y * 2. - 0.5 + gY ) * strideY;
            }
            else
            {
                x = (data->x * scale_x_y  - 0.5 * (scale_x_y - 1) + gX) * strideX;
                y = (data->y * scale_x_y  - 0.5 * (scale_x_y - 1) + gY) * strideY;
            }
            w = (data->w * data->w * 4.) * layer.anchorWidth[data->box_idx];
            h = (data->h * data->h * 4.) * layer.anchorHeight[data->box_idx];
            Boxes[boxIdx*4 + 0] = x - w/2.; /*x1*/
            Boxes[boxIdx*4 + 1] = y - h/2.; /*y1*/
            Boxes[boxIdx*4 + 2] = x + w/2.; /*x2*/
            Boxes[boxIdx*4 + 3] = y + h/2.; /*y2*/
            boxIdx++;
        }
        for(uint32_t label=0 ; label<numClasses ; label++)
        {
            sort(ScoreIndices[label].begin(), ScoreIndices[label].end(), scoreComapre);
        }
        Nms(
            numClasses,
            0,
            ClassNames, 
            ScoreIndices, Boxes.data(), cfg.iouThreshold,
            result,
            0
        );
    }
    else if (outputs_.size()>1)
    {
        // outputs = outputs_;
#ifdef DUMP_DATA
        uint32_t dumpSize = 0;
        for(int i=0;i<outputs_.size();i++)
        {
            // outputs[i]->Show();        
            // dxrt::DataDumpBin("output."+to_string(i)+".bin", outputs_[i]->data(), outputs_[i]->GetSize());            
            dxrt::DataDumpTxt("output."+to_string(i)+".txt", (float*)outputs_[i]->data(), outputs_[i]->shape()[1], outputs_[i]->shape()[2], data_align(outputs_[i]->shape()[3], 64));
            // dxrt::DataDumpTxt("output."+to_string(i)+".txt", (float*)outputs_[i]->data(), outputs_[i]->shape()[0], outputs_[i]->shape()[1], outputs_[i]->shape()[2]);
            // dumpSize+=outputs_[i]->GetSize();
        }
        // dxrt::DataDumpBin("output.bin", outputs_[0]->data(), dumpSize);
#endif
        for(int cls=0;cls<(int)numClasses;cls++)
        {
            ScoreIndices[cls].clear();
        }
        if(concatedTensors)
            FilterWithSort((float*)outputs_.front()->data());
        else
            FilterWithSort(outputs_);
        Nms(
            numClasses,
            0,
            ClassNames, 
            ScoreIndices, Boxes.data(), cfg.iouThreshold,
            result,
            0
        );
    }
    else
    {
        return PostProc((float*)outputs_.front()->data());
    }
    if(saveTo!=nullptr)
    {
        BoundingBox *boxes = (BoundingBox*)saveTo;
        memcpy(saveTo, &result[0], result.size()*sizeof(result[0]));
        boxes[result.size()].label = -1;
        if(result.empty()) boxes[0].label = -1;
    }
    Result = result;
    return result;
}