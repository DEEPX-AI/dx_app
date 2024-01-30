#include <algorithm>
#include "yolo.h"
#include "dxrt/util.h"

using namespace std;

// #define DUMP_DATA

void AnchorBoxParam::Show()
{
    cout << "    - AnchorBoxParam: [ " << num_grid_x << " x " << num_grid_y << " x " << num_boxes << "], width [";
    for(auto &w : width) cout << w << ", ";
    cout << "], height [";
    for(auto &h : height) cout << h << ", ";
    cout << "]" << endl;
}
void YoloParam::Show()
{
    cout << "  YoloParam: " << endl << "    - conf_threshold: " << conf_threshold << ", "
        << "score_threshold: " << score_threshold << ", "
        << "iou_threshold: " << iou_threshold << ", "
        << "num_classes: " << num_classes << ", "
        << "num_layers: " << num_layers << endl;
    for(auto &a : anchorBoxes) a.Show();
    cout << "    - classes: [";
    for(auto &c : class_names) cout << c << ", ";
    cout << "]" << endl;
}
Yolo::Yolo() {  }
Yolo::~Yolo()    
{
    if(Boxes!=nullptr)
    {
        delete [] Boxes;
        Boxes = nullptr;
    }
    if(Scores!=nullptr)
    {
        delete [] Scores;
        Scores = nullptr;        
    }
}
Yolo::Yolo(YoloParam &_cfg, std::vector<dxrt::Tensor> &_tensorData)
    :cfg(_cfg), TensorData(_tensorData)
{
    //setup number of boxes, classes, layers
    numClasses = cfg.num_classes;
    numBoxes = 0;
    for(auto &ab : cfg.anchorBoxes)
    {
        numBoxes += ab.num_grid_x * ab.num_grid_y * ab.num_boxes;
    }
    numLayers = cfg.num_layers;
    ClassNames = cfg.class_names;
    //allocate memory
    Boxes = new float[numBoxes * 4];
    Scores = new float[numBoxes * numClasses];
    cout << "YOLO created : " << numBoxes << " boxes, " << numClasses << " classes, " 
        << numLayers << " layers." << endl;
    cfg.Show();
    //prepare score indices
    for(size_t i=0; i<numClasses; i++)
    {
        vector<pair<float, int>> v;
        ScoreIndices.emplace_back(v);
    }
    //setup output layers
    for(auto &d : TensorData)
    {
        for(int layer=0; layer<numLayers; layer++)
        {
            auto anchorBox = cfg.anchorBoxes[layer];
            if(d.shape()[1]==anchorBox.num_grid_x && d.shape()[2]==anchorBox.num_grid_y)
            {
                if(d.shape()[3] == (4 + 1 + numClasses)*anchorBox.num_boxes)
                {
                    OutputLayer outputLayer = {0,};
                    outputLayer.boxes = anchorBox.num_boxes;
                    outputLayer.gridX = anchorBox.num_grid_x;
                    outputLayer.gridY = anchorBox.num_grid_y;
                    outputLayer.stride = cfg.image_size/anchorBox.num_grid_x;
                    outputLayer.anchorHeight.assign(outputLayer.boxes, 0);
                    outputLayer.anchorWidth.assign(outputLayer.boxes, 0);
                    for(int box=0; box<outputLayer.boxes; box++)
                    {
                        outputLayer.anchorHeight[box] = anchorBox.height[box];
                        outputLayer.anchorWidth[box] = anchorBox.width[box];
                    }
                    outputLayer.dataAlign = ((int)(d.shape()[2]/64) + 1)*64;
                    outputLayer.dataOffset = d.elem_size();
                    OutputLayers.emplace_back(outputLayer);
                }
            }
        }
    }
    for(auto &o : OutputLayers)
        o.Show();

}
void Yolo::FilterWithSort(float *org, vector<shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
    int grid;
    int x = 0, y = 1, w = 2, h = 3;
    float ScoreThreshold = cfg.score_threshold;
    float conf_threshold = cfg.conf_threshold;
    float rawThreshold = log(conf_threshold/(1-conf_threshold));
    float score, score1, anchor_grid, tmp, box_temp[4];
    float *data;
        for(int layer=0; layer<numLayers; layer++)
        {
            auto outputLayer = OutputLayers[layer];
            int stride = outputLayer.stride;
            int numGridX = outputLayer.gridX;
            int numGridY = outputLayer.gridY;
            // outputLayer.Show();

            for(int gY=0; gY<numGridY; gY++)
            {
                for(int gX=0; gX<numGridX; gX++)
                {
                    // if(org==nullptr)
                    // {
                    //     data = (float*)(outputs_[layer]->GetData() + 4*outputLayer.dataAlign*(gY*numGridX + gX));
                    // }
                    // else
                    // {
                    //     data = org + outputLayer.dataOffset/4 + outputLayer.dataAlign*(gY*numGridX + gX);
                    // }
                    for(int box=0; box<outputLayer.boxes; box++)
                    { 
                        bool boxDecoded = false;
                        data = (float*)(outputs_[layer]->data(gY, gX, box*(4+1+numClasses)));
                        // cout << boxIdx << ": " << hex << data << ", " << dec << gX << " x " << gY << ", " << dec << data[4] << ", " << sigmoid(data[4]) << endl;
                        if(data[4]>rawThreshold)
                        {
                            score1 = sigmoid(data[4]);
                            /* Step1 - obj_conf > CONF_THRESHOLD */
                            if(score1 > conf_threshold)
                            {
                                for(int cls=0; cls<numClasses;cls++)
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
                                            for (int ch=0; ch<4; ch++) /*Box and Score*/
                                            {
                                                tmp = sigmoid(data[ch]); /*sigmoid*/
                                                if (ch < 2) {
                                                    if (ch == 0) {
                                                        grid = gX;//outputLayer.gridX;
                                                    } else {
                                                        grid = gY;//outputLayer.gridY;
                                                    }
                                                    tmp = (tmp * 2. - 0.5 + grid) * outputLayer.stride; /*XY*/
                                                } else {
                                                    if (ch == 2) {
                                                        anchor_grid = outputLayer.anchorWidth[box];
                                                    } else {
                                                        anchor_grid = outputLayer.anchorHeight[box];
                                                    }
                                                    tmp = pow((tmp * 2.), 2) * anchor_grid; /*WH*/
                                                }
                                                box_temp[ch] = tmp;
                                            }
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
                        data += (4+1+numClasses);
                        // LOG_VALUE(boxIdx);
                        boxIdx++;
                    }
                }
            }
        }
    for(int cls=0;cls<numClasses;cls++)
    {
        sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), greater<>());
    }
}
vector< BoundingBox > Yolo::PostProc(vector<shared_ptr<dxrt::Tensor>> outputs_, void *saveTo)
{
    outputs = outputs_;
#ifdef DUMP_DATA
    for(int i=0;i<outputs.size();i++)
    {
        dxrt::DataDumpBin("output."+to_string(i)+".bin", outputs[i]->GetData(), outputs[i]->GetDataInfo().mem_size);
    }
#endif
    for(int cls=0;cls<numClasses;cls++)
    {
        ScoreIndices[cls].clear();
    }
    Result.clear();
    FilterWithSort(nullptr, outputs);
    Nms(
        numClasses,
        0,
        ClassNames, 
        ScoreIndices, Boxes, nullptr, cfg.iou_threshold,
        Result,
        0
    );
    if(saveTo!=nullptr)
    {
        BoundingBox *boxes = (BoundingBox*)saveTo;
        memcpy(saveTo, &Result[0], Result.size()*sizeof(Result[0]));
    }
    return Result;
}