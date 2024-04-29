#include <algorithm>
#include "ssd.h"
#include "dxrt/util.h"

// #define DUMP_DATA

using namespace std;
void PriorBoxParam::Show()
{
    cout << "  PriorBoxParam: " << endl << "    layers: " << num_layers << ", "
        << "min_scale: " << min_scale << ", "
        << "max_scale: " << max_scale << ", "
        << "center_variance: " << center_variance << ", "
        << "size_variance: " << size_variance << endl;
    for(auto &d : dim)
    {
        cout << "    " << d.num_grid_x << " x " << d.num_grid_y << " x " << d.num_boxes << endl;
    }
}
void SsdParam::Show()
{
    cout << "  SsdParam: " << endl << "    score_threshold: " << score_threshold << ", "
        << "iou_threshold: " << iou_threshold << ", " << "num_classes: " << num_classes 
        << ", " << "use_softmax:" << use_softmax << endl;
    cout << "    - classes: [";
    for(auto &c : class_names) cout << c << ", ";
    cout << "]" << endl;
    priorBoxes.Show();
}
Ssd::Ssd(SsdParam &_cfg)
    :cfg(_cfg)
{
    //setup number of boxes, classes, layers
    numClasses = cfg.num_classes;
    numBoxes = 0;
    for(auto &dim : cfg.priorBoxes.dim)
    {
        numBoxes += dim.num_grid_x*dim.num_grid_y*dim.num_boxes;
    }
    numLayers = cfg.priorBoxes.num_layers;
    ClassNames = cfg.class_names;
    //memory allocate
    Boxes = new float[numBoxes * 4];
    Scores = new float[numBoxes * numClasses];
    PriorBoxes = new float[numBoxes * 4];
    cout << "Ssd created : " << numBoxes << " boxes, " << numClasses << " classes, " 
        << numLayers << " layers." << endl;
    cfg.Show();
    //create prior boxes
    CreatePriorBoxes(cfg.priorBoxes.data_file);
    //prepare score indices
    for(size_t i=0; i<numClasses; i++)
    {
        vector<pair<float, int>> v;
        ScoreIndices.emplace_back(v);
    }
    //setup output layers
    for(int layer=0; layer<(int)numLayers; layer++)
    {
        OutputLayer outputLayer = {};
        auto priorBox = cfg.priorBoxes.dim[layer];
        outputLayer.boxes = priorBox.num_boxes;
        outputLayer.gridX = priorBox.num_grid_x;
        outputLayer.gridY = priorBox.num_grid_y;
        OutputLayers.emplace_back(outputLayer);
        outputLayer.Show();
    }
}

Ssd::Ssd() {}
Ssd::~Ssd()
{
    delete [] Boxes;
    delete [] Scores;
    delete [] PriorBoxes;
}

void Ssd::CreatePriorBoxes(const string &file)
{
    if(file.empty())
    {
        //temp
        float aspect_ratios[6] = {1.0, 2.0, 0.5, 3.0, 1./3.};
        cout << "Create Prior Boxes from SSD cfg." << endl;
        int m = numLayers;
        int box, k, num_boxes, idx = 0;
        int layer_h, layer_w, i, j;
        float min_scale = cfg.priorBoxes.min_scale;
        float max_scale = cfg.priorBoxes.max_scale;
        float scale, aspect_ratio, x, y, w, h;
        bool reduce_boxes_in_lowest_layer = cfg.priorBoxes.dim[0].num_boxes<cfg.priorBoxes.dim[1].num_boxes;
        vector<float> scales;
        scales.emplace_back(0.0);
        for(k=1; k<=m; k++)
        {
            scale = min_scale + (max_scale-min_scale)/(m-1)*(k-1);
            scales.emplace_back(scale);
        }
        for(k=1; k<=m; k++)
        {        
            num_boxes = cfg.priorBoxes.dim[k-1].num_boxes;
            layer_h = cfg.priorBoxes.dim[k-1].num_grid_y;
            layer_w = cfg.priorBoxes.dim[k-1].num_grid_x;
            for(i=0; i<layer_h; i++)
            {
                for(j=0; j<layer_w; j++)
                {
                    x = 1.0/layer_w*j + 1.0/layer_w*0.5;
                    y = 1.0/layer_h*i + 1.0/layer_h*0.5;
                    for(box=0; box<num_boxes;box++)
                    {            
                        if(box<num_boxes-1)
                        {
                            scale = scales[k];
                            aspect_ratio = aspect_ratios[box];
                            if(k==1 && reduce_boxes_in_lowest_layer && aspect_ratio==1.0)
                            {
                                scale = 0.1;
                            }
                        }
                        else
                        {
                            /* Extra Prior */
                            scale = sqrt(scales[k]*( (k==m)?1:scales[k+1]) );
                            aspect_ratio = 1.0;
                            if(k==1 && reduce_boxes_in_lowest_layer)
                            {
                                scale = scales[k];
                                aspect_ratio = aspect_ratios[box];
                            }
                        }
                        w = scale*sqrt(aspect_ratio);
                        h = scale/sqrt(aspect_ratio);
                        PriorBoxes[idx++] = x;
                        PriorBoxes[idx++] = y;
                        PriorBoxes[idx++] = w;
                        PriorBoxes[idx++] = h;
                    }
                }
            }
        }
        dxrt::DataDumpTxt("prior_boxes.generate.txt", PriorBoxes, 1, numBoxes, 4);
    }
    else
    {
        cout << "Create Prior Boxes from file: " << file << endl;
        int ret = access(file.c_str(),F_OK); /* Not OK : -1 */
        if (ret != -1) {
            std::ifstream fin(file, ios_base::binary);
            fin.read((char*)PriorBoxes, sizeof(float)*numBoxes*4);
            fin.close();
        } else {
            cout << __func__ << ": " << file << " doesn't exist." << endl;
            exit(-1);
        }
        dxrt::DataDumpTxt("prior_boxes.import.txt", PriorBoxes, 1, numBoxes, 4);
    }
}

void Ssd::FilterWithSoftmax(vector<shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
#if 0
    int x = 1, y = 0, w = 3, h = 2;
#else
    int x = 0, y = 1, w = 2, h = 3;
#endif
    float scoreThreshold = cfg.score_threshold;
    float *boxLocation, *classScore;
    float centerVariance = cfg.priorBoxes.center_variance;
    float sizeVariance = cfg.priorBoxes.size_variance;
    float center_x, center_y, width, height;
    for(int layer=0; layer<(int)numLayers; layer++)
    {
        auto outputLayer = OutputLayers[layer];
        int numGridX = outputLayer.gridX;
        int numGridY = outputLayer.gridY;
        int _numBoxes = outputLayer.boxes;
        int inc1 = 1;
        for(int gY=0; gY<numGridY; gY++)
        {
            for(int gX=0; gX<numGridX; gX++)
            {
                for(int box=0; box<_numBoxes; box++)
                { 
                    bool boxDecoded = false;
                    float sum = 0;
                    classScore = (float*)(outputs_[2*layer]->data(gY, gX, box*numClasses));
                    boxLocation = (float*)(outputs_[2*layer+1]->data(gY, gX, box*4));
                    for(int cls=0; cls<(int)numClasses;cls++)
                    {
                        sum += exp(classScore[cls*inc1]);
                    }
                    for(int cls=1; cls<(int)numClasses;cls++)
                    {
                        float score = exp(classScore[cls*inc1])*(1/sum);
                        if (score > scoreThreshold)
                        {
                            ScoreIndices[cls].emplace_back(score, boxIdx);
                            if(!boxDecoded)
                            {
                                float *boxOut = Boxes + boxIdx*4;
                                float *prior = PriorBoxes + boxIdx*4;
                                center_x = prior[0] + boxLocation[x*inc1]*centerVariance*prior[2];
                                center_y = prior[1] + boxLocation[y*inc1]*centerVariance*prior[3];
                                width = exp(boxLocation[w*inc1]*sizeVariance)*prior[2];
                                height = exp(boxLocation[h*inc1]*sizeVariance)*prior[3];
                                boxOut[0] = center_x - width/2;
                                boxOut[1] = center_y - height/2;
                                boxOut[2] = center_x + width/2;
                                boxOut[3] = center_y + height/2;
                                boxDecoded = true;
                            }
                        }
                    }
                    boxIdx++;
                }
            }
        }
    }
    for(int cls=1;cls<(int)numClasses;cls++)
    {
        sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), greater<>());
    }
}
void Ssd::FilterWithSigmoid(vector<shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
    int x = 1, y = 0, w = 3, h = 2;
    float scoreThreshold = cfg.score_threshold;
    float rawThreshold = log(scoreThreshold/(1-scoreThreshold));
    float *boxLocation, *classScore;
    float centerVariance = cfg.priorBoxes.center_variance;
    float sizeVariance = cfg.priorBoxes.size_variance;
    float center_x, center_y, width, height;
    for(int layer=0; layer<(int)numLayers; layer++)
    {
        auto outputLayer = OutputLayers[layer];
        int numGridX = outputLayer.gridX;
        int numGridY = outputLayer.gridY;
        int _numBoxes = outputLayer.boxes;
        int inc1 = 1;
        for(int gY=0; gY<numGridY; gY++)
        {
            for(int gX=0; gX<numGridX; gX++)
            {
                for(int box=0; box<_numBoxes; box++)
                { 
                    bool boxDecoded = false;
                    classScore = (float*)(outputs_[2*layer]->data(gY, gX, box*numClasses));
                    boxLocation = (float*)(outputs_[2*layer+1]->data(gY, gX, box*4));
                    for(int cls=1; cls<(int)numClasses;cls++)
                    {
                        float score = classScore[cls*inc1];
                        if (score > rawThreshold)
                        {
                            ScoreIndices[cls].emplace_back(sigmoid(score), boxIdx);
                            if(!boxDecoded)
                            {
                                float *boxOut = Boxes + boxIdx*4;
                                float *prior = PriorBoxes + boxIdx*4;
                                center_x = prior[0] + boxLocation[x*inc1]*centerVariance*prior[2];
                                center_y = prior[1] + boxLocation[y*inc1]*centerVariance*prior[3];
                                width = exp(boxLocation[w*inc1]*sizeVariance)*prior[2];
                                height = exp(boxLocation[h*inc1]*sizeVariance)*prior[3];
                                boxOut[0] = center_x - width/2;
                                boxOut[1] = center_y - height/2;
                                boxOut[2] = center_x + width/2;
                                boxOut[3] = center_y + height/2;
                                boxDecoded = true;
                            }
                        }
                    }
                    boxIdx++;
                }
            }
        }
    }
    for(int cls=1;cls<(int)numClasses;cls++)
    {
        sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), greater<>());
    }
}

vector< BoundingBox > Ssd::PostProc(vector<shared_ptr<dxrt::Tensor>> outputs_, void *saveTo)
{
    outputs = outputs_;
    for(int cls=1; cls<(int)numClasses; cls++)
    {
        ScoreIndices[cls].clear();
    }
    Result.clear();
    if(cfg.use_softmax)
        FilterWithSoftmax(outputs);
    else
        FilterWithSigmoid(outputs);
    Nms(
        numClasses,
        0,
        ClassNames, 
        ScoreIndices, Boxes, cfg.iou_threshold,
        Result,
        1
    );
    if(saveTo!=nullptr)
    {
        memcpy(saveTo, &Result[0], Result.size()*sizeof(Result[0]));
    }
    return Result;
}
