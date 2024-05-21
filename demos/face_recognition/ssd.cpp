#include <algorithm>
#include "ssd.h"
#include "dxrt/util.h"

using namespace std;
void PriorBoxParam::Show()
{
    cout << "  PriorBoxParam: " << endl
         << "    layers: " << num_layers << ", "
         << "min_scale: " << min_scale << ", "
         << "max_scale: " << max_scale << ", "
         << "center_variance: " << center_variance << ", "
         << "size_variance: " << size_variance << endl;
    for (auto &d : dim)
    {
        cout << "    " << d.num_grid_x << " x " << d.num_grid_y << " x " << d.num_boxes << endl;
    }
}
void SsdParam::Show()
{
    cout << "  SsdParam: " << endl
         << "    score_threshold: " << score_threshold << ", "
         << "iou_threshold: " << iou_threshold << ", "
         << "num_classes: " << num_classes
         << ", "
         << "use_softmax:" << use_softmax << endl;
    cout << "    - classes: [";
    for (auto &c : class_names)
        cout << c << ", ";
    cout << "]" << endl;
    priorBoxes.Show();
}
Ssd::Ssd(SsdParam &_cfg, std::vector<dxrt::Tensor> &_datainfo)
    : cfg(_cfg), datainfo(_datainfo)
{
    // setup number of boxes, classes, layers
    numClasses = cfg.num_classes;
    numBoxes = 0;
    for (auto &dim : cfg.priorBoxes.dim)
    {
        numBoxes += dim.num_grid_x * dim.num_grid_y * dim.num_boxes;
    }
    numLayers = cfg.priorBoxes.num_layers;
    ClassNames = cfg.class_names;
    // memory allocate
#if 0
    Boxes = new float[numBoxes * 4];
    Scores = new float[numBoxes * numClasses];
    PriorBoxes = new float[numBoxes * 4];
#else
    Boxes = vector<float>(numBoxes * 4);
    Scores = vector<float>(numBoxes * numClasses);
    PriorBoxes = vector<float>(numBoxes * 4);
#endif

    cout << "Ssd created : " << numBoxes << " boxes, " << numClasses << " classes, "
         << numLayers << " layers." << endl;
    cfg.Show();
    // create prior boxes
    CreatePriorBoxes(cfg.priorBoxes.data_file);
    // prepare score indices
    for (uint32_t i = 0; i < numClasses; i++)
    {
        vector<pair<float, int>> v;
        ScoreIndices.emplace_back(v);
    }
    // setup output layers
    for (uint32_t layer = 0; layer < numLayers; layer++)
    {
        OutputLayer outputLayer = {};
        auto priorBox = cfg.priorBoxes.dim[layer];
        outputLayer.boxes = priorBox.num_boxes;
        outputLayer.gridX = priorBox.num_grid_x;
        outputLayer.gridY = priorBox.num_grid_y;
        for (size_t i = 0; i < datainfo.size(); i++)
        {
            auto d = datainfo[i];
            if (d.name() == cfg.score_names[layer])
            {
                outputLayer.scoreAlign = 64;
                // outputLayer.scoreOffset = d.mem_offset;
                layerMap[d.name()] = i / 2;
            }
            else if (d.name() == cfg.loc_names[layer])
            {
                outputLayer.locAlign = 64;
                // outputLayer.locOffset = d.mem_offset;
                layerMap[d.name()] = i / 2;
            }
        }
        OutputLayers.emplace_back(outputLayer);
        outputLayer.Show();
    }
}

Ssd::Ssd() {}
Ssd::~Ssd() {}

void Ssd::CreatePriorBoxes(const string &file)
{
    cout << "Create Prior Boxes from file: " << file << endl;
    int ret = access(file.c_str(), F_OK); /* Not OK : -1 */
    if (ret != -1)
    {
        std::ifstream fin(file, ios_base::binary);
        fin.read((char *)&PriorBoxes[0], sizeof(float) * numBoxes * 4);
        fin.close();
    }
    else
    {
        cout << __func__ << ": " << file << " doesn't exist." << endl;
        exit(-1);
    }
}

static bool scoreComapre(const std::pair<float, int> &a, const std::pair<float, int> &b)
{
    if(a.first > b.first)
        return true;
    else
        return false;
};

void Ssd::FilterWithSoftmax(float *org, vector<shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
    int incLoc = 4;
    int incScore = numClasses;
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
    for (uint32_t layer = 0; layer < numLayers; layer++)
    {
        auto outputLayer = OutputLayers[layer];
        int numGridX = outputLayer.gridX;
        int numGridY = outputLayer.gridY;
        int tensorIdx = layerMap[cfg.score_names[layer]];
        // outputLayer.Show();
        // LOG_VALUE(tensorIdx);
        // outputs_[2*tensorIdx]->Show();
        int _numBoxes = outputLayer.boxes;
        int inc1 = 1;
        for (int gY = 0; gY < numGridY; gY++)
        {
            for (int gX = 0; gX < numGridX; gX++)
            {
                if (org == nullptr)
                {
                    classScore = (float *)(static_cast<uint8_t*>(outputs_[2 * tensorIdx + 1]->data()) + sizeof(float) * outputLayer.scoreAlign * (gY * numGridX + gX));
                    boxLocation = (float *)(static_cast<uint8_t*>(outputs_[2 * tensorIdx]->data()) + sizeof(float) * outputLayer.locAlign * (gY * numGridX + gX));
                }
                else
                {
                    classScore = org + outputLayer.scoreOffset / 4 + outputLayer.scoreAlign * (gY * numGridX + gX);
                    boxLocation = org + outputLayer.locOffset / 4 + outputLayer.locAlign * (gY * numGridX + gX);
                }
                for (int box = 0; box < _numBoxes; box++)
                {
                    bool boxDecoded = false;
                    float sum = 0;
                    for (uint32_t cls = 0; cls < numClasses; cls++)
                    {
                        sum += exp(classScore[cls * inc1]);
                    }
                    for (uint32_t cls = cfg.start_class; cls < numClasses; cls++)
                    {
                        float score = exp(classScore[cls * inc1]) * (1 / sum);
                        if (score > scoreThreshold)
                        {
                            ScoreIndices[cls].emplace_back(score, boxIdx);
                            if (!boxDecoded)
                            {
                                float *boxOut = &Boxes[boxIdx * 4];
                                float *prior = &PriorBoxes[boxIdx * 4];
                                center_x = prior[0] + boxLocation[x * inc1] * centerVariance * prior[2];
                                center_y = prior[1] + boxLocation[y * inc1] * centerVariance * prior[3];
                                width = exp(boxLocation[w * inc1] * sizeVariance) * prior[2];
                                height = exp(boxLocation[h * inc1] * sizeVariance) * prior[3];
                                boxOut[0] = center_x - width / 2;
                                boxOut[1] = center_y - height / 2;
                                boxOut[2] = center_x + width / 2;
                                boxOut[3] = center_y + height / 2;
                                boxDecoded = true;
                            }
                        }
                    }
                    boxLocation += incLoc;
                    classScore += incScore;
                    boxIdx++;
                }
            }
        }
    }
    for (uint32_t cls = 1; cls < numClasses; cls++)
    {
        sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), scoreComapre);
    }
}
void Ssd::FilterWithSigmoid(float *org, vector<shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
    int incLoc = 4;
    int incScore = numClasses;
    int x = 1, y = 0, w = 3, h = 2;
    float scoreThreshold = cfg.score_threshold;
    float rawThreshold = log(scoreThreshold / (1 - scoreThreshold));
    float *boxLocation, *classScore;
    float centerVariance = cfg.priorBoxes.center_variance;
    float sizeVariance = cfg.priorBoxes.size_variance;
    float center_x, center_y, width, height;
    for (uint32_t layer = 0; layer < numLayers; layer++)
    {
        auto outputLayer = OutputLayers[layer];
        int numGridX = outputLayer.gridX;
        int numGridY = outputLayer.gridY;
        // outputs_[2*layer]->Show();
        // outputs_[2*layer+1]->Show();
        // outputLayer.Show();
        int _numBoxes = outputLayer.boxes;
        int inc1 = 1;
        for (int gY = 0; gY < numGridY; gY++)
        {
            for (int gX = 0; gX < numGridX; gX++)
            {
                if (org == nullptr)
                {
                    classScore = (float *)(static_cast<uint8_t*>(outputs_[2 * layer]->data()) + sizeof(float) * outputLayer.scoreAlign * (gY * numGridX + gX));
                    boxLocation = (float *)(static_cast<uint8_t*>(outputs_[2 * layer + 1]->data()) + sizeof(float) * outputLayer.locAlign * (gY * numGridX + gX));
                }
                else
                {
                    classScore = org + outputLayer.scoreOffset / 4 + outputLayer.scoreAlign * (gY * numGridX + gX);
                    boxLocation = org + outputLayer.locOffset / 4 + outputLayer.locAlign * (gY * numGridX + gX);
                }
                for (int box = 0; box < _numBoxes; box++)
                {
                    bool boxDecoded = false;
                    for (uint32_t cls = 1; cls < numClasses; cls++)
                    {
                        float score = classScore[cls * inc1];
                        if (score > rawThreshold)
                        {
                            ScoreIndices[cls].emplace_back(sigmoid(score), boxIdx);
                            if (!boxDecoded)
                            {
                                float *boxOut = &Boxes[boxIdx * 4];
                                float *prior = &PriorBoxes[boxIdx * 4];
                                center_x = prior[0] + boxLocation[x * inc1] * centerVariance * prior[2];
                                center_y = prior[1] + boxLocation[y * inc1] * centerVariance * prior[3];
                                width = exp(boxLocation[w * inc1] * sizeVariance) * prior[2];
                                height = exp(boxLocation[h * inc1] * sizeVariance) * prior[3];
                                boxOut[0] = center_x - width / 2;
                                boxOut[1] = center_y - height / 2;
                                boxOut[2] = center_x + width / 2;
                                boxOut[3] = center_y + height / 2;
                                boxDecoded = true;
                            }
                        }
                    }
                    boxLocation += incLoc;
                    classScore += incScore;
                    boxIdx++;
                }
            }
        }
    }
    for (uint32_t cls = 1; cls < numClasses; cls++)
    {
        sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), scoreComapre);
    }
}

vector<BoundingBox> Ssd::PostProc(vector<shared_ptr<dxrt::Tensor>> outputs_, void *saveTo)
{
    outputs = outputs_;
    for (uint32_t cls = 1; cls < numClasses; cls++)
    {
        ScoreIndices[cls].clear();
    }
    Result.clear();
    if (cfg.use_softmax)
        FilterWithSoftmax(nullptr, outputs);
    else
        FilterWithSigmoid(nullptr, outputs);
    Nms(
        numClasses,
        0,
        ClassNames,
        ScoreIndices, &Boxes[0], cfg.iou_threshold,
        Result,
        1);
    if (saveTo != nullptr)
    {
        memcpy(saveTo, &Result[0], Result.size() * sizeof(Result[0]));
    }
    return Result;
}
