#pragma once

#include <string>
#include <vector>
#include "bbox.h"

using namespace std;

float CalcIOU(BoundingBox &a, BoundingBox &b);
float CalcIOU(float* box, float* truth);

void NmsOneClass(
    unsigned int cls,
    vector<string> &ClassNames,
    vector<vector<pair<float, int>>> &ScoreIndices,
    float *Boxes, float IouThreshold,
    vector<BoundingBox> &Result
);

void Nms(
    const size_t &numClass,
    const int &numDetectTotal,
    vector<string> &ClassNames,
    vector<vector<pair<float, int>>> &ScoreIndices,
    float *Boxes, const float &IouThreshold,
    vector<BoundingBox> &Result,
    int startClass
);
