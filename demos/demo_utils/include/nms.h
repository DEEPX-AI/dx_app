#pragma once

#include <string>
#include <vector>
#include "bbox.h"

float CalcIOU(float* box, float* truth);

void NmsOneClass(
    unsigned int cls,
    std::vector<std::string> &ClassNames,
    std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
    float *Boxes, float *Keypoints, float IouThreshold,
    std::vector<BoundingBox> &Result
);

void Nms(
    const size_t &numClass,
    const int &numDetectTotal,
    std::vector<std::string> &ClassNames,
    std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
    float *Boxes, float *Keypoints, const float &IouThreshold,
    std::vector<BoundingBox> &Result,
    int startClass
);
