#pragma once

#include <cstdint>
#include <string>
#include <vector>
struct SegmentationParam
{
    int classIndex;
    std::string className;
    uint8_t colorB;
    uint8_t colorG;
    uint8_t colorR;
};

void Segmentation(float *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses, const std::vector<int64_t>& shape);
void Segmentation(uint16_t *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses);
