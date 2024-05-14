#pragma once

#include <iostream>
#include <string>
#include <vector>

struct BoundingBox
{
    unsigned int label = -1;
    // std::string labelname;
    std::string name = "Unknown";
    std::string labelname;
    float score;
    float box[4];
    float feature[512];
    ~BoundingBox(void);
    BoundingBox(void);
    BoundingBox(unsigned int _label, std::string _labelname, float _score,
        float data1, float data2, float data3, float data4);
    void Show(void);
};