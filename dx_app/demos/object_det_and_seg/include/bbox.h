#pragma once

#include <iostream>
#include <string>
#include <vector>

struct BoundingBox
{
    unsigned int label;
    // std::string labelname;
    char labelname[20];
    float score;
    float box[4];
    ~BoundingBox(void);
    BoundingBox(void);
    BoundingBox(unsigned int _label, char _labelname[], float _score,
        float data1, float data2, float data3, float data4);
    void Show(void);
};