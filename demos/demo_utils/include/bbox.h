#pragma once

#include <iostream>
#include <string>
#include <vector>

struct BoundingBox
{
    // Member variables ordered by initialization sequence
    int label = -1;
    float score = 0.0f;
    std::string labelname = "";
    float box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float kpt[51] = {0.0f};

    BoundingBox() = default;
    ~BoundingBox() = default;
    
    // Constructor with initialization list
    BoundingBox(unsigned int _label, std::string const & _labelname, float _score,
        float data1, float data2, float data3, float data4, float *keypoints=nullptr);

    void Show(void);
};

/* For network packet communication */
typedef struct {
    int frameId;
    BoundingBox bboxes[100];
} BoundingBoxPacket_t;
