#include "bbox.h"

BoundingBox::BoundingBox(unsigned int _label, std::string const & _labelname, float _score,
        float data1, float data2, float data3, float data4, float *keypoints)
        : label(_label), score(_score), labelname(_labelname) 
    {
        box[0] = data1;
        box[1] = data2;
        box[2] = data3;
        box[3] = data4;
        if(keypoints)
            for (int i = 0; i < 51; i++) 
                kpt[i] = keypoints[i];
    }

void BoundingBox::Show() {
    std::cout << "    BBOX:" << labelname << "(" 
         << label << ") " << score << ", (" 
         << box[0] << ", " << box[1] << ", " 
         << box[2] << ", " << box[3] << ")" << std::endl;
}