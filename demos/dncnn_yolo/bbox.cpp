#include "string.h"
#include "bbox.h"

using namespace std;
BoundingBox::~BoundingBox(void) {}
BoundingBox::BoundingBox(void) {}
BoundingBox::BoundingBox(unsigned int _label, char _labelname[20], float _score,
        float data1, float data2, float data3, float data4)
    : label(_label), score(_score)
{
    box[0] = data1;
    box[1] = data2;
    box[2] = data3;
    box[3] = data4;
    strncpy(labelname, _labelname, 20);
}

void BoundingBox::Show(void)
{
    cout << "    BBOX:" << labelname << "(" \
            << label << ") " << score << ", (" \
            << box[0] << ", " << box[1] << ", " \
            << box[2] << ", " << box[3] << ")" << endl;
}