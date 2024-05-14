#include "string.h"
#include "bbox.h"

using namespace std;
BoundingBox::~BoundingBox(void) {}
BoundingBox::BoundingBox(void) {}
BoundingBox::BoundingBox(unsigned int _label, string _labelname, float _score,
                         float data1, float data2, float data3, float data4)
    : label(_label), score(_score)
{
    box[0] = data1;
    box[1] = data2;
    box[2] = data3;
    box[3] = data4;
    labelname = _labelname;
}

void BoundingBox::Show(void)
{
    cout << "    BBOX: " << name << ", " << labelname << "("
         << label << ") " << score << ", ("
         << box[0] << ", " << box[1] << ", "
         << box[2] << ", " << box[3] << ")" << endl;
}

float calc_dist_bbox(BoundingBox &a, BoundingBox &b)
{
    float ret = (a.box[0] - b.box[0]) * (a.box[0] - b.box[0]);
    ret += (a.box[1] - b.box[1]) * (a.box[1] - b.box[1]);
    return ret;
}
