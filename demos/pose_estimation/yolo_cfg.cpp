#include "yolo.h"

YoloLayerParam createYoloLayerParam(int _gx, int _gy, int _numB, const std::vector<float> &_vAnchorW, const std::vector<float> &_vAnchorH, const std::vector<int> &_vTensorIdx, float _sx = 0.f, float _sy = 0.f)
{
        YoloLayerParam s;
        s.numGridX = _gx;
        s.numGridY = _gy;
        s.numBoxes = _numB;
        s.anchorWidth = _vAnchorW;
        s.anchorHeight = _vAnchorH;
        s.tensorIdx = _vTensorIdx;
        s.scaleX = _sx;
        s.scaleY = _sy;
        return s;
}

YoloParam yolov5s6_pose_640 = {
    .height = 640,
    .width = 640,
    .confThreshold = 0.3,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 1,
    .layers = {
        createYoloLayerParam(80, 80, 3, { 19.0, 44.0, 38.0 }, { 27.0, 40.0, 94.0 }, { 1, 0 }),
        createYoloLayerParam(40, 40, 3, { 96.0, 86.0, 180.0 }, { 68.0, 152.0, 137.0 }, { 3, 2 }),
        createYoloLayerParam(20, 20, 3, { 140.0, 303.0, 238.0 }, { 301.0, 264.0, 542.0 }, { 5, 4 }),
        createYoloLayerParam(10, 10, 3, { 436.0, 739.0, 925.0 }, { 615.0, 380.0, 792.0 }, { 7, 6 })
    },
    .classNames = {"person"},
};

YoloParam yolov5s6_pose_1280 = {
    .height = 1280,
    .width = 1280,
    .confThreshold = 0.3,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 1,
    .layers = {
        createYoloLayerParam(160, 160, 3, { 19.0, 44.0, 38.0 }, { 27.0, 40.0, 94.0 }, { 1, 0 }),
        createYoloLayerParam(80, 80, 3, { 96.0, 86.0, 180.0 }, { 68.0, 152.0, 137.0 }, { 3, 2 }),
        createYoloLayerParam(40, 40, 3, { 140.0, 303.0, 238.0 }, { 301.0, 264.0, 542.0 }, { 5, 4 }),
        createYoloLayerParam(20, 20, 3, { 436.0, 739.0, 925.0 }, { 615.0, 380.0, 792.0 }, { 7, 6 })
    },
    .classNames = {"person"},
};
