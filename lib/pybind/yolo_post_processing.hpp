#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <iostream>
#include <functional>

namespace py = pybind11;

class YoloPostProcess
{
private:
    struct YoloLayerParam
    {
        float _scaleXY;
        int _stride;
        std::vector<float> _anchorWidth;
        std::vector<float> _anchorHeight;
    };

    struct YoloParam
    {
        float _confThreshold;
        float _scoreThreshold;
        float _iouThreshold;
        int _numClasses;
        int _numKeypoints;
        std::string _decodeMethod;
        std::string _boxFormat;
        std::vector<YoloLayerParam> _layers;
    };

    std::function<float(float)> _lastActivation;
    YoloParam _param;

    std::pair<float, float> _ratio;
    std::pair<float, float> _pad;

    static float CalcIOU(float *box1, float *box2);

    static bool ScoreCompare(std::pair<float, int> &a,
                             std::pair<float, int> &b);

    void NMS(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
             std::vector<float> &Boxes,
             std::vector<float> &Keypoints,
             std::vector<float> &Results);

    void ProcessPPU(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
                    std::vector<float> &Boxes,
                    std::vector<float> &Keypoints,
                    py::list &ie_output);
    void ProcessONNX(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
                     std::vector<float> &Boxes,
                     std::vector<float> &Keypoints,
                     py::list &ie_output);
    void ProcessRAW(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
                    std::vector<float> &Boxes,
                    py::list &ie_output);

public:
    YoloPostProcess(py::dict config);

    void SetConfig(py::dict config);

    py::array_t<float> Run(py::list ie_output,
                           std::pair<float, float> ratio, 
                           std::pair<float, float> pad);
};