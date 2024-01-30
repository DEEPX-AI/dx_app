#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include "bbox.h"
#include "yolo.h"
#include "dxrt/dxrt_api.h"

class ObjectDetection
{
public:
    ObjectDetection(dxrt::InferenceEngine &ie, std::pair<std::string, std::string> &videoSrc, int channel, 
                    int width, int height, int dstWidth, int dstHeight,
                    int posX, int posY, int numFrames);
    ObjectDetection(dxrt::InferenceEngine &ie, int channel, int destWidth, int destHeight, int posX, int posY);
    ~ObjectDetection();
    void threadFunc(int period);
    void threadFillBlank(int period);
    void Run(int period);
    void Stop();
    void Pause();
    void Play();
    void Contract();
    void Expand();
    cv::Mat &ResultFrame();
    cv::Mat &ExpandedFrame();
    std::pair<int, int> Position();
    pair<int, int> Resolution();
    uint64_t GetInferenceTime();
    uint64_t GetProcessingTime();
    int Channel();
    std::string &Name();
    void Toggle();
    void PostProc(std::vector<std::shared_ptr<dxrt::Tensor>>&);
    cv::Mat DrawBoundingBox(cv::Mat src, cv::Size originalSize, vector<BoundingBox> &bboxes, bool adjustRatio);
    friend std::ostream& operator<<(std::ostream&, const ObjectDetection&);
private:
    std::string _name;
    int _channel;
    int _targetFps = 30;
    uint64_t _inferTime = 0;
    uint64_t _processTime = 0;
    int _srcWidth;
    int _srcHeight;
    int _width;
    int _height;
    int _destWidth;
    int _destHeight;
    int _posX;
    int _posY;
    int _numFrames = 5;
    bool _offline;
    bool _image = false;
    bool _adjustRatio = false;
    bool _toggleDrawing = true;
    bool _isPause = false;
    bool _isExpanded = false;
    std::vector<cv::Scalar> _colorTable;
    std::pair<std::string, std::string> _videoSrc;
    cv::VideoCapture _cap;
    std::vector<cv::Mat> _src;
    std::vector<cv::Mat> _preprocessed;
    std::vector<cv::Mat> _dest;
    cv::Mat _resultFrame;
    cv::Mat _expandedFrame;
    cv::Mat _logo;
    dxrt::InferenceEngine &_ie;
    dxrt::Profiler &_profiler;
    std::thread _thread;
    std::atomic<bool> stop;
    std::atomic<int> _resultIdx;
    std::mutex _lock;
    std::mutex _frameLock;
    std::condition_variable _cv;
    std::vector<BoundingBox> _bboxes;
    Yolo yolo;
};