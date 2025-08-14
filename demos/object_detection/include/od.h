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
#include "yolo.h"
#include <dxrt/dxrt_api.h>
#include <common/objects.hpp>
#include <utils/videostream.hpp>
#include <queue>

class ObjectDetection
{
public:
    ObjectDetection(std::shared_ptr<dxrt::InferenceEngine> ie, std::pair<std::string, std::string> &videoSrc, int channel, 
                    int width, int height, int dstWidth, int dstHeight,
                    int posX, int posY, int numFrames);
    ObjectDetection(std::shared_ptr<dxrt::InferenceEngine> ie, int channel, int destWidth, int destHeight, int posX, int posY);
    ~ObjectDetection();
    void threadFunc(int period);
    void threadFillBlank(int period);
    void Run(int period);
    void Stop();
    void Pause();
    void Play();
    cv::Mat ResultFrame();
    std::pair<int, int> Position();
    std::pair<int, int> Resolution();
    uint64_t GetInferenceTime();
    uint64_t GetLatencyTime();
    uint64_t GetProcessingTime();
    uint64_t GetPostProcessCount();
    void SetZeroPostProcessCount();
    uint8_t* GetOutputMemory(){return outputMemory;};
    int Channel();
    std::string &Name();
    void Toggle();
    void PostProc(std::vector<std::shared_ptr<dxrt::Tensor>>&);
    dxapp::common::DetectObject GetScalingBBox(std::vector<BoundingBox>& bboxes);
    friend std::ostream& operator<<(std::ostream&, const ObjectDetection&);
private:
    std::shared_ptr<dxrt::InferenceEngine> _ie;
    dxrt::Profiler &_profiler;
    VideoStream _vStream;
    dxapp::common::Size_f _postprocPaddedSize;
    dxapp::common::Size_f _postprocScaleRatio;
    std::string _name;
    int _channel;
    int _targetFps = 30;
    uint64_t _inferenceTime = 0;
    uint64_t _latencyTime = 0;
    uint64_t _processTime = 0;
    uint64_t _duration_time = 0;
    uint64_t _processAverageTime = 0;
    uint64_t _processed_count = 0;
    uint64_t _ret_processed_count = 0;
    std::chrono::high_resolution_clock::time_point _fps_time_s;
    std::chrono::high_resolution_clock::time_point _fps_time_e;
    int _srcWidth;
    int _srcHeight;
    int _width;
    int _height;
    int _destWidth;
    int _destHeight;
    int _posX;
    int _posY;
    bool _offline;
    bool _toggleDrawing = true;
    bool _isPause = false;
    std::pair<std::string, std::string> _videoSrc;
    cv::VideoCapture _cap;
    std::vector<cv::Mat> _dest;
    cv::Mat _resultFrame;

    cv::Mat _logo;
    std::thread _thread;
    std::atomic<bool> stop;
    std::mutex _lock;
    std::mutex _frameLock;
    std::condition_variable _cv;
    std::vector<BoundingBox> _bboxes;
    Yolo yolo;

    uint8_t* outputMemory;
    int64_t output_length;
    std::vector<std::vector<int64_t>> output_shape;
    dxrt::DataType data_type;
};