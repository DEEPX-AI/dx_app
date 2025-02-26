#pragma once
#include <future>
#include <thread>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "post_process/yolo_post_processing.hpp"

#include "dxrt/dxrt_api.h"
#include "common/objects.hpp"
#include "utils/videostream.hpp"
#include "app_parser.hpp"

class DetectorApp
{
public :
    DetectorApp(std::shared_ptr<dxrt::InferenceEngine> inferenceEngine, AppSourceInfo sourceInfo, AppInputFormat input_format, AppType app_type, dxapp::yolo::Params& params, int channel, dxapp::common::Point position, dxapp::common::Size dst_size)
            : _inferenceEngine(inferenceEngine), _profiler(dxrt::Profiler::GetInstance()), _channel(channel + 1), _inputFormat(input_format), _appType(app_type), _params(params), _position(position), _dstSize(dst_size)
    {
        _videoPath = sourceInfo.inputPath;
        _inputType = sourceInfo.inputType;
        _name = "app" + std::to_string(_channel);
        _outName = _name + "_" + dxapp::common::getFileName(_videoPath);
        _processName = "proc_"+_name;
        _inferName = "infer_"+_name;

        _inputSize = dxapp::common::Size((int)_params._input_shape[2],(int)_params._input_shape[1]);

        _vStream = VideoStream(_inputType, _videoPath, sourceInfo.numOfFrames, _inputSize, _inputFormat, _dstSize, _inferenceEngine);        
        _srcSize = _vStream._srcSize;
        _postProcessing = dxapp::yolo::PostProcessing(_params, _inputSize, _srcSize, _dstSize);
        _resultFrame = cv::Mat(_dstSize._height, _dstSize._width, CV_8UC3, cv::Scalar(0, 0, 0));     
        _frame_count = 0;
        _processed_count = 0;
        outputsMemory = (uint8_t*)operator new(_inferenceEngine->output_size());
    };

    void runPostProcess(uint8_t* output_data, int64_t data_length)
    {
        std::unique_lock<std::mutex> _uniqueLock(_lock);
        _postProcessing.run(output_data, data_length);
        _processed_count += 1;
    };

    cv::Mat getResultFrame()
    {
        std::unique_lock<std::mutex> _uniqueLock(_getFrameLock);
        return _resultFrame;
    };

    dxapp::common::Point Position()
    {
        return _position;
    };

    dxapp::common::Size Resolution()
    {
        return _dstSize;
    };

    void makeThread()
    {
        _thread = std::thread(&DetectorApp::ppThreadFunction, this);
    };

    void notify_all()
    {
        _wait.store(false);
        std::cout << "[" << _name << "] : notify to this thread function. "<<std::endl;
        _cv.notify_all();
    };

    void joinThread()
    {
        while(!_thread.joinable())
        {
            usleep(10);
        }
        _thread.join();
    };

    void quitThread()
    {
        _quit.store(true);
    };

    bool quit()
    {
        return _quit.load();
    };

    void ppThreadFunction()
    {
        void* inf_data = nullptr;
        std::condition_variable _internal_cv;
        std::cout << "[" << _name << "] : entered post process thread function. "<<std::endl;
        {
            std::unique_lock<std::mutex> _uniqueLock(_lock);
            _cv.wait(_uniqueLock, [&](){return !_wait.load();});
        }
        cv::Mat outputImg = cv::Mat(_dstSize._height, _dstSize._width, CV_8UC3, cv::Scalar(0, 0, 0));

        while(!_quit.load())
        {    
            if(outputsMemory == nullptr)
            {
                usleep(100);
                continue;
            }
            _profiler.Start(_processName);    
            if(_frame_count - _processed_count < 10)// && !_quit.load())
            {
                inf_data = _vStream.GetInputStream();
                {
                    std::unique_lock<std::mutex> _uniqueLock(_lock);
                    _get_frame_result = _internal_cv.wait_for(_uniqueLock, std::chrono::seconds(3), [&](){ return inf_data!=nullptr; });
                    if(!_get_frame_result)
                    {
                        std::cout << "[DX-APP ERROR] The camera capture frame did not arrive properly." << std::endl;
                        std::cout << "[DX-APP ERROR] Shutting down the yolo_multi application." << std::endl;
                        _quit.store(true);
                        break;
                    }
                    else if(inf_data == nullptr)
                        continue;
                }
                _profiler.Start(_inferName);
                int req = _inferenceEngine->RunAsync(inf_data, (void*)this, (void*)outputsMemory);
                _frame_count += 1;

            }
            _profiler.End(_inferName);
            {
                std::unique_lock<std::mutex> _uniqueLock(_lock);
                dxapp::common::DetectObject results = _postProcessing.getResult();
                outputImg = _vStream.GetOutputStream(results);
                uint64_t fps = 1000000 / (_processTime + 0.1);
                std::string fpsCaption = "FPS : " + std::to_string((int)fps);
                cv::Size fpsCaptionSize = cv::getTextSize(fpsCaption, cv::FONT_HERSHEY_PLAIN, 3, 2, nullptr);
                cv::putText(outputImg, fpsCaption, cv::Point(outputImg.size().width - fpsCaptionSize.width, outputImg.size().height - fpsCaptionSize.height), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255),2);
                if(_appType == NONE)
                    std::cout << results <<std::endl;
            }
            {
                std::unique_lock<std::mutex> _uniqueLock(_getFrameLock);
               _resultFrame = outputImg;
            }
            _profiler.End(_processName);
            _processTime = _profiler.Get(_processName);
            _inferTime = _inferenceEngine->latency();
            usleep(1);
        };
        while(_frame_count != _processed_count)
        {
            usleep(1);
        }
        _vStream.Destructor();

    };

    void save()
    {
        std::string file_name = dxapp::common::getFileName(_videoPath);
        cv::imwrite("./result-" + _name + ".jpg", _resultFrame);
        std::cout << "save file : result-" << _name << ".jpg" <<std::endl;
    };

    uint8_t* get_outputMem(){
        return outputsMemory;
    };

    ~DetectorApp(){
        operator delete(outputsMemory);
    };

private:
    std::shared_ptr<dxrt::InferenceEngine> _inferenceEngine;
    dxrt::Profiler &_profiler;
    
    std::string _videoPath;
    std::string _name;
    std::string _outName;
    int _channel;
    int _numOfFrames;
    AppInputType _inputType;
    AppInputFormat _inputFormat;
    AppType _appType;
    dxapp::yolo::Params &_params;

    VideoStream _vStream;

    dxapp::common::Size _srcSize;
    dxapp::common::Size _inputSize;
    dxapp::common::Point _position;
    dxapp::common::Size _dstSize;

    cv::Mat _resultFrame;
    
    dxapp::yolo::PostProcessing _postProcessing;

    std::string _processName;
    std::string _inferName;
    uint64_t _inferTime = 0;
    uint64_t _processTime = 0;

    std::thread _thread;
    std::thread _ppThread;
    std::mutex _lock;
    std::mutex _getFrameLock;
    std::condition_variable _cv;
    std::atomic<bool> _wait = {true};
    std::atomic<bool> _quit = {false};

    unsigned long long _processed_count = 0;
    unsigned long long _frame_count = 0;

    bool _get_frame_result = true;
    
    uint8_t * outputsMemory = nullptr;

};

class Detector
{
public:
    Detector(const dxapp::AppConfig &_config):config(_config)
    {
        bool is_valid = dxapp::validationJsonSchema(config.modelInfo.c_str(), modelInfoSchema);
        if(!is_valid)
        {
            std::cout << config.modelInfo << std::endl;
            std::cout << "model params is invalid parsing" << std::endl;
            std::terminate();
        }

        readModelInfo(config.modelInfo.c_str());
        inferenceEngine = std::make_shared<dxrt::InferenceEngine>(modelPath);
        inputShape = inferenceEngine->inputs().front().shape();
        auto outputDataInfo = inferenceEngine->outputs();
        for(auto &info:outputDataInfo) 
        {
            outputShape.emplace_back(info.shape());
        }
        inputSize = inferenceEngine->input_size();
        if(inputShape[0]*inputShape[1]*inputShape[2]*inputShape[3] != inputSize)
            alignFactor = dxapp::common::get_align_factor(inputShape[1] * inputShape[3], 64);
        else
            alignFactor = false;

        params._classes = config.classes;
        params._numOfClasses = config.numOfClasses;
        params._input_shape = inputShape;
        params._outputShape = outputShape;

        size_t all_image_count = 0;
        for(auto const& source_info : config.sourcesInfo)
        {
            if(source_info.inputType == AppInputType::IMAGE)
                all_image_count++;
        }
        if(all_image_count == config.sourcesInfo.size())
            is_all_image = true;

        dxrt::DataType outputTensorType = inferenceEngine->outputs().front().type();
        switch (outputTensorType)
        {
        case dxrt::DataType::BBOX:
            params._ppu_format = dxapp::yolo::PPUFormat::BBOX;
            break;
        case dxrt::DataType::FACE:
            params._ppu_format = dxapp::yolo::PPUFormat::FACE;
            break;
        case dxrt::DataType::POSE:
            params._ppu_format = dxapp::yolo::PPUFormat::POSE;
            break;
        default:
            params._ppu_format = dxapp::yolo::PPUFormat::NONEPPU;
            break;
        }

        int div = dxapp::common::divideBoard(config.sourcesInfo.size());

        for(int i=0;i<(int)config.sourcesInfo.size();i++)
        {
            dxapp::common::Size dstSize = dxapp::common::Size(config.videoOutResolution._width/div, config.videoOutResolution._height/div);
            dxapp::common::Point dstPosition((config.videoOutResolution._width/div)*(i%div), (config.videoOutResolution._height/div)*(i/div));
            apps.emplace_back(std::make_shared<DetectorApp>(inferenceEngine, config.sourcesInfo[i], config.inputFormat, config.appType, 
                                    params, i, dstPosition, dstSize));
        }
        std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
        [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void* arg)
        {
            DetectorApp* app = (DetectorApp*)arg;
            int64_t dataLength = outputs.front()->shape().front();
            if(dxapp::common::compareVersions(DXRT_VERSION, "2.6.3"))
                dataLength = outputs.front()->shape()[1];
            app->runPostProcess(app->get_outputMem(), dataLength);
            return 0;
        };
        inferenceEngine->RegisterCallBack(postProcCallBack);
        
        resultView = cv::Mat(config.videoOutResolution._height, config.videoOutResolution._width, CV_8UC3, cv::Scalar(0, 0, 0));
    };

    ~Detector(void)
    {
        if(config.appType == OFFLINE)
        {
            if(saveThread.joinable())
                saveThread.join();
        }
    };

    void makeThread()
    {
        for(int i = 0;i<(int)apps.size();i++){
            apps[i]->makeThread();
        }
        if(config.appType == OFFLINE)
        {
            if(is_all_image)
            {
                std::cout << "[result save mode] ./xxx.jpg \n" 
                        << "Create Thread to save jpg " << std::endl;
                saveThread = std::thread(&Detector::saveImage, this);
            }
            else
            {
                std::cout << "[result save mode] ./result.avi \n" 
                        << "Create Thread to save avi " << std::endl;
                saveThread = std::thread(&Detector::saveResult, this);
            }
        }
    };

    void startThread()
    {
        for(int i=0;i<(int)apps.size();i++){
            apps[i]->notify_all();
        }
        if(config.appType == OFFLINE)
        {
            _wait.store(false);
            cv.notify_all();
        }
    };

    void joinThread()
    {
        for(int i=0;i<(int)apps.size();i++){
            apps[i]->joinThread();
        }
        if(config.appType == OFFLINE)
        {
            saveThread.join();
        }
    };

    void quitThread()
    {
        for(int i=0;i<(int)apps.size();i++){
            apps[i]->quitThread();
        }
    };

    bool status()
    {
        for(int i=0;i<(int)apps.size();i++){
            if(!apps[i]->quit())
            {
                return true;
            }
        }
        return false;
    };

    cv::Mat totalView()
    {
        for(int i=0; i<(int)apps.size();i++)
        {
            apps[i]->getResultFrame().copyTo(resultView(cv::Rect(apps[i]->Position()._x, apps[i]->Position()._y, apps[i]->Resolution()._width, apps[i]->Resolution()._height)));
        }
        return resultView;
    };

    void saveResult()
    {
        {
            std::unique_lock<std::mutex> _uniqueLock(lock);
            cv.wait(_uniqueLock, [&](){return !_wait.load();});
        }
        std::cout << "start Save Thread " << std::endl;
        std::cout << "Press 'q' key to quit " << std::endl;
        std::string saveFileName = "result.avi";
        cv::VideoWriter writer;
        writer.open(saveFileName, cv::VideoWriter::fourcc('M', 'J', 'P','G'), 30,  
                    cv::Size(config.videoOutResolution._width, config.videoOutResolution._height), true);
        if(!writer.isOpened())
        {
            std::cout << "Error : video writer for result.mp4 could not be opened. " << std::endl;
            return;
        }

        while(true)
        {
            writer << totalView();
            
            usleep(30000);
            if (!status()){
                std::cout << "quit save Thread "<< std::endl;
                break;
            }
        }
    };

    void saveImage()
    {
        {
            std::unique_lock<std::mutex> _uniqueLock(lock);
            cv.wait(_uniqueLock, [&](){return !_wait.load();});
        }
        while(true)
        {
            usleep(100000);
            for(int idx = 0; idx < static_cast<int>(apps.size()); idx++)
            {
                apps[idx]->save();
            }
            quitThread();
            break;
        }
    };

    void readModelInfo(const char* modelInfo)
    {
        rapidjson::Document doc;
        doc.Parse(modelInfo);
        modelPath = doc["path"].GetString();
        auto modelParam = doc["param"].GetObject();
        params._score_threshold = modelParam["score_threshold"].GetFloat();
        params._iou_threshold = modelParam["iou_threshold"].GetFloat();
        params._kpt_count = 0;
        
        std::string read = "";
        read = modelParam.HasMember("last_activation")?modelParam["last_activation"].GetString():"";
        if(read=="sigmoid")
            params._last_activation = dxapp::yolo::sigmoid;
        else if(read=="exp")
            params._last_activation = dxapp::yolo::exp;
        else
            params._last_activation = [](float x){return x;};
        
        read = modelParam["decoding_method"].GetString();
        if(read=="yolox")
            params._decode_method = dxapp::yolo::Decode::YOLOX;
        else if(read=="yolo_scale")
            params._decode_method = dxapp::yolo::Decode::YOLOSCALE;
        else if(read=="yolo_pose")
            params._decode_method = dxapp::yolo::Decode::YOLO_POSE;
        else if(read=="scrfd")
            params._decode_method = dxapp::yolo::Decode::SCRFD;
        else if(read=="yolov8")
            params._decode_method = dxapp::yolo::Decode::YOLOV8;
        else if(read=="yolov9")
            params._decode_method = dxapp::yolo::Decode::YOLOV9;
        else if(read=="custom_decode")
            params._decode_method = dxapp::yolo::Decode::CUSTOM_DECODE;
        else
            params._decode_method = dxapp::yolo::Decode::YOLO_BASIC;
        
        read = modelParam["box_format"].GetString();
        if(read=="corner")
            params._box_format = dxapp::yolo::BBoxFormat::XYX2Y2;
        else
            params._box_format = dxapp::yolo::BBoxFormat::CXCYWH;
        
        if(params._decode_method == dxapp::yolo::Decode::YOLO_POSE)
        {
            read = modelParam["kpt_order"].GetString();
            if(read=="end")
                params._kpt_order = dxapp::yolo::KeyPointOrder::BBOX_FRONT;
            else
                params._kpt_order = dxapp::yolo::KeyPointOrder::KPT_FRONT;
            params._kpt_count = modelParam["kpt_count"].GetInt();
        }
        else if(params._decode_method == dxapp::yolo::Decode::SCRFD)
        {
            params._kpt_count = modelParam["kpt_count"].GetInt();
        }

        for(auto &d:modelParam["layer"].GetArray()){
            auto o = d.GetObject();
            dxapp::yolo::Layers l = {};
            l.name = o["name"].GetString();
            l.stride = o["stride"].GetInt();
            l.scale = o.HasMember("scale")? o["scale"].GetFloat() : 0;
            for(auto &w:o["anchor_width"].GetArray()){
                l.anchor_width.emplace_back(w.GetFloat());
            }
            for(auto &h:o["anchor_height"].GetArray()){
                l.anchor_height.emplace_back(h.GetFloat());
            }
            params._layers.emplace_back(l);
        }
    };
    
    dxapp::AppConfig config;
    std::vector<int64_t> inputShape;
    std::vector<std::vector<int64_t>> outputShape;
    int inputSize;
    int alignFactor;
    bool is_all_image = false;
    
    std::shared_ptr<dxrt::InferenceEngine> inferenceEngine;
    dxapp::yolo::Params params;
    std::string modelPath;

    std::vector<std::shared_ptr<DetectorApp>> apps;

    cv::Mat resultView;
    std::thread saveThread;
    std::mutex lock;
    std::condition_variable cv;
    std::atomic<bool> _wait = {true};
private:
    const char* modelInfoSchema = R"""(
            {
                "type": "object",
                "properties":{
                    "path": {
                        "type": "string"
                    },
                    "param": {
                        "type": "object",
                        "properties": {
                            "score_threshold": {
                                "type": "number"
                            },
                            "iou_threshold": {
                                "type": "number"
                            },
                            "last_activation": {
                                "type": "string"
                            },
                            "decoding_method": {
                                "type": "string"
                            },
                            "box_format": {
                                "type": "string"
                            },
                            "kpt_order": {
                                "type": "string"
                            },
                            "kpt_count": {
                                "type": "number"
                            },
                            "layer": {
                                "type": "array",
                                "items":{
                                    "type": "object",
                                    "properties":{
                                        "name":{
                                            "type": "string"
                                        },
                                        "stride":{
                                            "type": "number"
                                        },
                                        "anchor_width":{
                                            "type": "array",
                                            "items": "number"
                                        },
                                        "anchor_height":{
                                            "type": "array",
                                            "items": "number"
                                        },
                                        "scale":{
                                            "type": "number"
                                        }
                                    },
                                    "required": [
                                        "name", "stride", "anchor_width", "anchor_height"
                                    ]
                                }
                            }
                        },
                        "required": [
                            "score_threshold", "iou_threshold", "decoding_method", "box_format", "layer"
                        ]
                    }
                },
                "required":[
                    "path", "param"
                ]
            }
        )""";

};
