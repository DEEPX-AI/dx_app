#include "od.h"
#include "yolo.h"
#include <utils/common_util.hpp>

using namespace std;
using namespace cv;

extern YoloParam yoloParam;

ObjectDetection::ObjectDetection(std::shared_ptr<dxrt::InferenceEngine> ie, std::pair<string, string> &videoSrc, int channel, 
        int width, int height, int destWidth, int destHeight,
        int posX, int posY, int numFrames)
: _ie(ie), _profiler(dxrt::Profiler::GetInstance()), _channel(channel + 1),
    _width(width), _height(height), _destWidth(destWidth), _destHeight(destHeight), 
    _posX(posX), _posY(posY), _videoSrc(videoSrc)
{
    AppInputType inputType = AppInputType::VIDEO;
    if(_videoSrc.second == "camera")
        inputType = AppInputType::CAMERA;
    else if(_videoSrc.second == "image")
        inputType = AppInputType::IMAGE;
    else if(_videoSrc.second == "rtsp")
        inputType = AppInputType::RTSP;
#if __riscv
    else if(_videoSrc.second == "isp")
        inputType = AppInputType::ISP;
#endif
    else
        inputType = AppInputType::VIDEO;
    auto inputShape = _ie->inputs().front().shape();
    auto npuShape = dxapp::common::Size((int)inputShape[1],(int)inputShape[1]);
    auto dstShape = dxapp::common::Size(_destWidth, _destHeight);

    _vStream = VideoStream(inputType, _videoSrc.first, numFrames, npuShape, AppInputFormat::IMAGE_BGR, dstShape, _ie);
    auto srcShape = _vStream._srcSize;
    _srcWidth = srcShape._width;
    _srcHeight = srcShape._height;
    _name = "app" + to_string(_channel);
    dxapp::common::Size_f _postprocRatio;
    _postprocRatio._width = (float)dstShape._width/srcShape._width;
    _postprocRatio._height = (float)dstShape._height/srcShape._height;

    float _preprocRatio = std::min((float)npuShape._width/srcShape._width, (float)npuShape._height/srcShape._height);
    
    if(srcShape == npuShape)
    {
        _postprocPaddedSize._width = 0.f;
        _postprocPaddedSize._height = 0.f;
    }
    else
    {
        dxapp::common::Size resizeShpae((int)(srcShape._width * _preprocRatio), (int)(srcShape._height * _preprocRatio));
        _postprocPaddedSize._width = (npuShape._width - resizeShpae._width) / 2.f;
        _postprocPaddedSize._height = (npuShape._height - resizeShpae._height) / 2.f;
    }

    _postprocScaleRatio = dxapp::common::Size_f(_postprocRatio._width/_preprocRatio, _postprocRatio._height/_preprocRatio);
    
    _resultFrame = cv::Mat(_destHeight, _destWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    yolo = Yolo(yoloParam);
    yolo.LayerReorder(ie->outputs());

    outputMemory = (uint8_t*)operator new(_ie->output_size());
    output_length = 0;
    for(auto &o:_ie->outputs())
    {
        output_shape.emplace_back(o.shape());
    }
    data_type = _ie->outputs().front().type();

    _fps_time_s = std::chrono::high_resolution_clock::now();
    _fps_time_e = std::chrono::high_resolution_clock::now();
}
ObjectDetection::ObjectDetection(std::shared_ptr<dxrt::InferenceEngine> ie, int channel, int destWidth, int destHeight, int posX, int posY)
: _ie(ie), _profiler(dxrt::Profiler::GetInstance()), _channel(channel+1), _destWidth(destWidth), _destHeight(destHeight), _posX(posX), _posY(posY)
{
    _name = "app" + to_string(_channel);
    if(dxapp::common::pathValidation("./sample/dx_colored_logo.png"))
    {
        _logo = cv::imread("./sample/dx_colored_logo.png", cv::IMREAD_COLOR);
        cv::resize(_logo, _resultFrame, cv::Size(_destWidth, _destHeight), 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        _resultFrame = cv::Mat(_destHeight, _destWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    outputMemory = nullptr;
}
ObjectDetection::~ObjectDetection() {}
dxapp::common::DetectObject ObjectDetection::GetScalingBBox(vector<BoundingBox>& bboxes)
{
    dxapp::common::DetectObject result;
    result._num_of_detections = bboxes.size();
    for (auto& b : bboxes)
    {
        dxapp::common::BBox box;
        box._xmin = (b.box[0] - _postprocPaddedSize._width) * _postprocScaleRatio._width;
        box._ymin = (b.box[1] - _postprocPaddedSize._height) * _postprocScaleRatio._height;
        box._xmax = (b.box[2] - _postprocPaddedSize._width) * _postprocScaleRatio._width;
        box._ymax = (b.box[3] - _postprocPaddedSize._height) * _postprocScaleRatio._height;
        box._width = (b.box[2] - b.box[0]) * _postprocScaleRatio._width;
        box._height = (b.box[3] - b.box[1]) * _postprocScaleRatio._height;
        box._kpts.emplace_back(dxapp::common::Point_f(-1 , -1, -1));
    
        dxapp::common::Object object;
        object._bbox = box;
        object._conf = b.score;
        object._classId = b.label;
        object._name = b.labelname;
        result._detections.emplace_back(object);
    }
    return result;
}
void ObjectDetection::threadFunc(int period)
{
    string cap = "cap" + to_string(_channel);
    string proc = "proc" + to_string(_channel);
#if 0
    char caption[100] = {0,};
    float fps = 0.f; double infCount = 0.0;
#endif
    _profiler.Add(cap);
    _profiler.Add(proc);
    cv::Mat member_temp;
    while(1)
    {        
        if(stop) break;
        _profiler.Start(proc);
        _profiler.Start(cap);
        auto input = _vStream.GetInputStream();
        _fps_time_s = std::chrono::high_resolution_clock::now();
        int req = _ie->RunAsync(input, (void*)this, (void*)outputMemory);
#if 0
        _ie->Wait(req); /* optional */
#endif
        auto s = std::chrono::high_resolution_clock::now();
        vector<BoundingBox> bboxes;
        dxapp::common::DetectObject bboxes_objects;
        {
            unique_lock<mutex> lk(_lock);
            if(!_bboxes.empty() && _toggleDrawing)
            {
                bboxes = vector<BoundingBox>(_bboxes);
                bboxes_objects = GetScalingBBox(bboxes);
            }
        }
        member_temp = _vStream.GetOutputStream(bboxes_objects);
            
#if 0
        fps += 1000000.0 / _inferTime;
        infCount++;
        float resultFps = round((fps/infCount) * 100) / 100;
        
        snprintf(caption, sizeof(caption), " / %.2f FPS", _channel, resultFps);
        cv::rectangle(member_temp, Point(0, 0), Point(230, 34), Scalar(0, 0, 0), cv::FILLED);
        cv::putText(member_temp, caption, Point(56, 21), 0, 0.7, cv::Scalar(255,255,255), 2, LINE_AA);
#else
        cv::rectangle(member_temp, Point(0, 0), Point(40, 20), Scalar(0, 0, 0), cv::FILLED);
#endif
        cv::putText(member_temp, to_string(_channel), Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, LINE_AA);

        
        _inferenceTime = _ie->inference_time();
        _latencyTime = _ie->latency();
        
        _profiler.End(cap);
        int64_t t = (period*1000 - _profiler.Get(cap))/1000;
        if(t<0 || t>period) t = 0;
#ifdef __linux__
        usleep(t*1000);
#elif _WIN32
        Sleep(t);
#endif
        auto e = std::chrono::high_resolution_clock::now();
        
        if(_processed_count > 0)
        {
            unique_lock<mutex> lk(_frameLock);
            member_temp.copyTo(_resultFrame);
            _processAverageTime = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count();
            // uint64_t new_average = ((_processAverageTime * _processed_count) + _processTime)/(_processed_count + 1);
            // _processAverageTime = new_average;
            if(_isPause){
                _cv.wait(lk);
            }
        }

        _profiler.End(proc);
    }
    _profiler.Erase(cap);
    _profiler.Erase(proc);
    cout << _channel << " ended." << endl;
}
void ObjectDetection::threadFillBlank(int period)
{
    while(1)
    {        
        if(stop) break;
#ifdef __linux__
        usleep(period * 1000);
#elif _WIN32
        Sleep(period);
#endif
    }
    cout << _channel << " ended." << endl;
}
void ObjectDetection::Run(int period)
{
    stop = false;
    if(_videoSrc.first.empty())
        _thread = thread(&ObjectDetection::threadFillBlank, this, period);
    else
        _thread = thread(&ObjectDetection::threadFunc, this, period);
}
void ObjectDetection::Stop()
{
    stop = true;
    _thread.join();
}
void ObjectDetection::Pause()
{
    unique_lock<mutex> lk(_frameLock);
    if(!_isPause)
        _isPause = true;
}
void ObjectDetection::Play()
{
    unique_lock<mutex> lk(_frameLock);
    if(_isPause){
        _isPause = false;
        _cv.notify_all();
    }
}
cv::Mat ObjectDetection::ResultFrame()
{
    unique_lock<mutex> lk(_frameLock);
    cv::Mat out = _resultFrame.clone();
    return out;
}
pair<int, int> ObjectDetection::Position()
{
    return make_pair(_posX, _posY);
}
pair<int, int> ObjectDetection::Resolution()
{
    return make_pair(_destWidth, _destHeight);
}
uint64_t ObjectDetection::GetLatencyTime()
{
    return _latencyTime;
}
uint64_t ObjectDetection::GetInferenceTime()
{
    return _inferenceTime;
}
uint64_t ObjectDetection::GetProcessingTime()
{
    return _processAverageTime;
}
int ObjectDetection::Channel()
{
    return _channel;
}
string &ObjectDetection::Name()
{
    return _name;
}
void ObjectDetection::Toggle()
{
    _toggleDrawing = !_toggleDrawing;
}
void ObjectDetection::PostProc(vector<shared_ptr<dxrt::Tensor>> &outputs)
{
    unique_lock<mutex> lk(_lock);
    _bboxes = yolo.PostProc(outputs);
}
void ObjectDetection::PostProc(void* outputs, int output_length)
{
    unique_lock<mutex> lk(_lock);
    _bboxes = yolo.PostProc(outputs, output_shape, data_type, output_length);
    _fps_time_e = std::chrono::high_resolution_clock::now();
    _processed_count++;
    _duration_time = std::chrono::duration_cast<std::chrono::microseconds>(_fps_time_e - _fps_time_s).count();
}
ostream& operator<<(ostream& os, const ObjectDetection& od)
{
    os << od._name << ": " << od._channel << ", "
        << od._videoSrc.first << ", " << od._videoSrc.second << ", "
        << od._channel << ", " << od._targetFps << ", "
        << od._width << ", " << od._height << ", "
        << od._destWidth << ", " << od._destHeight << ", "
        << od._posX << ", " << od._posY << ", "
        << od._offline << ", " << od._cap.get(CAP_PROP_FPS);
    return os;
}
