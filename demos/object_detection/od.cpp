#include "display.h"
#include "od.h"
#include "yolo.h"

using namespace std;
using namespace cv;

extern void *PreProc(cv::Mat &src, cv::Mat &dest, bool keepRatio=true, bool bgr2rgb=true, uint8_t padValue=0);
extern YoloParam yoloParam;

ObjectDetection::ObjectDetection(dxrt::InferenceEngine &ie, std::pair<string, string> &videoSrc, int channel, 
        int width, int height, int destWidth, int destHeight,
        int posX, int posY, int numFrames)
: _ie(ie), _profiler(dxrt::Profiler::GetInstance()), _videoSrc(videoSrc), _channel(channel + 1),
    _width(width), _height(height), _destWidth(destWidth), _destHeight(destHeight), 
    _posX(posX), _posY(posY)
{
    _name = "app" + to_string(_channel);
    _offline = _videoSrc.second=="offline"?true:false;
    _image = _videoSrc.second=="image"?true:false;
    if(_videoSrc.second == "camera")
    {
        _cap.open(_videoSrc.first, CAP_V4L2);
        _cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
        _cap.set(CAP_PROP_FRAME_WIDTH, 1280); // camera capture resolution is hard-coded
        _cap.set(CAP_PROP_FRAME_HEIGHT, 720); // camera capture resolution is hard-coded
        _cap.set(CAP_PROP_FPS, 30);
        _srcWidth = _cap.get(CAP_PROP_FRAME_WIDTH);
        _srcHeight = _cap.get(CAP_PROP_FRAME_HEIGHT);
        if(_destWidth/_destHeight != _srcWidth/_srcHeight)
        {
            _adjustRatio = true;
        }
    }
    else if(_image)
    {
        cv::Mat tempImage = cv::imread(_videoSrc.first, cv::IMREAD_COLOR);
        _srcWidth = tempImage.size().width;
        _srcHeight = tempImage.size().height;
        if(_destWidth/_destHeight != _srcWidth/_srcHeight)
        {
            _adjustRatio = true;
        }
        _numFrames = 1;
        _offline = true;
    }
    else
    {
        _cap.open(_videoSrc.first);
        _cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
        _srcWidth = _cap.get(CAP_PROP_FRAME_WIDTH);
        _srcHeight = _cap.get(CAP_PROP_FRAME_HEIGHT);
        int totalFrames = _cap.get(CAP_PROP_FRAME_COUNT);
        if(_offline)
        {            
            if(numFrames>0)
            {
                _numFrames = min(totalFrames, numFrames);
            }
            else
            {
                _numFrames = totalFrames;
            }
        }
        if(_destWidth/_destHeight != _srcWidth/_srcHeight)
        {
            _adjustRatio = true;
        }
    }
    if(!_cap.isOpened() && !_image)
    {
        cout << "Error: Can't open " << _videoSrc.first << endl;
        DXRT_ASSERT(1==0, "capture error");
    }
    for(int i=0;i<_numFrames;i++)
    {
        _src.emplace_back(cv::Mat(_srcHeight, _srcWidth, CV_8UC3));
        _preprocessed.emplace_back(cv::Mat(_height, _width, CV_8UC3));
        _dest.emplace_back(cv::Mat(_destHeight, _destWidth, CV_8UC3));
    }
    if(_offline && !_image)
    {
        cout << "Preprocessing " << _videoSrc.first << endl;
        for(int i=0;i<_numFrames;i++)
        {
            _cap >> _src[i];
            PreProc(_src[i], _preprocessed[i], true, true, 114);
            cv::resize(_src[i], _dest[i], cv::Size(_destWidth, _destHeight), 0, 0, cv::INTER_LINEAR);
        }
    }
    else if(_image)
    {
        cout << "Preprocessing " << _videoSrc.first << endl;
        for(int i=0;i<_numFrames;i++)
        {
            _src[i] = cv::imread(_videoSrc.first, cv::IMREAD_COLOR);
            PreProc(_src[i], _preprocessed[i], true, true, 114);
            cv::resize(_src[i], _dest[i], cv::Size(_destWidth, _destHeight), 0, 0, cv::INTER_LINEAR);
        }
    }
    _colorTable = GetObjectColors(0);
    _resultFrame = cv::Mat(_destHeight, _destWidth, CV_8UC3);
    _expandedFrame = cv::Mat(_srcHeight, _srcWidth, CV_8UC3);
    yolo = Yolo(yoloParam);
}
ObjectDetection::ObjectDetection(dxrt::InferenceEngine &ie, int channel, int destWidth, int destHeight, int posX, int posY)
: _ie(ie), _profiler(dxrt::Profiler::GetInstance()), _channel(channel+1), _destWidth(destWidth), _destHeight(destHeight), _posX(posX), _posY(posY)
{
    _name = "app" + to_string(_channel);
    /***
     * Create blank
     * 1. load logo image
     * 2. resize logo image
    */
    _logo = cv::imread("./sample/dx_colored_logo.png", cv::IMREAD_COLOR);
    cv::resize(_logo, _resultFrame, cv::Size(_destWidth, _destHeight), 0, 0, cv::INTER_LINEAR);
    // cv::resize(_logo, _expandedFrame, cv::Size(_srcWidth, _srcHeight), 0, 0, cv::INTER_LINEAR);
    _expandedFrame = _logo.clone();
}
ObjectDetection::~ObjectDetection() {}
void ObjectDetection::threadFunc(int period)
{
    int idx = 0, prevIdx = 0;
    string cap = "cap" + to_string(_channel);
    char caption[100];
    float fps = 0.f; double infCount = 0.0;
    string infTime = "inf" + to_string(_channel);
    _profiler.Add(cap);
    _profiler.Add(infTime);
    _resultIdx = 0;
    while(1)
    {        
        if(stop) break;
        _profiler.Start(cap);
        prevIdx = idx;
        if(!_offline)
        {            
            _cap >> _src[idx];            
            if(_cap.get(CAP_PROP_FRAME_COUNT)>0)
            {
                if(_cap.get(CAP_PROP_POS_FRAMES)>=_cap.get(CAP_PROP_FRAME_COUNT))
                {
                    _cap.set(CAP_PROP_POS_FRAMES, 0);
                }
            }            
            PreProc(_src[idx], _preprocessed[idx], true, true, 114);
            cv::resize(_src[idx], _dest[idx], cv::Size(_destWidth, _destHeight), 0, 0, cv::INTER_LINEAR);
        }
        {
            int req = _ie.RunAsync(_preprocessed[idx].data, (void*)this);
#if 1
            _ie.Wait(req); /* optional */
#endif
            _inferTime = _ie.latency();
            // cout << _inferTime << endl;
        }

        vector<BoundingBox> bboxes;
        {
            unique_lock<mutex> lk(_lock);
            bboxes = vector<BoundingBox>(_bboxes);
        }
        {
            unique_lock<mutex> lk(_frameLock);
            _resultFrame = DrawBoundingBox(_dest[idx].clone(), _src[idx].size(), bboxes, _adjustRatio);
            if(_isExpanded){
                _expandedFrame = DrawBoundingBox(_src[idx].clone(), _src[idx].size(), bboxes, false);
            }
            

            // for record appearence 
            // cv::rectangle(_resultFrame, Point(0, 0), Point(130, 34), Scalar(0, 0, 0), cv::FILLED);
            // cv::putText(_resultFrame, caption, Point(40, 22), 0, 0.7, cv::Scalar(255, 255, 255), 2);
            // cv::circle(_resultFrame, Point(17, 17), 10, Scalar(0, 0, 255), -1);
#if 0
            fps += 1000000.0 / _inferTime;
            infCount++;
            float resultFps = round((fps/infCount) * 100) / 100;
            
            snprintf(caption, sizeof(caption), " / %.2f FPS", _channel, resultFps);
            cv::rectangle(_resultFrame, Point(0, 0), Point(230, 34), Scalar(0, 0, 0), cv::FILLED);
            cv::putText(_resultFrame, caption, Point(56, 21), 0, 0.7, cv::Scalar(255,255,255), 2, LINE_AA);
#else
            cv::rectangle(_resultFrame, Point(0, 0), Point(76, 34), Scalar(0, 0, 0), cv::FILLED); 
            if(_isExpanded){
                cv::rectangle(_expandedFrame, Point(0, 0), Point(76, 34), Scalar(0, 0, 0), cv::FILLED); 
            }
#endif
            cv::putText(_resultFrame, " # " + to_string(_channel), Point(0, 21), 7, 0.7, cv::Scalar(255, 255, 255), 2, LINE_AA);
            if(_isExpanded){
                cv::putText(_expandedFrame, " # " + to_string(_channel), Point(0, 21), 7, 0.7, cv::Scalar(255, 255, 255), 2, LINE_AA); 
            }

            if(_isPause){
                _cv.wait(lk);
            }
        }
            
        idx = (idx + 1) % _numFrames;
        _profiler.End(cap);
        _processTime = _profiler.Get(cap);
        int64_t t = (period*1000 - _processTime)/1000;
        if(t<0 || t>period) t = 0;
        usleep(t*1000);
    }
    _profiler.Erase(infTime);
    cout << _channel << " ended." << endl;
}
void ObjectDetection::threadFillBlank(int period)
{
    while(1)
    {        
        if(stop) break;
        
        usleep(period * 1000);
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
    std::unique_lock lk(_frameLock);
    if(!_isPause)
        _isPause = true;
}
void ObjectDetection::Play()
{
    std::unique_lock lk(_frameLock);
    if(_isPause){
        _isPause = false;
        _cv.notify_all();
    }
}
void ObjectDetection::Contract()
{
    if(_isExpanded)
        _isExpanded = false;
}
void ObjectDetection::Expand()
{
    if(!_isExpanded)
        _isExpanded = true;
}
cv::Mat &ObjectDetection::ResultFrame()
{
    unique_lock<mutex> lk(_frameLock);
    return _resultFrame;
}
cv::Mat &ObjectDetection::ExpandedFrame()
{
    unique_lock<mutex> lk(_frameLock);
    return _expandedFrame;
}
pair<int, int> ObjectDetection::Position()
{
    return make_pair(_posX, _posY);
}
pair<int, int> ObjectDetection::Resolution()
{
    return make_pair(_destWidth, _destHeight);
}
uint64_t ObjectDetection::GetInferenceTime()
{
    return _inferTime;
}
uint64_t ObjectDetection::GetProcessingTime()
{
    return _processTime;
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
cv::Mat ObjectDetection::DrawBoundingBox(cv::Mat src, cv::Size originalSize, vector<BoundingBox> &bboxes, bool adjustRatio)
{   
    cv::Mat dst;
    if(!bboxes.empty() && _toggleDrawing)
    {
        DisplayBoundingBox(src, bboxes, _height, _width, "", "",
                        cv::Scalar(0, 0, 255), _colorTable, "", 0, -1, true, (float)originalSize.width, (float)originalSize.height);
    }
    dst = src;

    return dst;
}
void ObjectDetection::PostProc(vector<shared_ptr<dxrt::Tensor>> &outputs)
{
    unique_lock<mutex> lk(_lock);
    _bboxes = yolo.PostProc(outputs);
}
ostream& operator<<(ostream& os, const ObjectDetection& od)
{
    os << od._name << ": " << od._channel << ", "
        << od._videoSrc.first << ", " << od._videoSrc.second << ", "
        << od._channel << ", " << od._targetFps << ", "
        << od._srcWidth << ", " << od._srcHeight << ", "
        << od._width << ", " << od._height << ", "
        << od._destWidth << ", " << od._destHeight << ", "
        << od._posX << ", " << od._posY << ", "
        << od._numFrames << ", " << od._numFrames << ", "
        << od._offline << ", " << od._cap.get(CAP_PROP_FPS);
    return os;
}