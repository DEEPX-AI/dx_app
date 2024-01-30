#pragma once
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <common/objects.hpp>


typedef enum{
    PRELOAD = 0,
    RUNTIME,
}SrcMode;

class VideoStream
{
public:
    SrcMode _srcMode;
    AppInputType _srcType;
    std::string _srcPath;
    dxapp::common::Size _srcSize;
    AppInputFormat _npuColorFormat;        
    dxapp::common::Size _npuSize;
    int _preLoadNum;
    int _cam_fps;
    int _cam_width;
    int _cam_height;
    int _tot_frame;
    int _cur_frame;
    
    cv::Mat _frame;
    cv::VideoCapture _video;
    std::vector<cv::Mat> _srcImg;
    std::vector<cv::Mat> _preImg;

    cv::Mat _srcImg_runtime;
    cv::Mat _preImg_runtime;    
    
//for GetInputStream
    int _preGetCnt = 0;
    int _fillPadvalue;
    float _preprocRatio;
    dxapp::common::Size _resizeSize;
    float _dw, _dh;
    int _top, _bottom, _left, _right;

//for GetOutputStream
    dxapp::common::Size _dstSize;
    int _outGetCnt = 0;


    VideoStream(AppInputType srcType, std::string srcPath, int preLoadNum, dxapp::common::Size npuSize, AppInputFormat npuColorFormat, dxapp::common::Size dstSize, int cam_fps = 30, int cam_width = 1280, int cam_height = 720)
            : _srcType(srcType), _srcPath(srcPath), _npuSize(npuSize), _npuColorFormat(npuColorFormat), _dstSize(dstSize), _cam_fps(cam_fps), _cam_width(cam_width), _cam_height(cam_height)
    {
    //Capture
        switch(_srcType)
        {
            case IMAGE :
                _srcMode = PRELOAD;
                _preLoadNum = 1;
                _frame = cv::imread(srcPath, cv::IMREAD_COLOR);
                _srcSize._width = _frame.cols;
                _srcSize._height = _frame.rows;                                
            break;

            case VIDEO :    

                if(preLoadNum != -1)
                {
                    _preLoadNum = preLoadNum;
                    _srcMode = PRELOAD;                    
                }
                else
                {
                    _srcMode = RUNTIME;
                }

                _video.open(srcPath);            
                if(!_video.isOpened())
                {
                    std::cout << "Error: file " << srcPath << " could not be opened." <<std::endl;
                    return;
                }               
                _srcSize._width = _video.get(cv::CAP_PROP_FRAME_WIDTH);
                _srcSize._height = _video.get(cv::CAP_PROP_FRAME_HEIGHT);
                _tot_frame = _video.get(cv::CAP_PROP_FRAME_COUNT);
            break;

            case CAMERA :
                _srcMode = RUNTIME;
                _video.open(srcPath, cv::CAP_V4L2);
                _video.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
                _video.set(cv::CAP_PROP_FRAME_WIDTH, _cam_width);
                _video.set(cv::CAP_PROP_FRAME_HEIGHT, _cam_height);
                _video.set(cv::CAP_PROP_FPS, _cam_fps);
                if(!_video.isOpened())
                {
                    std::cout << "Error: camera could not be opened." <<std::endl;
                    return;
                }
                _srcSize._width = _video.get(cv::CAP_PROP_FRAME_WIDTH);
                _srcSize._height = _video.get(cv::CAP_PROP_FRAME_HEIGHT);
                _cam_fps = _video.get(cv::CAP_PROP_FPS);
            break;

            case ISP :
                _srcMode = RUNTIME;
                std::cout << "Error: isp could not be opened." <<std::endl;
                return;
            break;
            
            case ETHERNET :
                _srcMode = RUNTIME;
            break;
        }

    //Preprocess 
        _preprocRatio = std::min((float)_npuSize._width/_srcSize._width, (float)_npuSize._height/_srcSize._height);
        _resizeSize = dxapp::common::Size((int)(_srcSize._width * _preprocRatio), (int)(_srcSize._height * _preprocRatio));
        _dw = (_npuSize._width - _resizeSize._width) / 2.f;
        _dh = (_npuSize._height - _resizeSize._height) / 2.f;

        _top = std::max((int)std::round(_dh - 0.1), 0);
        _bottom = std::max((int)std::round(_dh + 0.1), 0);
        _left = std::max((int)std::round(_dw - 0.1), 0);
        _right = std::max((int)std::round(_dw + 0.1), 0);
        if(_npuSize == _srcSize)
        {
            _fillPadvalue = -1;
        }
        else 
        {
            _fillPadvalue = 114;
        }

    //PreloadNum
        if(_srcMode == PRELOAD) 
        {
            for(int i = 0; i < _preLoadNum; i++)
            {
                Preprocess();
            }
        }        

    };
    VideoStream(){};
    ~VideoStream(){};

    //Capture
    cv::Mat ImgCapture(void)
    {        
        switch(_srcType)
        {
            case IMAGE :                       
                _frame = cv::imread(_srcPath, cv::IMREAD_COLOR);
            break;
            
            case VIDEO:
                _video >> _frame;                                                    
                _cur_frame = _video.get(cv::CAP_PROP_POS_FRAMES);

                if( (_cur_frame > 0) && (_cur_frame >= _tot_frame) )
                {
                    _video.set(cv::CAP_PROP_POS_FRAMES, 0);                
                }            
            break;

            case CAMERA :
                _video >> _frame;                                                    
            break;

            case ISP :

            break;

            default :
            break;
        }
        return _frame.clone();
    };

    //GetInputStream  
    void ImgPreResize(cv::Mat& src, cv::Mat& dst)
    {
        if(_fillPadvalue < 0)
        { // resize src image
            cv::resize(src, dst, cv::Size(_npuSize._width, _npuSize._height), 0, 0, cv::INTER_LINEAR);
        }
        else
        { // make letterbox 
            cv::Mat resized = cv::Mat(_resizeSize._height, _resizeSize._width, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::resize(src, resized, cv::Size(_resizeSize._width, _resizeSize._height), 0, 0, cv::INTER_LINEAR);
            cv::copyMakeBorder(resized, dst, _top, _bottom, _left, _right, cv::BORDER_CONSTANT, cv::Scalar(_fillPadvalue,_fillPadvalue,_fillPadvalue));
        }
    };    

    void ImgCvtColor(cv::Mat& img, AppInputFormat& inputFormat)
    {
        switch(inputFormat)
        {
            case IMAGE_RGB :
                cv::cvtColor(img.clone(), img, cv::COLOR_BGR2RGB);
            break;
            case IMAGE_GRAY :
                cv::cvtColor(img.clone(), img, cv::COLOR_BGR2GRAY);
            break;
            case IMAGE_BGR :
                cv::cvtColor(img.clone(), img, cv::COLOR_RGB2BGR);
            break;
        }
    };    

    void Preprocess(void)
    {   
        cv::Mat srcImg;
        cv::Mat preImg;

        srcImg = ImgCapture();       
        ImgPreResize(srcImg, preImg);              
        ImgCvtColor(preImg, _npuColorFormat);

        if(_srcMode == RUNTIME)
        {
            _srcImg_runtime = srcImg;
            _preImg_runtime = preImg;   
        }
        else
        {
            _srcImg.emplace_back(srcImg);
            _preImg.emplace_back(preImg);
        }
    };

    unsigned char *GetInputStream(void)
    {
        cv::Mat img;

        if(_srcMode == RUNTIME)
        {
            Preprocess();
            img = _preImg_runtime;
        }         
        else
        {
            img = _preImg[_preGetCnt];
            _preGetCnt++;            
            if(_preGetCnt>=_preLoadNum)
            {
                _preGetCnt = 0;
            }
        }

        return img.data;        
    };

    //GetOutputStream
    void ImgPostResize(cv::Mat& src, cv::Mat& dst)
    {
        if(_srcSize == _dstSize)
        {
            dst = src;
        }
        else
        {
            cv::resize(src, dst, cv::Size(_dstSize._width, _dstSize._height), 0, 0, cv::INTER_LINEAR);
        }
    };

    //Display
    void DrawBox(cv::Mat& dst, dxapp::common::Object& obj)
    {
        cv::rectangle(dst, cv::Rect(obj._bbox._xmin, obj._bbox._ymin, obj._bbox._width, obj._bbox._height), dxapp::common::color_table[obj._classId], 2);
    };

    void DrawCaption(cv::Mat dst, dxapp::common::Object& obj)
    {
        int textBaseLine = 0;
        auto textSize = cv::getTextSize(obj._name, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &textBaseLine);
        cv::rectangle(dst, cv::Point(obj._bbox._xmin, obj._bbox._ymin - textSize.height),
                            cv::Point(obj._bbox._xmin + textSize.width, obj._bbox._ymin),
                            dxapp::common::color_table[obj._classId], cv::FILLED);
        cv::putText(dst, obj._name, cv::Point(obj._bbox._xmin, obj._bbox._ymin), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255));       
    };

    cv::Mat GetOutputStream(dxapp::common::DetectObject& obj)
    {
        cv::Mat dstImg;
        if(_srcMode == RUNTIME)        
        {
            ImgPostResize(_srcImg_runtime, dstImg);        
        }
        else
        {
            ImgPostResize(_srcImg[_outGetCnt], dstImg);                    
            _outGetCnt++;            
            if(_outGetCnt>=_preLoadNum)
            {
                _outGetCnt = 0;
            }
        }

        for(size_t i=0;i<obj._detections.size();i++)
        {
            DrawBox(dstImg, obj._detections[i]);
            DrawCaption(dstImg, obj._detections[i]);                    
        } 

        return dstImg;

    };

};
