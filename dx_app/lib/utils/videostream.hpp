#pragma once
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <common/objects.hpp>
#include "dxrt/dxrt_api.h"
#include "color_table.hpp"

#if __riscv
#include "v1/isp_eyenix.h"
#endif


typedef enum{
    PRELOAD = 0,
    RUNTIME,
}SrcMode;

class VideoStream
{
public:
    AppInputType _srcType;
    std::string _srcPath;
    int _preLoadNum;
    dxapp::common::Size _npuSize;
    AppInputFormat _npuColorFormat;        
    dxapp::common::Size _dstSize;
    std::shared_ptr<dxrt::InferenceEngine> _inferenceEngine;
    int _cam_fps;
    int _cam_width;
    int _cam_height;

    dxapp::common::Size _srcSize;
    int _tot_frame;
    int _cur_frame;
    
    SrcMode _srcMode;
    cv::Mat _frame;
    cv::VideoCapture _video;
    std::vector<cv::Mat> _srcImg;
    std::vector<cv::Mat> _preImg;

    cv::Mat _srcImg_runtime;
    cv::Mat _preImg_runtime;    

    
#if __riscv       
    UYV_PARAMS_S gUyvParams;
    
    vector<uint64_t> inputAddr;
    uint32_t inputSize;    
    vector<dxrt::Tensors> vinputs;
    shared_ptr<dxrt::Device> device;
    
    UYV_DATA_S *pUyvData;

    int cdma_cnt;
    int bufIdx;    
#endif

//for GetInputStream
    int _preGetCnt = 0;
    int _fillPadvalue;
    float _preprocRatio;
    dxapp::common::Size _resizeSize;
    float _dw, _dh;
    int _top, _bottom, _left, _right;

//for GetOutputStream
    int _outGetCnt = 0;

    VideoStream(AppInputType srcType, std::string srcPath, int preLoadNum, dxapp::common::Size npuSize, AppInputFormat npuColorFormat, dxapp::common::Size dstSize, std::shared_ptr<dxrt::InferenceEngine> inferenceEngine, int cam_fps = 30, int cam_width = 1280, int cam_height = 720)
            : _srcType(srcType), _srcPath(srcPath), _preLoadNum(preLoadNum), _npuSize(npuSize), _npuColorFormat(npuColorFormat), _dstSize(dstSize), _inferenceEngine(inferenceEngine), _cam_fps(cam_fps), _cam_width(cam_width), _cam_height(cam_height)
    {
    //Capture
        switch(_srcType)
        {
            case IMAGE :
                _srcMode = PRELOAD;
                _preLoadNum = 1;
                _frame = cv::imread(_srcPath, cv::IMREAD_COLOR);
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

                _video.open(_srcPath);            
                if(!_video.isOpened())
                {
                    std::cout << "Error: file " << _srcPath << " could not be opened." <<std::endl;
                }               
                _srcSize._width = _video.get(cv::CAP_PROP_FRAME_WIDTH);
                _srcSize._height = _video.get(cv::CAP_PROP_FRAME_HEIGHT);
                _tot_frame = _video.get(cv::CAP_PROP_FRAME_COUNT);
            break;

            case CAMERA :
                _srcMode = RUNTIME;
                _video.open(_srcPath, cv::CAP_V4L2);
                _video.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
                _video.set(cv::CAP_PROP_FRAME_WIDTH, _cam_width);
                _video.set(cv::CAP_PROP_FRAME_HEIGHT, _cam_height);
                _video.set(cv::CAP_PROP_FPS, _cam_fps);
                if(!_video.isOpened())
                {
                    std::cout << "Error: camera could not be opened." <<std::endl;
                }
                _srcSize._width = _video.get(cv::CAP_PROP_FRAME_WIDTH);
                _srcSize._height = _video.get(cv::CAP_PROP_FRAME_HEIGHT);
                _cam_fps = _video.get(cv::CAP_PROP_FPS);
            break;

            case ISP :
                _srcMode = RUNTIME;

    #if __riscv
                _srcSize._width = _npuSize._width;
                _srcSize._height = _npuSize._height;

                device = dxrt::CheckDevices().front();
                DXRT_ASSERT(device!=nullptr, "no device");
                           
                // Eyenix ISP Setup Part
                if(ENX_VSYS_Init() != 0){
                    printf("ENX_VSYS_Init failed\n");
                    exit(0);
                }

                if(ENX_UYV_CAPTURE_Init(&gUyvParams) != 0){
                    printf("ENX_UYV_CAPTURE_Init failed\n");
                    exit(0);
                }

                if(ENX_UYV_CAPTURE_CH_Start(0, &gUyvParams.UyvChnParams[0]) != 0) {
                    printf("ENX_UYV_CAPTURE_CH_Start failed\n");
                    exit(0);
                }

                SetUYVResolution(_npuSize._width,_npuSize._height);

                InitOSD();

                pUyvData = (UYV_DATA_S *)malloc(sizeof(UYV_DATA_S));
                memset(pUyvData,0,sizeof(UYV_DATA_S));

                vinputs = _inferenceEngine->inputs(0);
                for(auto &inputs:_inferenceEngine->inputs(0))
                {                
                    inputAddr.push_back(inputs.front().phy_addr());                
                }

                inputSize = _inferenceEngine->input_size();

                cdma_cnt = 0;
                _cur_frame = 0;
                bufIdx = 0;
    #else
                std::cout << "Error: isp could not be opened." <<std::endl;
    #endif                
            break;

            case ETHERNET :
                _srcMode = RUNTIME;
            break;

            default :
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
    ~VideoStream(){
    };

    void Destructor(void)
    {
#if __riscv
        if(_srcType == ISP)
        {
            if(pUyvData != NULL) {
                free(pUyvData);
                pUyvData = NULL;
            }

            if(ENX_UYV_CAPTURE_CH_Stop(0) != 0){
                printf("ENX_UYV_CAPTURE_CH_Stop failed\n");
            }

            ENX_UYV_CAPTURE_Exit();
            ENX_VSYS_Exit();

            DeinitOSD();

        }
#endif  
    };

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

            case ETHERNET :
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
            default :
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
    
    void* GetInputStream(void)
    {
        cv::Mat img;

        if(_srcType == ISP)
        {
#if __riscv                              
            Capture_Thread();

            bufIdx = _cur_frame%2;
            uint32_t npu_cdma_data[3] = {
                inputAddr[bufIdx], // dest
                pUyvData->PhysAddr, // src
                inputSize // size
            };
            if (0 != device->Process(dxrt::DXRT_CMD_SOC_CUSTOM, npu_cdma_data)){
                cdma_cnt++;
                if (cdma_cnt>10){
                    cout << "Don't use DMA COPY method" << endl;
                    exit(1);
                }
            }

            _cur_frame++;            
            return vinputs[bufIdx].front().data();
#endif            
        }
        else
        {
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
        }
        return (void *)img.data;        
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

    void DrawCaption(cv::Mat& dst, dxapp::common::Object& obj)
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
        if(_srcType == ISP)
        {
#if __riscv                   
            ClearFontBuf();
            ClearBoxBuf();
            for(size_t i=0;i<obj._detections.size();i++)
            {
                DrawBox((int)i, obj._detections[i]);
                DrawCaption((int)i, obj._detections[i]);
            }
            UpdateOSD();  
#endif
        }
        else
        {
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
        }

        return dstImg;
    };

#if __riscv        
    void rgb2ycbcr(cv::Scalar color, int &y, int &cb, int &cr)
    {
        int r, g, b;
        double temp_y, temp_cb, temp_cr;

        r = color[2];
        g = color[1];
        b = color[0];

        temp_y = (0.257*(double)r) + (0.504*(double)g) + (0.098 * (double)b) + 16;    
        temp_cb = -(0.148*(double)r) - (0.291*(double)g) + (0.439 * (double)b) + 128;
        temp_cr = (0.439*(double)r) - (0.368*(double)g) - (0.071 * (double)b) + 128;    

        y = (int)temp_y;
        cb = (int)temp_cb;
        cr = (int)temp_cr;
    }

    void Capture_Thread(void)
    {
        Int32 ret = 0;
        struct timeval timeout;
        fd_set readFds;
        Int32 uyvFds[MAX_NPU_CH];
        Int32 maxFd = 0;
        Int32 nCh = 0;

        FD_ZERO(&readFds);
        
        // printf("%s start ch=%d\n", __func__, nCh);

        while(1)
        {
            FD_ZERO(&readFds);

            while((uyvFds[nCh] = ENX_UYV_CAPTURE_CH_GetFd(nCh)) < 0) {
                printf("wait stream from video .. \n");
                usleep(100);
                continue;
            }

            if (maxFd < uyvFds[nCh]) {
                maxFd = uyvFds[nCh];
            }

            FD_SET(uyvFds[nCh], &readFds);

            timeout.tv_sec  = 0;
            timeout.tv_usec = 100 * 1000; //100ms

            ret = select(maxFd+1, &readFds, NULL, NULL, &timeout);

            if (ret < 0) { 
                printf("Sample Channel select err %d \n", ret);
                break;
            } else if (0 == ret) { 
                printf("Sample Channel select timeout %d \n", ret);
                continue;
            } else { 

                if(uyvFds[nCh] < 0 || uyvFds[nCh] > maxFd) {
                    printf("Sample Channel fd set failed uyvFds %d \n", uyvFds[nCh]);
                    break;
                }

                if (FD_ISSET(uyvFds[nCh], &readFds)) 
                { 
                    ret = ENX_UYV_CAPTURE_CH_GetFrame(nCh, pUyvData);	// Lock
                    if(ret != 0) {
                        printf("ENX_VENC_CH_GetFrame failed\n");
                        continue;
                    }
                    break;
                } // FD_ISSET
            } // select ret
        sleep(0);
        } // while
    }    

    void DrawBox(int box_id, dxapp::common::Object& obj)
    {
        int y, cb, cr;

        if( (obj._bbox._xmin>0) && (obj._bbox._ymin>0) && (box_id < MAX_BOX) )
        {
            BOXBUF[box_id].BoxOn = true;
            BOXBUF[box_id].x_min = obj._bbox._xmin; //xmin
            BOXBUF[box_id].y_min = obj._bbox._ymin; //ymin
            BOXBUF[box_id].x_max = obj._bbox._xmax; //xmax
            BOXBUF[box_id].y_max = obj._bbox._ymax; //ymax

            BOXATTRBUF[box_id].Mode    = 0;        // Tone
            BOXATTRBUF[box_id].Fill    = 0;    // not filled
            BOXATTRBUF[box_id].modVal  = 0;        // tone 100%

            rgb2ycbcr(dxapp::common::color_table[obj._classId], y, cb, cr);
            
            BOXATTRBUF[box_id].ColorY  = y;
            BOXATTRBUF[box_id].ColorCb = cb;
            BOXATTRBUF[box_id].ColorCr = cr;
        }    

    }

    void DrawCaption(int box_id, dxapp::common::Object& obj)
    {

        if( (obj._bbox._xmin>0) && (obj._bbox._ymin>0) && (box_id < MAX_BOX) )
        {
            char cstr[25];
            strcpy(cstr, obj._name.c_str());
            int slength = strlen(cstr);

            int strXp = (double)(obj._bbox._xmin)*MAX_CHAR/SCREEN_WIDTH;
            int strYp = (double)(obj._bbox._ymin)*MAX_LINE/SCREEN_HEIGHT;
            
            if( ( (strXp + slength) < MAX_CHAR) && (strYp < MAX_LINE) )
            {       
                for(int j=0;j<slength;++j)
                {

                    FONTBUF[strYp][strXp + j].enChar  = true;       // Change char
                    FONTBUF[strYp][strXp + j].enAttr  = true;       // Change color
                    FONTBUF[strYp][strXp + j].enAlpha = true;       // Change alpha
                    FONTBUF[strYp][strXp + j].Alpha   = 0;          // Not Used          
                    FONTBUF[strYp][strXp + j].Attr    = 0;          // Not Used
                    FONTBUF[strYp][strXp + j].Char    = cstr[j];    // space
                }
            }
        }
    }
#endif

};
