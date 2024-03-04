#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <getopt.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syslog.h>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include "display.h"
#endif
#include "dxrt/dxrt_api.h"
#include "yolo.h"
#include "v4l2.h"
#include "socket.h"
#include "fb.h"

using namespace std;
#ifdef USE_OPENCV
using namespace cv;
#endif

#define ISP_INPUT_DMA_COPY
#define ISP_PHY_ADDR   (0xA0000000)
#define ISP_INPUT_ZEROCOPY
#define POSTPROC_SIMULATION_DEVICE_OUTPUT
#define DISPLAY_WINDOW_NAME "Object Detection"
#define INPUT_CAPTURE_PERIOD_MS 30

#define CAMERA_FRAME_WIDTH 960
#define CAMERA_FRAME_HEIGHT 540
#define FRAME_BUFFERS 5
#define DMA_IOCTL_DATA		_IOW('T', 3, struct npu_dma_ioctl_data_copy)

// pre/post parameter table
extern YoloParam yolov5s_320, yolov5s_512, yolov5s_640, yolov5s_512_concat, yolox_s_512, yolov7_640, yolov7_512, yolov4_608;
extern YoloParam yolov5s_640_ppu;
YoloParam yoloParams[] = {
    [0] = yolov5s_320,
    [1] = yolov5s_512,
    [2] = yolov5s_640,
    [3] = yolov5s_512_concat,
    [4] = yolox_s_512,
    [5] = yolov7_640,
    [6] = yolov7_512,
    [7] = yolov4_608,
    [8] = yolov5s_640_ppu,
};

namespace dxrt {
    extern uint64_t MemIfPhyBase;
}

struct npu_dma_ioctl_data_copy
{
    uint32_t input_addr;
    uint32_t uyv_addr;
    uint32_t input_size;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
static struct option const opts[] = {
    { "model", required_argument, 0, 'm' },
    { "dncnn", required_argument, 0, 'd' },
    { "image", required_argument, 0, 'i' },
    { "video", required_argument, 0, 'v' },
    { "write", required_argument, 0, 'w' },
    { "camera", no_argument, 0, 'c' },
    { "isp", no_argument, 0, 'x' },
    { "bin",  required_argument, 0, 'b' },
    { "sim", required_argument, 0, 's' },
    { "async", no_argument, 0, 'a' },
    { "ethernet", no_argument, 0, 'e' },
    { "param", required_argument, 0, 'p' },
    { "loop", no_argument, 0, 'l' },    
    { "help", no_argument, 0, 'h' },
    { 0, 0, 0, 0 }
};
const char* usage =
"yolo demo\n"
"  -m, --model     define yolo model path\n"
"  -d, --dncnn     define dncnn model path\n"
"  -i, --image     use image file input\n"
"  -v, --video     use video file input\n"
"  -w, --write     write result frames to a video file\n"
"  -c, --camera    use camera input\n"
"  -x, --isp       use ISP input\n"
"  -b, --bin       use binary file input\n"
"  -s, --sim       use pre-defined npu output binary file input( perform post-proc. only )\n"
"  -a, --async     asynchronous inference\n"
"  -e, --ethernet  use ethernet input\n"
"  -p, --param      pre/post-processing parameter selection\n"
"  -l, --loop      loop test\n"
"  -h, --help      show help\n"
;
void help()
{
    cout << usage << endl;    
}

#ifdef USE_OPENCV
void *PreProc(cv::Mat &src, cv::Mat &dest, bool keepRatio=true, bool bgr2rgb=true, uint8_t padValue=0)
{
    cv::Mat Resized;
    if(keepRatio)
    {
        float dw, dh;
        uint16_t top, bottom, left, right;
        float ratioDest = (float)dest.cols/dest.rows;
        float ratioSrc = (float)src.cols/src.rows;
        int newWidth, newHeight;
        if(ratioSrc < ratioDest)
        {
            newHeight = dest.rows;
            newWidth = newHeight * ratioSrc;
        }
        else
        {
            newWidth = dest.cols;
            newHeight = newWidth / ratioSrc;
        }
        cv::Mat src2 = cv::Mat(newWidth, newHeight, CV_8UC3);
        cv::resize(src, src2, Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
        dw = (dest.cols - src2.cols)/2.;
        dh = (dest.rows - src2.rows)/2.;
        top    = (uint16_t)round(dh - 0.1);
        bottom = (uint16_t)round(dh + 0.1);
        left   = (uint16_t)round(dw - 0.1);
        right  = (uint16_t)round(dw + 0.1);
        cv::copyMakeBorder(src2, dest, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(padValue,padValue,padValue));
    }
    else
    {
        cv::resize(src, dest, Size(dest.cols, dest.rows), 0, 0, cv::INTER_LINEAR);
    }
    if(bgr2rgb)
    {
        cv::cvtColor(dest, dest, COLOR_BGR2RGB);
    }
    return (void*)dest.data;
}
#endif
bool stopFlag = false;
void RequestToStop(int sig)
{
    stopFlag = true;
}
bool GetStopFlag()
{
    return stopFlag;
}

int main(int argc, char *argv[])
{
    int optCmd, loops = 1, paramIdx = 0;
    string modelPath="", dncnnPath="", videoFile="";  
    bool cameraInput = false;
    int model_input_h = 0, model_input_w = 0;
    vector<unsigned long> inputPtr;
#ifdef USE_OPENCV
    auto objectColors = GetObjectColors();
#endif    

    if(argc==1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }

    while ((optCmd = getopt_long(argc, argv, "m:d:i:v:w:cxb:s:aep:l:n:h", opts,
        NULL)) != -1) {
        switch (optCmd) {
            case '0':
                break;
            case 'm':
                modelPath = strdup(optarg);
                break;
            case 'd':
                dncnnPath = strdup(optarg);
                break;
            case 'v':
                videoFile = strdup(optarg);
                break;
            case 'c':
                cameraInput = true;
                break;
            case 'p':
                paramIdx = stoi(optarg);
                break;
            case 'h':
            default:
                help();
                exit(0);
                break;
        }
    }
    LOG_VALUE(modelPath);
    LOG_VALUE(videoFile);
    LOG_VALUE(cameraInput);

    if(modelPath.empty())
    {
        cout << "Error: no model argument." << endl;
        help();
        return -1;
    }

    string captionModel = dxrt::StringSplit(modelPath, "/").back() + ", 30fps";    

    auto dncnnEngine = dxrt::InferenceEngine(dncnnPath);
    auto yoloEngine = dxrt::InferenceEngine(modelPath);

    auto input_shape = dncnnEngine.inputs().front().shape();
    model_input_h = input_shape[1];
    model_input_w = input_shape[2];
    
    auto yoloParam = yoloParams[paramIdx];
    Yolo yolo = Yolo(yoloParam);
    auto& profiler = dxrt::Profiler::GetInstance();
#if USE_OPENCV
    if(!videoFile.empty() || cameraInput)
    {
        bool pause = false;
        cv::VideoCapture cap;
        cv::VideoWriter writer;
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS], yoloPreProcFrame[FRAME_BUFFERS];
        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(model_input_h, model_input_w, CV_8UC3, cv::Scalar(0, 0, 0));
            yoloPreProcFrame[i] = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        int idx = 0, targetIdx = 0, key;
        if(!videoFile.empty())
        {
            cap.open(videoFile);
            cap.set(CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH);
            cap.set(CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT);
            if(!cap.isOpened())
            {
                cout << "Error: file " << videoFile << " could not be opened." <<endl;
                return -1;
            }
        }
        else
        {
            cap.open(0, cv::CAP_V4L2);
            cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
            cap.set(CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH);
            cap.set(CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT);
            if(!cap.isOpened())
            {
                cout << "Error: camera could not be opened." <<endl;
                return -1;
            }
        }
        cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        if(true)
        {
            namedWindow(DISPLAY_WINDOW_NAME);
            moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
            int callBackCnt = 0; // debug
            queue<pair<vector<BoundingBox>, int>> bboxesQueue;
            vector<BoundingBox> bboxes;
            mutex lk;
            cv::Mat improvedOutput = cv::Mat::zeros(model_input_h, model_input_w, CV_8UC3);
            
            /* Denoise PostProcessing Code */
            std::function<int(vector<shared_ptr<dxrt::Tensor>>, void*)> dncnnPPCallback = \
                [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
                {
                    /* PostProc */
                    float *data = (float *)outputs.front()->data();
                    
                    profiler.Start("dncnn");
                    lk.lock();
                    for (int y = 0; y < model_input_h; y++)
                    {
                        for (int x = 0; x < model_input_w; x++)
                        {
                            for (int c = 0; c < 3; c++)
                            {
                                float value = data[(y * model_input_w + x) * 64 + c] * 255.f;
                                if (value < 0.f)
                                    value = 0.f;
                                else if (value > 255.f)
                                    value = 255.f;
                                improvedOutput.data[(y * model_input_w + x) * 3 + c] = (uint8_t)value;
                            }
                        }
                    }
                    lk.unlock();
                    profiler.End("dncnn");
                    /* Restore raw frame index from tensor */
                    return 0;
                };
            
            /* OD PostProcessing Code */
            std::function<int(vector<shared_ptr<dxrt::Tensor>>, void*)> odPPCallback = \
                [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
                {
                    profiler.Start("od");
                        /* PostProc */
                        auto result = yolo.PostProc(outputs);
                        /* Restore raw frame index from tensor */
                        lk.lock();
                        bboxesQueue.push(
                            make_pair(
                                result, 
                                (uint64_t) arg
                            )
                        );
                        lk.unlock();
                    profiler.End("od");
                    return 0;
                };

            dncnnEngine.RegisterCallBack(dncnnPPCallback);
            yoloEngine.RegisterCallBack(odPPCallback);
            Mat outFrame = cv::Mat::zeros(cv::Size(CAMERA_FRAME_WIDTH * 2, CAMERA_FRAME_HEIGHT), CV_8UC3);
            Mat dnYoloFrame = cv::Mat::zeros(cv::Size(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT), CV_8UC3);
            Mat YoloFrame = cv::Mat::zeros(cv::Size(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT), CV_8UC3);
            while(1)
            {
                profiler.Start("cap");
                if(!pause)
                {
                    cap >> frame[idx];
                    if(frame[idx].empty()) cap.set(CAP_PROP_POS_FRAMES, 0), cap >> frame[idx];
                    cv::resize(frame[idx].clone(), frame[idx], Size(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT), cv::INTER_LINEAR);
                    cv::resize(frame[idx], resizedFrame[idx], Size(model_input_w, model_input_h), cv::INTER_LINEAR);
                    dncnnEngine.RunAsync(resizedFrame[idx].data, (void*)idx);
                    lk.lock();
                    cv::resize(improvedOutput.clone(), dnYoloFrame, Size(frame[idx].size().width, frame[idx].size().height), cv::INTER_LINEAR);
                    lk.unlock();
                    
                    /* write your yolo object detection model preprocessing code */
                    PreProc(dnYoloFrame, yoloPreProcFrame[idx], true, true, 114);
                    yoloEngine.RunAsync(yoloPreProcFrame[idx].data, (void*)idx);
                    lk.lock();
                    if(!bboxesQueue.empty())
                    {
                        bboxes = bboxesQueue.front().first;
                        targetIdx = bboxesQueue.front().second;
                        bboxesQueue.pop();
                    }
                    DisplayBoundingBox(dnYoloFrame, bboxes, yoloParam.height, yoloParam.width, "", "",
                        cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true, (float)frame[idx].size().width, (float)frame[idx].size().height);
                    lk.unlock();

                    /* integrate result frames */
                    dnYoloFrame.copyTo(outFrame(cv::Rect(0, 0, CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT)));
                    frame[idx].copyTo(outFrame(cv::Rect(CAMERA_FRAME_WIDTH, 0, CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT)));
                    
                    cv::imshow(DISPLAY_WINDOW_NAME, outFrame);
                    (++idx)%=FRAME_BUFFERS;
                }
                profiler.End("cap");
                int64_t t = (INPUT_CAPTURE_PERIOD_MS*1000 - profiler.Get("cap"))/1000;
                if(t<0 || t>INPUT_CAPTURE_PERIOD_MS) t = 0;
                key = cv::waitKey(max((int64_t)1, t));
                
                std::cout << 
                "======================================" << std::endl <<
                " DNCNN Inference Time = " << dncnnEngine.inference_time() << "us" << std::endl <<
                " YOLO Inference Time = " << yoloEngine.inference_time() << "us" << std::endl <<
                " Process Time = " << profiler.Get("cap") << "us" << std::endl <<
                std::endl;
                if(key == 0x20) //'p'
                {
                    pause = !pause;
                }
                else if(key == 0x1B) //'ESC'
                {
                    break;
                }
            }
            sleep(1);
        }
    }
#endif

    cout << dncnnEngine.name() << " : latency " << dncnnEngine.latency() << "us, " << dncnnEngine.inference_time() << "us" << endl;

    return 0;
}
