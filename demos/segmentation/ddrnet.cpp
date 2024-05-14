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

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "display.h"
#include "dxrt/dxrt_api.h"

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////////////////////////////////////////////
#define PREPROC_KEEP_IMG_RATIO false
#define DISPLAY_WINDOW_NAME "Segmentation: DDRNet"
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
#define FRAME_BUFFERS 5

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif
struct SegmentationParam
{
    int classIndex;
    string className;
    uint8_t colorB;
    uint8_t colorG;
    uint8_t colorR;
};
SegmentationParam segCfg[] = {
    {0, "background", 0, 0, 0, }, /* Skip */
    {1, "foot", 0, 128, 0, },
    {2, "body", 0, 0, 128, },
};
/////////////////////////////////////////////////////////////////////////////////////////////////

static struct option const opts[] = {
    { "model", required_argument, 0, 'm' },
    { "image", required_argument, 0, 'i' },
    { "video", required_argument, 0, 'v' },
    { "camera", no_argument, 0, 'c' },
    { "bin",  required_argument, 0, 'b' },
    { "sim", required_argument, 0, 's' },
    { "async", no_argument, 0, 'a' },
    { "iomode", no_argument, 0, 'o' },
    { "help", no_argument, 0, 'h' },
    { "width", required_argument, 0, 'x' },
    { "height", required_argument, 0, 'y' },
    { 0, 0, 0, 0 }
};
const char* usage =
"Image Segmentation Demo\n"
"  -m, --model     define dxnn model path\n"
"  -i, --image     use image file input\n"
"  -v, --video     use video file input\n"
"  -c, --camera    use camera input\n"
"  -b, --bin       use binary file input\n"
"  -s, --sim       use pre-defined npu output binary file input( perform post-proc. only )\n"
"  -a, --async     asynchronous inference\n"
"  -o, --iomode    I/O only mode (not perform inference directly)\n"
"  -x, --width     Input image width\n"
"  -y, --height    Input image height\n"
"  -h, --help      show help\n"
;
void help()
{
    cout << usage << endl;    
}

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
        cv::Mat src2;
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

void Segmentation(uint16_t *input, cv::Mat &color, SegmentationParam *cfg, int numClasses)
{
    uint8_t *output = color.data;
    for(int h=0;h<color.rows;h++)
    {
        for(int w=0;w<color.cols;w++)
        {
            for(int cls=1; cls<numClasses; cls++)
            {
                if(input[color.cols*h + w] == (uint16_t)cls)
                {
                    output[3*color.cols*h + 3*w + 0] = cfg[cls].colorB;
                    output[3*color.cols*h + 3*w + 1] = cfg[cls].colorG;
                    output[3*color.cols*h + 3*w + 2] = cfg[cls].colorR;
                }
            }
        }
    }
}
void Segmentation(uint16_t *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses)
{
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            for(int cls=1; cls<numClasses; cls++)
            {
                if(input[cols*h + w] == (uint16_t)cls)
                {
                    output[3*cols*h + 3*w + 0] = cfg[cls].colorB;
                    output[3*cols*h + 3*w + 1] = cfg[cls].colorG;
                    output[3*cols*h + 3*w + 2] = cfg[cls].colorR;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int optCmd;
    int inputWidth = 0, inputHeight = 0;
    string modelPath="", imgFile="", videoFile="", binFile="", simFile="";
    bool cameraInput = false, asyncInference = false;

    if(argc==1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }

    while ((optCmd = getopt_long(argc, argv, "m:i:v:cb:s:x:y:aoh", opts,
        NULL)) != -1) {
        switch (optCmd) {
            case '0':
                break;
            case 'm':
                modelPath = strdup(optarg);
                break;
            case 'i':
                imgFile = strdup(optarg);
                break;
            case 'v':
                videoFile = strdup(optarg);
                break;
            case 'c':
                cameraInput = true;
                break;
            case 'b':
                binFile = strdup(optarg);
                break;
            case 's':
                simFile = strdup(optarg);
                break;
            case 'x':
                inputWidth = stoi(optarg);
                break;
            case 'y':
                inputHeight = stoi(optarg);
                break;
            case 'a':
                asyncInference = true;
                break;
            case 'h':
            default:
                help();
                exit(0);
                break;
        }
    }
    
    if(inputWidth==0) inputWidth = 768;
    if(inputHeight==0) inputHeight = 384;
    LOG_VALUE(inputWidth);
    LOG_VALUE(inputHeight);
    LOG_VALUE(modelPath);
    LOG_VALUE(videoFile);
    LOG_VALUE(binFile);
    LOG_VALUE(simFile);
    LOG_VALUE(cameraInput);
    LOG_VALUE(asyncInference);

    dxrt::InferenceEngine ie(modelPath);

    auto& profiler = dxrt::Profiler::GetInstance();
    if(!imgFile.empty())
    {
        cv::Mat frame = cv::imread(imgFile, IMREAD_COLOR);
        profiler.Start("pre");
        cv::Mat resizedFrame = cv::Mat(inputHeight, inputWidth, CV_8UC3);
        PreProc(frame, resizedFrame, PREPROC_KEEP_IMG_RATIO);
        profiler.End("pre");
        profiler.Start("main");
        auto outputs = ie.Run(resizedFrame.data);
        profiler.End("main");
        LOG_VALUE(outputs.size());
        profiler.Start("post-segment");
        cv::Mat result = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        Segmentation((uint16_t*)outputs[0]->data(), result.data, result.rows, result.cols, segCfg, 3);
        profiler.End("post-segment");
        profiler.Start("post-blend");
        cv::resize(result, result, Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);
        frame = frame + result;
        // cv::addWeighted( frame, 1.0, result, 0.5, 0.0, blendResultToRaw);
        profiler.End("post-blend");
        cout << dec << inputWidth << "x" << inputHeight << " <- " << frame.cols << "x" << frame.rows << endl;
        cv::imwrite("result-blend-to-raw.jpg", frame);
        cv::imwrite("result-segmentation.jpg", result);
        cv::imwrite("resized.jpg", resizedFrame);
        cv::imshow("segmentation", frame);
        cv::waitKey(0);
        profiler.Show();
        return 0;
    }
    else if(!videoFile.empty() || cameraInput)
    {
        bool pause = false;
        cv::VideoCapture cap;
        cv::VideoWriter writer;
        cv::TickMeter tm;
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS], segFrame[FRAME_BUFFERS];
        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
            segFrame[i] = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        int inIdx = 0, outIdx = 0, key;
        if(!videoFile.empty())
        {
            cap.open(videoFile);
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
        int fps = cap.get(CAP_PROP_FPS);
        float capInterval = 1000./fps;
        queue<int> outIdxQueue;
        mutex lk;
        cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        namedWindow(DISPLAY_WINDOW_NAME);
        moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
        std::function<int(vector<shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
            [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
            {
                profiler.Start("post");
                uint64_t id = *(uint64_t*) arg;
                segFrame[id].setTo(cv::Scalar(0,0,0));
                Segmentation((uint16_t*)outputs[0]->data(), segFrame[id].data, segFrame[id].rows, segFrame[id].cols, segCfg, 3);
                lk.lock();
                outIdxQueue.push(id);
                lk.unlock();
                profiler.End("post");
                return 0;
            };
        ie.RegisterCallBack(postProcCallBack);
        profiler.Start("cap");
        Mat outFrame;
        while(1)
        {
            tm.reset();
            tm.start();
            if(!pause)
            {
                cap >> frame[inIdx];                    
                if(frame[inIdx].empty()) break;
                profiler.Start("pre");
                PreProc(frame[inIdx], resizedFrame[inIdx], PREPROC_KEEP_IMG_RATIO);
                profiler.End("pre");
                profiler.Start("main");
                int reqId = ie.RunAsync(resizedFrame[inIdx].data, (void*)(intptr_t)inIdx);
                UNUSEDVAR(reqId);
                profiler.End("main");
                if(!outIdxQueue.empty())
                {
                    lk.lock();
                    outIdx = outIdxQueue.front();
                    outIdxQueue.pop();
                    lk.unlock();
                    outFrame = frame[outIdx];
                    cv::Mat add;
                    cv::resize(segFrame[outIdx], add, Size(outFrame.cols, outFrame.rows), 0, 0, cv::INTER_LINEAR);
                    cv::addWeighted(outFrame, 1.0, add, 0.5, 0.0, outFrame);
                }
                else
                {
                    outIdx = inIdx;
                    outFrame = frame[outIdx];
                }
                cv::imshow(DISPLAY_WINDOW_NAME, outFrame);
                (++inIdx)%=FRAME_BUFFERS;
            }
            key = cv::waitKey(1);
            if(key == 0x20) //'p'
            {
                pause = !pause;
            }
            else if(key == 0x1B) //'ESC'
            {
                break;
            }
            tm.stop();
            double elapsed = tm.getTimeMilli();
            if (elapsed < capInterval)
            {
                cv::waitKey( max(1, (int)(capInterval - elapsed)) );
            }
        }
        sleep(1);
        profiler.Show();
        return 0;
    }

    if(!binFile.empty())
    {
        float *buf = new float[50*1024*1024/sizeof(float)];
#if 1   
        uint8_t *input_buffer = new uint8_t(inputWidth * inputHeight * 3);
        dxrt::DataFromFile(binFile, input_buffer);
        auto outputs = ie.Run(input_buffer);
#else
        auto inputTensor = ie.GetInput(0x1000);
        dxrt::DataFromFile(binFile, inputTensor[0]->GetData());
        auto outputs = ie.Run(inputTensor);
#endif        
        if(outputs.size()==0)
        {
            cout << "Error. Invalid output detected." << endl;
            return -1;
        }
        delete [] buf;
        return 0;
    }

    return 0;
}