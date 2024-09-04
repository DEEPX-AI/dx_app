#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syslog.h>

#include <opencv2/opencv.hpp>

#include "display.h"
#include "dxrt/dxrt_api.h"
#include "yolo.h"
#include "isp.h"
#include "v4l2.h"
#include "osd_eyenix.h"
#include "socket.h"

using namespace std;
using namespace cv;

#define ISP_PHY_ADDR   (0x9D000000)
#define ISP_INPUT_ZEROCOPY
#define POSTPROC_SIMULATION_DEVICE_OUTPUT
#define DISPLAY_WINDOW_NAME "Pose Estimation"
#define INPUT_CAPTURE_PERIOD_MS 30
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
#define FRAME_BUFFERS 5

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif

// pre/post parameter table
extern YoloParam yolov5s6_pose_640, yolov5s6_pose_1280;
YoloParam yoloParams[] = {
    [0] = yolov5s6_pose_640,
    [1] = yolov5s6_pose_1280
};

/////////////////////////////////////////////////////////////////////////////////////////////////
static struct option const opts[] = {
    { "model", required_argument, 0, 'm' },
    { "image", required_argument, 0, 'i' },
    { "video", required_argument, 0, 'v' },
    { "write", required_argument, 0, 'w' },
    { "camera", no_argument, 0, 'c' },
    { "isp", no_argument, 0, 'x' },
    { "bin",  required_argument, 0, 'b' },
    { "async", no_argument, 0, 'a' },
    { "param", required_argument, 0, 'p' },
    { "loop", no_argument, 0, 'l' },
    { "help", no_argument, 0, 'h' },
    { 0, 0, 0, 0 }
};
const char* usage =
"pose estimation demo\n"
"  -m, --model     define model path\n"
"  -i, --image     use image file input\n"
"  -v, --video     use video file input\n"
"  -w, --write     write result frames to a video file\n"
"  -c, --camera    use camera input\n"
"  -x, --isp       use ISP input\n"
"  -b, --bin       use binary file input\n"
"  -a, --async     asynchronous inference\n"
"  -p, --param      pre/post-processing parameter selection\n"
"  -l, --loop      loop test\n"
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
        cv::Mat src2 = cv::Mat(newHeight, newWidth, CV_8UC3);
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
bool stopFlag = false;
void RequestToStop(int sig)
{
    UNUSEDVAR(sig);
    stopFlag = true;
}
bool GetStopFlag()
{
    return stopFlag;
}

int main(int argc, char *argv[])
{
    int i = 1, loops = 1, paramIdx = 0;
    string modelPath="", imgFile="", videoFile="", binFile="", simFile="", videoOutFile="";        
    bool cameraInput = false, ispInput = false,
        asyncInference = false, writeFrame = false;
    vector<unsigned long> inputPtr;
    auto objectColors = GetObjectColors();

    if(argc==1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }
    
    while (i < argc) {
        std::string arg(argv[i++]);
        if (arg == "-m")
                                modelPath = strdup(argv[i++]);
        else if (arg == "-i")
                                imgFile = strdup(argv[i++]);
        else if (arg == "-v")
                                videoFile = strdup(argv[i++]);
        else if (arg == "-w")
        {
                                videoOutFile = strdup(argv[i++]);
                                writeFrame = true;
        }
        else if (arg == "-c")
                                cameraInput = true;
        else if (arg == "-x")
                                ispInput = true;
        else if (arg == "-b")
                                binFile = strdup(argv[i++]);
        else if (arg == "-p")
                                paramIdx = stoi(argv[i++]);
        else if (arg == "-l")
                                loops = stoi(argv[i++]);
        else if (arg == "-a")
                                asyncInference = true;
        else if (arg == "-h")
                                help(), exit(0);
        else
                                help(), exit(0);
    }
    if (modelPath.empty())
    {
        help(), exit(0);
    }
    if (imgFile.empty()&&videoFile.empty()&&!cameraInput&&!ispInput&&binFile.empty())
    {
        help(), exit(0);
    }
    
    LOG_VALUE(modelPath);
    LOG_VALUE(videoFile);
    LOG_VALUE(imgFile);
    LOG_VALUE(binFile);
    LOG_VALUE(cameraInput);
    LOG_VALUE(ispInput);
    LOG_VALUE(asyncInference);

    dxrt::InferenceEngine ie(modelPath);
    auto yoloParam = yoloParams[paramIdx];
    Yolo yolo = Yolo(yoloParam);
    yolo.LayerReorder(ie.outputs());
    auto& profiler = dxrt::Profiler::GetInstance();
    if(!imgFile.empty())
    {
        vector<shared_ptr<dxrt::Tensor>> outputs;
        cv::Mat frame = cv::imread(imgFile, IMREAD_COLOR);
        profiler.Start("pre");
        cv::Mat resizedFrame = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3);
        PreProc(frame, resizedFrame, true, true, 114);
        profiler.End("pre");
        if(!asyncInference)
        {
            profiler.Start("main");
            outputs = ie.Run(resizedFrame.data);
            profiler.End("main");
        }
        else
        {
            vector<uint64_t> tmp = {};
            profiler.Start("main");            
            int reqId = ie.RunAsync(resizedFrame.data);
            LOG_VALUE(reqId);
            outputs = ie.Wait(reqId);
            profiler.End("main");
        }
        profiler.Start("post");
        auto result = yolo.PostProc(outputs);
        profiler.End("post");
        yolo.ShowResult();
        DisplayBoundingBox(frame, result, yoloParam.height, yoloParam.width, \
            "", "", cv::Scalar(0, 0, 255), objectColors, "result.jpg", 0, -1, true);
        std::cout << "save file : result.jpg " << std::endl;
        profiler.Show();
        return 0;
    }
    else if(!videoFile.empty() || cameraInput)
    {
        bool pause = false;
        cv::VideoCapture cap;
        cv::VideoWriter writer;
        cv::TickMeter tm;
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS];
        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        int idx = 0, key;
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
        cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        if(!writeFrame)
        {
            namedWindow(DISPLAY_WINDOW_NAME, WINDOW_NORMAL);
            moveWindow(DISPLAY_WINDOW_NAME, 0, 0);

            queue<pair<vector<BoundingBox>, int>> bboxesQueue;
            vector<BoundingBox> bboxes;
            mutex lk;
            std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
                [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
                {
                    profiler.Start("post");
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
                    cap >> frame[idx];                    
                    if(frame[idx].empty()) break;
                    profiler.Start("pre");
                    PreProc(frame[idx], resizedFrame[idx], true, true, 114);
                    profiler.End("pre");
                    profiler.Start("main");
                    int reqId = ie.RunAsync(resizedFrame[idx].data, (void*)(intptr_t)idx);
                    UNUSEDVAR(reqId);
                    profiler.End("main");
                    lk.lock();
                    if(!bboxesQueue.empty())
                    {
                        bboxes = bboxesQueue.front().first;
                        outFrame = frame[bboxesQueue.front().second];
                        bboxesQueue.pop();
                    }
                    else
                    {
                        outFrame = frame[idx];
                    }
                    lk.unlock();
                    DisplayBoundingBox(outFrame, bboxes, yoloParam.height, yoloParam.width, "", "",
                        cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                    cv::imshow(DISPLAY_WINDOW_NAME, outFrame);
                    (++idx)%=FRAME_BUFFERS;
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
        }
        else
        {
            DXRT_ASSERT(!videoOutFile.empty(), "video output file must be configured.");
            writer.open(
                videoOutFile, VideoWriter::fourcc('M','J','P','G'), (int)cap.get(CAP_PROP_FPS), 
                cv::Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), true
            );
            if(!writer.isOpened())
            {
                cout << "Error: video writer for " << videoFile << " could not be opened." <<endl;
                return -1;
            }
            float fps;
            int textBaseline = 0;
            while(1)
            {
                cout << "frame " << cap.get(CAP_PROP_POS_FRAMES) << " / " << cap.get(CAP_PROP_FRAME_COUNT) << endl;
                profiler.Start("cap");
                    cap >> frame[idx];
                    if(frame[idx].empty()) break;
                profiler.End("cap");
                profiler.Start("pre");
                    PreProc(frame[idx], resizedFrame[idx], true, true, 114);
                profiler.End("pre");
                profiler.Start("main");
                    auto outputs = ie.Run(resizedFrame[idx].data);
                profiler.End("main");
                profiler.Start("post");
                    if(outputs.size()>0)
                    {
                        auto result = yolo.PostProc(outputs);
                        DisplayBoundingBox(frame[idx], result, yoloParam.height, yoloParam.width, "", "",
                            cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);                        
                    }
                    fps = 1; 
                    ostringstream oss;
                    oss << setprecision(2) << fixed << fps;
                    string text = "FPS:" + oss.str();
                    auto textSize = cv::getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 1, &textBaseline);
                    cv::rectangle( 
                        frame[idx],
                        Point( 25, 25-textSize.height ), 
                        Point( 25 + textSize.width, 35 ), 
                        Scalar(172, 81, 99),
                        cv::FILLED);
                    cv::putText(
                        frame[idx], text, Point( 30, 30 ), 
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255));
                profiler.End("post");
                profiler.Start("writer");
                    writer << frame[idx];
                profiler.End("writer");
                (++idx)%=FRAME_BUFFERS;
            }
        }
        profiler.Show();
        return 0;
    }
    if(!binFile.empty())
    {
        int cnt = 0;
        do {
            vector<uint8_t> inputBuf(ie.input_size(), 0);
            dxrt::DataFromFile(binFile, inputBuf.data());
            cv::Mat frame(yoloParam.height, yoloParam.width, CV_8UC3, inputBuf.data());
            cv::imwrite("debug.jpg", frame);
            auto outputs = ie.Run(inputBuf.data());
            if(!outputs.empty())
            {
                auto result = yolo.PostProc(outputs);
                yolo.ShowResult();
            }
            cnt++;
        } while(loops<0?1:(cnt<loops));
        return 0;
    }
    return 0;
}