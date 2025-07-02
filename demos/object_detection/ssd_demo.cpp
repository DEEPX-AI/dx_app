#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#ifdef __linux__
#include <syslog.h>
#endif

#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>

#include "display.h"
#include "dxrt/dxrt_api.h"
#include "ssd.h"

using namespace std;
using namespace cv;

#define ISP_PHY_ADDR   (0x8F000000)
#define ISP_INPUT_ZEROCOPY
// #define ISP_DEBUG_BY_OPENCV
#define POSTPROC_SIMULATION_DEVICE_OUTPUT
#define DISPLAY_WINDOW_NAME "Object Detection"
#define INPUT_CAPTURE_PERIOD_MS 30
#define CAMERA_FRAME_WIDTH 800
#define CAMERA_FRAME_HEIGHT 600
#define FRAME_BUFFERS 5

static int memfd;
static unsigned char *ispBuf = NULL;

#ifdef __linux__
unsigned char *InitISPMapping(size_t mapSize, off_t phyAddr)
{
    memfd = open("/dev/mem", O_RDWR);
    if(memfd < 0){
        printf("mem open error\n");
        return nullptr;
    }
    ispBuf = (unsigned char *)mmap(
        0,                                              // addr
        mapSize,                        // len
        PROT_READ|PROT_WRITE,   // prot
        MAP_SHARED,                             // flags
        memfd,                                  // fd
        phyAddr             // offset
    );
    return ispBuf;
};

void DeinitISPMapping(size_t imgSize)
{
    munmap(ispBuf, imgSize);
    close(memfd);
};
#endif

// pre/post parameter table
extern SsdParam mv1_ssd_300, mv2_ssd_320, mv1_ssd_512;
SsdParam ssdParams[] = {
    mv1_ssd_300, // index = 0
    mv2_ssd_320, // index = 1
    mv1_ssd_512, // index = 2
};

/////////////////////////////////////////////////////////////////////////////////////////////////

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

int main(int argc, char *argv[])
{
    int loops=1, paramIdx=0;
    string modelPath="", imgFile="", videoFile="", binFile="", simFile="";
    bool cameraInput = false, ispInput = false, asyncInference = false;
    auto objectColors = GetObjectColors();


    std::string app_name = "ssd object detection model demo";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
    ("m, model", "define model path", cxxopts::value<std::string>(modelPath))
    ("i, image", "use image file input", cxxopts::value<std::string>(imgFile))
    ("v, video", "use video file input", cxxopts::value<std::string>(videoFile))
    ("c, camera", "use camera input", cxxopts::value<bool>(cameraInput)->default_value("false"))
    ("x, isp", "use ISP input", cxxopts::value<bool>(ispInput)->default_value("false"))
    ("b, bin", "use binary file input", cxxopts::value<std::string>(binFile))
    ("s, sim", "use pre-defined npu output binary file input( perform post-proc. only )", cxxopts::value<std::string>(simFile))
    ("a, async", "asynchronous inference", cxxopts::value<bool>(asyncInference))
    ("p, param", "pre/post-processing parameter selection", cxxopts::value<int>(paramIdx))
    ("l, loop", "loop test", cxxopts::value<int>(loops)->default_value("1"))
    ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help") || modelPath.empty())
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    LOG_VALUE(modelPath);
    LOG_VALUE(videoFile);
    LOG_VALUE(imgFile);
    LOG_VALUE(binFile);
    LOG_VALUE(simFile);
    LOG_VALUE(cameraInput);
    LOG_VALUE(ispInput);
    LOG_VALUE(asyncInference);

    dxrt::InferenceEngine ie(modelPath);
    auto ssdParam = ssdParams[paramIdx];
    Ssd ssd = Ssd(ssdParam);
    auto& profiler = dxrt::Profiler::GetInstance();
    if(!imgFile.empty())
    {
        cv::Mat frame = cv::imread(imgFile, IMREAD_COLOR);
        profiler.Start("pre");
        cv::Mat resizedFrame = cv::Mat(ssdParam.image_size, ssdParam.image_size, CV_8UC3);
        PreProc(frame, resizedFrame, false, true);
        profiler.End("pre");
        cv::imwrite("resized.jpg", resizedFrame);
        profiler.Start("main");
        auto outputs = ie.Run(resizedFrame.data);
        profiler.End("main");
            // for(auto &output:outputs)
            //     output->Show();
            profiler.Start("post");
            auto result = ssd.PostProc(outputs);
            profiler.End("post");
            ssd.ShowResult();
            DisplayBoundingBox(frame, result, -1, -1, \
                "", "", cv::Scalar(0, 0, 255), objectColors, "result.jpg", 0, -1, true);
        // cv::waitKey(0);
        
        return 0;
    }
    else if(!videoFile.empty() || cameraInput)
    {
        bool pause = false;
        cv::VideoCapture cap;
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS];
        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(ssdParam.image_size, ssdParam.image_size, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        int idx = 0, prevIdx = 0, key;
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
            cap.open(0);
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
        namedWindow(DISPLAY_WINDOW_NAME);
        moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
        profiler.Start("cap");
        while(1)
        {
            profiler.End("cap");
            profiler.Start("cap");
            if(!pause)
            {
                cap >> frame[idx];
                if(frame[idx].empty()) break;                
                profiler.Start("pre");
                PreProc(frame[idx], resizedFrame[idx], true, true, 114);
                profiler.End("pre");
                profiler.Start("main");
                auto outputs = ie.Run(resizedFrame[idx].data);
                profiler.End("main");
                // cout << "  - " << outputs.size() << endl;
                if(outputs.size()>0)
                {
                    profiler.Start("post");
                    auto result = ssd.PostProc(outputs);
                    // cv::Mat outFrame;
                    // outFrame = *(cv::Mat*)(outputs[0]->GetDestination());
                    // ssd.ShowResult();
                    // DisplayBoundingBox(outFrame, result, ssdParam.image_size, ssdParam.image_size, "", "",
                    //     cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                    DisplayBoundingBox(frame[prevIdx], result, ssdParam.image_size, ssdParam.image_size, "", "",
                        cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                    profiler.End("post");
                    cv::imshow(DISPLAY_WINDOW_NAME, frame[prevIdx]);
                }
                prevIdx = idx;
                (++idx)%=FRAME_BUFFERS;                    
            }
            key = cv::waitKey(INPUT_CAPTURE_PERIOD_MS);
            if(key == 0x20) //'p'
            {
                pause = !pause;
            }
            else if(key == 0x1B) //'ESC'
            {
                break;
            }
        }
#ifdef __linux
        sleep(1);
#elif _WIN32
        Sleep(1000);
#endif

        // ie.Show();
        
        return 0;
    }
    if(!binFile.empty())
    {
        int cnt = 0;
        do {
            vector<uint8_t> inputBuf(ie.input_size(), 0);
            dxrt::DataFromFile(binFile, inputBuf.data());
            cv::Mat frame = cv::Mat(ssdParam.image_size, ssdParam.image_size, CV_8UC3);
            cv::imwrite("debug.jpg", frame);
            auto outputs = ie.Run(inputBuf.data());
            if(!outputs.empty())
            {
                auto result = ssd.PostProc(outputs);
                ssd.ShowResult();
                cnt++;
            }
        } while(loops<0?1:(cnt<loops));
        return 0;
    }
    return 0;
}
