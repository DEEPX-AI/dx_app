#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#include <syslog.h>
#include "fb.h"
#elif _WIN32
#include <Windows.h>
#endif

#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>

#include "dxrt/dxrt_api.h"
#include "utils/common_util.hpp"

#include "display.h"
#include "yolo.h"
#include "v4l2.h"
#include "socket.h"

using namespace std;
using namespace cv;

#define ISP_INPUT_DMA_COPY
#define ISP_PHY_ADDR   (0xA0000000)
#define ISP_INPUT_ZEROCOPY
// #define ISP_DEBUG_BY_RAWFILE
// #define ISP_DEBUG_BY_OPENCV
#define POSTPROC_SIMULATION_DEVICE_OUTPUT
#define DISPLAY_WINDOW_NAME "Object Detection"
#define INPUT_CAPTURE_PERIOD_MS 30

// camera frame resolution (1920, 1080), (1280, 720), (800, 600)
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
#define FRAME_BUFFERS 5
#define DMA_IOCTL_DATA		_IOW('T', 3, struct npu_dma_ioctl_data_copy)

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif

#define PERFORMANCE_TEST false

// pre/post parameter table
extern YoloParam yolov5s_320, yolov5s_512, yolov5s_640, yolox_s_512, yolov7_640, yolov7_512, yolox_s_640;
YoloParam yoloParams[] = {
    yolov5s_320,
    yolov5s_512,
    yolov5s_640,
    yolox_s_512,
    yolov7_640,
    yolov7_512,
    yolox_s_640,
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

#if __riscv
pthread_mutex_t lock_uyv = PTHREAD_MUTEX_INITIALIZER;
int uyv_cnt = 0;
#endif

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
DXRT_TRY_CATCH_BEGIN
    int loops = -1, paramIdx = 0;
    string modelPath="", imgFile="", videoFile="", binFile="", simFile="", videoOutFile="", OSDstr="", rtspPath="";  
    bool cameraInput = false, ispInput = false, 
        asyncInference = false, writeFrame = false, performanceTest = false;
    vector<unsigned long> inputPtr;
    auto objectColors = GetObjectColors();

    
    std::string app_name = "yolo object detection demo";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
    ("m, model", "define model path", cxxopts::value<std::string>(modelPath))
    ("i, image", "use image file input", cxxopts::value<std::string>(imgFile))
    ("v, video", "use video file input", cxxopts::value<std::string>(videoFile))
    ("w, write", "write result frames to a video file", cxxopts::value<bool>(writeFrame)->default_value("false"))
    ("c, camera", "use camera input", cxxopts::value<bool>(cameraInput)->default_value("false"))
    ("r, rtsp", "use rtsp input", cxxopts::value<std::string>(rtspPath))
    ("x, isp", "use ISP input", cxxopts::value<bool>(ispInput)->default_value("false"))
    ("b, bin", "use binary file input", cxxopts::value<std::string>(binFile))
    ("s, sim", "use pre-defined npu output binary file input( perform post-proc. only )", cxxopts::value<std::string>(simFile))
    ("a, async", "asynchronous inference", cxxopts::value<bool>(asyncInference)->default_value("false"))
    ("p, param", "pre/post-processing parameter selection", cxxopts::value<int>(paramIdx)->default_value("-1"))
    ("l, loop", "loop test", cxxopts::value<int>(loops)->default_value("0"))
    ("performance", "for performance inference", cxxopts::value<bool>(performanceTest)->default_value("false"))
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
    LOG_VALUE(rtspPath);
    LOG_VALUE(imgFile);
    LOG_VALUE(binFile);
    LOG_VALUE(simFile);
    LOG_VALUE(cameraInput);
    LOG_VALUE(ispInput);
    LOG_VALUE(asyncInference);

    auto captionModel = dxapp::common::getFileName(modelPath);

    dxrt::InferenceEngine ie(modelPath);
    auto yoloParam = yoloParams[paramIdx];
    Yolo yolo = Yolo(yoloParam);
    yolo.LayerReorder(ie.outputs());
    auto& profiler = dxrt::Profiler::GetInstance();
    if(!imgFile.empty())
    {
        vector<vector<shared_ptr<dxrt::Tensor>>> vOutputs;
        vector<int> requests;
        cv::Mat frame = cv::imread(imgFile, IMREAD_COLOR);
        profiler.Start("pre");
        cv::Mat resizedFrame = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3);
        PreProc(frame, resizedFrame, true, true, 114);
        profiler.End("pre");
        if(!asyncInference)
        {
            profiler.Start("main");
            vOutputs.emplace_back( ie.Run(resizedFrame.data) );
            profiler.End("main");
        }
        else
        {
            int loop;
            if(loops<0)loops=1;
            for(loop=0;loop<loops;loop++)
            {                
                profiler.Start("main");            
                requests.emplace_back(
                    ie.RunAsync(resizedFrame.data)
                );
                profiler.End("main");
            }
            for(auto &request:requests)
            {
                vOutputs.emplace_back( ie.Wait(request));
            }
        }
        for(auto &outputs:vOutputs)
        {
            profiler.Start("post");
            auto result = yolo.PostProc(outputs);
            profiler.End("post");
            yolo.ShowResult();
            DisplayBoundingBox(frame, result, yoloParam.height, yoloParam.width, \
                "", "", cv::Scalar(0, 0, 255), objectColors, "result.jpg", 0, -1, true);            
            std::cout << "save file : result.jpg " << std::endl;
        }
    }
    else if(!videoFile.empty() || !rtspPath.empty() || cameraInput)
    {
        bool pause = false;
        double total_frames = 0;
        cv::VideoCapture cap;
        cv::VideoWriter writer;
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS];

        int64_t previous_average_time = 0;
        auto time_s = std::chrono::high_resolution_clock::now();
        auto time_e = std::chrono::high_resolution_clock::now();
        uint64_t postprocessed_count = 0;
        uint64_t duration_time = 0;
        int font_size = 0;

        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        int idx = 0, key = 0;
        if(!videoFile.empty())
        {
            cap.open(videoFile);
            if(!cap.isOpened())
            {
                cout << "Error: file " << videoFile << " could not be opened." <<endl;
                return -1;
            }
            total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
            font_size = cap.get(cv::CAP_PROP_FRAME_HEIGHT) / 100;
        }
        else
        {
            if(cameraInput)
#ifdef __linux__
                cap.open(0, cv::CAP_V4L2);
#elif _WIN32
                cap.open(0);
#endif
            else
                cap.open(rtspPath);
            cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
            if(!cap.isOpened())
            {
                cout << "Error: camera could not be opened." <<endl;
                return -1;
            }
            font_size = cap.get(cv::CAP_PROP_FRAME_HEIGHT) / 100;
        }
        cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        if(!writeFrame)
        {
            namedWindow(DISPLAY_WINDOW_NAME);
            moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
            queue<pair<vector<BoundingBox>, int>> bboxesQueue;
            vector<BoundingBox> bboxes;
            mutex lk;
            std::function<int(vector<shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
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
                    time_e = std::chrono::high_resolution_clock::now();
                    duration_time = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s).count();
                    postprocessed_count++;
                    profiler.End("post");
                    return 0;
                };
            ie.RegisterCallBack(postProcCallBack);
            profiler.Start("cap");
            Mat outFrame;
            while(1)
            {
                profiler.Start("cap");
                profiler.Start("main");
                if(!pause)
                {
                    if(cap.get(cv::CAP_PROP_POS_FRAMES) > total_frames - FRAME_BUFFERS)
                    {
                        if(loops > 0)
                            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    }
                    cap >> frame[idx];
                    if(frame[idx].empty())
                    {
                        break;
                    }
                    profiler.Start("pre");
                    PreProc(frame[idx], resizedFrame[idx], true, true, 114);
                    profiler.End("pre");
                    time_s = std::chrono::high_resolution_clock::now();
                    int reqId = ie.RunAsync(resizedFrame[idx].data, (void*)(intptr_t)idx);
                    UNUSEDVAR(reqId);
                    profiler.End("main");
                    if(postprocessed_count > 0 && bboxesQueue.size() > 0)
                    {
                        lk.lock();
                        bboxes = bboxesQueue.front().first;
                        outFrame = frame[bboxesQueue.front().second];
                        bboxesQueue.pop();
                        lk.unlock();
                        DisplayBoundingBox(outFrame, bboxes, yoloParam.height, yoloParam.width, "", "",
                            cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                        cv::Size mainCaptionSize = cv::getTextSize(captionModel, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, nullptr);
                        cv::rectangle(outFrame, Point(0, 0), Point(mainCaptionSize.width+ 40, mainCaptionSize.height + 50), Scalar(0, 0, 0), cv::FILLED);
                        cv::putText(outFrame, captionModel, cv::Point(20, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
                        auto new_average_time = ((previous_average_time * postprocessed_count) + duration_time)/ (postprocessed_count + 1);
                        previous_average_time = new_average_time;
                        uint64_t fps = 1000000 / new_average_time;
                        std::string fpsCaption = "FPS : " + std::to_string((int)fps);
                        cv::Size fpsCaptionSize = cv::getTextSize(fpsCaption, cv::FONT_HERSHEY_PLAIN, font_size, 2, nullptr);
                        // cv::putText(outFrame, fpsCaption, cv::Point(outFrame.size().width - fpsCaptionSize.width, outFrame.size().height - fpsCaptionSize.height), cv::FONT_HERSHEY_PLAIN, font_size, cv::Scalar(255, 255, 255),2);
                        cv::imshow(DISPLAY_WINDOW_NAME, outFrame);
                    }
                    (++idx)%=FRAME_BUFFERS;
                }
                profiler.End("cap");
                int64_t t = (INPUT_CAPTURE_PERIOD_MS*1000 - profiler.Get("cap"))/1000;
                if(t<0 || t>INPUT_CAPTURE_PERIOD_MS) t = 0;

                if(performanceTest)
                    key = cv::waitKey(1);
                else
                    key = cv::waitKey(max((int64_t)1, t));
                
                if(key == 'p') //'p'
                {
                    pause = !pause;
                }
                else if(key == 0x1B || key == 'q') //'ESC'
                {
                    break;
                }
            }
#ifdef __linux__
            sleep(1);
#elif _WIN32
            Sleep(1000);
#endif
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
                if (frame[idx].empty()) break;
                profiler.End("cap");
                profiler.Start("pre");
                PreProc(frame[idx], resizedFrame[idx], true, true, 114);
                profiler.End("pre");
                profiler.Start("main");
                auto outputs = ie.Run(resizedFrame[idx].data);
                profiler.End("main");
                profiler.Start("post");
                if (outputs.size() > 0)
                {
                    auto result = yolo.PostProc(outputs);
                    DisplayBoundingBox(frame[idx], result, yoloParam.height, yoloParam.width, "", "",
                        cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                }
                fps = 1000000. / ie.latency();
                ostringstream oss;
                oss << setprecision(2) << fixed << fps;
                string text = "FPS:" + oss.str();
                auto textSize = cv::getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 1, &textBaseline);
                cv::rectangle(
                    frame[idx],
                    Point(25, 25 - textSize.height),
                    Point(25 + textSize.width, 35),
                    Scalar(172, 81, 99),
                    cv::FILLED);
                cv::putText(
                    frame[idx], text, Point(30, 30),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
                profiler.End("post");
                profiler.Start("writer");
                writer << frame[idx];
                profiler.End("writer");
                (++idx) %= FRAME_BUFFERS;
            }
        }
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
                DisplayBoundingBox(frame, result, yoloParam.height, yoloParam.width, \
                    "", "", cv::Scalar(0, 0, 255), objectColors, "result.jpg", 0, -1, true);
            }
            cnt++;
        } while(loops<0?1:(cnt<loops));
    }
    if(!simFile.empty())
    {
        // simulation for concated single output tensor
        vector<float> buf(ie.output_size()/sizeof(float), 0);
        dxrt::DataFromFile(simFile, buf.data());
        profiler.Start("post");
        auto result = yolo.PostProc(buf.data());
        profiler.End("post");
    }
    if(ispInput)
    {
        signal(SIGINT, RequestToStop);   
        {
#if __riscv
            // struct npu_dma_ioctl_data_copy *npu_cdma_data = (struct npu_dma_ioctl_data_copy *)malloc(sizeof(struct npu_dma_ioctl_data_copy));
            UYV_PARAMS_S gUyvParams;
            UYV_THR_PARAM_S gUyvThrParam[MAX_NPU_CH];
            THR_HNDL_S guyvhndl[MAX_NPU_CH];
            auto device = dxrt::CheckDevices().front();
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

            SetUYVResolution(yoloParam.width,yoloParam.height);

            InitOSD();

            UYV_DATA_S *pUyvData = NULL;
            pUyvData = (UYV_DATA_S *)malloc(sizeof(UYV_DATA_S));
            memset(pUyvData,0,sizeof(UYV_DATA_S));
            gUyvThrParam[0].nCh = 0;
            gUyvThrParam[0].bThrRunning = TRUE;
            gUyvThrParam[0].pUyvData =pUyvData ;

            // UYV Thread
            if(ENX_UTIL_thrCreate(&guyvhndl[0], get_uyv_thread, 1, 0, (void *)&gUyvThrParam[0]) < 0) {
                return -1;
            }
    #ifdef ISP_DEBUG_BY_OPENCV        
            cv::Mat frame = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3);
    #endif

    #ifdef ISP_INPUT_DMA_COPY
                cout << "use dma copy" << endl;
    #endif
            auto vInputs = ie.inputs(0);
            vector<uint64_t> inputAddr;
            for(auto &inputs:ie.inputs(0))
            {                
                inputAddr.push_back(inputs.front().phy_addr());                
            }
            for(auto &addr:inputAddr)
                cout << hex << "input addr: " << addr << dec << endl;
            uint64_t inputSize = ie.input_size();
            int cdma_cnt = 0;
            int current_uyv_cnt = 0;
            int profiler_total_flag = 1;
            int cnt = 0;
            int req = -1;
            int callBackCnt = 0;
            int bufIdx = 0;
            std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
                [&](std::vector<shared_ptr<dxrt::Tensor>> outputs, void* arg)
                {
                    float fps;
                    callBackCnt++;
                    profiler.Start("post");
    #ifdef ISP_DEBUG_BY_OPENCV
                    static int cnt = 0;
                    cv::Mat frame(yoloParam.height, yoloParam.width, CV_8UC3, inputs.front()->GetData());
                    // cv::imwrite("isp"+((loops>0)?to_string(callBackCnt):to_string(0))+".jpg", frame);
    #endif
                    auto result = yolo.PostProc(outputs);
                    profiler.Start("osd");
                    SendResultOSD(nputime, OSDstr, yoloParam, result);
                    profiler.End("osd");
    #ifdef ISP_DEBUG_BY_OPENCV
                    if(callBackCnt<20)
                    {
                        DisplayBoundingBox(frame, result, yoloParam.height, yoloParam.width, \
                            "", "", cv::Scalar(0, 0, 255), objectColors, "isp"+((loops>0)?to_string(callBackCnt):to_string(0))+".jpg", 0, -1, true);
                    }
    #endif
                    profiler.End("post");                    
                    return 0;
                };
            ie.RegisterCallBack(postProcCallBack);
            do {
                if(GetStopFlag()) {
                    gUyvThrParam[0].bThrRunning = FALSE;
                    if(guyvhndl[0].hndl)	ENX_UTIL_thrJoin(&guyvhndl[0]);
                    break;
                }
                if (profiler_total_flag) profiler.Start("total");
                int measure = 0;
                pthread_mutex_lock(&lock_uyv);
                if (current_uyv_cnt == uyv_cnt) {
                    profiler_total_flag = 0;
                    pthread_mutex_unlock(&lock_uyv);
                    sleep(0);
                    continue;
                }
                profiler_total_flag = 1;
                current_uyv_cnt = uyv_cnt;
                pthread_mutex_unlock(&lock_uyv);

    #ifdef ISP_INPUT_DMA_COPY
                profiler.Start("cdma");
                bufIdx = cnt%2;
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
                        break;
                    }
                    continue;
                }
                // cv::Mat frame(yoloParam.height, yoloParam.width, CV_8UC3, (UYV_DATA_S *)(pUyvData->pVirtAddr)); // to check input frame
                // cv::imwrite("isp.jpg", frame); // to check input frame
                // exit(1);
                profiler.End("cdma");
                profiler.Start("main");
                auto outputs = ie.Run(vInputs[bufIdx].front().data());
                profiler.End("main");
    #ifdef ISP_DEBUG_BY_OPENCV
                memcpy(frame.data, (UYV_DATA_S *)(pUyvData->pVirtAddr), ie.input_size());
    #endif
    #else
                profiler.Start("main");
                req = ie.RunAsync((UYV_DATA_S *)(pUyvData->pVirtAddr));
                profiler.End("main");
    #ifdef ISP_DEBUG_BY_OPENCV
                memcpy(frame.data, (UYV_DATA_S *)(pUyvData->pVirtAddr), ie.input_size());
    #endif
    #endif
                cnt++;
                profiler.End("total");                
                nputime[NPU_INFERENCE] = ie.inference_time();
            // } while(loops<0?1:(cnt<loops));
            } while(1);

            if (gUyvThrParam[0].bThrRunning == TRUE){
                gUyvThrParam[0].bThrRunning = FALSE;
                if(guyvhndl[0].hndl)	ENX_UTIL_thrJoin(&guyvhndl[0]);
            }

            usleep(1000*1000);

            if(ENX_UYV_CAPTURE_CH_Stop(0) != 0){
                printf("ENX_UYV_CAPTURE_CH_Stop failed\n");
            }

            ENX_UYV_CAPTURE_Exit();
            ENX_VSYS_Exit();

            // if(npu_cdma_data){
            //     free(npu_cdma_data);
            // }
            DeinitOSD();
#endif
        }
    }

    cout << ie.name() << " : latency " << ie.latency() << "us, " << ie.inference_time() << "us" << endl;
DXRT_TRY_CATCH_END
    return 0;
}
