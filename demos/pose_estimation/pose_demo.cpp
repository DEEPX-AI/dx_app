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
#include "isp.h"
#include "v4l2.h"
#include "osd_eyenix.h"
#include "socket.h"

using namespace std;
#ifdef USE_OPENCV
using namespace cv;
#endif

#define ISP_PHY_ADDR   (0x9D000000)
#define ISP_INPUT_ZEROCOPY
// #define ISP_DEBUG_BY_RAWFILE
// #define ISP_DEBUG_BY_OPENCV
#define POSTPROC_SIMULATION_DEVICE_OUTPUT
#define DISPLAY_WINDOW_NAME "Pose Estimation"
#define INPUT_CAPTURE_PERIOD_MS 30
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
#define FRAME_BUFFERS 5

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
    { "sim", required_argument, 0, 's' },
    { "async", no_argument, 0, 'a' },
    { "ethernet", no_argument, 0, 'e' },
    { "param", required_argument, 0, 'p' },
    { "loop", no_argument, 0, 'l' },
    { "numbuf", required_argument, 0, 'n' },
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
"  -s, --sim       use pre-defined npu output binary file input( perform post-proc. only )\n"
"  -a, --async     asynchronous inference\n"
"  -e, --ethernet  use ethernet input\n"
"  -p, --param      pre/post-processing parameter selection\n"
"  -l, --loop      loop test\n"
"  -n, --numbuf    number of memory buffers for inference\n"
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
    int optCmd, loops = 1, paramIdx = 0, numBuf = 1;
    string modelPath="", imgFile="", videoFile="", binFile="", simFile="", videoOutFile="";        
    bool cameraInput = false, ispInput = false, ethernetInput = false,
        asyncInference = false, writeFrame = false;
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

    while ((optCmd = getopt_long(argc, argv, "m:i:v:w:cxb:s:aep:l:n:h", opts,
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
            case 'w':
                videoOutFile = strdup(optarg);
                writeFrame = true;
                break;
            case 'c':
                cameraInput = true;
                break;
            case 'x':
                ispInput = true;
                break;
            case 'b':
                binFile = strdup(optarg);
                break;
            case 's':
                simFile = strdup(optarg);
                break;
            case 'a':
                asyncInference = true;
                break;
            case 'e':
                ethernetInput = true;
                break;
            case 'p':
                paramIdx = stoi(optarg);
                break;
            case 'l':
                loops = stoi(optarg);
                break;
            case 'n':
                numBuf = stoi(optarg);
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
    LOG_VALUE(imgFile);
    LOG_VALUE(binFile);
    LOG_VALUE(simFile);
    LOG_VALUE(cameraInput);
    LOG_VALUE(ispInput);
    LOG_VALUE(asyncInference);

    if(modelPath.empty())
    {
        cout << "Error: no model argument." << endl;
        help();
        return -1;
    }

    auto ie = dxrt::InferenceEngine(modelPath);
    auto yoloParam = yoloParams[paramIdx];
    Yolo yolo = Yolo(yoloParam);
    auto& profiler = dxrt::Profiler::GetInstance();
#if USE_OPENCV
    if(!imgFile.empty())
    {
        vector<shared_ptr<dxrt::Tensor>> outputs;
        cv::Mat frame = cv::imread(imgFile, IMREAD_COLOR);
        profiler.Start("pre");
        cv::Mat resizedFrame = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3);
        PreProc(frame, resizedFrame, true, true, 114);
        profiler.End("pre");
        cv::imwrite("resized.jpg", resizedFrame);
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
        // cv::waitKey(0);
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

            int callBackCnt = 0; // debug
            queue<pair<vector<BoundingBox>, int>> bboxesQueue;
            vector<BoundingBox> bboxes;
            mutex lk;
            std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
                [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
                {
                    // callBackCnt++; // debug
                    profiler.Start("post");
                    // cout << "      >> callback " << callBackCnt << endl; // debug
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
                    // LOG_VALUE(profiler.Get("post"));
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
                    int reqId = ie.RunAsync(resizedFrame[idx].data, (void*)idx);
                    // ie.Wait(reqId);
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
                // LOG_VALUE(elapsed);
                if (elapsed < capInterval)
                {
                    // LOG_VALUE(capInterval-elapsed);
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
                    // if(cap.get(CAP_PROP_POS_FRAMES)>30*15) break; // temp
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
                    // fps = 1000000. / ie.GetNpuPerf(0);
                    fps = 1; // TODO
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
                    // cout << fps << endl; // debug
                    // cv::imwrite("tmp.jpg", frame[idx]); break; // debug 
                profiler.End("post");
                profiler.Start("writer");
                    writer << frame[idx];
                profiler.End("writer");
                (++idx)%=FRAME_BUFFERS;
            }
        }
        // ie.Show();
        profiler.Show();
        return 0;
    }
#endif
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
    if(!simFile.empty())
    {
// #ifdef POSTPROC_SIMULATION_DEVICE_OUTPUT
//         // simulation for device output tensors
//         auto outputs = ie.GetInput().front()->GetOutput();
//         dxrt::DataFromFile(simFile, outputs.front()->data(), ie.output_size());
//         profiler.Start("post");
//         auto result = yolo.PostProc(outputs);
//         profiler.End("post");
//         yolo.ShowResult();
//         profiler.Show();
// #else
//         // simulation for concated single output tensor
//         float *buf = new float[10*1024*1024/sizeof(float)];
//         dxrt::DataFromFile(simFile, buf);
//         profiler.Start("post");
//         auto result = yolo.PostProc(buf);
//         profiler.End("post");
//         delete buf;
// #endif
//         return 0;
    }
//     if(ispInput)
//     {
//         signal(SIGINT, RequestToStop);
//         if(dxrt::DeviceVariant()=="DX_L1")
//         {
//             DXRT_ASSERT(numBuf==1, "number of buffers should be set to 1 for DX-L1 ISP demo.");
//             InitOSD();
//             void *ispBufPtr = (void*)InitISPMapping(
//                 ie.GetFeatureSize(),
//                 ISP_PHY_ADDR
//             );
//             DXRT_ASSERT(ispBufPtr!=nullptr, "fail to init ISP");
//             int fd, cnt = 0, frameCnt, req;
//             int callBackCnt = -1;                        
//             std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, std::vector<std::shared_ptr<dxrt::Tensor>>)> postProcCallBack = \
//                 [&](std::vector<shared_ptr<dxrt::Tensor>> outputs, std::vector<shared_ptr<dxrt::Tensor>> inputs)
//                 {
//                     float fps;
//                     ostringstream oss;
//                     // callBackCnt++; // debug
//                     profiler.Start("post");
//                     // cout << "      >> callback " << callBackCnt << endl; // debug
// #ifdef ISP_DEBUG_BY_OPENCV
//                     static int cnt = 0;
//                     cv::Mat frame(yoloParam.height, yoloParam.width, CV_8UC3, inputs.front()->data());
//                     // cv::imwrite("isp"+((loops>0)?to_string(callBackCnt):to_string(0))+".jpg", frame);
// #endif
//                     auto result = yolo.PostProc(outputs);
//                     EyenixOSD(result, yoloParam.classNames, yoloParam.height, yoloParam.width);
//                     fps = 1000000. / ie.GetNpuPerf(0);
//                     oss << setprecision(2) << fixed << fps;
//                     string text = "FPS:" + oss.str();
//                     EyenixOSD_setString(2, 2, text.c_str() );
//                     // cout << ie.GetNpuPerf() << endl;
// #ifdef ISP_DEBUG_BY_OPENCV
//                     if(callBackCnt<20)
//                     {
//                         DisplayBoundingBox(frame, result, yoloParam.height, yoloParam.width, \
//                             "", "", cv::Scalar(0, 0, 255), objectColors, "isp"+((loops>0)?to_string(callBackCnt):to_string(0))+".jpg", 0, -1, true);
//                     }
// #endif
//                     profiler.End("post");
//                     return 0;
//                 };
//             ie.RegisterCallBack(postProcCallBack);
//             do {
//                 if(GetStopFlag()) break;
//                 profiler.Start("main");
//                 ie.Run(ISP_PHY_ADDR, ispBufPtr);
//                 if(asyncInference) ie.Wait(); /* need for asynchronous inference */
//                 profiler.End("main");
//                 ++cnt;
//             } while(loops<0?1:(cnt<loops));
//             usleep(1000000);
//             profiler.Show();
//             DeinitISPMapping(ie.GetFeatureSize());
//             DeinitOSD();
//         }
//         else if(dxrt::DeviceVariant()=="DX_L2")
//         {
//             DXRT_ASSERT(numBuf==4, "number of buffers should be set to 4 for DX-L2 ISP demo.");
//             int fd, cnt = 0, frameCnt, ret;
//             int callBackCnt = -1;
// #ifdef ISP_DEBUG_BY_OPENCV
//             vector<uint8_t *> ispDebugBuffers;
//             for(int i=0;i<20;i++)
//             {
//                 ispDebugBuffers.emplace_back( new uint8_t[ie.input_size()] );
//             }
// #endif
//             std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, std::vector<std::shared_ptr<dxrt::Tensor>>)> postProcCallBack = \
//                 [&](std::vector<shared_ptr<dxrt::Tensor>> outputs, std::vector<shared_ptr<dxrt::Tensor>> inputs)
//                 {
//                     callBackCnt++; // debug
//                     profiler.Start("post");
//                     // cout << "      >> callback " << callBackCnt << endl; // debug
//                     auto result = yolo.PostProc(outputs);
//                     /* TODO : OSD should be implemented. */
//                     // std::cout << "[" << callBackCnt << "] " << dec << result.size() << " boxes." << std::endl;
//                     // for(int i=0;i<(int)result.size();i++)
//                     // {
//                     //     result[i].Show();
//                     // }
// #ifdef ISP_DEBUG_BY_OPENCV
//                     if(callBackCnt<20)
//                     {
//                         memcpy(ispDebugBuffers[callBackCnt], inputs.front()->data(), ie.input_size());
//                         cv::Mat frame(yoloParam.height, yoloParam.width, CV_8UC3, ispDebugBuffers[callBackCnt]);
//                         DisplayBoundingBox(frame, result, yoloParam.height, yoloParam.width, \
//                             "", "", cv::Scalar(0, 0, 255), objectColors, "isp"+((loops>0)?to_string(callBackCnt):to_string(0))+".jpg", 0, -1, true);
//                     }
// #endif
//                     profiler.End("post");
//                     // LOG_VALUE(profiler.Get("post"));
//                     return 0;
//                 };
//             ie.RegisterCallBack(postProcCallBack);
//             for(int i=0;i<numBuf;i++)
//             {
//                 inputPtr.emplace_back((unsigned long)ie.GetInputPtr(i));
//             }
//             V4L2CaptureWorker captureWorker("/dev/video11", yoloParam.height, yoloParam.width, numBuf, inputPtr);
//             do {
//                 if(GetStopFlag()) break;
//                 frameCnt = captureWorker.GetFrameId();
//                 if(frameCnt<0)
//                 {
//                     continue;
//                 }
//                 // cout << "  >> frame " << frameCnt << ", " << cnt << " ==" << endl; // debug
//                 profiler.Start("main");
//                 ie.Run(frameCnt);
//                 ie.Wait();                
// #ifdef ISP_DEBUG_BY_OPENCV
//                 usleep(500000); /* timing margin for isp debug */
// #endif
//                 profiler.End("main");
//                 ++cnt;
//             } while(loops<0?1:(cnt<loops));
//             captureWorker.Stop();
//             usleep(1000000);
//             profiler.Show();
//         }
//     }
    // if(ethernetInput)
    // {
    //     signal(SIGINT, RequestToStop);
    //     int sock, ret, idx, cnt = 0;
    //     int inputSize = ie.input_size();
    //     BoundingBoxPacket_t bboxPacket;
    //     atomic<bool> stopRecv(false);
    //     ssize_t bytes, bytes_recv, bytes_send;

    //     /* Socket for Server */
    //     int server_socket, client_socket;
    //     struct sockaddr_in server_addr, client_addr;        
    //     socklen_t client_addr_size;
    //     server_socket = socket(AF_INET, SOCK_STREAM, 0);
    //     DXRT_ASSERT(server_socket!=-1, "socket() error");
    //     int opt_val = 1;
    //     int opt_len = sizeof(int);
    //     setsockopt(server_socket, IPPROTO_TCP, TCP_NODELAY, (void*)&opt_val, opt_len);
    //     setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (void*)&opt_val, opt_len);
    //     memset(&server_addr, 0, sizeof(server_addr));
    //     server_addr.sin_family = AF_INET;
    //     server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    //     server_addr.sin_port = htons(atoi("8080"));
    //     DXRT_ASSERT(
    //         bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr))!=-1,
    //         "bind() error"
    //     );
    //     DXRT_ASSERT(listen(server_socket, 5)!=-1, "listen() error");        

    //     for(int i=0;i<numBuf;i++)
    //     {
    //         inputPtr.emplace_back((unsigned long)ie.GetInputPtr(i));
    //     }
    //     std::function<void(size_t)> recvFunc = \
    //         [&](size_t size)
    //         {
    //             static int frameId = 0; // internal frame id
    //             profiler.Start("recv");
    //             bytes_recv = recv(client_socket, (void*)inputPtr[frameId], size, MSG_WAITALL);
    //             if(bytes_recv!=size) stopRecv = true;
    //             cout << "[" << cnt << ", " << frameId << " ]" << endl;
    //             ie.Run(frameId);
    //             profiler.End("recv");                
    //             (++frameId)%=numBuf;
    //         };
    //     std::function<int(vector<shared_ptr<dxrt::Tensor>>, vector<shared_ptr<dxrt::Tensor>>)> postProcCallBack = \
    //         [&](vector<shared_ptr<dxrt::Tensor>> outputs, vector<shared_ptr<dxrt::Tensor>> inputs)
    //         {
    //             profiler.Start("post");
    //                 auto result = yolo.PostProc(outputs, (void*)bboxPacket.bboxes);
    //                 bboxPacket.frameId = inputs.front()->GetBufId();
    //                 send(client_socket, (void*)&bboxPacket, sizeof(BoundingBoxPacket_t), 0);
    //             profiler.End("post");
    //             return 0;
    //         };
    //     ie.RegisterCallBack(postProcCallBack);

    //     while(1)
    //     {
    //         if(GetStopFlag()) break;
    //         cout << "==== Standby." << endl;
    //         client_addr_size = sizeof(client_addr);
    //         client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_addr_size);
    //         DXRT_ASSERT(client_socket!=-1, "accept() error.");
    //         send(client_socket, &yoloParam.height, sizeof(int), 0);
    //         send(client_socket, &yoloParam.width, sizeof(int), 0);
    //         stopRecv = false;
    //         while(1)
    //         {
    //             if(stopRecv) break;
    //             profiler.Start("cap");
    //             recvFunc(inputSize);
    //             profiler.End("cap");
    //             cnt++;
    //         }
    //         close(client_socket);
    //         cout << "==== Closed connection from client." << endl;
    //         // break; // temp.
    //     }
        
    //     profiler.Show();
    //     usleep(1000*1000);
    //     close(server_socket);
    // }

    return 0;
}