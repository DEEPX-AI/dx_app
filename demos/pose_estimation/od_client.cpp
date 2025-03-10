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
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>

#include <opencv2/opencv.hpp>
#include "display.h"
#include "dxrt/dxrt_api.h"
#include "socket.h"
#include <future>

using namespace std;
using namespace cv;

#define CAMERA_FRAME_WIDTH 800
#define CAMERA_FRAME_HEIGHT 600
#define INPUT_CAPTURE_PERIOD_MS 33
#define FRAME_BUFFERS 8

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
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
    int loops = 1;
    int height = 0, width = 0, inputSize = 0, timeIntervalMs = 30;
    string imgFile="", videoFile="", binFile="", ethernetServerIP="";
    bool cameraInput = false, socketInput = false, asyncInference = false;
    ssize_t bytes, bytes_recv, bytes_send;
    auto objectColors = GetObjectColors();

        
    std::string app_name = "yolo pose object detection client demo";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
    ("i, image", "use image file input", cxxopts::value<std::string>(imgFile))
    ("v, video", "use video file input", cxxopts::value<std::string>(videoFile))
    ("c, camera", "use camera input", cxxopts::value<bool>()->default_value("false"))
    ("b, bin", "use binary file input", cxxopts::value<std::string>(binFile))
    ("a, async", "asynchronous inference", cxxopts::value<bool>()->default_value("false"))
    ("l, loop", "loop test", cxxopts::value<int>(loops)->default_value("1"))
    ("x, width", "input image width of model", cxxopts::value<int>(width))
    ("y, height", "input image height of model", cxxopts::value<int>(height))
    ("p, ip", "server IP", cxxopts::value<std::string>(ethernetServerIP))
    ("t, time", "request time interval (ms)", cxxopts::value<int>(timeIntervalMs))
    ("h, help", "print usage")
    ;
    
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help") || ethernetServerIP.empty())
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    
    LOG_VALUE(videoFile);
    LOG_VALUE(imgFile);
    LOG_VALUE(binFile);
    LOG_VALUE(cameraInput);
    LOG_VALUE(asyncInference);    
    LOG_VALUE(ethernetServerIP);

    /* Socket for Client */
    int sock;
    struct sockaddr_in serv_addr;    
    sock = socket(PF_INET, SOCK_STREAM, 0);
    DXRT_ASSERT(sock!=-1, "socket() error");
    int opt_val = 1;
    int opt_len = sizeof(int);
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (void*)&opt_val, opt_len);
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(ethernetServerIP.c_str());
    serv_addr.sin_port = htons(atoi("8080"));
    DXRT_ASSERT(connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr))!=-1, "connect() error");
    recv(sock, &height, sizeof(int), MSG_WAITALL);
    recv(sock, &width, sizeof(int), MSG_WAITALL);
    inputSize = 3*height*width;
    cout << "Connected to server " << ethernetServerIP 
        << " : height " << height << ", width " << width
        << " input size " << inputSize << "Bytes" << endl;
    DXRT_ASSERT(height>0, "invalid height");
    DXRT_ASSERT(width>0, "invalid width");
    DXRT_ASSERT(inputSize>0, "invalid input size");

    auto& profiler = dxrt::Profiler::GetInstance();
    function<vector<BoundingBox>(BoundingBoxPacket_t *)> ExtractBboxes = \
        [&](BoundingBoxPacket_t *packet)
        {
            BoundingBox *bboxPtr = packet->bboxes;
            vector<BoundingBox> bboxes;
            while(1)
            {
                if(bboxPtr->label<0) break;
                bboxes.emplace_back(*bboxPtr);
                bboxPtr++;
            }
            return bboxes;
        };
    if(!imgFile.empty())
    {
        ssize_t bytes;
        BoundingBoxPacket_t bboxPacket;
        vector<shared_ptr<dxrt::Tensor>> outputs;
        cv::Mat frame = cv::imread(imgFile, IMREAD_COLOR);
        profiler.Start("pre");
        cv::Mat resizedFrame = cv::Mat(height, width, CV_8UC3);
        PreProc(frame, resizedFrame, true, true, 114);
        profiler.End("pre");
        cv::imwrite("resized.jpg", resizedFrame);
        profiler.Start("main");
            profiler.Start("send");
            int id = 0;
            bytes = send(sock, &id, sizeof(int), 0);
            bytes = send(sock, resizedFrame.data, inputSize, 0);
            DXRT_ASSERT(bytes==inputSize, "send failed");
            profiler.End("send");
            cout << "Sent " << bytes << " bytes." << endl;
            profiler.Start("recv");
            bytes = recv(sock, (void*)&bboxPacket, sizeof(BoundingBoxPacket_t), MSG_WAITALL);
            DXRT_ASSERT(bytes==sizeof(BoundingBoxPacket_t), "recv failed");
            profiler.End("recv");
            cout << "Received " << bytes << " bytes." << endl;
        profiler.End("main");
        vector<BoundingBox> bboxes = ExtractBboxes(&bboxPacket);
        for(auto &bbox:bboxes) bbox.Show();
        DisplayBoundingBox(frame, bboxes, height, width, \
            "", "", cv::Scalar(0, 0, 255), objectColors, "result.jpg", 0, -1, true);
        
        return 0;
    }
    else if(!videoFile.empty() || cameraInput)
    {
        cv::VideoCapture cap;
        signal(SIGINT, RequestToStop);
        uint64_t tSend;
        BoundingBoxPacket_t bboxPacket[FRAME_BUFFERS];
        vector<BoundingBox> curBboxes;
        int key, capture = 0, show = 0;
        ssize_t bytes, bytes_recv, bytes_send;
        atomic<int> curFrameId(0);
        queue<BoundingBoxPacket_t*> bboxesQueue;
        mutex lk;
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS];        
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
        cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
            frame[i] = cv::Mat(cap.get(CAP_PROP_FRAME_HEIGHT), cap.get(CAP_PROP_FRAME_WIDTH), CV_8UC3, cv::Scalar(0, 0, 0));
        }
        std::function<int(int, int, int)> tune = \
            [&](int org, int max_val, int diff)
            {
                int ret = org - diff;
                if(ret<0) ret += max_val;
                return ret;
            };
        std::function<void(void)> recvLoop = \
            [&](void)
            {
                static int id = 0;
                while(1)
                {
                    if(GetStopFlag()) break;
                    profiler.Start("recv");
                    bytes_recv = recv(sock, (void*)&bboxPacket[id], sizeof(BoundingBoxPacket_t), MSG_WAITALL);
                    if(bytes_recv!=sizeof(BoundingBoxPacket_t)) stopFlag = true;
                    profiler.End("recv");
                    // cout << "recv: " << id << endl;
                    lk.lock();
                    bboxesQueue.push(&bboxPacket[id]);
                    if(bboxesQueue.size()>10) bboxesQueue.pop();
                    lk.unlock();
                    (++id)%=FRAME_BUFFERS;
                }
            };
        std::function<void(void)> sendLoop = \
            [&](void)
            {
                static int id = 0;
                static int loop = 0;                
                while(1)
                {
                    if(GetStopFlag()) break;
                    cout << "[" << loop << ", " << id << " ]" << endl;                    
                    profiler.Start("send");
                    bytes_send = send(sock, resizedFrame[curFrameId].data, inputSize, 0);
                    if(bytes_send!=inputSize) stopFlag = true;
                    profiler.End("send");
                    /* timing margin for next capture */
                    int64_t t = timeIntervalMs*1000 - profiler.Get("cap");
                    // LOG_VALUE(t);
                    if(t>0) usleep(t);
                    (++id)%=FRAME_BUFFERS;            
                    loop++;
                }
            };        
        thread {recvLoop}.detach();
        thread {sendLoop}.detach();        
        while(1)
        {
            if(GetStopFlag()) break;
            profiler.Start("main");
            cap >> frame[capture];
            if(frame[capture].empty()) break;
            profiler.Start("pre");
            PreProc(frame[capture], resizedFrame[capture], true, true, 114);
            profiler.End("pre");
            curFrameId = capture;
            show = tune(capture, FRAME_BUFFERS, 3);
            {
                BoundingBoxPacket_t *curPacket=nullptr;
                vector<BoundingBox> bboxes;
                lk.lock();
                if(!bboxesQueue.empty())
                {
                    curPacket = bboxesQueue.back();
                }
                lk.unlock();
                if(curPacket!=nullptr)
                {
                    bboxes = ExtractBboxes(curPacket);
                    DisplayBoundingBox(frame[show], bboxes, height, width, "", "",
                        cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                }
            }
            cv::imshow("OD", frame[show]);
            profiler.End("main");
            int64_t t = (INPUT_CAPTURE_PERIOD_MS*1000 - profiler.Get("main"))/1000;
            if(t>INPUT_CAPTURE_PERIOD_MS)
                t = INPUT_CAPTURE_PERIOD_MS;
            // LOG_VALUE(t);
            key = cv::waitKey(max((int64_t)1, t));
            if(key == 0x1B) //'ESC'
            {
                break;
            }
            (++capture)%=FRAME_BUFFERS;
        }
        
        return 0;
    }
    if(!binFile.empty())
    {
        return 0;
    }

    close(sock);

    return 0;
}