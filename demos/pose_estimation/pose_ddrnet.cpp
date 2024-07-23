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
#define DISPLAY_WINDOW_NAME "YOLO Pose + DDRNet"
#define INPUT_CAPTURE_PERIOD_MS 30
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
#define FRAME_BUFFERS 5
#define SEG_INPUT_WIDTH 768
#define SEG_INPUT_HEIGHT 384

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif

// pre/post parameter table
extern YoloParam yolov5s6_pose_640, yolov5s6_pose_1280;
struct SegmentationParam
{
    int classIndex;
    string className;
    uint8_t colorB;
    uint8_t colorG;
    uint8_t colorR;
};
SegmentationParam segCfg[] = {
    {0, "background", 0, 0, 0, },
    {1, "foot", 0, 128, 0, },
    {2, "body", 0, 0, 128, },
};

/////////////////////////////////////////////////////////////////////////////////////////////////
const char* usage =
    "pose estimation with ddrnet demo\n"
    "[*]  -m0, --posemodel          yolo pose model file path\n"
    "[*]  -m1, --segmodel           yolo segmentation model file path\n"
    "     -i,  --image             use image file input\n"
    "     -v, --video     use video file input\n"
    "     -w, --write     write result frames to a video file\n"
    "     -c, --camera    use camera input\n"
    "     -a, --async     asynchronous inference\n"
    "     -h, --help      show help\n";

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
    int i = 1;
    string pose_model_path="", seg_model_path="", imgFile="", videoFile="", binFile="", simFile="", videoOutFile="";        
    bool cameraInput = false, asyncInference = false;
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
        if (arg == "-m0")
                                pose_model_path = strdup(argv[i++]);
        else if (arg == "-m1")
                                seg_model_path = strdup(argv[i++]);
        else if (arg == "-i")
                                imgFile = strdup(argv[i++]);
        else if (arg == "-v")
                                videoFile = strdup(argv[i++]);
        else if (arg == "-w")
                                videoOutFile = strdup(argv[i++]);
        else if (arg == "-c")
                                cameraInput = true;
        else if (arg == "-a")
                                asyncInference = true;
        else if (arg == "-h")
                                help(), exit(0);
        else
                                help(), exit(0);
    }
    if (pose_model_path.empty() || seg_model_path.empty())
    {
        help(), exit(0);
    }
    if (imgFile.empty()&&videoFile.empty()&&!cameraInput)
    {
        help(), exit(0);
    }
    
    LOG_VALUE(pose_model_path);
    LOG_VALUE(seg_model_path);
    LOG_VALUE(videoFile);
    LOG_VALUE(imgFile);
    LOG_VALUE(cameraInput);

    string captionModel = "YOLOv5 pose + DDRNet , 30fps";
    dxrt::InferenceEngine pose(pose_model_path);
    dxrt::InferenceEngine seg(seg_model_path);
    auto yoloParam = yolov5s6_pose_640;
    Yolo yolo = Yolo(yoloParam);
    if(pose.outputs().front().type() == dxrt::DataType::POSE)
        yolo.LayerInverse();
    auto& profiler = dxrt::Profiler::GetInstance();
    cv::Mat frame[FRAME_BUFFERS];
    cv::Mat poseInput[FRAME_BUFFERS];
    cv::Mat segInput[FRAME_BUFFERS];
    cv::Mat segFrame[FRAME_BUFFERS];
    for(int i=0;i<FRAME_BUFFERS;i++)
    {
        poseInput[i] = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3, cv::Scalar(0, 0, 0));
        segInput[i] = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        segFrame[i] = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    if(!imgFile.empty())
    {
        vector<shared_ptr<dxrt::Tensor>> poseOutputs, segOutputs;
        frame[0] = cv::imread(imgFile, IMREAD_COLOR);
        profiler.Start("pre");
        PreProc(frame[0], poseInput[0], true, true, 114);
        PreProc(frame[0], segInput[0], false);
        profiler.End("pre");
        if(!asyncInference)
        {
            profiler.Start("main");
            poseOutputs = pose.Run(poseInput[0].data);
            segOutputs = seg.Run(segInput[0].data);
            profiler.End("main");
        }
        else
        {
            vector<uint64_t> tmp = {};
            profiler.Start("main");            
            int poseReq = pose.RunAsync(poseInput[0].data);
            int segReq = seg.RunAsync(segInput[0].data);
            poseOutputs = pose.Wait(poseReq);
            segOutputs = seg.Wait(segReq);
            profiler.End("main");
        }
        profiler.Start("post-pose");
        auto poseResults = yolo.PostProc(poseOutputs);
        profiler.End("post-pose");
        DisplayBoundingBox(frame[0], poseResults, yoloParam.height, yoloParam.width, \
            "", "", cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
        yolo.ShowResult();
        profiler.Start("post-seg");
        segFrame[0].setTo(cv::Scalar(0,0,0));
        Segmentation((uint16_t*)segOutputs[0]->data(), segFrame[0].data, segFrame[0].rows, segFrame[0].cols, segCfg, 3);
        cv::Mat add;
        cv::resize(segFrame[0], add, Size(frame[0].cols, frame[0].rows), 0, 0, cv::INTER_LINEAR);
        cv::addWeighted(frame[0], 1.0, add, 1.0, 0.0, frame[0]);
        profiler.End("post-seg");
        cv::imwrite("result.jpg", frame[0]);
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
        int inIdx = 0, key;
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
        if(!videoOutFile.empty()){
            writer.open(videoOutFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                        cv::Size(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT), true);
            if(!writer.isOpened()){
                cout << "Error: writer could not be opened." << endl;
            }
        }
        int fps = cap.get(CAP_PROP_FPS);
        float capInterval = 1000./fps;
        
        cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        namedWindow(DISPLAY_WINDOW_NAME);
        moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
        queue<pair<vector<BoundingBox>, int>> poseQueue;
        vector<BoundingBox> bboxes;
        mutex poseLock;
        std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> posePostProc = \
            [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
            {
                profiler.Start("post-pose");
                auto result = yolo.PostProc(outputs);
                poseLock.lock();
                poseQueue.push(
                    make_pair(result,(uint64_t) arg)
                );
                poseLock.unlock();
                profiler.End("post-pose");
                return 0;
            };
        pose.RegisterCallBack(posePostProc);
        queue<int> segQueue;
        mutex segLock;
        std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> segPostProc = \
            [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
            {
                profiler.Start("post-seg");
                segFrame[(uint64_t)arg].setTo(cv::Scalar(0,0,0));
                Segmentation((uint16_t*)outputs[0]->data(), segFrame[(uint64_t)arg].data, segFrame[(uint64_t)arg].rows, segFrame[(uint64_t)arg].cols, segCfg, 3);
                segLock.lock();
                segQueue.push((uint64_t)arg);
                segLock.unlock();
                profiler.End("post-seg");                
                return 0;
            };
        seg.RegisterCallBack(segPostProc);
        Mat outFrame, add;
        int poseIdx, segIdx;
DemoLoop:
        cap.set(CAP_PROP_POS_FRAMES, 0);
        inIdx = 0;
        while(1)
        {
            tm.reset();
            tm.start();
            if(!pause)
            {
                cap >> frame[inIdx];                    
                if(frame[inIdx].empty()) break;
                profiler.Start("pre");
                PreProc(frame[inIdx], poseInput[inIdx], true, true, 114);
                PreProc(frame[inIdx], segInput[inIdx], false);
                profiler.End("pre");
                profiler.Start("main");
                int poseReq = pose.RunAsync(poseInput[inIdx].data, (void*)(intptr_t)inIdx);
                int segReq = seg.RunAsync(segInput[inIdx].data, (void*)(intptr_t)inIdx);
                UNUSEDVAR(poseReq);
                UNUSEDVAR(segReq);
                profiler.End("main");
                profiler.Start("display");
                poseLock.lock();
                if(!poseQueue.empty())
                {
                    bboxes = poseQueue.front().first;
                    poseIdx = poseQueue.front().second;
                    poseQueue.pop();                    
                }
                else
                {
                    poseIdx = inIdx;
                }                
                poseLock.unlock();
                segLock.lock();
                if(!segQueue.empty())
                {
                    segIdx = segQueue.front();
                    segQueue.pop();
                }
                else
                {
                    segIdx = inIdx;
                }
                segLock.unlock();
                outFrame = frame[poseIdx];
                DisplayBoundingBox(outFrame, bboxes, yoloParam.height, yoloParam.width, "", "",
                    cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                cv::resize(segFrame[segIdx], add, Size(outFrame.cols, outFrame.rows), 0, 0, cv::INTER_LINEAR);
                cv::addWeighted(outFrame, 1.0, add, 1.0, 0.0, outFrame);
                cv::rectangle(outFrame, Point(0, 0), Point(400, 40), Scalar(0, 0, 0), cv::FILLED);
                cv::putText(outFrame, captionModel, Point(20, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
                profiler.End("display");
                cv::imshow(DISPLAY_WINDOW_NAME, outFrame);
                if(!videoOutFile.empty()){
                    writer << outFrame;
                }
                (++inIdx)%=FRAME_BUFFERS;
            }
            tm.stop();
            double elapsed = tm.getTimeMilli();
            if (elapsed < capInterval)
            {
                key = cv::waitKey( max(1, (int)(capInterval - elapsed)) );
            }
            else
            {
                key = cv::waitKey(1);
            }
            if(key == 0x20) //'p'
            {
                pause = !pause;
            }
            else if(key == 0x1B) //'ESC'
            {
                RequestToStop(0);
                break;
            }
        }
        if(!GetStopFlag()) goto DemoLoop;
        sleep(1);
        profiler.Show();
        return 0;
    }

    return 0;
}