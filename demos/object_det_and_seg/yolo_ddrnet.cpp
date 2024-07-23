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
#include <opencv2/core.hpp>

#include "display.h"
#include "dxrt/dxrt_api.h"
#include "yolo.h"
#include "segmentation.h"

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////////////////////////////////////////////
#define DISPLAY_WINDOW_NAME "OD + Seg."
#define INPUT_CAPTURE_PERIOD_MS 40
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
#define OD_INPUT_WIDTH 512
#define OD_INPUT_HEIGHT 512
#define SEG_INPUT_WIDTH 768
#define SEG_INPUT_HEIGHT 384

YoloLayerParam createYoloLayerParam(int _gx, int _gy, int _numB, const std::vector<float> &_vAnchorW, const std::vector<float> &_vAnchorH, const std::vector<int> &_vTensorIdx, float _sx = 0.f, float _sy = 0.f)
{
        YoloLayerParam s;
        s.numGridX = _gx;
        s.numGridY = _gy;
        s.numBoxes = _numB;
        s.anchorWidth = _vAnchorW;
        s.anchorHeight = _vAnchorH;
        s.tensorIdx = _vTensorIdx;
        s.scaleX = _sx;
        s.scaleY = _sy;
        return s;
}

YoloParam odCfg = {
    .height = 512,
    .width = 512,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
            createYoloLayerParam(64, 64, 3, { 10.0, 16.0, 33.0 }, { 13.0, 30.0, 23.0 }, { 0 }),
            createYoloLayerParam(32, 32, 3, { 30.0, 62.0, 59.0 }, { 61.0, 45.0, 119.0 }, { 1 }),
            createYoloLayerParam(16, 16, 3, { 116.0, 156.0, 373.0 }, { 90.0, 198.0, 326.0 }, { 2 })
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};

SegmentationParam segCfg[] = {
    {0, "background", 0, 0, 0, }, /* Skip */
    {1, "foot", 0, 128, 0, },
    {2, "body", 0, 0, 128, },
};

const char* usage =
    "Object Detection with Image Segmentation Demo\n"
    "[*]  -m0, --od_modelpath      object detection model include path (yolov5s_512)\n"
    "[*]  -m1, --seg_modelpath     segmentation model include path (ddrnet)\n"
    "     -i,  --image             use image file input\n"
    "     -v,  --video             use video file input\n"
    "     -c,  --camera            use camera input\n"
    "[*]  -a,  --async             asynchronous inference\n"
    "     -h,  --help              show help\n";
void help()
{
    cout << usage << endl;    
}

void *PreProc(cv::Mat& src, cv::Mat &dest, bool keepRatio=true, bool bgr2rgb=true, uint8_t padValue=0)
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

void Segmentation(uint16_t *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses)
{
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            int cls = input[cols*h + w];
            if(cls<numClasses)
            {
                output[3*cols*h + 3*w + 0] = cfg[cls].colorB;
                output[3*cols*h + 3*w + 1] = cfg[cls].colorG;
                output[3*cols*h + 3*w + 2] = cfg[cls].colorR;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int i = 1;
    string imgFile="", videoFile="", binFile="", simFile="";
    string od_modelpath = "", seg_modelpath = "";
    bool cameraInput = false;
    auto objectColors = GetObjectColors(0);

    if(argc==1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }

    while (i < argc) {
        std::string arg(argv[i++]);
        if (arg == "-m0")
                                od_modelpath = strdup(argv[i++]);
        else if (arg == "-m1")
                                seg_modelpath = strdup(argv[i++]);
        else if (arg == "-i")
                                imgFile = strdup(argv[i++]);
        else if (arg == "-v")
                                videoFile = strdup(argv[i++]);
        else if (arg == "-c")
                                cameraInput = true;
        else if (arg == "-h")
                                help(), exit(0);
        else
                                help(), exit(0);
    }
    if (od_modelpath.empty() || seg_modelpath.empty())
    {
        help(), exit(0);
    }
    if (imgFile.empty()&&videoFile.empty()&&!cameraInput)
    {
        help(), exit(0);
    }

    LOG_VALUE(od_modelpath);
    LOG_VALUE(seg_modelpath);
    LOG_VALUE(imgFile);
    LOG_VALUE(videoFile);
    LOG_VALUE(cameraInput);
    
    dxrt::InferenceEngine ieOD(od_modelpath);
    dxrt::InferenceEngine ieSEG(seg_modelpath);
    
    Yolo yolo = Yolo(odCfg);
    if(ieOD.outputs().front().type() == dxrt::DataType::BBOX)
        yolo.LayerInverse();

    auto& profiler = dxrt::Profiler::GetInstance();
    if(!imgFile.empty())
    {
        cv::Mat frame, segInput, odInput;
        /* Capture */
        frame = cv::imread(imgFile, IMREAD_COLOR);

        /* PreProcessing */
        profiler.Start("pre");
        odInput = cv::Mat(OD_INPUT_HEIGHT, OD_INPUT_WIDTH, CV_8UC3);
        segInput = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3);
        PreProc(frame, odInput, true, true, 114);
        PreProc(frame, segInput, false);
        profiler.End("pre");

        /* Main */
        profiler.Start("main");

        auto OdOutputTensors = ieOD.Run(odInput.data);
        auto SegOutputTensors = ieSEG.Run(segInput.data);

        profiler.End("main");

        /* PostProcessing : Object Detection */
        profiler.Start("post-obj.detection");
        auto OdResult = yolo.PostProc(OdOutputTensors);
        profiler.End("post-obj.detection");
        yolo.ShowResult();
        DisplayBoundingBox(frame, OdResult, odCfg.height, odCfg.width,
                "", "", cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);    
        
        /* PostProcessing : Segmentation */
        profiler.Start("post-segment");
        cv::Mat SegResult = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        Segmentation((uint16_t*)SegOutputTensors[0]->data(), SegResult.data, SegResult.rows, SegResult.cols, segCfg, 3);
        profiler.End("post-segment");

        /* PostProcessing : Blend Image */
        profiler.Start("post-blend");
        cv::Mat add;
        cv::resize(SegResult, add, Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);
        cv::addWeighted(frame, 1.0, add, 1.0, 0.0, frame);
        profiler.End("post-blend");
        cout << dec << SEG_INPUT_WIDTH << "x" << SEG_INPUT_HEIGHT << " <- " << frame.cols << "x" << frame.rows << endl;

        /* Save & Show */
        cv::imwrite("result.jpg", frame);
        std::cout << "save file : result.jpg " << std::endl;
        profiler.Show();
        return 0;
    }
    else if(!videoFile.empty() || cameraInput)
    {
        cv::VideoCapture cap;
        cv::Mat frame[5];
        cv::Mat odInput, segInput;
        int idx = 0, prevIdx = 0;
        odInput = cv::Mat(OD_INPUT_HEIGHT, OD_INPUT_WIDTH, CV_8UC3);
        segInput = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3);
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
        namedWindow(DISPLAY_WINDOW_NAME);
        moveWindow(DISPLAY_WINDOW_NAME, 0, 0);

        profiler.Start("cap");
        while (waitKey(INPUT_CAPTURE_PERIOD_MS)<0 && cap.isOpened()) {
            profiler.End("cap");
            profiler.Start("cap");
            /* Capture */
            cap >> frame[idx];
            if(frame[idx].empty()) break;

            /* PreProcessing */
            profiler.Start("pre");
            PreProc(frame[idx], odInput, true, true, 114);
            PreProc(frame[idx], segInput, false);
            profiler.End("pre");

            /* Main */        
            profiler.Start("main");

            auto OdOutputTensors = ieOD.Run(odInput.data);
            auto SegOutputTensors = ieSEG.Run(segInput.data);
            
            profiler.End("main");

            /* PostProcessing : Object Detection */
            profiler.Start("post-od");
            if(!OdOutputTensors.empty())
            {
                auto OdResult = yolo.PostProc(OdOutputTensors);
                DisplayBoundingBox(frame[prevIdx], OdResult, odCfg.height, odCfg.width,
                    "", "", cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);    
            }
            profiler.End("post-od");            
            
            /* PostProcessing : Segmentation, Blend Image */
            if(!SegOutputTensors.empty())
            {
                profiler.Start("post-segment");
                cv::Mat SegResult = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
                Segmentation((uint16_t*)SegOutputTensors[0]->data(), SegResult.data, SegResult.rows, SegResult.cols, segCfg, 3);
                profiler.End("post-segment");
                profiler.Start("post-blend");
                cv::resize(SegResult, SegResult, Size(frame[prevIdx].cols, frame[prevIdx].rows), 0, 0, cv::INTER_LINEAR);
                frame[prevIdx] = frame[prevIdx] + SegResult;
                profiler.End("post-blend");
            }

            /* Display */
            cv::imshow(DISPLAY_WINDOW_NAME, frame[prevIdx]);
            prevIdx = idx;
            (++idx)%=5;
        }
        profiler.Show();
        return 0;
    }

    return 0;
}