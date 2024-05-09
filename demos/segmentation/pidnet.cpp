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
#define DISPLAY_WINDOW_NAME "PIDNet"
#define INPUT_CAPTURE_PERIOD_MS 30
#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640
#define NUM_CLASSES 19
#define CAMERA_FRAME_WIDTH 800
#define CAMERA_FRAME_HEIGHT 600
#define FRAME_BUFFERS 10

struct SegmentationParam
{
    int classIndex;
    string className;
    uint8_t colorB;
    uint8_t colorG;
    uint8_t colorR;
};
SegmentationParam segCfg[] = {
    {	0	,	"road",	128	,	64	,	128	,	},
    {	1	,	"sidewalk",	244	,	35	,	232	,	},
    {	2	,	"building",	70	,	70	,	70	,	},
    {	3	,	"wall",	102	,	102	,	156	,	},
    {	4	,	"fence",	190	,	153	,	153	,	},
    {	5	,	"pole",	153	,	153	,	153	,	},
    {	6	,	"traffic light",	51	,	255	,	255	,	},
    {	7	,	"traffic sign",	220	,	220	,	0	,	},
    {	8	,	"vegetation",	107	,	142	,	35	,	},
    {	9	,	"terrain",	152	,	251	,	152	,	},
    {	10	,	"sky",	255	,	0	,	0	,	},
    {	11	,	"person",	0	,	51	,	255	,	},
    {	12	,	"rider",	255	,	0	,	0	,	},
    {	13	,	"car",	255	,	51	,	0	,	},
    {	14	,	"truck",	255	,	51	,	0	,	},
    {	15	,	"bus",	255	,	51	,	0	,	},
    {	16	,	"train",	0	,	80	,	100	,	},
    {	17	,	"motorcycle",	0	,	0	,	230	,	},
    {	18	,	"bicycle",	119	,	11	,	32	,	},
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

void Segmentation(float *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses)
{
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            int maxIdx = 0;
            for (int c=0;c<numClasses;c++){
                if(input[(cols*h + w)*64 + maxIdx] < input[(cols*h + w)*64 + c]){
                    maxIdx = c;
                }
            }
            output[3*cols*h + 3*w + 2] = cfg[maxIdx].colorB;
            output[3*cols*h + 3*w + 1] = cfg[maxIdx].colorG;
            output[3*cols*h + 3*w + 0] = cfg[maxIdx].colorR;
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

    if(inputWidth==0) inputWidth = MODEL_WIDTH;
    if(inputHeight==0) inputHeight = MODEL_HEIGHT;
    LOG_VALUE(inputWidth);
    LOG_VALUE(inputHeight);
    LOG_VALUE(modelPath);
    LOG_VALUE(videoFile);
    LOG_VALUE(binFile);
    LOG_VALUE(simFile);
    LOG_VALUE(cameraInput);
    LOG_VALUE(asyncInference);

    auto ie = dxrt::InferenceEngine(modelPath);

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
        Segmentation((float*)outputs[0]->data(), result.data, result.rows, result.cols, segCfg, NUM_CLASSES);
        profiler.End("post-segment");
        profiler.Start("post-blend");
        cv::resize(result, result, Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);
        cv::addWeighted( frame, 0.5, result, 0.5, 0.0, frame);
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
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS], result[FRAME_BUFFERS];
        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
            result[i] = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
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
                PreProc(frame[idx], resizedFrame[idx], PREPROC_KEEP_IMG_RATIO);
                profiler.End("pre");
                profiler.Start("main");
                auto outputs = ie.Run(resizedFrame[idx].data);
                profiler.End("main");
                if(outputs.size()>0)
                {                    
                    profiler.Start("post-segment");
                    cv::Mat resultExpand, outFrameBlend;
                    cv::Mat outFrame = frame[prevIdx];
                    Segmentation(
                        (float*)outputs[0]->data(), result[idx].data, 
                        result[idx].rows, result[idx].cols, segCfg, NUM_CLASSES);
                    profiler.End("post-segment");
                    profiler.Start("post-blend");
                    cv::resize(result[idx], resultExpand, Size(frame[idx].cols, frame[idx].rows), 0, 0, cv::INTER_LINEAR);
                    cv::addWeighted( outFrame, 0.5, resultExpand, 0.5, 0.0, outFrameBlend);
                    profiler.End("post-blend");
                    cv::imshow(DISPLAY_WINDOW_NAME, outFrameBlend);
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
        sleep(1);
        profiler.Show();
        return 0;
    }
    if(!binFile.empty())
    {
        uint8_t *input_buffer = new uint8_t(inputWidth * inputHeight * 3);
        dxrt::DataFromFile(binFile, input_buffer);
        auto outputs = ie.Run(input_buffer);
        if(outputs.size()==0)
        {
            cout << "Error. Invalid output detected." << endl;
            return -1;
        }
        return 0;
    }

    return 0;
}