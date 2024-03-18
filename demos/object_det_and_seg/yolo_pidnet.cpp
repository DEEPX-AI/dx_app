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
#include "yolo.h"
#include "segmentation.h"

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////////////////////////////////////////////
#define DISPLAY_WINDOW_NAME "YOLO + PIDNet"
#define INPUT_CAPTURE_PERIOD_MS 30
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
#define SEG_NUM_CLASSES 19
#define OD_INPUT_WIDTH 512
#define OD_INPUT_HEIGHT 512
#define SEG_INPUT_WIDTH 1024
#define SEG_INPUT_HEIGHT 512
#define NUM_VIDEO_FILES 3
#define FRAME_BUFFERS 10

string videoFiles[NUM_VIDEO_FILES] = {
    "stuttgart-00.avi",
    "stuttgart-01.avi",
    "stuttgart-02.avi",
};

YoloParam odCfg = {
    .image_size = OD_INPUT_WIDTH,
    .conf_threshold = 0.25,
    .score_threshold = 0.3,
    .iou_threshold = 0.4, 
    .num_classes = 80,
    .num_layers = 3,
    .anchorBoxes = {
        {
            .num_grid_x = 64,
            .num_grid_y = 64,
            .width = { 10.0, 16.0, 33.0 },
            .height = { 13.0, 30.0, 23.0 },
            .num_boxes = 3,
        },
        {
            .num_grid_x = 32,
            .num_grid_y = 32,
            .width = { 30.0, 62.0, 59.0 },
            .height = { 61.0, 45.0, 119.0 },
            .num_boxes = 3,
        },
        {
            .num_grid_x = 16,
            .num_grid_y = 16,
            .width = { 116.0, 156.0, 373.0 },
            .height = { 90.0, 198.0, 326.0 },
            .num_boxes = 3,
        },
    },
    .class_names = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
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
enum TASK
{
    TASK_OD = 0,
    TASK_SEG = 1,
    TASK_MAX = 2,
};
/////////////////////////////////////////////////////////////////////////////////////////////////

const char* usage =
    "Object Detection with Image Segmentation Demo\n"
    "[*]  -m0, --od_modelpath      object detection model include path (yolov5s_512)\n"
    "[*]  -m1, --seg_modelpath     segmentation model include path (pidnet)\n"
    "     -i,  --image             use image file input\n"
    "     -v,  --video             use video file input\n"
    "     -c,  --camera            use camera input\n"
    "     -h,  --help              show help\n";

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
void Segmentation(uint16_t *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses)
{
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            int cls = input[cols*h + w];
            if(cls<numClasses)
            {
                output[3*cols*h + 3*w + 2] = cfg[cls].colorB;
                output[3*cols*h + 3*w + 1] = cfg[cls].colorG;
                output[3*cols*h + 3*w + 0] = cfg[cls].colorR;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int i = 1;
    int optCmd;
    int inputWidth = 0, inputHeight = 0;
    string imgFile="", videoFile="", binFile="", simFile="";
    string od_modelpath = "", seg_modelpath = "";
    bool pcieInput = false, cameraInput = false;
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

    vector<vector<int>> deviceNumbers;
    vector<dxrt::InferenceOption> options;
    vector<dxrt::InferenceEngine*> ie;

    dxrt::InferenceEngine ieOD(od_modelpath);
    dxrt::InferenceEngine ieSEG(seg_modelpath);
    auto dataInfo = ieOD.outputs();
    Yolo yolo = Yolo(odCfg, dataInfo);

    auto& profiler = dxrt::Profiler::GetInstance();
    cv::VideoCapture caps[NUM_VIDEO_FILES];
    if(videoFile=="0")
    {
        for(int i=0;i<NUM_VIDEO_FILES;i++)
        {
            caps[i].open(videoFiles[i]);
            if(!caps[i].isOpened())
            {
                cout << "Error: file " << videoFiles[i] << " could not be opened." <<endl;
                return -1;
            }
        }        
    }
    if(!imgFile.empty())
    {
        cv::Mat frame;
        cv::Mat resizedFrame[TASK_MAX];
        /* Capture */
        frame = cv::imread(imgFile, IMREAD_COLOR);

        /* PreProcessing */
        profiler.Start("pre");
        resizedFrame[TASK_OD] = cv::Mat(OD_INPUT_HEIGHT, OD_INPUT_WIDTH, CV_8UC3);
        resizedFrame[TASK_SEG] = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3);
        PreProc(frame, resizedFrame[TASK_OD], true, true, 114);
        PreProc(frame, resizedFrame[TASK_SEG], false);
        profiler.End("pre");

        /* Main */
        profiler.Start("main");
        auto OdOutputTensors = ieOD.Run(resizedFrame[TASK_OD].data);
        auto SegOutputTensors = ieSEG.Run(resizedFrame[TASK_SEG].data);
        profiler.End("main");

        /* PostProcessing : Object Detection */
        profiler.Start("post-obj.detection");
        auto OdResult = yolo.PostProc(OdOutputTensors);
        profiler.End("post-obj.detection");
        yolo.ShowResult();
        DisplayBoundingBox(frame, OdResult, odCfg.image_size, odCfg.image_size, \
            "", "", cv::Scalar(0, 0, 255), objectColors, "result-od.jpg", 0, -1, true);    
        
        /* PostProcessing : Segmentation */
        profiler.Start("post-segment");
        cv::Mat SegResult = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        Segmentation((uint16_t*)SegOutputTensors[0]->data(), SegResult.data, SegResult.rows, SegResult.cols, segCfg, SEG_NUM_CLASSES);
        profiler.End("post-segment");

        /* PostProcessing : Blend Image */
        profiler.Start("post-blend");
        cv::resize(SegResult, SegResult, Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);
        // frame = 0.5*frame + 0.5*SegResult;
        cv::addWeighted(frame, 0.5, SegResult, 0.5, 0.0, frame);
        profiler.End("post-blend");
        cout << dec << SEG_INPUT_WIDTH << "x" << SEG_INPUT_HEIGHT << " <- " << frame.cols << "x" << frame.rows << endl;

        /* Save & Show */
        cv::imwrite("result-od-seg.jpg", frame);
        cv::imwrite("result-seg.jpg", SegResult);
        cv::imwrite("resized-seg.jpg", resizedFrame[TASK_SEG]);
        cv::imwrite("resized-od.jpg", resizedFrame[TASK_OD]);
        cv::imshow(DISPLAY_WINDOW_NAME, frame);
        cv::waitKey(0);
        profiler.Show();
        return 0;
    }
    else if(!videoFile.empty() || cameraInput)
    {
        int videoIdx = 0, idx = 0, prevIdx = 0, key;
        bool pause = false, stop = false;
        cv::VideoCapture cap;
        cv::Mat frame[FRAME_BUFFERS], SegResult[FRAME_BUFFERS];
        cv::Mat resizedFrame[TASK_MAX];
        vector<BoundingBox> prevOdResult, OdResult;
        resizedFrame[TASK_OD] = cv::Mat(OD_INPUT_HEIGHT, OD_INPUT_WIDTH, CV_8UC3);
        resizedFrame[TASK_SEG] = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3);
        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            SegResult[i] = cv::Mat(SEG_INPUT_HEIGHT, SEG_INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        while(!stop && videoIdx>=0)
        {
            if(!videoFile.empty())
            {
                if(videoFile!="0")
                {
                    cap.open(videoFile);
                    if(!cap.isOpened())
                    {
                        cout << "Error: file " << videoFile << " could not be opened." <<endl;
                        return -1;
                    }
                    videoIdx = -1;
                }
                else
                {
                    cap.open(videoFiles[videoIdx]);
                    if(!cap.isOpened())
                    {
                        cout << "Error: file " << videoFiles[videoIdx] << " could not be opened." <<endl;
                        return -1;
                    }
                    (++videoIdx)%=NUM_VIDEO_FILES;
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
                videoIdx = -1;
            }
            cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
            cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
            namedWindow(DISPLAY_WINDOW_NAME);
            moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
            profiler.Start("cap");
            while(!stop)
            // while (waitKey(INPUT_CAPTURE_PERIOD_MS)<0 && cap.isOpened())
            {        
                profiler.End("cap");
                profiler.Start("cap");
                if(!pause)
                {
                    /* Capture */
                    cap >> frame[idx];
                    if(frame[idx].empty()) break;
                    cv::Mat outFrameBlend = frame[idx].clone();

                    /* PreProcessing */
                    profiler.Start("pre");
                    PreProc(frame[idx], resizedFrame[TASK_OD], true, true, 114);
                    PreProc(frame[idx], resizedFrame[TASK_SEG], false);
                    profiler.End("pre");

                    /* Main */        
                    profiler.Start("main");
                    auto SegOutputTensors = ieSEG.Run(resizedFrame[TASK_SEG].data);
                    auto OdOutputTensors = ieOD.Run(resizedFrame[TASK_OD].data);
                    profiler.End("main");
                    
                    /* PostProcessing : Segmentation, Blend Image */
                    if(!SegOutputTensors.empty())
                    {
                        profiler.Start("post-segment");
                        cv::Mat SegResultExpand;
                        // cv::Mat outFrame = *(cv::Mat*)(SegOutputTensors[0]->GetDestination());
                        // cv::Mat outFrame = frame[prevIdx];
                        cv::Mat outFrame = frame[prevIdx];

                        Segmentation((uint16_t*)SegOutputTensors[0]->data(), SegResult[idx].data, 
                            SegResult[idx].rows, SegResult[idx].cols, segCfg, SEG_NUM_CLASSES);
                        profiler.End("post-segment");
                        profiler.Start("post-blend");
                        cv::resize(SegResult[idx], SegResultExpand, Size(frame[idx].cols, frame[idx].rows), 0, 0, cv::INTER_LINEAR);
                        cv::addWeighted( outFrame, 0.5, SegResultExpand, 0.5, 0.0, outFrameBlend);
                        profiler.End("post-blend");
                        /* PostProcessing : Object Detection */
                        profiler.Start("post-od");
                        if(!OdOutputTensors.empty())
                        {
                            OdResult = yolo.PostProc(OdOutputTensors);
                            DisplayBoundingBox(outFrameBlend, prevOdResult, odCfg.image_size, odCfg.image_size, "", "",
                                cv::Scalar(0, 0, 255), objectColors, "", 0, -1, true);
                            prevOdResult = OdResult;
                        }                
                        profiler.End("post-od");
                    }
                    /* Display */
                    cv::imshow(DISPLAY_WINDOW_NAME, outFrameBlend);
                    prevIdx = idx;
                    (++idx)%=FRAME_BUFFERS;
                    key = cv::waitKey(INPUT_CAPTURE_PERIOD_MS);
                    if(key == 0x20) //'p'
                    {
                        pause = !pause;
                    }
                    else if(key == 0x1B) //'ESC'
                    {
                        stop = true;
                        break;
                    }
                }
            }
        }
        profiler.Show();
        sleep(1);
        return 0;
    }

    return 0;
}