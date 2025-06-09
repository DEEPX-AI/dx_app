#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#include <syslog.h>
#endif
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cxxopts.hpp>

#include "display.h"
#include "dxrt/dxrt_api.h"

#include "utils/common_util.hpp"

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////////////////////////////////////////////
#define PREPROC_KEEP_IMG_RATIO false
#define DISPLAY_WINDOW_NAME "PIDNet"
#define INPUT_CAPTURE_PERIOD_MS 30
#define NUM_CLASSES 19
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080
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

void Segmentation(float *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses, const std::vector<int64_t>& shape)
{
    bool need_transpose = shape[1] == numClasses? true : false;
    int compare_max_idx, compare_channel_idx;
    int pitch = shape[3];
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            int maxIdx = 0;
            for (int c=0;c<numClasses;c++)
            {
                if(need_transpose)
                {
                    compare_max_idx = w + (cols * h) + (maxIdx * rows * cols);
                    compare_channel_idx = w + (cols * h) + (c * rows * cols);
                }
                else
                {
                    compare_max_idx = maxIdx + ((cols * h) + w) * pitch;
                    compare_channel_idx = c + ((cols * h) + w) * pitch;
                }
                if(input[compare_max_idx] < input[compare_channel_idx])
                {
                    maxIdx = c;
                }
            }
            output[3*cols*h + 3*w + 2] = cfg[maxIdx].colorB;
            output[3*cols*h + 3*w + 1] = cfg[maxIdx].colorG;
            output[3*cols*h + 3*w + 0] = cfg[maxIdx].colorR;
        }
    }
}

void Segmentation(uint16_t *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses)
{
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            int class_index = input[cols*h + w];
            if(class_index >= numClasses && class_index < 0)
                continue;
            output[3*cols*h + 3*w + 2] = cfg[class_index].colorB;
            output[3*cols*h + 3*w + 1] = cfg[class_index].colorG;
            output[3*cols*h + 3*w + 0] = cfg[class_index].colorR;
        }
    }
}

int main(int argc, char *argv[])
{
DXRT_TRY_CATCH_BEGIN
    int inputWidth = 0, inputHeight = 0;
    string modelPath="", imgFile="", videoFile="", binFile="", simFile="";
    bool cameraInput = false, asyncInference = false;
    bool usingOrt = false;

    std::string app_name = "segmentation model demo";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
    ("m, model", "define dxnn model path", cxxopts::value<std::string>(modelPath))
    ("i, image", "use image file input", cxxopts::value<std::string>(imgFile))
    ("width, input_widht", "input width size(default : 640)", cxxopts::value<int>(inputWidth)->default_value("640"))
    ("height, input_height", "input height size(default : 640)", cxxopts::value<int>(inputHeight)->default_value("640"))
    ("v, video", "use video file input", cxxopts::value<std::string>(videoFile))
    ("c, camera", "use camera input", cxxopts::value<bool>(cameraInput)->default_value("false"))
    ("a, async", "asynchronous inference", cxxopts::value<bool>(asyncInference)->default_value("false"))
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
    LOG_VALUE(cameraInput);
    LOG_VALUE(asyncInference);

    dxrt::InferenceEngine ie(modelPath);

    if(dxapp::common::checkOrtLinking()) 
    {
        usingOrt = true;
    }
    bool is_argmax = ie.outputs().front().type() == dxrt::DataType::UINT16 ? true : false;
    
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
        profiler.Start("post-segment");
        cv::Mat result = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        if(is_argmax)
            Segmentation((uint16_t*)outputs[0]->data(), result.data, result.rows, result.cols, segCfg, NUM_CLASSES);
        else
            Segmentation((float*)outputs[0]->data(), result.data, result.rows, result.cols, segCfg, NUM_CLASSES, outputs[0]->shape());
        profiler.End("post-segment");
        profiler.Start("post-blend");
        cv::resize(result, result, Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);
        cv::addWeighted( frame, 0.5, result, 0.5, 0.0, frame);
        profiler.End("post-blend");
        cout << dec << inputWidth << "x" << inputHeight << " <- " << frame.cols << "x" << frame.rows << endl;
        cv::imwrite("result.jpg", frame);
        std::cout << "save file : result.jpg " << std::endl;
        
        return 0;
    }
    else if(!videoFile.empty() || cameraInput)
    {
        bool pause = false;
        cv::VideoCapture cap;
        cv::Mat frame[FRAME_BUFFERS], resizedFrame[FRAME_BUFFERS], result[FRAME_BUFFERS];
        dxrt::TensorPtrs _outputs;

        int64_t previous_average_time = 0;
        auto time_s = std::chrono::high_resolution_clock::now();
        auto time_e = std::chrono::high_resolution_clock::now();
        uint64_t postprocessed_count = 0;
        uint64_t duration_time = 0;
        int font_size = 0;

        for(int i=0;i<FRAME_BUFFERS;i++)
        {
            resizedFrame[i] = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
            result[i] = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
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
        }
        else
        {
#ifdef __linux__
            cap.open(0, cv::CAP_V4L2);
#elif _WIN32
            cap.open(0);
#endif
            cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
            if(!cap.isOpened())
            {
                cout << "Error: camera could not be opened." <<endl;
                return -1;
            }
        }
        font_size = cap.get(cv::CAP_PROP_FRAME_HEIGHT) / 100;
        cout << "FPS: " << dec << (int)cap.get(CAP_PROP_FPS) << endl;
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        namedWindow(DISPLAY_WINDOW_NAME);
        moveWindow(DISPLAY_WINDOW_NAME, 0, 0);

        std::vector<std::vector<uint8_t>> output_buffers;
        if(asyncInference)
        {
            for(int buffer_length = 0; buffer_length < FRAME_BUFFERS; buffer_length++)
            {
                output_buffers.emplace_back(std::vector<uint8_t>(ie.output_size()));
            }
        }

        std::queue<int> idx_queue;
        std::mutex lk;

        std::function<int(vector<shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
                [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *arg)
                {
                    profiler.Start("copy");
                    /* PostProc */
                    lk.lock();
                    int arg_idx = *(int*)arg;
                    memcpy((void*)output_buffers[arg_idx].data(), outputs[0]->data(), ie.output_size());
                    idx_queue.push(arg_idx);
                    lk.unlock();
                    time_e = std::chrono::high_resolution_clock::now();
                    duration_time = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s).count();
                    postprocessed_count++;
                    profiler.End("copy");
                    return 0;
                };
        if(asyncInference)
            ie.RegisterCallBack(postProcCallBack);

        profiler.Start("cap");
        while(1)
        {
            profiler.End("cap");
            profiler.Start("cap");
            if(!pause)
            {
                cap >> frame[idx];
                if(cap.get(cv::CAP_PROP_POS_FRAMES) == cap.get(cv::CAP_PROP_FRAME_COUNT))
                {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    cap >> frame[idx];
                }
                
                profiler.Start("pre");
                PreProc(frame[idx], resizedFrame[idx], PREPROC_KEEP_IMG_RATIO);
                profiler.End("pre");
                profiler.Start("main");
                time_s = std::chrono::high_resolution_clock::now();
                if(asyncInference)
                    auto req = ie.RunAsync(resizedFrame[idx].data, &idx);
                else
                {
                    _outputs = ie.Run(resizedFrame[idx].data);
                    time_e = std::chrono::high_resolution_clock::now();
                    duration_time = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s).count();
                    postprocessed_count++;
                }
                if(postprocessed_count > 0)
                {
                    int current_idx = asyncInference ? idx_queue.front():idx;
                    profiler.Start("post-segment");
                    cv::Mat resultExpand, outFrameBlend;
                    cv::Mat outFrame = frame[current_idx];
                    lk.lock();
                    if(asyncInference)
                    {
                        if(is_argmax)
                        {
                            Segmentation((uint16_t*)output_buffers[current_idx].data(), result[current_idx].data,
                                    result[current_idx].rows, result[current_idx].cols, segCfg, NUM_CLASSES);
                        }
                        else
                        {
                            Segmentation((float*)output_buffers[current_idx].data(), result[current_idx].data,
                                    result[current_idx].rows, result[current_idx].cols, segCfg, NUM_CLASSES, ie.GetOutputs().front().shape());
                        }
                    }
                    else
                    {
                        if(is_argmax)
                        {
                            Segmentation((uint16_t*)_outputs[0]->data(), result[current_idx].data,
                                    result[current_idx].rows, result[current_idx].cols, segCfg, NUM_CLASSES);
                        }
                        else
                        {
                            Segmentation((float*)_outputs[0]->data(), result[current_idx].data,
                                    result[current_idx].rows, result[current_idx].cols, segCfg, NUM_CLASSES, ie.GetOutputs().front().shape());
                        }
                    }
                    lk.unlock();
                    profiler.End("main");
                    profiler.End("post-segment");
                    profiler.Start("post-blend");
                    cv::resize(result[current_idx], resultExpand, Size(frame[current_idx].cols, frame[current_idx].rows), 0, 0, cv::INTER_LINEAR);
                    cv::addWeighted( outFrame, 0.5, resultExpand, 0.5, 0.0, outFrameBlend);
                    profiler.End("post-blend");
                    auto new_average_time = ((previous_average_time * postprocessed_count) + duration_time)/ (postprocessed_count + 1);
                    previous_average_time = new_average_time;
                    uint64_t fps = 1000000 / new_average_time;
                    std::string fpsCaption = "FPS : " + std::to_string((int)fps);
                    cv::Size fpsCaptionSize = cv::getTextSize(fpsCaption, cv::FONT_HERSHEY_PLAIN, font_size, 2, nullptr);
                    // cv::putText(outFrameBlend, fpsCaption, cv::Point(outFrameBlend.size().width - fpsCaptionSize.width, outFrameBlend.size().height - fpsCaptionSize.height), cv::FONT_HERSHEY_PLAIN, font_size, cv::Scalar(255, 255, 255),2);
                    cv::imshow(DISPLAY_WINDOW_NAME, outFrameBlend);
                    if(asyncInference)
                        idx_queue.pop();
                }
                (++idx)%=FRAME_BUFFERS;
            }
            key = cv::waitKey(INPUT_CAPTURE_PERIOD_MS);
            if(key == 'q' || key == 0x1B)
            {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        return 0;
    }
DXRT_TRY_CATCH_END
    return 0;
}