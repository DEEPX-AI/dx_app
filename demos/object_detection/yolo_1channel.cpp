#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cxxopts.hpp>

#include "display.h"
#include "yolo.h"
#include "image.h"

#include <dxrt/dxrt_api.h>
#include <utils/common_util.hpp>
#include <utils/color_table.hpp>

#define DISPLAY_WINDOW_NAME "Object Detection"
#define FRAME_BUFFERS 10

// pre/post parameter table
extern YoloParam yolov5s_320, yolov5s_512, yolov5s_640, 
yolov7_512, yolov7_640, yolov8_640, yolox_s_512, yolov5s_face_640, yolov3_512, yolov4_416,
yolov9_640;
std::vector<YoloParam> yoloParams = {
    yolov5s_320,
    yolov5s_512,
    yolov5s_640,
    yolov7_512,
    yolov7_640,
    yolov8_640,
    yolox_s_512,
    yolov5s_face_640,
    yolov3_512,
    yolov4_416,
    yolov9_640
};

/////////////////////////////////////////////////////////////////////////////////////////////////
struct OdEstimationArgs {
    std::vector<std::vector<BoundingBox>> od_results;
    std::vector<std::vector<int64_t>> od_output_shape;
    std::mutex lk;
    int od_process_count = 0;
    int frame_idx = 0;
};

int main(int argc, char *argv[])
{
DXRT_TRY_CATCH_BEGIN
    std::string imgFile="", videoFile="";
    std::string modelpath = "";
    int parameter = 0;
    bool cameraInput = false;
    bool visualize = false;
    auto objectColors = dxapp::common::color_table;
    bool loop = false; // default loop 1 time, if not set, will exit when video ends
    bool fps_only = false;
    int target_fps = 0;
    int frame_skip = -1;

    auto frame_rate_end_time = std::chrono::high_resolution_clock::now();

    std::string app_name = "object detection model demo";
    cxxopts::Options options("yolo", app_name + " application usage ");
    options.add_options()
    ("m, model", "define dxnn model path", cxxopts::value<std::string>(modelpath))
    ("p, parameter", "define object detection parameter \n"
                     "0: yolov5s_320, 1: yolov5s_512, 2: yolov5s_640, 3: yolov7_512, 4: yolov7_640, 5: yolov8_640, 6: yolox_s_512, 7: yolov5s_face_640, 8: yolov3_512, 9: yolov4_416, 10: yolov9_640", 
                     cxxopts::value<int>(parameter)->default_value("0"))
    ("i, image", "use image file input", cxxopts::value<std::string>(imgFile))
    ("v, video", "use video file input", cxxopts::value<std::string>(videoFile))
    ("c, camera", "use camera input", cxxopts::value<bool>(cameraInput)->default_value("false"))
    ("l, loop", "loop video file, if not set, will exit when video ends", cxxopts::value<bool>(loop)->default_value("false"))
    ("fps_only", "will not visualize, only show fps", cxxopts::value<bool>(fps_only)->default_value("false"))
    ("target_fps", "Adjusts FPS by skipping frames or sleeping if the video is faster or slower than the target FPS", 
                     cxxopts::value<int>(target_fps)->default_value("0"))
    ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help") || modelpath.empty())
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    
    LOG_VALUE(modelpath);
    LOG_VALUE(videoFile);
    LOG_VALUE(imgFile);
    LOG_VALUE(cameraInput);
    
    dxrt::InferenceOption op_od;
    op_od.devices.push_back(0); 

    dxrt::InferenceEngine ie(modelpath, op_od);

    if(!dxapp::common::minversionforRTandCompiler(&ie))
    {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not compatible with the version of the runtime. Please compile the model again." << std::endl;
        return -1;
    }
    
    auto odCfg = yoloParams[parameter];
    Yolo yolo = Yolo(odCfg);
    if(!yolo.LayerReorder(ie.GetOutputs()))
        return -1;

    cv::Mat odInput, frame;
    cv::Mat frames[FRAME_BUFFERS];
    odInput = cv::Mat(odCfg.height, odCfg.width, CV_8UC3);

    /** -1 : not started, 1 : started */
    int display_start = -1;
    int display_exit = -1;
    bool app_quit = false;

    OdEstimationArgs od_args;
    
    std::vector<std::vector<int64_t>> output_shape;
    for(auto &o:ie.GetOutputs())
    {
        output_shape.emplace_back(o.shape());
    }

    od_args.od_output_shape = output_shape;

    od_args.od_results = std::vector<std::vector<BoundingBox>>(FRAME_BUFFERS);

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> od_postProcCallBack = 
                [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void *arg)
    {
        auto arguments = (OdEstimationArgs*)arg;
        {
            std::unique_lock<std::mutex> lk(arguments->lk);
            int index = arguments->od_process_count;
            if(index >= FRAME_BUFFERS) {
                index = index % FRAME_BUFFERS;
            }else if (index < 0) {
                index = 0;
            }

            auto od_result = yolo.PostProc(outputs);
            arguments->od_results[index] = od_result;
            arguments->od_process_count = arguments->od_process_count + 1;
            arguments->frame_idx = arguments->frame_idx + 1;
        }
        return 0;
    };

    std::function<void(int)> display_result_thread = [&](int idx) {
        int index = 0;
        while (display_start == -1) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        if(fps_only) {
            printf("Waiting for inference to complete...\n");
        }
        if(visualize && !fps_only) {
            cv::namedWindow("Result", cv::WINDOW_NORMAL);
            cv::resizeWindow("Result", 960, 540);
        }
        while (!app_quit) {
            if (display_exit == 1) {
                break;
            }
            {
                std::unique_lock<std::mutex> lk(od_args.lk);
                index = od_args.frame_idx - 1;
                if(index >= FRAME_BUFFERS) {
                    index = index % FRAME_BUFFERS;
                }else if (index < 0) {
                    index = 0;
                }
            }
            if (od_args.od_process_count > 0) {
                cv::Mat display = frames[index].clone();
                if(display.empty()) {
                    continue;
                }
                // Draw object detection results
                DisplayBoundingBox(display, od_args.od_results[index], odCfg.height, odCfg.width, objectColors, odCfg.postproc_type, true);
                if(visualize && !fps_only) {
                    cv::imshow("Result", display);
                    if(cv::waitKey(1) == 'q') {
                        display_exit = 1;
                        app_quit = true;
                    }
                }
                else if(!visualize) {
                    if(od_args.od_process_count == idx)
                    {
                        yolo.ShowResult();
                        cv::imwrite("result.jpg", display);
                        std::cout << "Result saved to result.jpg" << std::endl;
                        display_exit = 1;
                        app_quit = true;
                    }
                }
                frame_rate_end_time = std::chrono::high_resolution_clock::now();
            }
        }
        
        display_start = 1;
        cv::destroyAllWindows();
    };

    ie.RegisterCallback(od_postProcCallBack);

    if(!imgFile.empty())
    {
        /* Capture */

        frame = cv::imread(imgFile, cv::IMREAD_COLOR);
        int index = 0;
        int loop_count = 1;
        if (loop) {
            loop_count = 100;
            printf("Loop image file Inference %d times\n", loop_count);
        }
        std::vector<std::vector<uint8_t>> odOutputs(FRAME_BUFFERS);
        std::vector<cv::Mat> odInputs(FRAME_BUFFERS);
        for(int i=0;i<FRAME_BUFFERS;i++) {
            odOutputs[i] = std::vector<uint8_t>(ie.GetOutputSize());
            odInputs[i] = cv::Mat(odCfg.height, odCfg.width, CV_8UC3);
        }
        std::thread display_result_thread_obj(display_result_thread, loop_count);

        auto s = std::chrono::high_resolution_clock::now();
        for(int i=0; i<loop_count; i++)
        {
            frames[index] = frame;
            /* PreProcessing */
            PreProc(frames[index], odInputs[index], true, true, 114);
            std::ignore = ie.RunAsync(odInputs[index].data, &od_args, (void*)odOutputs[index].data());
            index = (index + 1) % FRAME_BUFFERS;
            if(i == 0)
                display_start = 1;
        }

        while(true) {
            if(od_args.od_process_count == loop_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        display_result_thread_obj.join();

        auto e = std::chrono::high_resolution_clock::now();

        std::cout << "[DXAPP] [INFO] total time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / loop_count << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : " << loop_count / (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0) << std::endl;

        return 0;
    }

    if(!videoFile.empty() || cameraInput)
    {
        cv::VideoCapture cap;
        int camera_frame_width = 0;
        int camera_frame_height = 0;
        int index = 0, frame_count = 0;
        visualize = true;

        std::vector<std::vector<uint8_t>> odOutputs(FRAME_BUFFERS);
        std::vector<cv::Mat> odInputs(FRAME_BUFFERS);
        for(int i=0;i<FRAME_BUFFERS;i++) {
            odOutputs[i] = std::vector<uint8_t>(ie.GetOutputSize());
            odInputs[i] = cv::Mat(odCfg.height, odCfg.width, CV_8UC3);
        }

        std::thread display_result_thread_obj(display_result_thread, 0);

        if(cameraInput)
        {
#if _WIN32
            cap.open(0);
#elif __linux__
            cap.open(0, cv::CAP_V4L2);
#endif
            camera_frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            camera_frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            if(!cap.isOpened())
            {
                std::cout << "Error: camera could not be opened." << std::endl;
                return -1;
            }
        }
        else
        {
            cap.open(videoFile);
            if(!cap.isOpened())
            {
                std::cout << "Error: file " << videoFile << " could not be opened." << std::endl;
                return -1;
            }
        }
        std::cout << "VideoCapture FPS: " << std::dec << (int)cap.get(cv::CAP_PROP_FPS) << std::endl;
        std::cout << "VideoCapture Resolution: " << camera_frame_width << " x " << camera_frame_height << std::endl;

        auto s = std::chrono::high_resolution_clock::now();

        while(true) {
            cv::Mat frame;
            cap >> frame;

            if(frame_skip > 0) {
                cap.set(cv::CAP_PROP_POS_FRAMES, cap.get(cv::CAP_PROP_POS_FRAMES) + frame_skip);
            }
            
            if(frame.empty()) {
                if(loop)
                {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    cap >> frame;
                }
                else
                {
                    printf("\nVideo file ended.\n");
                    display_exit = 1;
                    app_quit = true;
                    break;
                }
            }

            frames[index] = frame;

            /* PreProcessing */
            PreProc(frame, odInputs[index], true, true, 114);

            std::ignore = ie.RunAsync(odInputs[index].data, &od_args, (void*)odOutputs[index].data());

            index = (index + 1) % FRAME_BUFFERS;
            frame_count++;

            if(display_start == -1) {
                display_start = 1;
            }

            if(app_quit) {
                break;
            }
            if(!cameraInput && target_fps > 0) {
                auto frame_rate_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_rate_end_time - s).count() / frame_count;
                if(frame_rate_duration > 0 && frame_rate_duration < 1000000 / target_fps) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1000000 / target_fps - frame_rate_duration));
                }
                else if (frame_rate_duration > 1000000 / target_fps) {
                    frame_skip = (int)(target_fps - (1000000 / frame_rate_duration));
                }
                else {
                    frame_skip = 0;
                }
            }
        }
        while(true) {
            if(od_args.od_process_count == frame_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        display_result_thread_obj.join();

        auto e = std::chrono::high_resolution_clock::now();
        std::cout << "[DXAPP] [INFO] total time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / frame_count << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : " << frame_count / (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0) << std::endl;

        return 0;
    }

DXRT_TRY_CATCH_END

    return 0;
}