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

#include "segmentation.h"
#include "image.h"

#include <dxrt/dxrt_api.h>
#include <utils/common_util.hpp>

#define DISPLAY_WINDOW_NAME "Only Segmentation"
#define FRAME_BUFFERS 5

/**
 * 19 classes for cityscapes datasets
 */
SegmentationParam segmentation_config_19classes[] = {
    {0, "road", 128, 64, 128},
    {1, "sidewalk", 244, 35, 232},
    {2, "building", 70, 70, 70},
    {3, "wall", 102, 102, 156},
    {4, "fence", 190, 153, 153},
    {5, "pole", 153, 153, 153},
    {6, "traffic light", 51, 255, 255},
    {7, "traffic sign", 220, 220, 0},
    {8, "vegetation", 107, 142, 35},
    {9, "terrain", 152, 251, 152},
    {10, "sky", 255, 0, 0},
    {11, "person", 0, 51, 255},
    {12, "rider", 255, 0, 0},
    {13, "car", 255, 51, 0},
    {14, "truck", 255, 51, 0},
    {15, "bus", 255, 51, 0},
    {16, "train", 0, 80, 100},
    {17, "motorcycle", 0, 0, 230},
    {18, "bicycle", 119, 11, 32}
};

SegmentationParam segmentation_config_3classes[] = {
    {0, "background", 0, 0, 0, },
    {1, "foot", 0, 128, 0, },
    {2, "body", 0, 0, 128, },
};

struct SegmentationArgs {
    std::vector<cv::Mat> seg_results;
    std::mutex output_process_lk;
    int seg_process_count = 0;
    int frame_idx = 0;
};

int main(int argc, char *argv[])
{
DXRT_TRY_CATCH_BEGIN
    std::string imgFile="", videoFile="", cameraPath="";
    std::string modelpath = "";
    int parameter = 0;
    int segmentation_model_input_width = 0;
    int segmentation_model_input_height = 0;
    bool cameraInput = false;
    bool visualize = false;
    bool loop = false;
    bool fps_only = false;
    int numClasses = 0;
    int target_fps = 0;
    int frame_skip = -1;

    auto frame_rate_end_time = std::chrono::high_resolution_clock::now();

    std::string app_name = "segmentation model demo";
    cxxopts::Options options("segmentation", app_name + " application usage ");
    options.add_options()
    ("m, model", "(* required) define dxnn model path", cxxopts::value<std::string>(modelpath))
    ("p, parameter", "(* required) define segmentation parameter \n"
                      "0: 19 classes ", cxxopts::value<int>(parameter)->default_value("0"))
    ("i, image", "use image file input", cxxopts::value<std::string>(imgFile))
    ("v, video", "use video file input", cxxopts::value<std::string>(videoFile))
    ("c, camera", "use camera input", cxxopts::value<bool>(cameraInput)->default_value("false"))
    ("camera_path", "use camera input (provide camera path)", cxxopts::value<std::string>(cameraPath)->default_value("/dev/video0"))
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
    LOG_VALUE(parameter);
    LOG_VALUE(imgFile);
    LOG_VALUE(videoFile);
    LOG_VALUE(cameraInput);
    
    dxrt::InferenceOption op_seg;
    op_seg.devices.push_back(0); 

    dxrt::InferenceEngine ieSEG(modelpath, op_seg);
    if(!dxapp::common::minversionforRTandCompiler(&ieSEG))
    {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not compatible with the version of the runtime. Please compile the model again." << std::endl;
        return -1;
    }

    SegmentationParam *segCfg;

    if(parameter == 0)
    {
        segmentation_model_input_width = ieSEG.GetInputs().front().shape()[2];
        segmentation_model_input_height = ieSEG.GetInputs().front().shape()[1];
        segCfg = segmentation_config_19classes;
        numClasses = 19;
    }
    else
    {
        segmentation_model_input_width = 768;
        segmentation_model_input_height = 384;
        segCfg = segmentation_config_3classes;
        numClasses = 3;
    }
    
    cv::Mat segInput, frame;
    cv::Mat frames[FRAME_BUFFERS];
    segInput = cv::Mat(segmentation_model_input_height, segmentation_model_input_width, CV_8UC3);

    /** -1 : not started, 1 : started */
    int display_start = -1;
    int display_exit = -1;
    bool app_quit = false;

    SegmentationArgs seg_args;
    seg_args.seg_results = std::vector<cv::Mat>(FRAME_BUFFERS);

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> seg_postProcCallBack = 
                [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void *arg)
    {
        auto arguments = (SegmentationArgs*)arg;
        {
            std::unique_lock<std::mutex> lk(arguments->output_process_lk);
            int index = arguments->frame_idx;
            if(index >= FRAME_BUFFERS) {
                index = index % FRAME_BUFFERS;
            }else if (index < 0) {
                index = 0;
            }
            cv::Mat seg_result = cv::Mat(segmentation_model_input_height, segmentation_model_input_width, CV_8UC3, cv::Scalar(0, 0, 0));
            if(outputs.front()->type() == dxrt::DataType::UINT16)
                Segmentation((uint16_t*)outputs.front()->data(), seg_result.data, seg_result.rows, seg_result.cols, segCfg, numClasses);
            else if(outputs.front()->type() == dxrt::DataType::FLOAT)
                Segmentation((float*)outputs.front()->data(), seg_result.data, seg_result.rows, seg_result.cols, segCfg, numClasses, outputs.front()->shape());
            arguments->seg_results[index] = seg_result;
            arguments->seg_process_count = arguments->seg_process_count + 1;
            arguments->frame_idx = arguments->frame_idx + 1;
        }
        return 0;
    };

    std::function<void(int)> display_result_thread = [&](int idx) {
        int index = 0;
        // Pre-allocate resized segmentation result to avoid repeated memory allocation
        cv::Mat resized_seg;
        
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
                std::unique_lock<std::mutex> lk(seg_args.output_process_lk);
                index = seg_args.frame_idx - 1;
                if(index >= FRAME_BUFFERS) {
                    index = index % FRAME_BUFFERS;
                }else if (index < 0) {
                    index = 0;
                }
            }
            if (seg_args.seg_process_count > 0) {
                cv::Mat display = frames[index].clone();
                if(display.empty()) {
                    continue;
                }
                // Overlay segmentation results - reuse resized_seg memory
                cv::resize(seg_args.seg_results[index], resized_seg, display.size());
                cv::addWeighted(display, 0.6, resized_seg, 0.4, 0, display);

                if(visualize && !fps_only) {
                    cv::imshow("Result", display);
                    if(cv::waitKey(1) == 'q') {
                        display_exit = 1;
                        app_quit = true;
                    }
                }
                else if(!visualize) {
                    if(seg_args.seg_process_count == idx)
                    {
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

    ieSEG.RegisterCallback(seg_postProcCallBack);

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
        
        std::vector<std::vector<uint8_t>> segOutputs(FRAME_BUFFERS);
        std::vector<cv::Mat> segInputs(FRAME_BUFFERS);
        for(int i=0;i<FRAME_BUFFERS;i++) {
            segOutputs[i] = std::vector<uint8_t>(ieSEG.GetOutputSize());
            segInputs[i] = cv::Mat(segmentation_model_input_height, segmentation_model_input_width, CV_8UC3);
        }

        std::thread display_result_thread_obj(display_result_thread, loop_count);

        auto s = std::chrono::high_resolution_clock::now();
        for(int i=0; i<loop_count; i++) 
        {
            frames[index] = frame;
            /* PreProcessing */
            PreProc(frames[index], segInputs[index], false);
            std::ignore = ieSEG.RunAsync(segInputs[index].data, &seg_args, (void*)segOutputs[index].data());
            index = (index + 1) % FRAME_BUFFERS;
            if(i == 0)
                display_start = 1;
        }

        while(true) {
            if(seg_args.seg_process_count == loop_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        display_result_thread_obj.join();

        auto e = std::chrono::high_resolution_clock::now();

        std::cout << "[DXAPP] [INFO] total time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / loop_count << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : " << seg_args.seg_process_count / (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0) << std::endl;

        return 0;
    }

    if(!videoFile.empty() || cameraInput)
    {

        cv::VideoCapture cap;
        int camera_frame_width = 0;
        int camera_frame_height = 0;
        int index = 0, frame_count = 0;
        visualize = true;

        std::vector<std::vector<uint8_t>> segOutputs(FRAME_BUFFERS);
        for(int i=0;i<FRAME_BUFFERS;i++) {
            segOutputs[i] = std::vector<uint8_t>(ieSEG.GetOutputSize());
        }

        std::thread display_result_thread_obj(display_result_thread, 0);

        if(cameraInput)
        {
#if _WIN32
            cap.open(0);
#elif __linux__
            cap.open(cameraPath, cv::CAP_V4L2);
#endif
            if(!cap.isOpened())
            {
                std::cout << "Error: camera could not be opened." << std::endl;
                return -1;
            }
            camera_frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            camera_frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        }
        else
        {
            cap.open(videoFile);
            if(!cap.isOpened())
            {
                std::cout << "Error: file " << videoFile << " could not be opened." << std::endl;
                return -1;
            }
            camera_frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            camera_frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        }
        std::cout << "VideoCapture FPS: " << std::dec << (int)cap.get(cv::CAP_PROP_FPS) << std::endl;
        std::cout << "VideoCapture Resolution: " << camera_frame_width << " x " << camera_frame_height << std::endl;

        auto s = std::chrono::high_resolution_clock::now();

        while(true) {
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
                    std::cout << "\nVideo file ended.\n" << std::endl;
                    display_exit = 1;
                    app_quit = true;
                    break;
                }
            }

            frames[index] = frame;

            /* PreProcessing */
            PreProc(frame, segInput, false);

            std::ignore = ieSEG.RunAsync(segInput.data, &seg_args, (void*)segOutputs[index].data());

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
            if(seg_args.seg_process_count == frame_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        display_result_thread_obj.join();

        auto e = std::chrono::high_resolution_clock::now();
        std::cout << "[DXAPP] [INFO] total time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / seg_args.seg_process_count << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : " << seg_args.seg_process_count / (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0) << std::endl;

        return 0;
    }

DXRT_TRY_CATCH_END

    return 0;
}