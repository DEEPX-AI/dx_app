#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "display.h"
#include "yolo.h"
#include "segmentation.h"
#include "image.h"

#include <dxrt/dxrt_api.h>
#include <utils/common_util.hpp>
#include <utils/color_table.hpp>

#define DISPLAY_WINDOW_NAME "OD + Segmentation"
#define FRAME_BUFFERS 15

// pre/post parameter table
extern YoloParam yolov5s_320, yolov5s_512, yolov5s_640, yolov7_640, yolov8_640, yolox_s_512, yolov5s_face_640, yolov9_640;
std::vector<YoloParam> yoloParams = {
    yolov5s_320,
    yolov5s_512,
    yolov5s_640,
    yolov7_640,
    yolov8_640,
    yolox_s_512,
    yolov5s_face_640,
    yolov9_640,
};

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

const char* usage =
    "object detection with image segmentation demo usage \n"
    "Usage:\n"
    "  od_segmentation [OPTION...]\n"
    "  -m0, --od_modelpath arg     (* required) object detection model include path\n"
    "  -m1, --seg_modelpath arg    (* required) segmentation model include path \n"
    "  -p0, --od_parameter arg     (* required) object detection parameter\n" 
    "                              0: yolov5s_320, 1: yolov5s_512, 2: yolov5s_640, 3: yolov7_640, 4: yolov8_640, 5: yolox_s_512, 6: yolov5s_face_640, 7: yolov9_640\n"
    "  -p1, --seg_parameter arg    (* required) segmentation parameter for cityscapes\n"
    "                              0: 19 classes (default)\n"
    "  -i,  --image arg            use image file input\n"
    "  -v,  --video arg            use video file input\n"
    "  -c,  --camera               use camera input\n"
    "       --camera_path          provide camera path(default /dev/video0)\n"
    "  -l,  --loop                 loop video file, if not set, will exit when video ends\n"
    "       --fps_only             will not visualize, only show fps\n"
    "       --target_fps           Adjusts FPS by skipping frames or sleeping if the video is faster or slower than the target FPS\n"
    "  -h,  --help                 show help\n";
void help()
{
    std::cout << usage << std::endl;    
}

struct OdSegmentationArgs {
    std::vector<cv::Mat> seg_results;
    std::vector<std::vector<BoundingBox>> od_results;
    std::vector<std::vector<int64_t>> od_output_shape;
    std::mutex lk;
    int od_process_count = 0;
    int seg_process_count = 0;
    int frame_idx = 0;
};


int main(int argc, char *argv[])
{
DXRT_TRY_CATCH_BEGIN
    int i = 1;
    std::string imgFile="", videoFile="", cameraPath="/dev/video0";
    std::string od_modelpath = "", seg_modelpath = "";
    int od_parameter = 0, seg_parameter = 0;
    int segmentation_model_input_width = 0;
    int segmentation_model_input_height = 0;
    bool cameraInput = false;
    bool visualize = false;
    bool loop = false;
    bool fps_only = false;
    int target_fps = 0;
    int frame_skip = -1;

    auto frame_rate_end_time = std::chrono::high_resolution_clock::now();

    auto objectColors = dxapp::common::color_table;
    int numClasses = 0;
    if(argc==1)
    {
        std::cout << "Error: no arguments." << std::endl;
        help();
        return -1;
    }
    while (i < argc) {
        std::string arg(argv[i++]);
        if (arg == "-m0" || arg == "--od_modelpath")
                                od_modelpath = argv[i++];
        else if (arg == "-m1" || arg == "--seg_modelpath")
                                seg_modelpath = argv[i++];
        else if (arg == "-p0" || arg == "--od_parameter")
                                od_parameter = std::stoi(argv[i++]);
        else if (arg == "-p1" || arg == "--seg_parameter")
                                seg_parameter = std::stoi(argv[i++]);
        else if (arg == "-i" || arg == "--image")
                                imgFile = argv[i++];
        else if (arg == "-v" || arg == "--video")
                                videoFile = argv[i++];
        else if (arg == "-c" || arg == "--camera")
                                cameraInput = true;
        else if (arg == "-l" || arg == "--loop")
                                loop = true;
        else if (arg == "--fps_only")
                                fps_only = true;
        else if (arg == "--target_fps")
                                target_fps = std::stoi(argv[i++]);
        else if (arg == "--camera_path")
                                cameraPath = argv[i++];
        else if (arg == "-h" || arg == "--help")
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
    
    dxrt::InferenceOption op_od;
    op_od.devices.push_back(0); 
    op_od.boundOption = 1; 
    
    dxrt::InferenceOption op_seg;
    op_seg.devices.push_back(0); 
    op_seg.boundOption = 5; 

    dxrt::InferenceEngine ieOD(od_modelpath, op_od);
    dxrt::InferenceEngine ieSEG(seg_modelpath, op_seg);
    if(!dxapp::common::minversionforRTandCompiler(&ieOD) 
        || !dxapp::common::minversionforRTandCompiler(&ieSEG))
    {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not compatible with the version of the runtime. Please compile the model again." << std::endl;
        return -1;
    }

    SegmentationParam *segCfg;

    if(seg_parameter == 0)
    {
        segmentation_model_input_width = ieSEG.GetInputs().front().shape()[2];
        segmentation_model_input_height = ieSEG.GetInputs().front().shape()[1];
        segCfg = segmentation_config_19classes;
        numClasses = 19;
    }
    
    YoloParam odCfg = yoloParams[od_parameter];

    Yolo yolo = Yolo(odCfg);
    if(!yolo.LayerReorder(ieOD.GetOutputs()))
        return -1;
    
    cv::Mat odInput, segInput, frame;
    cv::Mat frames[FRAME_BUFFERS];
    odInput = cv::Mat(odCfg.height, odCfg.width, CV_8UC3);
    segInput = cv::Mat(segmentation_model_input_height, segmentation_model_input_width, CV_8UC3);

    /** -1 : not started, 1 : started */
    int display_start = -1;
    int display_exit = -1;
    bool app_quit = false;
    OdSegmentationArgs od_seg_args;

    std::vector<std::vector<int64_t>> output_shape;
    for(auto &o:ieOD.GetOutputs())
    {
        output_shape.emplace_back(o.shape());
    }
    od_seg_args.seg_results = std::vector<cv::Mat>(FRAME_BUFFERS);
    od_seg_args.od_results = std::vector<std::vector<BoundingBox>>(FRAME_BUFFERS);

    od_seg_args.od_output_shape = output_shape;

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> od_postProcCallBack = 
                [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void *arg)
    {
        auto arguments = (OdSegmentationArgs*)arg;
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

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> seg_postProcCallBack = 
                [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void *arg)
    {
        auto arguments = (OdSegmentationArgs*)arg;
        {
            std::unique_lock<std::mutex> lk(arguments->lk);
            int index = arguments->seg_process_count;
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
        }
        return 0;
    };

    std::function<void(int)> display_result_thread = [&](int idx) {
        int od_index = 0, seg_index = 0;
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
                std::unique_lock<std::mutex> lk(od_seg_args.lk);
                od_index = od_seg_args.od_process_count - 1;
                if(od_index >= FRAME_BUFFERS) {
                    od_index = od_index % FRAME_BUFFERS;
                }else if (od_index < 0) {
                    od_index = 0;
                }
                seg_index = od_seg_args.seg_process_count - 1;
                if(seg_index >= FRAME_BUFFERS) {
                    seg_index = seg_index % FRAME_BUFFERS;
                }else if (seg_index < 0) {
                    seg_index = 0;
                }
            }
            if (od_seg_args.seg_process_count > 0 && od_seg_args.od_process_count > 0) {
                cv::Mat display = frames[od_index].clone();
                if(display.empty()) {
                    continue;
                }
                // Overlay segmentation results - reuse resized_seg memory
                cv::resize(od_seg_args.seg_results[seg_index], resized_seg, display.size());
                cv::addWeighted(display, 0.6, resized_seg, 0.4, 0, display);

                // Draw object detection results
                DisplayBoundingBox(display, od_seg_args.od_results[od_index], odCfg.height, odCfg.width, objectColors, odCfg.postproc_type, true);
                if(visualize && !fps_only) {
                    cv::imshow("Result", display);
                    if(cv::waitKey(1) == 'q') {
                        display_exit = 1;
                        app_quit = true;
                    }
                }
                else if(!visualize) {
                    if(od_seg_args.od_process_count == idx)
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

    ieOD.RegisterCallback(od_postProcCallBack);
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
        std::vector<std::vector<uint8_t>> odOutputs(FRAME_BUFFERS);
        std::vector<std::vector<uint8_t>> segOutputs(FRAME_BUFFERS);
        for(int i=0;i<FRAME_BUFFERS;i++) {
            odOutputs[i] = std::vector<uint8_t>(ieOD.GetOutputSize());
            segOutputs[i] = std::vector<uint8_t>(ieSEG.GetOutputSize());
        }
        std::thread display_result_thread_obj(display_result_thread, loop_count);

        auto s = std::chrono::high_resolution_clock::now();
        for(int i=0; i<loop_count; i++)
        {
            frames[index] = frame;
            /* PreProcessing */
            PreProc(frame, odInput, true, true, 114);
            PreProc(frame, segInput, false);

            std::ignore = ieSEG.RunAsync(segInput.data, &od_seg_args, (void*)segOutputs[index].data());
            std::ignore = ieOD.RunAsync(odInput.data, &od_seg_args, (void*)odOutputs[index].data());
            index = (index + 1) % FRAME_BUFFERS;
            if(i == 0)
                display_start = 1;
        }

        while(true) {
            if(od_seg_args.seg_process_count == loop_count && od_seg_args.od_process_count == loop_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        display_result_thread_obj.join();

        auto e = std::chrono::high_resolution_clock::now();

        std::cout << "[DXAPP] [INFO] total time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / loop_count << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : " << od_seg_args.od_process_count / (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0) << std::endl;

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
        std::vector<std::vector<uint8_t>> segOutputs(FRAME_BUFFERS);
        for(int i=0;i<FRAME_BUFFERS;i++) {
            odOutputs[i] = std::vector<uint8_t>(ieOD.GetOutputSize());
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
            PreProc(frame, odInput, true, true, 114);
            PreProc(frame, segInput, false);

            std::ignore = ieSEG.RunAsync(segInput.data, &od_seg_args, (void*)segOutputs[index].data());
            std::ignore = ieOD.RunAsync(odInput.data, &od_seg_args, (void*)odOutputs[index].data());

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
            if((od_seg_args.seg_process_count == frame_count) && (od_seg_args.od_process_count == frame_count)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        display_result_thread_obj.join();

        auto e = std::chrono::high_resolution_clock::now();
        std::cout << "[DXAPP] [INFO] total time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : " << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / od_seg_args.od_process_count << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : " << od_seg_args.od_process_count / (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0) << std::endl;

        return 0;
    }
    
DXRT_TRY_CATCH_END

    return 0;
}