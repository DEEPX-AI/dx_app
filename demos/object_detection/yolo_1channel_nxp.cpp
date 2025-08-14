#include <sys/mman.h>
#include <unistd.h>
#include <syslog.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <termios.h>

#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>

#include <dxrt/dxrt_api.h>
#include <utils/color_table.hpp>
#include <utils/common_util.hpp>

#include "display.h"
#include "yolo.h"
#include "image.h"
#include "nxp.h"

#define FRAME_BUFFERS 5

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif

// pre/post parameter table
extern YoloParam yolov8_640;
std::vector<YoloParam> yoloParams = {
    yolov8_640
};

int lb_kbhit(void)
{
	struct termios oldt, newt;
	int ch;

	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;

    newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);

	ch = getchar();
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    std::cout << std::hex << ch << std::dec << std::endl;
	return ch;
}

void keycodeThread(void* args)
{
    auto return_code = (int*)args;
    int keycode = 255;

    while(keycode != 0x71)
    {
        std::cout << "press key[ quit:q, left:l, right:r, up:u, down:d ] >> " << std::endl; 
        keycode = lb_kbhit();
        if(keycode == 'c')
            *return_code = keycode;
    }

    *return_code = -1;
    return;
}

int main(int argc, char *argv[])
{
DXRT_TRY_CATCH_BEGIN
    int paramIdx = 0;
    std::string modelPath="";
    auto objectColors = dxapp::common::color_table;
    
    std::string app_name = "yolo object detection demo (nxp demo)";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
    ("m, model", "define model path", cxxopts::value<std::string>(modelPath))
    ("p, param", "pre/post-processing parameter selection", cxxopts::value<int>(paramIdx)->default_value("4"))
    ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help") || modelPath.empty())
    {
        std::cout << options.help() << std::endl;
        return -1;
    }

    LOG_VALUE(modelPath);
    LOG_VALUE(paramIdx);

    std::string captionModel = dxrt::StringSplit(modelPath, "/").back();

    dxrt::InferenceEngine ie(modelPath);
    if(!dxapp::common::minversionforRTandCompiler(&ie))
    {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not compatible with the version of the runtime. Please compile the model again." << std::endl;
        return -1;
    }
    auto yoloParam = yoloParams[paramIdx];
    Yolo yolo = Yolo(yoloParam);
    if(!yolo.LayerReorder(ie.outputs()))
        return -1;
    
    int fb_num = 0;
    char str[64];
    int keycode = 255;

    int camera_fb = 0;

    int frame_count = 0;
    int drawresult_count = 0;
    int postproc_count = 0;

    std::vector<uint8_t *> buffer(VDMA_CNN_BUF_MAX);


    sprintf(str, "/dev/fb%d", fb_num);

    FrameBuffer dx_fb(str, 1);
    camera_fb = open("/dev/video0", O_RDWR);
    if(nxp::print_caps(camera_fb))
        return 1;

    if(nxp::init_mmap(camera_fb, buffer.data()))
        return 1;
    
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::vector<uint8_t> camera_buffer(VDMA_CNN_IMG_HSIZE * VDMA_CNN_IMG_VSIZE * 3);

    auto th = std::thread(&keycodeThread, &keycode);
    cv::Mat resizedFrame = cv::Mat(yoloParam.height, yoloParam.width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    std::queue<std::pair<std::vector<BoundingBox>, int>> bboxesQueue;
    std::vector<BoundingBox> bboxes;
    std::mutex lk;
    int idx = 0;
    double capture_time = 0.f, preproc_time = 0.f, postproc_time = 0.f, drawresult_time = 0.f, inference_time = 0.f;
    auto inference_start = std::chrono::high_resolution_clock::now();
    auto inference_end = std::chrono::high_resolution_clock::now();

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
        [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void *arg)
        {
            auto start = std::chrono::high_resolution_clock::now();
            inference_end = start;
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count();
            inference_time += duration / 1000;
            /* PostProc */
            auto result = yolo.PostProc(outputs);
            postproc_count++;
            /* Restore raw frame index from tensor */
            {
                std::unique_lock<std::mutex> u_lock(lk);
                bboxesQueue.push(
                    std::make_pair(
                        result, 
                        (uint64_t) arg
                    )
                );
                // LOG_VALUE(result.size())
            }
            auto end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            postproc_time += duration / 1000;
            // std::cout << "postprocess image  -------- > " << duration << " us " << std::endl;
            return 0;
        };

    ie.RegisterCallBack(postProcCallBack);
    int capture_idx=0;
    std::string file_name = "";
    while(keycode >= 0)
    {
        auto start = std::chrono::high_resolution_clock::now();
        nxp::capture_image(camera_fb, buffer.data(), camera_buffer.data());
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        capture_time += duration / 1000;
        // std::cout << "capture image  -------- > " << duration << " us " << std::endl;
        
        auto frame = cv::Mat(cv::Size(VDMA_CNN_IMG_HSIZE, VDMA_CNN_IMG_VSIZE), CV_8UC3, camera_buffer.data());
        // if(keycode == 'c')
        // {
        //     // file_name = "capture"
        //     dxapp::common::dumpBinary(camera_buffer.data(), VDMA_CNN_IMG_HSIZE * VDMA_CNN_IMG_VSIZE * 3, "draw_capture"+std::to_string(capture_idx)+".bin");
        //     keycode = 0;
        //     capture_idx++;
        // }
        start = std::chrono::high_resolution_clock::now();
        PreProc(frame, resizedFrame, false, false, 114);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        preproc_time += duration / 1000;
        inference_start = end;
        // std::cout << "preprocess image  -------- > " << duration << " us " << std::endl;
        int reqId = ie.RunAsync(resizedFrame.data, &idx);
        UNUSEDVAR(reqId);
        {
            std::unique_lock<std::mutex> u_lock(lk);
            
            if(!bboxesQueue.empty())
            {
                bboxes = bboxesQueue.front().first;
                bboxesQueue.pop();
                start = std::chrono::high_resolution_clock::now();
                dx_fb.EraseBoxes(0, bboxes, VDMA_CNN_IMG_VSIZE, VDMA_CNN_IMG_HSIZE);
                dx_fb.DrawBoxes(0, bboxes, objectColors, yoloParam.height, yoloParam.width);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                // std::cout << "draw result image  -------- > " << duration << " us " << std::endl;
                drawresult_time += duration / 1000;
                drawresult_count++;
                
                if(keycode == 'c')
                {
                    // file_name = "capture"
                    dxapp::common::dumpBinary(dx_fb.get_data(), 1920 * 1080 * 4, "draw_capture"+std::to_string(capture_idx)+".bin");
                    keycode = 0;
                    capture_idx++;
                }
            }
        }
        frame_count++;
    }
    th.join();
    dx_fb.Clear();
    std::cout << "=== unit is microseconds ===" << std::endl;
    LOG_VALUE(capture_time / frame_count)
    LOG_VALUE(preproc_time / frame_count)
    LOG_VALUE(inference_time / postproc_count)
    LOG_VALUE(postproc_time / postproc_count)
    LOG_VALUE(drawresult_time / drawresult_count)
    std::cout << "============================" << std::endl;
    LOG_VALUE(frame_count)
    LOG_VALUE(postproc_count)
    LOG_VALUE(drawresult_count)
    std::cout << "============================" << std::endl;

    if(camera_fb)
    {
        close(camera_fb);
    }
    
    std::cout << ie.name() << " : latency " << ie.latency() << "us, " << ie.inference_time() << "us" << std::endl;

DXRT_TRY_CATCH_END

    return 0;
}
