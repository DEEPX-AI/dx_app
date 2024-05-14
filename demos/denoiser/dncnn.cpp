#include <cmath>
#include <chrono>
#include <future>
#include <iostream>
#include <algorithm>
#include <string>
#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "dxrt/dxrt_api.h"

#define INPUT_CAPTURE_PERIOD_MS 1
#define DISPLAY_WINDOW_NAME "DENOISE"

int WIDTH = 512;
int HEIGHT = 512;

int CAMERA_FRAME_WIDTH = 800;
int CAMERA_FRAME_HEIGHT = 600;

const char *usage =
    "DNCNN demo\n"
    "  -m, --model     define model path\n"
    "  -i, --image     use image file input\n"
    "  -v, --video     use video file input\n"
    "  -c, --camera    use camera input\n"
    "(opt) --mean      set noise parameter (default 0.0)\n"
    "(opt) --std       set noise parameter (default 15.0)\n"
    "  -h, --help      show help\n";

void help()
{
    std::cout << usage << std::endl;
}


cv::Mat get_noise_image(cv::Mat src, double _mean, double _std)
{
    cv::Mat add_weight;
    cv::Mat dst; 
    cv::Mat gaussian_noise = cv::Mat(src.size(),CV_16SC3);

    cv::randn(gaussian_noise, cv::Scalar::all(_mean), cv::Scalar::all(_std));

    src.convertTo(add_weight,CV_16SC3);
    addWeighted(add_weight, 1.0, gaussian_noise, 1.0, 0.0, add_weight);
    add_weight.convertTo(dst, src.type());
    
    return dst;
}

int main(int argc, char *argv[])
{
    int arg_idx = 1;
    std::string modelPath = "", imgFile = "", videoFile = "";
    bool cameraInput = false;
    double mean = 0.0, std = 15.0;
    int input_w = 0, input_h = 0;
    std::mutex lock;

    if (argc == 1)
    {
        std::cout << "Error: no arguments." << std::endl;
        help();
        return -1;
    }

    while (arg_idx < argc) {
        std::string arg(argv[arg_idx++]);
        if (arg == "-m" || arg == "--model")
                        modelPath = strdup(argv[arg_idx++]);
        else if (arg == "-i" || arg == "--image")
                        imgFile = strdup(argv[arg_idx++]);
        else if (arg == "-v" || arg == "--video")
                        videoFile = strdup(argv[arg_idx++]);
        else if (arg == "-c" || arg == "--camera")
                        cameraInput = true;
        else if (arg == "--mean")
                        mean = std::stod(argv[arg_idx++]);
        else if (arg == "--std")
                        std = std::stod(argv[arg_idx++]);
        else if (arg == "-h" || arg == "--help")
                        help(), exit(0);
        else
                        help(), exit(0);
    }

    dxrt::InferenceEngine ie(modelPath);
    auto& profiler = dxrt::Profiler::GetInstance();

    auto input_shape = ie.inputs().front().shape();
    input_w = input_shape[2], input_h = input_shape[1];
    
    int div = input_w / 2;
    
    cv::namedWindow(DISPLAY_WINDOW_NAME);

    if (!imgFile.empty())
    {
        cv::Mat frame = cv::imread(imgFile, cv::IMREAD_COLOR);
        cv::Mat noised_frame;
        cv::Mat output_frame = cv::Mat::zeros(input_h, input_w, CV_8UC3);
        cv::Mat view = cv::Mat::zeros(input_h, input_w, CV_8UC3);
        cv::resize(frame, frame, cv::Size(input_w, input_h), cv::INTER_LINEAR);

        while(true)
        {
            noised_frame = get_noise_image(frame, mean, std);
            
            profiler.Start("all");
            auto outputs = ie.Run(noised_frame.data);
            float *data = (float *)outputs.front()->data();
            for (int y = 0; y < input_h; y++)
            {
                for (int x = 0; x < input_w; x++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        float value = data[(y * input_w + x) * 64 + c] * 255.f;
                        if (value < 0.f)
                            value = 0.f;
                        else if (value > 255.f)
                            value = 255.f;
                        output_frame.data[(y * input_w + x) * 3 + c] = (uint8_t)value;
                    }
                }
            }
            profiler.End("all");
            int64_t t = profiler.Get("all");
            std::cout << 
            "======================================" << std::endl <<
            " processing time = " << t/1000 << "ms" <<
            ", FPS = " << round((1000000/t) * 100)/100 << std::endl;
            
            view = output_frame.clone();
            if(div > 0)
                noised_frame(cv::Rect(0, 0, div, input_h)).copyTo(view(cv::Rect(0, 0, div, input_h)));

            cv::line(view, cv::Point(div, 0), cv::Point(div, input_h), cv::Scalar(0, 0, 0), 2);
            

            cv::imshow(DISPLAY_WINDOW_NAME, view);

            int key = cv::waitKey(INPUT_CAPTURE_PERIOD_MS);
            
            if (key == 0x1B)
            {
                break;
            }
            else if (key > '0' && key < '9')
            {
                std::unique_lock<mutex> _lock(lock);
                std = (key - '0') * 10.0;
            }
            else if (key == 'a')
            {
                std::unique_lock<mutex> _lock(lock);
                div -= 10;
                if (div < 0)
                {
                    div = input_w/2;
                }
            }
            else if (key == 'd')
            {
                std::unique_lock<mutex> _lock(lock);
                div += 10;
                if (div > input_w)
                {
                    div = input_w/2;
                }
            }

            
        }

    }
    else if (!videoFile.empty() || cameraInput)
    {
        cv::VideoCapture cap;

        if (!videoFile.empty())
        {
            cap.open(videoFile);
            if (!cap.isOpened())
            {
                std::cout << "Error: file " << videoFile << " could not be opened." << std::endl;
                return -1;
            }
        }
        else
        {
            cap.open(0, cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT);
            if (!cap.isOpened())
            {
                std::cout << "Error: camera could not be opened." << std::endl;
                return -1;
            }
        }
        cv::Mat frame, resized_fame;
        cv::Mat output_frame = cv::Mat::zeros(input_h, input_w, CV_8UC3);
        cv::Mat view = cv::Mat::zeros(input_h, input_w, CV_8UC3);
        cv::Mat noised_frame;

        while (1)
        {
            cap >> frame;
            if (frame.empty())
            {
            	cap.open(videoFile);
                if (!cap.isOpened())
                {
                    std::cout << "Error: file " << videoFile << " could not be opened." << std::endl;
                        return -1;
                }
	            cap >> frame;

            }
            profiler.Start("all");
            cv::resize(frame, resized_fame, cv::Size(input_w, input_h), cv::INTER_LINEAR);
            
            noised_frame = get_noise_image(resized_fame, mean, std);

            auto outputs = ie.Run(noised_frame.data);
            float *data = (float *)outputs.front()->data();
            
            for (int y = 0; y < input_h; y++)
            {
                for (int x = 0; x < input_w; x++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        float value = data[(y * input_w + x) * 64 + c] * 255.f;
                        if (value < 0.f)
                            value = 0.f;
                        else if (value > 255.f)
                            value = 255.f;
                        output_frame.data[(y * input_w + x) * 3 + c] = (uint8_t)value;
                    }
                }
            }
            profiler.End("all");
            int64_t t = profiler.Get("all");
            std::cout << 
            "======================================" << std::endl <<
            " processing time = " << t/1000 << "ms" <<
            ", FPS = " << round((1000000/t) * 100)/100 << std::endl;

            view = output_frame.clone();
            if (div > 0)
                noised_frame(cv::Rect(0, 0, div, input_h)).copyTo(view(cv::Rect(0, 0, div, input_h)));

            cv::line(view, cv::Point(div, 0), cv::Point(div, input_h), cv::Scalar(0, 0, 0), 2);

            cv::imshow(DISPLAY_WINDOW_NAME, view);
            
            int key = cv::waitKey(INPUT_CAPTURE_PERIOD_MS);
            
            if (key == 0x1B)
            {
                break;
            }
            else if (key > '0' && key < '9')
            {
                std::unique_lock<mutex> _lock(lock);
                std = (key - '0') * 10.0;
            }
            else if (key == 'a')
            {
                std::unique_lock<mutex> _lock(lock);
                div -= 10;
                if (div < 0)
                {
                    div = input_w/2;
                }
            }
            else if (key == 'd')
            {
                std::unique_lock<mutex> _lock(lock);
                div += 10;
                if (div > input_w)
                {
                    div = input_w/2;
                }
            }

        }
    }
    return 1;
}
