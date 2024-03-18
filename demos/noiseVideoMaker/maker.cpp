#include <cmath>
#include <chrono>
#include <future>
#include <iostream>
#include <algorithm>
#include <string>
#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#define INPUT_CAPTURE_PERIOD_MS 1
#define DISPLAY_WINDOW_NAME "output"

const char *usage =
    "DNCNN demo\n"
    "  -v, --video     use video file input\n"
    "  -o, --output    save noised video file\n"
    "  -n, --num       number of frames to save\n"
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
    std::string modelPath = "", videoFile = "", outputFile = "";
    int num_frames = 100;
    double mean = 10.0, std = 80.0;
    std::mutex lock;

    if (argc == 1)
    {
        std::cout << "Error: no arguments." << std::endl;
        help();
        return -1;
    }

    while (arg_idx < argc) {
        std::string arg(argv[arg_idx++]);
        if (arg == "-v" || arg == "--video")
                        videoFile = strdup(argv[arg_idx++]);
        else if (arg == "-o" || arg == "--output")
                        outputFile = strdup(argv[arg_idx++]);
        else if (arg == "-n" || arg == "--num")
                        num_frames = std::stoi(argv[arg_idx++]);
        else if (arg == "-h" || arg == "--help")
                        help(), exit(0);
        else
                        help(), exit(0);
    }
    if (!videoFile.empty())
    {
        cv::VideoCapture cap;
        cv::VideoWriter writer;

        if (!videoFile.empty())
        {
            cap.open(videoFile);
            if (!cap.isOpened())
            {
                std::cout << "Error: file " << videoFile << " could not be opened." << std::endl;
                return -1;
            }
        }
        writer.open(
                outputFile, cv::VideoWriter::fourcc('M','J','P','G'), (int)cap.get(cv::CAP_PROP_FPS), 
                cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)), true
            );

        cv::Mat frame;
        cv::Mat noised_frame;
        int s = 0;
        while (1)
        {
            cap >> frame;
            if (cap.get(cv::CAP_PROP_POS_FRAMES) > num_frames)
            {
                break;
            }
            
            noised_frame = get_noise_image(frame, mean, std);
            writer << noised_frame;
            printf("%d  ", ++s);

        }
    }
    return 1;
}
