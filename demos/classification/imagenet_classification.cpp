#include <future>
#include <thread>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>

#include "dxrt/dxrt_api.h"

// Replace macros with constexpr constants
constexpr int NPU_ID = 0;
constexpr int IMAGE_WIDTH = 224;
constexpr int IMAGE_HEIGHT = 224;

// for visualization
constexpr const char* MODEL_NAME = "EfficientNetB0";
constexpr const char* CHIP_NAME = "DX-M1";
constexpr double TOPS = 23.0;

std::string format_imagenet_info(int count) {
    return "ImageNet 2012  " + std::to_string(count);
}

std::string format_accuracy_info(double accuracy) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << "Accuracy (%)  " << accuracy;
    return oss.str();
}

std::string format_framerate_info(double frame_rate) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << "Frame Rate (fps)  " << frame_rate;
    return oss.str();
}

std::vector<std::string> get_list(const std::string& file_path)
{
    std::ifstream f(file_path);
    std::vector<std::string> list;
    std::string element;
    while(!f.eof())
    {
        getline(f, element);
        if(element=="") continue;
        list.emplace_back(element);
    }
    f.close();
    return list;
}

std::vector<std::string> split(const std::string& str, char separator)  
{  
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, separator)) {
        result.push_back(token);
    }
    
    // Ensure we have exactly 2 elements for backward compatibility
    if (result.size() < 2) {
        result.resize(2);
    }
    
    return result;
}

cv::Mat make_board(int count, double accuracy, double latency)
{
    // visualize count, accuracy, frame_rate, etc.
    double frame_rate = 1 / latency;
    double acc = accuracy * 100;

    int font = cv::FONT_HERSHEY_COMPLEX;
    float s1 = 0.7f;
    int th1 = 2;
    int stride = 40;
    auto linetype = cv::LINE_AA;
    cv::Scalar color(255, 255, 255);

    cv::Point pt(18, 105);
    cv::Mat board(260, 300, CV_8UC3, cv::Scalar(179, 102, 0));

    cv::putText(board, format_imagenet_info(count), pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, format_accuracy_info(acc), pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, format_framerate_info(frame_rate), pt, font, s1, color, th1, linetype);

    return board;
}

void rearrange_for_im2col(const uint8_t *src, uint8_t *dst)
{
    constexpr int size = IMAGE_WIDTH * 3;
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        memcpy(&dst[y * (size + 32)], &src[y * size], size);
}

std::vector<uint8_t> preprocess(const std::string& image_path, const std::string& based_path)
{
    std::vector<uint8_t> tensor(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    
    std::string full_path = based_path + "/" + image_path;
    cv::Mat image = cv::imread(full_path, cv::IMREAD_COLOR);
    
    cv::Mat resized;
    cv::Mat input(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, tensor.data());

    if (image.cols == IMAGE_WIDTH && image.rows == IMAGE_HEIGHT) {
        resized = image;
    } else {
        cv::resize(image, resized, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    }
    
    cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);
    return tensor;
}

int inference(const std::string& model_path, const std::vector<std::string>& image_gt_list, 
              const std::string& based_path, int *count, double *accuracy, double *latency, 
              bool *exit_flag, std::vector<bool>& results)
{
    // initialize inference engine
    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(model_path, io);

    std::future<std::vector<uint8_t>> input_future;
    std::vector<std::string> image_gt = split(image_gt_list[0], ' ');
    input_future = std::async(std::launch::async, preprocess, image_gt[0], based_path);

    int correct = 0;
    int i = 0;
    int inference_count = 0;
    while(true)
    {
        auto input = input_future.get();
        int gt = std::stoi(image_gt[1]); // Use std::stoi instead of atoi
        if((i+1) % image_gt_list.size() < image_gt_list.size())
        {
            image_gt = split(image_gt_list[(i+1) % image_gt_list.size()], ' ');
            input_future = std::async(std::launch::async, preprocess, image_gt[0], based_path);
        }

        double tick_run = cv::getTickCount();
        auto output = ie.Run(input.data());
        double time_run = ((double)cv::getTickCount() - tick_run) / cv::getTickFrequency();

        int ret = *(uint16_t*)output.front()->data();
        inference_count++;
        if (gt == ret)
        {
            correct++;
            results[i] = true;
        }

        *count = i + 1;
        *accuracy = (double)correct / inference_count;
        *latency = time_run;
        if (*exit_flag)
            break;
        i = (i+1) % image_gt_list.size();
    }
    return 0;
}

void visualize(const std::string& model_path, const std::string& image_list_path, 
               const std::string& based_image_path)
{
    int count = 0;
    double accuracy = 0;
    double latency = 0;
    bool exit_flag = false;
    std::vector<std::string> image_gt_list = get_list(image_list_path);

    std::vector<bool> results(image_gt_list.size());
    std::future<int> value_future = std::async(std::launch::async, inference, model_path, image_gt_list, based_image_path, &count, &accuracy, &latency, &exit_flag, std::ref(results));

    cv::Mat image;
    cv::Mat constant;
    std::string window_name = "ImageNet Classification";

    std::vector<std::string> image_gt;

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::moveWindow(window_name, 0, 0);

    while (true)
    {
        if (count == static_cast<int>(image_gt_list.size())) count = 0;
        
        image_gt = split(image_gt_list[count], ' ');
        
        image = cv::imread(based_image_path+"/"+image_gt[0], cv::IMREAD_ANYCOLOR);
        cv::resize(image, constant, cv::Size(260,260));

        auto board = make_board(count, accuracy, latency);
        cv::Mat view;
        cv::hconcat(board, constant, view);
        
        cv::imshow(window_name, view);

        int key = cv::waitKey(10);
        if (key == 27 || key == 'q')
        {
            exit_flag = true;
            break;
        }
    }
    
    value_future.get();
}

int main(int argc, char *argv[])
{
    std::string model_path = "";
    std::string based_image_path = "";
    std::string image_list_path = "";
    
    std::string app_name = "imagenet_classification_demo";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
        ("m, model_path", "(* required) classification model file (.dxnn, required)", cxxopts::value<std::string>(model_path))
        ("p, base_path", "(* required) input image files directory (required)", cxxopts::value<std::string>(based_image_path))
        ("i, image_list", "(* required) imagenet image list txt file (required)", cxxopts::value<std::string>(image_list_path))
        ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    if(model_path.empty())
    {
        std::cerr << "[ERROR] Model path is required. Use -m or --model_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    if(based_image_path.empty())
    {
        std::cerr << "[ERROR] Image path is required. Use -p or --base_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    if(image_list_path.empty()) 
    {
        std::cerr << "[ERROR] Image list path is required. Use -i or --image_list option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    visualize(model_path, image_list_path, based_image_path);
    return 0;
}