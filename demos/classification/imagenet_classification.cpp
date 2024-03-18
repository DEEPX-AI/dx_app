#include <getopt.h>
#include <future>
#include <thread>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dxrt/dxrt_api.h"

#define DEFAULT_MODEL_PATH "/dxrt/models/DX_M1/v2p0/efficientnet-b0_argmax"
#define DEFAULT_IMAGE_PATH "/dxrt/dataset/imagenet/val/"
#define DEFAULT_IMAGE_LIST "/dxrt/dataset/imagenet/val_map.txt"

#define NPU_ID 0
#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224

// for visualization
#define MODEL_NAME "EfficientNetB0"
#define CHIP_NAME "DX-M1"
#define TOPS 23.0

template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

std::vector<std::string> get_list(std::string file_path)
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

std::string* split(std::string str, char seperator)  
{  
    int currIndex = 0, i = 0;  
    int startIndex = 0, endIndex = 0;  
    std::string* result = new std::string[2];
    while (i <= (int)str.length())  
    {  
        if (str[i] == seperator || i == (int)str.length())  
        {  
            endIndex = i;  
            std::string subStr = "";  
            subStr.append(str, startIndex, endIndex - startIndex);  
            result[currIndex] = subStr;  
            currIndex += 1;  
            startIndex = endIndex + 1;  
        }  
        i++;  
    }
    return result;
}  

cv::Mat make_board(int count, double accuracy, double latency)
{
    // visualize count, accuracy, frame_rate, etc.
    double frame_rate = 1 / latency;
    double acc = accuracy * 100;

    int font = cv::FONT_HERSHEY_COMPLEX;
    float s1 = 0.7;
    int th1 = 2;
    int stride = 40;
    auto linetype = cv::LINE_AA;
    cv::Scalar color(255, 255, 255);

    cv::Point pt(18, 105);
    cv::Mat board(260, 300, CV_8UC3, cv::Scalar(179, 102, 0));

    cv::putText(board, string_format("ImageNet 2012  %d", count), pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, string_format("Accuracy (%)  %.1f", acc), pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, string_format("Frame Rate (fps)  %.0f", frame_rate), pt, font, s1, color, th1, linetype);

    return board;
}

std::string get_imagenet_name(int index)
{
    return string_format("ILSVRC2012_val_%08d", index + 1);
}

void rearrange_for_im2col(uint8_t *src, uint8_t *dst)
{
    constexpr int size = IMAGE_WIDTH * 3;
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        memcpy(&dst[y * (size + 32)], &src[y * size], size);
}

uint8_t *preprocess(std::string image_path, std::string based_path)
{
    auto image = cv::imread(based_path+"/"+image_path);
    cv::Mat resized, input;
    if (image.cols == IMAGE_WIDTH && image.rows == IMAGE_HEIGHT)
    {
        resized = image;
    }
    else
    {
        cv::resize(image, resized, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    }
    cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);
    uint8_t *tensor = new uint8_t[IMAGE_HEIGHT * (IMAGE_WIDTH * 3 + 32)];
    rearrange_for_im2col(input.data, tensor);
    return tensor;
}

int inference(std::string model_path, std::vector<std::string> image_gt_list, std::string based_path, int *count, double *accuracy, double *latency, bool *exit_flag, bool *results)
{
    // initialize inference engine
    auto ie = dxrt::InferenceEngine(model_path);
    std::string* image_gt = new std::string[2];

    std::future<uint8_t *> input_future;
    image_gt = split(image_gt_list[0], '\t');
    input_future = std::async(std::launch::async, preprocess, image_gt[0], based_path);

    int correct = 0;
    for (int i = 0; i < (int)image_gt_list.size(); i++)
    {
        auto input = input_future.get();
        int gt = atoi(image_gt[1].c_str());
        if(i + 1 < (int)image_gt_list.size())
        {
            image_gt = split(image_gt_list[i+1], '\t');
            input_future = std::async(std::launch::async, preprocess, image_gt[0], based_path);
        }

        double tick_run = cv::getTickCount();
        auto output = ie.Run(input);
        double time_run = ((double)cv::getTickCount() - tick_run) / (double)cv::getTickFrequency();
        if(input!=NULL)
            delete input;

        int ret = *(int*)output.front()->data();
        if (gt == ret)
        {
            correct++;
            results[i] = true;
        }

        *count = i + 1;
        *accuracy = (double)correct / (double)(i + 1);
        *latency = time_run;
        if (*exit_flag)
            break;
    }
    return 0;
}

void visualize(std::string model_path, std::string image_list_path, std::string based_image_path)
{
    int count = 0;
    double accuracy = 0;
    double latency = 0;
    bool exit_flag = false;
    std::vector<std::string> image_gt_list = get_list(image_list_path);
    bool* results = new bool[image_gt_list.size()];
    std::future<int> value_future = std::async(std::launch::async, inference, model_path, image_gt_list, based_image_path,&count, &accuracy, &latency, &exit_flag, results);

    cv::Mat image, constant;
    std::string window_name = "ImageNet Classification";

    std::string* image_gt = new std::string[2];

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::moveWindow(window_name, 0, 0);

    while (1)
    {
        if (count == (int)image_gt_list.size()) break;

        std::future<uint8_t *> input_future;
        image_gt = split(image_gt_list[count], '\t');
        image = cv::imread(based_image_path+"/"+image_gt[0], cv::IMREAD_ANYCOLOR);
        cv::resize(image, constant, cv::Size(260,260));

        auto board = make_board(count, accuracy, latency);
        cv::Mat view;
        cv::hconcat(board, constant, view);
        // cv::imwrite("./result.jpg", view);
        
        cv::imshow(window_name, view);

        int key = cv::waitKey(10);
        if (key == 27)
        {
            exit_flag = true;
            break;
        }

    }
    value_future.get();
}

static struct option const opts[] = {
    {"model", required_argument, 0, 'm'},
    {"path", required_argument, 0, 'p'},
    {"image", required_argument, 0, 'i'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}};

const char *usage =
    "ImageNet Classification Demo\n"
    "  -m, --model        define model path\n"
    "  -p, --path         based image file path\n"
    "  -i, --image        ImageNet image list path\n"
    "  -h, --help         show help\n";

void help()
{
    std::cout << usage << std::endl;
}

int main(int argc, char *argv[])
{
    std::string model_path = DEFAULT_MODEL_PATH;
    std::string based_image_path = DEFAULT_IMAGE_PATH;
    std::string image_list_path = DEFAULT_IMAGE_LIST;

    if (argc == 1)
    {
        std::cout << "Error: no arguments." << std::endl;
        help();
        return -1;
    }

    int optCmd;
    while ((optCmd = getopt_long(argc, argv, "m:p:i:h", opts, NULL)) != -1)
    {
        switch (optCmd)
        {
        case '0':
            break;
        case 'm':
            model_path = strdup(optarg);
            break;
        case 'p':
            based_image_path = strdup(optarg);
            break;
        case 'i':
            image_list_path = strdup(optarg);
            break;
        case 'h':
        default:
            help();
            exit(0);
            break;
        }
    }

    visualize(model_path, image_list_path, based_image_path);
    return 0;
}