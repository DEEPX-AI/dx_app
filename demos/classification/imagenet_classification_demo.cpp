#include <future>
#include <thread>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>

#include "dxrt/dxrt_api.h"
#include "rapidjson/error/en.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"

#define DEFAULT_MODEL_PATH "/dxrt/m1/efficientnet-b0_argmax"
#define DEFAULT_IMAGE_PATH "/dxrt/m1/imagenet/imagenet_val/"
#define DEFAULT_LABEL_PATH "/dxrt/m1/imagenet/imagenet_val.json"
#define DEFAULT_GRID_PATH "/dxrt/m1/imagenet/grid_8x5/"

#define NPU_ID 0
#define NUM_IMAGES 50000
#define NUM_BUFFS 500
#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224
#define GRID_WIDTH 8
#define GRID_HEIGHT 5
#define GRID_UNIT (GRID_WIDTH*GRID_HEIGHT)

// for visualization
#define MODEL_NAME "EfficientNetB0"
#define CHIP_NAME "DX-M1"
#define TOPS 23.0

using namespace std;
using namespace rapidjson;

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

cv::Mat make_board(int count, double accuracy, double fps, int board_height)
{
    // visualize count, accuracy, fps, etc.
    double acc = accuracy * 100;

    int font = cv::FONT_HERSHEY_COMPLEX;
    float s1 = 1.0, s2 = 2.5;
    int th1 = 2, th2 = 6;
    int stride = 200;
    auto linetype = cv::LINE_AA;
    cv::Scalar color(255, 255, 255);

    cv::Point pt(32, 250);
    cv::Mat board(board_height, 400, CV_8UC3, cv::Scalar(179, 102, 0));

    cv::putText(board, "ImageNet 2012", pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, "Accuracy (%)", pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, "Frame Rate (fps)", pt, font, s1, color, th1, linetype);
    pt.y += stride;

    pt.y = 340;
    cv::putText(board, string_format(" %d", count), pt, font, s2, color, th2, linetype);
    pt.y += stride;
    cv::putText(board, string_format(" %.2f", acc), pt, font, s2, color, th2, linetype);
    pt.y += stride;
    cv::putText(board, string_format(" %.0f", fps), pt, font, s2, color, th2, linetype);
    pt.y += stride;

    pt.y = 850;
    stride = 50;
    cv::putText(board, MODEL_NAME, pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, string_format("%d x %d", IMAGE_WIDTH, IMAGE_HEIGHT), pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, CHIP_NAME, pt, font, s1, color, th1, linetype);
    pt.y += stride;
    cv::putText(board, string_format("%.2f FPS/TOPS", fps / TOPS), pt, font, s1, color, th1, linetype);
    pt.y += stride;

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

uint8_t *preprocess(std::string image_path, int image_index)
{
    auto name = get_imagenet_name(image_index);
    auto image = cv::imread(image_path + name + ".PNG");
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

vector<void*> preprocessAll(string image_path)
{
    vector<void*> ret;
    for(int i=0;i<NUM_IMAGES;i++)
    {
        if(i%1000==0)
        {
            cout << "preprocessing: " << i << endl;
        }
        ret.emplace_back(
            (void*)preprocess(image_path, i)
        );
    }
    return ret;
}

void GenerateGT(string labelPath, int numImages, int *GroundTruth)
{
    std::ifstream f(labelPath);
    string json((istreambuf_iterator<char>(f)), (istreambuf_iterator<char>()));
    Document doc;
    doc.Parse(json.c_str());
    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    doc.Accept(writer);
    for(int i=0; i<numImages; i++)
    {
        string dataset_name = get_imagenet_name(i);
        GroundTruth[i] = doc[dataset_name.c_str()].GetInt();
    }    
}

int main(int argc, char *argv[])
{
    std::string model_path = DEFAULT_MODEL_PATH;
    std::string image_path = DEFAULT_IMAGE_PATH;
    std::string label_path = DEFAULT_LABEL_PATH;
    std::string grid_path = DEFAULT_GRID_PATH;
    
    std::string app_name = "imagenet_classification_grid_demo";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
        ("m, model_path", "classification model file (.dxnn, required)", cxxopts::value<std::string>(model_path)->default_value(DEFAULT_MODEL_PATH))
        ("i, image", "input image files directory (required)", cxxopts::value<std::string>(image_path)->default_value(DEFAULT_IMAGE_PATH))
        ("l, label", "input ground truth label json file (required)", cxxopts::value<std::string>(label_path)->default_value(DEFAULT_LABEL_PATH))
        ("g, grid", "imagenet grid image files directory (required)", cxxopts::value<std::string>(grid_path)->default_value(DEFAULT_GRID_PATH))
        ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help") || model_path.empty())
    {
        std::cout << options.help() << endl;
        exit(0);
    }

    int GroundTruth[NUM_IMAGES];
    int Classification[NUM_IMAGES];
    mutex resultLock;
    int numBuf = NUM_BUFFS;
    dxrt::InferenceEngine ie(model_path);
    auto& profiler = dxrt::Profiler::GetInstance();
    atomic<int> gridIdx {0};
    atomic<int> gInfCnt {0};
    atomic<int> correct {0};
    double accuracy = 0;
    double fps = 0;
    atomic<bool> exit_flag {false};
    bool results[NUM_IMAGES] = {false};
    std::function<int(vector<shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
        [&](vector<shared_ptr<dxrt::Tensor>> outputs, void *args)
        {
            int id = (uint64_t)args;
            gInfCnt.store(id);
            Classification[id] = *((int*)(outputs.front()->data()));
            if(Classification[id]==GroundTruth[id])
            {
                ++correct;
                results[id] = true;
                accuracy = (double)correct.load()/(double)(id+1);                
            }
            if(id%GRID_UNIT==GRID_UNIT-1)
            {
                gridIdx.store(id/GRID_UNIT);
            }
            return 0;
        };
    ie.RegisterCallBack(postProcCallBack);
    GenerateGT(label_path, NUM_IMAGES, GroundTruth);
    auto inputs = preprocessAll(image_path);
    cv::Mat grids[NUM_IMAGES/GRID_UNIT];
    for(int i=0;i<NUM_IMAGES/GRID_UNIT;i++)
    {
        grids[i] = cv::imread(grid_path + string_format("%05d.JPEG", i));
    }
    std::string window_name = "ImageNet Classification";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::resizeWindow(window_name, 1920, 1080);
    //cv::moveWindow(window_name, 0, 0);

    thread ( [&](void) {
        while(1)
        {
            volatile int cnt = 0;
            correct.store(0);
            while(!exit_flag.load())
            {
                int reqId = 0;
                profiler.Start("inf");
                for(int i=0;i<numBuf-1;i++)
                {
                    reqId = ie.RunAsync(inputs[cnt], (void*)(intptr_t)cnt);
                    cnt++;
                }
                reqId = ie.RunAsync(inputs[cnt], (void*)(intptr_t)cnt);
                cnt++;
                ie.Wait(reqId);
                profiler.End("inf");
                // fps = 1000*1000/ie.GetNpuPerf(0, true) + 1000*1000/ie.GetNpuPerf(1, true); /* NPU only FPS */
                fps = 1000*1000/(profiler.Get("inf")/(numBuf-1)); /* system FPS */
                if(cnt==NUM_IMAGES) break;
            }
        }
    }).detach();

    thread ( [&](void) {        
        cv::Mat grid;
        int grid_size = GRID_WIDTH * GRID_HEIGHT;
        while(!exit_flag.load())
        {
            while (1)
            {
                int grid_index = gridIdx.load();
                int count = gInfCnt.load();
                {
                    grid = grids[grid_index];
                    for (int index = 0; index < grid_size; index++)
                    {
                        int gx = index % GRID_WIDTH * IMAGE_WIDTH;
                        int gy = index / GRID_WIDTH * IMAGE_HEIGHT;
                        cv::Rect rect(cv::Point2i(gx + 4, gy + 4), cv::Point2i(gx + IMAGE_WIDTH - 4, gy + IMAGE_HEIGHT - 4));
                        cv::Scalar color(0, 64, 255);
                        if (results[grid_index * grid_size + index])
                            color = cv::Scalar(64, 255, 0);
                        cv::rectangle(grid, rect, color, 4);
                    }
                }
                auto board = make_board(count, accuracy, fps, grid.rows);
                cv::Mat view;
                cv::hconcat(board, grid, view);
                cv::imshow(window_name, view);

                int key = cv::waitKey(1);
                if (key == 27)
                {
                    exit_flag.store(true);
                    break;
                }
            }
        }
    }).detach();

    while(!exit_flag.load());
#ifdef __linux__
    sleep(1);
#elif _WIN32
    Sleep(1000);
#endif

    return 0;
}