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
#include <cxxopts.hpp>

#include "dxrt/dxrt_api.h"
#include "utils/common_util.hpp"

using namespace std;

int getArgMax(float* output_data, int number_of_classes)
{
    int max_idx = 0;
    for(int i=0;i<number_of_classes;i++)
    {
        if(output_data[max_idx] < output_data[i])
        {
            max_idx = i;
        }
    }
    return max_idx;
}

int main(int argc, char *argv[])
{
    string modelPath="", imgFile="";    
    bool loopTest = false;
    int input_w = 224, input_h = 224, input_c = 3, class_size = 1000;

    std::string app_name = "classification";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()
        ("m, model_path", "classification model file (.dxnn, required)", cxxopts::value<std::string>(modelPath))
        ("i, image_path", "input image file path(jpg, png, jpeg ...)", cxxopts::value<std::string>(imgFile))
        ("width, input_width", "input width size (default : 224)", cxxopts::value<int>(input_w)->default_value("224"))
        ("height, intpu_height", "input height size (default : 224)", cxxopts::value<int>(input_h)->default_value("224"))
        ("class, class_size", "number of classes (default : 1000)", cxxopts::value<int>(class_size)->default_value("1000"))
        ("l, loop", "loops to test", cxxopts::value<bool>(loopTest)->default_value("false"))
        ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help") || modelPath.empty())
    {
        std::cout << options.help() << endl;
        exit(0);
    }
    LOG_VALUE(modelPath)
    LOG_VALUE(imgFile)
    LOG_VALUE(loopTest)
    
    dxrt::InferenceEngine ie(modelPath);

    // for align64
    int align_factor = dxapp::common::get_align_factor((input_w * input_c),64);

    if(!imgFile.empty())
    {
        bool usingOrt = dxapp::common::checkOrtLinking();
        
        do 
        {
            cv::Mat image, resized, input;
            image = cv::imread(imgFile, cv::IMREAD_COLOR);
            cv::resize(image, resized, cv::Size(input_w, input_h));
            cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);
            
            vector<uint8_t> inputBuf(ie.input_size());

            if(usingOrt)
            {
                memcpy(&inputBuf[0], &input.data[0], ie.input_size());
            }
            else
            {
                for(int y = 0; y < input_h; y++)
                {
                    memcpy(&inputBuf[y*(input_w * input_c + align_factor)], &input.data[y * input_w * input_c], input_w * input_c);
                }
            }
            auto outputs = ie.Run(inputBuf.data());
            
            if(!outputs.empty())
            {
                if(ie.outputs().front().type() == dxrt::DataType::FLOAT)
                {
                    if(usingOrt)
                        class_size = outputs.front()->shape()[1];
                    auto result = getArgMax((float*)outputs.front()->data(), class_size);
                    cout << "Top1 Result : class " << result << endl;
                }
                else
                {
                    auto result = *(uint16_t*)outputs.front()->data();
                    cout << "Top1 Result : class " << result << endl;
                }
            }
        } while(loopTest);
    }
    return 0;
}