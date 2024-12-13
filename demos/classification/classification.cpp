#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <getopt.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syslog.h>

#include <opencv2/opencv.hpp>

#include "dxrt/dxrt_api.h"

using namespace std;

constexpr int input_w = 224, input_h = 224, input_c = 3;

static struct option const opts[] = {
    { "model", required_argument, 0, 'm' },
    { "image",  required_argument, 0, 'i' },    
    { "loop", no_argument, 0, 'l' },
    { "help", no_argument, 0, 'h' },
    { 0, 0, 0, 0 }
};
const char* usage =
"simple classification demo from image file\n"
"please using argmax model\n"
"  -m, --model     define model path\n"
"  -i, --image     using image file input\n"
"  -l, --loop      loop test\n"
"  -h, --help      show help\n"
;
void help()
{
    cout << usage << endl;    
}

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
    int optCmd;
    string modelPath="", imgFile="";    
    bool loopTest = false;

    if(argc==1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }    

    while ((optCmd = getopt_long(argc, argv, "m:i:alh", opts,
        NULL)) != -1) {
        switch (optCmd) {
            case '0':
                break;
            case 'm':
                modelPath = strdup(optarg);
                break;
            case 'i':
                imgFile = strdup(optarg);
                break;
            case 'l':
                loopTest = true;
                break;
            case 'h':
            default:
                help();
                exit(0);
                break;
        }
    }
    LOG_VALUE(modelPath);
    LOG_VALUE(imgFile);

    // for align64
    int align_factor = ((int)(input_w * input_c))&(-64);
    align_factor = (input_w * input_c) - align_factor;

    if(!imgFile.empty())
    {
        dxrt::InferenceEngine ie(modelPath);
        bool usingOrt = false;
        for (const auto& task_str : ie.task_order()) {
            if (task_str.find("cpu") != std::string::npos) {
                usingOrt = true;
                break;
            }
        }
        
        do 
        {
            cv::Mat image, resized, input;
            image = cv::imread(imgFile, cv::IMREAD_COLOR);
            cv::resize(image, resized, cv::Size(input_w, input_h));
            cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);
            
            vector<uint8_t> inputBuf(ie.input_size());

            if(usingOrt)
            {
                ie.outputs().front().type();
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
                if(usingOrt && ie.outputs().front().type() == dxrt::DataType::FLOAT)
                {
                    int batch_size = outputs.front()->shape()[0];
                    int class_size = outputs.front()->shape()[1];
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