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
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cmath>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "dxrt/dxrt_api.h"
#include "miware_include/MI_diagnosis_label.hpp"

using namespace std;
using namespace diagnosis;


static struct option const opts[] = {
    { "model", required_argument, 0, 'm' },
    { "input", required_argument, 0, 'i' },
    { "help", no_argument, 0, 'h' },
    { 0, 0, 0, 0 }
};
const char* usage =
"simple classification demo from binary file\n"
"please using argmax model\n"
"  -m, --model     define model path\n"
"  -i, --input     define input data path\n"
"  -h, --help      show help\n"
;
void help()
{
    cout << usage << endl;    
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

int DataDump(void* ptr, int dump_size, string file_name)
{
    ofstream outfile(file_name, ios::binary);
    if(!outfile.is_open())
    {
        cout << "can not open file "<< endl;
        return -1;
    }
    outfile.write((char*) ptr, dump_size);
    outfile.close();
    return 1;
}

void readCSV(std::ifstream* file, std::string filePath, float* dst)
{
    string value;
    file->open(filePath);
    for(int i=0; i<2500; i++){
        std::getline(*file, value);
        dst[i] = stof(value);
    }
}

void readBinary(std::string filePath, float* dst, int size, int elemSize)
{
    FILE *fp = NULL;
    fp = fopen(filePath.c_str(), "rb");
    fread((void*)dst, size, elemSize, fp);
    fclose(fp);
}

void calibration(float* src, int8_t* dst, double multiply, double sum, int size){
    for(int i=0; i<size; i++)
    {
        src[i] = std::round((src[i] * multiply) + sum);
        if(src[i] < -128)
            dst[i] = -128;
        else if(src[i] > 127)
            dst[i] = 127;
        else
            dst[i] = (int8_t)src[i];
    }
}

// get argmax
int postProcessing(float* output, int size){
    int max_idx = 0;   
    for(int i=0;i<size;i++){
        if(output[max_idx]<output[i]){
            max_idx = i;
        }
    }
    return max_idx;
}
// get softmax
std::vector<float> softmax(float* data, int size){
    std::vector<float> result;
    std::vector<double> exp_data;
    double exp_data_sum = 0.f;
    for(int i=0;i<size;i++){
        exp_data.emplace_back(std::exp(data[i]));
    }
    exp_data_sum = (double)std::accumulate(exp_data.begin(), exp_data.end(), 0.0);
    for(int i=0;i<size;i++){
        result.emplace_back((float)(exp_data[i]/exp_data_sum));
    }
    return result;
}

void im2col(int8_t* src, int8_t* dst, int input_h,int input_w,int output_h,int output_w,int stride_h,int stride_w,int kernel_h,int kernel_w,int output_channel)
{
    const int CHANNEL_UNIT = 64; 

    const int in_channel_size = kernel_h * kernel_w * output_channel;
    const int npu_channel_shape = std::ceil(static_cast<float>(in_channel_size) / CHANNEL_UNIT) * CHANNEL_UNIT;

    for (int c_idx = 0; c_idx < std::ceil(static_cast<float>(in_channel_size) / CHANNEL_UNIT); ++c_idx) {
        for (int h = 0; h < output_h; ++h) {
            for (int w = 0; w < output_w; ++w) {
                int window_start_addr = w * stride_w * output_channel + h * stride_h * output_channel * input_w;
                int in_c_iter_num = (c_idx == std::ceil(static_cast<float>(in_channel_size) / CHANNEL_UNIT) - 1)
                                        ? in_channel_size % CHANNEL_UNIT
                                        : CHANNEL_UNIT;
                if (in_c_iter_num == 0) {
                    in_c_iter_num = CHANNEL_UNIT;
                }
                for (int c = 0; c < in_c_iter_num; ++c) {
                    int im2col_c_idx = c_idx * CHANNEL_UNIT + c;
                    int in_c_idx = im2col_c_idx / output_channel;
                    int in_h_idx = in_c_idx / kernel_w;
                    int in_w_idx = in_c_idx % kernel_w;
                    dst[c_idx * output_h * output_w * CHANNEL_UNIT + h * output_w * CHANNEL_UNIT + w * CHANNEL_UNIT + c] =
                        src[window_start_addr + (im2col_c_idx % output_channel) + (in_w_idx * output_channel) + (in_h_idx * input_w * output_channel)];
                }
            }
        }
    }

}

int main(int argc, char *argv[])
{
    int optCmd;
    string modelPath="";
    string inputPath = "";

    if(argc==1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }    

    while ((optCmd = getopt_long(argc, argv, "m:i:h", opts,
        NULL)) != -1) {
        switch (optCmd) {
            case '0':
                break;
            case 'm':
                modelPath = strdup(optarg);
                break;
            case 'i':
                inputPath = strdup(optarg);
                break;
            case 'h':
            default:
                help();
                exit(0);
                break;
        }
    }
    
    LOG_VALUE(modelPath);
    LOG_VALUE(inputPath);

    if(!inputPath.empty())
    {
        auto ie = dxrt::InferenceEngine(modelPath);
        float* inputRegacy = new float[2500];
        int8_t* calibratedTensor = new int8_t[2500];
        int8_t* inputTensor = new int8_t[158144];

        if(std::filesystem::is_directory(inputPath))
        {
            int correct = 0; int fail = 0; int total = 0;
            for (auto& keyValue:diagnosis_labels){
                total++;
                std::string inputCls = inputPath + "/" + keyValue.second;
                if(std::filesystem::is_directory(inputCls))
                {
                    std::ifstream file;
                    for(auto &inputFile:std::filesystem::directory_iterator(inputCls))
                    {
                        readCSV(&file, inputFile.path(), inputRegacy);
                        calibration(inputRegacy, calibratedTensor, 0.12669560313224792, 30.071500778198242, 2500);
                        im2col(calibratedTensor, inputTensor, 2500, 1, 2471, 1, 1, 1, 30, 1, 1);
                        auto outputs = ie.Run(inputTensor);
                        auto output_ptr = outputs.back()->data();
                        int outputChannel = ie.outputs().front().shape()[3];
                        auto softmax_output = softmax((float*)output_ptr, outputChannel);
                        int argmax = postProcessing(&softmax_output.front(),outputChannel);
                        std::cout << " GT : " << keyValue.second << " OUTPUT : " << diagnosis_labels.at(argmax) << std::endl;
                        if(argmax == keyValue.first){
                            correct++;
                        }else{
                            fail++;
                        }
                        std::cout << " correct : " << correct << ", fail : " << fail <<std::endl;
                        std::cout << std::endl;
                        file.close();            
                    }
                }
            }
        }
        else
        {
            std::filesystem::path filepath = inputPath;
            if(filepath.extension() == ".bin")
            {
                readBinary(inputPath, inputRegacy, 2500, sizeof(float));
                calibration(inputRegacy, calibratedTensor, 0.12669560313224792, 30.071500778198242, 2500);
                DataDump(calibratedTensor, 2500, "./pre_output.bin" );
                im2col(calibratedTensor, inputTensor, 2500, 1, 2471, 1, 1, 1, 30, 1, 1);
                DataDump(inputTensor, 2471 * 64, "./im2col_output.bin");
                auto outputs = ie.Run(inputTensor);
                auto output_ptr = outputs.back()->data();
                DataDump(output_ptr, outputs.back()->elem_size(), "./npu_output.bin");
            }
        }
    }
    return 0;
}