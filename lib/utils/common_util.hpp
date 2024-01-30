#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <utils/color_table.hpp>

namespace dxapp
{
namespace common
{
    int get_align_factor(int length, int based)
    {
        return based - (length - (length & (-based)));
    }
    
    template<typename T>
    void readBinary(std::string filePath, T* dst, int elemSize)
    {
        std::FILE *fp = NULL;
        fp = std::fopen(filePath.c_str(), "rb");
        std::fseek(fp, 0, SEEK_END);
        auto size = ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        fread((void*)dst, size, elemSize, fp);
        fclose(fp);
    }

    void dumpBinary(void *ptr, int dump_size, std::string file_name)
    {
        std::ofstream outfile(file_name, std::ios::binary);
        if(!outfile.is_open())
        {
            std::cout << "cna not open file " << file_name << std::endl;
            std::terminate();
        }
        outfile.write((char*)ptr, dump_size);
        outfile.close();
    }
    
    void readCSV(std::string filePath, float* dst, int size)
    {
        std::ifstream file;
        std::string value;
        file.open(filePath);
        for(int i=0; i<size; i++){
            std::getline(file, value);
            dst[i] = std::stof(value);
        }
        file.close();
    }
    
    int divideBoard(int numImages)
    {
        int ret_Div = 1;
        if(numImages < 2) ret_Div = 1;
        else if(numImages < 5) ret_Div = 2;
        else if(numImages < 10) ret_Div = 3;
        else if(numImages < 17) ret_Div = 4;
        else if(numImages < 26) ret_Div = 5;
        else if(numImages < 37) ret_Div = 6;
        else if(numImages < 50) ret_Div = 7;
        return ret_Div;
    }

    template<typename T>
    void show(std::vector<T> vec)
    {
        std::cout << "\n[ ";
        for(auto &v:vec)
        {
            std::cout << std::dec << v << ", " ;
        }
        std::cout << " ]" << std::endl;
    };

    void drawBox(cv::Mat& dst, dxapp::common::Object obj)
    {
        cv::rectangle(dst, cv::Rect(obj._bbox._xmin, obj._bbox._ymin, obj._bbox._width, obj._bbox._height), dxapp::common::color_table[obj._classId], 2);
    };

    void drawLabel(cv::Mat& dst, dxapp::common::Object obj)
    {
        int textBaseLine = 0;
        auto textSize = cv::getTextSize(obj._name, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &textBaseLine);
        cv::rectangle(dst, cv::Point(obj._bbox._xmin, obj._bbox._ymin - textSize.height),
                            cv::Point(obj._bbox._xmin + textSize.width, obj._bbox._ymin),
                            dxapp::common::color_table[obj._classId], cv::FILLED);
        cv::putText(dst, obj._name, cv::Point(obj._bbox._xmin, obj._bbox._ymin), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255));
        
    };

} // namespace common
} // namespace dxapp 