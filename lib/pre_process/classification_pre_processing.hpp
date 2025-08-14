#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include <common/objects.hpp>

#include <opencv2/opencv.hpp>

namespace dxapp
{
namespace classification
{
    struct PreConfig {
        std::vector<int64_t> _dstShape;
        uint64_t _dstSize;
        AppInputFormat _inputFormat;
    };
    
    inline void image_pre_processing(cv::Mat& src, uint8_t* input_tensor, PreConfig config){
        cv::Mat dst;
        cv::resize(src.clone(), dst, cv::Size(config._dstShape[1], config._dstShape[1]), 0, 0, cv::INTER_LINEAR);
        if(config._inputFormat == AppInputFormat::IMAGE_RGB)
            cv::cvtColor(dst.clone(), dst, cv::COLOR_BGR2RGB);
        else if(config._inputFormat == AppInputFormat::IMAGE_GRAY && config._dstShape[3] == 1)
            cv::cvtColor(dst.clone(), dst, cv::COLOR_BGR2GRAY);
        memmove(input_tensor, dst.data, config._dstSize);
    };
    
} // namespace classification
} // namespace dxapp
