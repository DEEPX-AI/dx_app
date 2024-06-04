#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "common/objects.hpp"

#include <opencv2/opencv.hpp>

namespace dxapp
{
namespace classification
{
    struct PreConfig {
        std::vector<int64_t> _dstShape;
        int64_t _dstSize;
        AppInputFormat _inputFormat;
        int _alignFactor;
    };
    
    inline void data_pre_processing(uint8_t* src, uint8_t* dst, PreConfig config)
    {
        int copy_size = (int)(config._dstShape[2] * config._dstShape[3]);
        for(int y=0; y<config._dstShape[2]; ++y)
        {
            memcpy(&dst[y * (copy_size + config._alignFactor)],
                   &src[y * copy_size], 
                   copy_size
                   );
        }
    };

    inline void image_pre_processing(cv::Mat& src, uint8_t* input_tensor, PreConfig config){
        cv::Mat dst;
        cv::resize(src.clone(), dst, cv::Size(config._dstShape[1], config._dstShape[2]), 0, 0, cv::INTER_LINEAR);
        if(config._inputFormat == AppInputFormat::IMAGE_RGB)
            cv::cvtColor(dst.clone(), dst, cv::COLOR_BGR2RGB);
        else if(config._inputFormat == AppInputFormat::IMAGE_GRAY && config._dstShape[3] == 1)
            cv::cvtColor(dst.clone(), dst, cv::COLOR_BGR2GRAY);
        
        if(config._alignFactor>0)
            data_pre_processing(dst.data, input_tensor, config);
        else
            memmove(input_tensor, dst.data, config._dstSize);
    };
    
} // namespace classification
} // namespace dxapp
