#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "dxapp_api.hpp"

#include <opencv2/opencv.hpp>

namespace dxapp
{
namespace classification
{
    struct PreConfig {
        std::vector<int64_t> _dstShape;
        AppInputFormat _inputFormat;
        int _alignFactor;
        bool _needIm2Col;
    };

    void image_pre_processing(cv::Mat src, uint8_t* input_tensor, PreConfig config){
        cv::Mat dst;
        cv::resize(src.clone(), dst, cv::Size(config._dstShape[1], config._dstShape[2]), 0, 0, cv::INTER_LINEAR);
        if(config._inputFormat == AppInputFormat::IMAGE_RGB)
            cv::cvtColor(dst.clone(), dst, cv::COLOR_BGR2RGB);
        else if(config._inputFormat == AppInputFormat::IMAGE_GRAY && config._dstShape[3] == 1)
            cv::cvtColor(dst.clone(), dst, cv::COLOR_BGR2GRAY);
        
        if(config._alignFactor>0)
        {
            int copy_size = config._dstShape[1] * config._dstShape[3];
            for(int y=0;y<config._dstShape[2];y++){
                memcpy(&input_tensor[y*(copy_size + config._alignFactor)], 
                       &dst.data[y * copy_size], copy_size);
            }
        }
        else
        {
            input_tensor = dst.data;
        }
    };
    void binary_pre_processing(void* src, void* dst, PreConfig config)
    {
        // TODO : add im2col or array align function 
    };
        
    
} // namespace classification
} // namespace dxapp
