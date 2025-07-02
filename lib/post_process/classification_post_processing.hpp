#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <map>

#include "dxrt/dxrt_api.h"
#include "common/objects.hpp"
#include "utils/common_util.hpp"

#include <opencv2/opencv.hpp>

namespace dxapp
{
namespace classification
{
    struct PostConfig {
        std::vector<int64_t> _outputShape;
        AppOutputType _outputType;
        std::map<uint16_t, std::string> _classes;
    };

    template<typename T>
    inline int getArgMax(T* input, int size)
    {
        int max_idx = 0;
        max_idx = 0;
        for(int i=0;i<size;i++){
            if(input[max_idx]<input[i]){
                max_idx = i;
            }
        }
        return max_idx;
    }

    inline std::vector<float> getSoftmax(float* data, int size){
        std::vector<float> result;
        std::vector<double> exp_data;
        double exp_data_sum = 0.0;
        
        for(int i=0;i<size;i++){
            exp_data.emplace_back(std::exp(data[i]));
        }

        exp_data_sum = (double)std::accumulate(exp_data.begin(), exp_data.end(), 0.0);

        for(int i=0;i<size;i++){
            result.emplace_back((float)(exp_data[i]/exp_data_sum));
        }
        return result;
    }
    
    inline common::ClsObject postProcessing(std::vector<std::shared_ptr<dxrt::Tensor>> outputs, PostConfig config){
        common::ClsObject result;
        if(config._outputType==AppOutputType::OUTPUT_ARGMAX)
        {
            result._top1 = *(uint16_t*)outputs.front()->data();
            result._name = config._classes.at(result._top1);
        }
        else
        {
            result._scores = classification::getSoftmax((float*)outputs.back()->data(), config._classes.size());
            result._top1 = classification::getArgMax(result._scores.data(), config._classes.size());
            result._name = config._classes.at(result._top1);
        }
        return result;
    }

    inline cv::Mat resultViewer(std::string path, common::ClsObject result)
    {
        cv::Mat viewer;
        char comments[100];

        cv::Mat image, resized, input;
        if (dxapp::common::getExtension(path) == "bin")
        {
            viewer = cv::Mat(cv::Size(224, 224), CV_8UC3);
            dxapp::common::readBinary(path, viewer.data);
        }
        else
        {
            viewer = cv::imread(path, cv::IMREAD_COLOR);
        }
        cv::resize(viewer.clone(), viewer, cv::Size(640, 640), 0, 0, 1);
        snprintf(comments, 100, "Top1 Result : class %d (%s)", result._top1, result._name.c_str());
        cv::putText(viewer, comments, cv::Point(10,viewer.size().height-20), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255,255,0), 2);
        return viewer;
    }

} // namespace classification
} // namespace dxapp
