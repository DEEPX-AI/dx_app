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
namespace yolo
{
    class PreProcessing
    {
    private:
        dxapp::common::Size _oldSize;
        dxapp::common::Size _newSize;
        AppInputFormat _inputFormat;
        int _fillPadvalue = 114;
        float _preprocRatio;
        dxapp::common::Size _resizeSize;
        float _dw, _dh;
        int _top, _bottom, _left, _right;
        /* data */
    public:
        PreProcessing(dxapp::common::Size oldSize, dxapp::common::Size newSize, AppInputFormat inputFormat)
           :_oldSize(oldSize), _newSize(newSize), _inputFormat(inputFormat)
           {
                _preprocRatio = std::min((float)_newSize._width/_oldSize._width, (float)_newSize._height/_oldSize._height);
                _resizeSize = dxapp::common::Size((int)(_oldSize._width * _preprocRatio), (int)(_oldSize._height * _preprocRatio));
                _dw = (_newSize._width - _resizeSize._width) / 2.f;
                _dh = (_newSize._height - _resizeSize._height) / 2.f;

                _top = std::max((int)std::round(_dh - 0.1), 0);
                _bottom = std::max((int)std::round(_dh + 0.1), 0);
                _left = std::max((int)std::round(_dw - 0.1), 0);
                _right = std::max((int)std::round(_dw + 0.1), 0);
                if(newSize == oldSize)
                {
                    _fillPadvalue = -1;
                }
                else 
                {
                    _fillPadvalue = 114;
                }
                
           };
           PreProcessing(){};
           ~PreProcessing(){};
        void run(cv::Mat& src, cv::Mat& dst);
        
    };

    void PreProcessing::run(cv::Mat& src, cv::Mat& dst)
    {
        cv::Mat resized;
        if(_fillPadvalue < 0)
        { // resize src image
            cv::resize(src, dst, cv::Size(_newSize._width, _newSize._height), 0, 0, cv::INTER_LINEAR);
        }
        else
        { // make letterbox 
            cv::resize(src, resized, cv::Size(_resizeSize._width, _resizeSize._height), 0, 0, cv::INTER_LINEAR);
            cv::copyMakeBorder(resized, dst, _top, _bottom, _left, _right, cv::BORDER_CONSTANT, cv::Scalar(_fillPadvalue,_fillPadvalue,_fillPadvalue));
        }
        if(_inputFormat==AppInputFormat::IMAGE_RGB){
            cv::cvtColor(dst.clone(),dst, cv::COLOR_BGR2RGB);
        }
    };

} // namespace yolo

} // namespace dxapp
