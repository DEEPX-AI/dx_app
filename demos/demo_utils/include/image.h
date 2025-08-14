#pragma once

#include <opencv2/opencv.hpp>


void PreProc(cv::Mat& src, cv::Mat &dest, bool keepRatio=true, bool bgr2rgb=true, uint8_t padValue=0);