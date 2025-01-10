#pragma once

#include <list>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <map>
#include <time.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "bbox.h"

void DisplayBoundingBox(cv::Mat &frame, std::vector<BoundingBox> &result, 
    float OriginHeight, float OriginWidth, std::string frameTitle, std::string frameText, 
    cv::Scalar UniformColor, std::vector<cv::Scalar> ObjectColors,
    std::string OutputImgFile, int DisplayDuration, int Category=-1, bool ImageCenterAligned=false);
std::vector<cv::Scalar> GetObjectColors(int type=0);