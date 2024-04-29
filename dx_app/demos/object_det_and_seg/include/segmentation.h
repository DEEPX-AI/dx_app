#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <time.h>
#include <thread>
#include <mutex>
#include <condition_variable>

struct SegmentationParam
{
    int classIndex;
    std::string className;
    uint8_t colorB;
    uint8_t colorG;
    uint8_t colorR;
};
