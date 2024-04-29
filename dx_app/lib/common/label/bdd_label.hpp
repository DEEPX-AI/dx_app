#pragma once

#include <map>

namespace dxapp
{
namespace bdd
{
    static int bdd_numClasses = 10;
    // For Object Detection, The Field ID range from 1 instead of 0
    static std::map<uint16_t, std::string> bdd_od_labels = {
            {1  , "pedestrian"},
            {2  , "rider"},
            {3  , "car"},
            {4  , "truck"},
            {5  , "bus"},
            {6  , "train"},
            {7  , "motorcycle"},
            {8  , "bicycle"},
            {9  , "traffic light"},
            {10 , "traffic sign"},
    };
}
} // namespace dxapp