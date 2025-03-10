#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <common/objects.hpp>

namespace dxapp
{
namespace common
{
    
    inline float calcIoU(dxapp::common::BBox a, dxapp::common::BBox b)
    {
        float a_w = a._xmax - a._xmin;
        float a_h = a._ymax - a._ymin;
        float b_w = b._xmax - b._xmin;
        float b_h = b._ymax - b._ymin;
        float overlap_w = std::min(a._xmax, b._xmax) - std::max(a._xmin, b._xmin);
        if(overlap_w < 0) overlap_w = 0.f;
        float overlap_h = std::min(a._ymax, b._ymax) - std::max(a._ymin, b._ymin);
        if(overlap_h < 0) overlap_h = 0.f;

        float overlap_area = overlap_w * overlap_h;
        float a_area = a_w * a_h;
        float b_area = b_w * b_h;

        return overlap_area / (a_area + b_area - overlap_area);
    }

    inline dxapp::common::Object scalingObject(dxapp::common::Object src, dxapp::common::Size_f padSize, dxapp::common::Size_f scaleRatio)
    {
        dxapp::common::Object dst = src;
        dst._bbox._xmin = (src._bbox._xmin - padSize._width) * scaleRatio._width;
        dst._bbox._ymin = (src._bbox._ymin - padSize._height) * scaleRatio._height;
        dst._bbox._xmax = (src._bbox._xmax - padSize._width) * scaleRatio._width;
        dst._bbox._ymax = (src._bbox._ymax - padSize._height) * scaleRatio._height;
        dst._bbox._width = src._bbox._width * scaleRatio._width;
        dst._bbox._height = src._bbox._height * scaleRatio._height;
        for(auto &kpt:dst._bbox._kpts)
        {
            if(kpt._z > 0)
            {
                kpt._x = (kpt._x - padSize._width) * scaleRatio._width;
                kpt._y = (kpt._y - padSize._height) * scaleRatio._height;
            }
        }
        return dst;
    };

    inline void nms(std::vector<dxapp::common::BBox> rawBoxes, std::vector<std::vector<std::pair<float, int>>> &scoreIndices, float iou_threshold, std::map<uint16_t, std::string> &classes, dxapp::common::Size_f padSize, dxapp::common::Size_f scaleRatio, dxapp::common::DetectObject &result)
    {
        for(size_t idx=0;idx<scoreIndices.size();idx++) // class 
        {
            auto& _indices = scoreIndices[idx];
            for(size_t j=0;j<_indices.size();j++)  
            {
                if(_indices[j].first == 0.0f) continue;

                for(size_t k=j+1;k<_indices.size();k++)
                {
                    if(_indices[k].first == 0.f) continue;
                    float iou = calcIoU(rawBoxes[_indices[j].second], rawBoxes[_indices[k].second]);
                    if(iou >= iou_threshold)
                    {
                        _indices[k].first = 0.0f;
                    }
                }
            }
        }
        for(size_t idx=0;idx<scoreIndices.size();idx++)
        {
            auto _indices = scoreIndices[idx];
            for(size_t j=0;j<_indices.size();j++)
            {
                if(_indices[j].first > 00.f)
                {
                    dxapp::common::Object obj;
                    obj._bbox = rawBoxes[_indices[j].second];
                    obj._conf = _indices[j].first;
                    obj._classId = static_cast<int>(idx);
                    obj._name = classes.at(static_cast<int>(idx));

                    result._detections.emplace_back(scalingObject(obj, padSize, scaleRatio));
                }
            }
        }
        std::sort(result._detections.begin(),result._detections.end(), 
                    [](const dxapp::common::Object &a, const dxapp::common::Object &b)
                    {
                    return a._conf > b._conf;
                    });
        result._num_of_detections = result._detections.size();
    };
    
    

} // namespace decode
} // namespace dxapp 