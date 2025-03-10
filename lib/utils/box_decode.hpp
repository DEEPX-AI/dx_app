#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <common/objects.hpp>

namespace dxapp
{
namespace decode
{
    inline dxapp::common::BBox yoloBasicDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        UNUSEDVAR(scale);
        auto data = datas[0];
        dxapp::common::BBox box_temp;
        box_temp._xmin = (activation(data[0]) * 2. - 0.5 + grid._x ) * stride; //center x
        box_temp._ymin = (activation(data[1]) * 2. - 0.5 + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
        dxapp::common::BBox result;
        result._xmin = box_temp._xmin - box_temp._width / 2.f;
        result._ymin = box_temp._ymin - box_temp._height / 2.f;
        result._xmax = box_temp._xmin + box_temp._width / 2.f;
        result._ymax = box_temp._ymin + box_temp._height / 2.f;
        result._width = box_temp._width;
        result._height = box_temp._height;

        return result;
    };

    inline dxapp::common::BBox yoloScaledDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        auto data = datas[0];
        dxapp::common::BBox box_temp;
        box_temp._xmin = (activation(data[0] * scale - 0.5 * (scale - 1)) + grid._x ) * stride; //center x
        box_temp._ymin = (activation(data[1] * scale - 0.5 * (scale - 1)) + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
        dxapp::common::BBox result;
        result._xmin = box_temp._xmin - box_temp._width / 2.f;
        result._ymin = box_temp._ymin - box_temp._height / 2.f;
        result._xmax = box_temp._xmin + box_temp._width / 2.f;
        result._ymax = box_temp._ymin + box_temp._height / 2.f;
        result._width = box_temp._width;
        result._height = box_temp._height;

        return result;
    };

    inline dxapp::common::BBox yoloXDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        UNUSEDVAR(anchor);
        UNUSEDVAR(scale);
        auto data = datas[0];
        dxapp::common::BBox box_temp;
        box_temp._xmin = (data[0] + grid._x ) * stride; //center x
        box_temp._ymin = (data[1] + grid._y ) * stride; //center y
        box_temp._width = activation(data[2]) * stride;
        box_temp._height = activation(data[3]) * stride;
        dxapp::common::BBox result;
        result._xmin = box_temp._xmin - box_temp._width / 2.f;
        result._ymin = box_temp._ymin - box_temp._height / 2.f;
        result._xmax = box_temp._xmin + box_temp._width / 2.f;
        result._ymax = box_temp._ymin + box_temp._height / 2.f;
        result._width = box_temp._width;
        result._height = box_temp._height;
        result._kpts = {dxapp::common::Point_f(-1, -1, -1)};
        return result;
    };

    inline dxapp::common::BBox yoloPoseDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float kpt_length)
    {
        auto data = datas[0];
        auto keypoints_0_3 = datas[1];
        auto keypoints_4_16 = datas[2];
        dxapp::common::BBox box_temp;
        std::vector<dxapp::common::Point_f> p_temp;
        box_temp._xmin = (activation(data[0]) * 2. - 0.5 + grid._x ) * stride; //center x
        box_temp._ymin = (activation(data[1]) * 2. - 0.5 + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
        for(int kptIdx = 0; kptIdx < (int)kpt_length; ++kptIdx)
        {
            int idx = kptIdx * 3;
            float *keypointsData;
            if(kptIdx < 4)
                keypointsData = keypoints_0_3;
            else
            {
                keypointsData = keypoints_4_16;    
                idx = (kptIdx - 4) * 3;
            }
            if(activation(keypointsData[idx + 2])<0.5)
            {
                box_temp._kpts.emplace_back(dxapp::common::Point_f(-1, -1));
            }
            else
            {
                box_temp._kpts.emplace_back(dxapp::common::Point_f(
                                (keypointsData[idx + 0] * 2 - 0.5 + grid._x) * stride,
                                (keypointsData[idx + 1] * 2 - 0.5 + grid._y) * stride,
                                0.5f
                                ));
            }
        }
        dxapp::common::BBox result;
        result._xmin = box_temp._xmin - box_temp._width / 2.f;
        result._ymin = box_temp._ymin - box_temp._height / 2.f;
        result._xmax = box_temp._xmin + box_temp._width / 2.f;
        result._ymax = box_temp._ymin + box_temp._height / 2.f;
        result._width = box_temp._width;
        result._height = box_temp._height;
        result._kpts = box_temp._kpts;

        return result;
    };

    inline dxapp::common::BBox SCRFDDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float kpt_length)
    {
        auto data = datas[0];
        auto keypoints = datas[1];
        dxapp::common::BBox box_temp;
        std::vector<dxapp::common::Point_f> p_temp;
        box_temp._xmin = (grid._x - data[0]) * stride;
        box_temp._ymin = (grid._y - data[1]) * stride;
        box_temp._xmax = (grid._x + data[2]) * stride;
        box_temp._ymax = (grid._y + data[3]) * stride;
        box_temp._width = box_temp._xmax - box_temp._xmin;
        box_temp._height = box_temp._ymax - box_temp._ymin;

        for(int kptIdx = 0; kptIdx < (int)kpt_length; ++kptIdx)
        {
            int idx = kptIdx * 2;
            box_temp._kpts.emplace_back(dxapp::common::Point_f(
                            (grid._x + keypoints[idx + 0]) * stride,
                            (grid._y + keypoints[idx + 1]) * stride,
                            0.5f
                            ));
        }

        return box_temp;
    };

    inline dxapp::common::BBox yoloCustomDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        /**
         * @brief adding your decode method
         * 
         * example code ..
         * 
         *      auto data = datas[0];
         *      dxapp::common::BBox box_temp;
         *      box_temp._xmin = (activation(data[0]) * 2. - 0.5 + grid._x ) * stride; //center x
         *      box_temp._ymin = (activation(data[1]) * 2. - 0.5 + grid._y ) * stride; //center y
         *      box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
         *      box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
         *      dxapp::common::BBox result = {
         *              ._xmin=box_temp._xmin - box_temp._width / 2.f,
         *              ._ymin=box_temp._ymin - box_temp._height / 2.f,
         *              ._xmax=box_temp._xmin + box_temp._width / 2.f,
         *              ._ymax=box_temp._ymin + box_temp._height / 2.f,
         *              ._width = box_temp._width,
         *              ._height = box_temp._height,
         *      };
         * 
         */

        dxapp::common::BBox result;

        return result;
    };

} // namespace decode
} // namespace dxapp 