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
    inline dxapp::common::BBox yoloBasicDecode(std::function<float(float)> activation, std::vector<float> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        UNUSEDVAR(scale);
        dxapp::common::BBox box_temp;
        box_temp._xmin = (activation(datas[0]) * 2. - 0.5 + grid._x ) * stride; //center x
        box_temp._ymin = (activation(datas[1]) * 2. - 0.5 + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(datas[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(datas[3]) * 2.f), 2) * anchor._height;
        dxapp::common::BBox result;
        result._xmin = box_temp._xmin - box_temp._width / 2.f;
        result._ymin = box_temp._ymin - box_temp._height / 2.f;
        result._xmax = box_temp._xmin + box_temp._width / 2.f;
        result._ymax = box_temp._ymin + box_temp._height / 2.f;
        result._width = box_temp._width;
        result._height = box_temp._height;

        return result;
    };

    inline dxapp::common::BBox yoloScaledDecode(std::function<float(float)> activation, std::vector<float> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        dxapp::common::BBox box_temp;
        box_temp._xmin = (activation(datas[0] * scale - 0.5 * (scale - 1)) + grid._x ) * stride; //center x
        box_temp._ymin = (activation(datas[1] * scale - 0.5 * (scale - 1)) + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(datas[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(datas[3]) * 2.f), 2) * anchor._height;
        dxapp::common::BBox result;
        result._xmin = box_temp._xmin - box_temp._width / 2.f;
        result._ymin = box_temp._ymin - box_temp._height / 2.f;
        result._xmax = box_temp._xmin + box_temp._width / 2.f;
        result._ymax = box_temp._ymin + box_temp._height / 2.f;
        result._width = box_temp._width;
        result._height = box_temp._height;

        return result;
    };

    inline dxapp::common::BBox yoloFaceDecode(std::function<float(float)> activation, std::vector<float> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        UNUSEDVAR(scale);
        dxapp::common::BBox box_temp;
        box_temp._xmin = (activation(datas[0]) * 2. - 0.5 + grid._x ) * stride; //center x
        box_temp._ymin = (activation(datas[1]) * 2. - 0.5 + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(datas[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(datas[3]) * 2.f), 2) * anchor._height;
        dxapp::common::BBox result;
        result._xmin = box_temp._xmin - box_temp._width / 2.f;
        result._ymin = box_temp._ymin - box_temp._height / 2.f;
        result._xmax = box_temp._xmin + box_temp._width / 2.f;
        result._ymax = box_temp._ymin + box_temp._height / 2.f;
        result._width = box_temp._width;
        result._height = box_temp._height;
        for(int i = 0; i < 5; i++)
        {
            float kx = datas[4 + (i * 2)] * anchor._width + (grid._x * stride);
            float ky = datas[5 + (i * 2)] * anchor._height + (grid._y * stride);
            result._kpts.emplace_back(dxapp::common::Point_f(kx, ky, 0.5f));
        }

        return result;
    };

    inline dxapp::common::BBox yoloXDecode(std::function<float(float)> activation, std::vector<float> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        UNUSEDVAR(anchor);
        UNUSEDVAR(scale);
        dxapp::common::BBox box_temp;
        box_temp._xmin = (datas[0] + grid._x ) * stride; //center x
        box_temp._ymin = (datas[1] + grid._y ) * stride; //center y
        box_temp._width = activation(datas[2]) * stride;
        box_temp._height = activation(datas[3]) * stride;
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

    inline dxapp::common::BBox yoloCustomDecode(std::function<float(float)> activation, std::vector<float> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        (void)activation; // Suppress unused parameter warning
        (void)datas;      // Suppress unused parameter warning
        (void)grid;       // Suppress unused parameter warning
        (void)anchor;     // Suppress unused parameter warning
        (void)stride;     // Suppress unused parameter warning
        (void)scale;      // Suppress unused parameter warning
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