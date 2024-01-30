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
    dxapp::common::BBox yoloBasicDecode(std::function<float(float)> activation, float* data, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        dxapp::common::BBox box_temp;
        box_temp._xmin = (activation(data[0]) * 2. - 0.5 + grid._x ) * stride; //center x
        box_temp._ymin = (activation(data[1]) * 2. - 0.5 + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
        dxapp::common::BBox result = {
            ._xmin=box_temp._xmin - box_temp._width / 2.f,
            ._ymin=box_temp._ymin - box_temp._height / 2.f,
            ._xmax=box_temp._xmin + box_temp._width / 2.f,
            ._ymax=box_temp._ymin + box_temp._height / 2.f,
            ._width = box_temp._width,
            ._height = box_temp._height,
        };

        return result;
    };

    dxapp::common::BBox yoloScaledDecode(std::function<float(float)> activation, float* data, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        dxapp::common::BBox box_temp;
        box_temp._xmin = (activation(data[0] * scale - 0.5 * (scale - 1)) + grid._x ) * stride; //center x
        box_temp._ymin = (activation(data[1] * scale - 0.5 * (scale - 1)) + grid._y ) * stride; //center y
        box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
        box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
        dxapp::common::BBox result = {
            ._xmin=box_temp._xmin - box_temp._width / 2.f,
            ._ymin=box_temp._ymin - box_temp._height / 2.f,
            ._xmax=box_temp._xmin + box_temp._width / 2.f,
            ._ymax=box_temp._ymin + box_temp._height / 2.f,
            ._width = box_temp._width,
            ._height = box_temp._height,
        };

        return result;
    };

    dxapp::common::BBox yoloXDecode(std::function<float(float)> activation, float* data, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        dxapp::common::BBox box_temp;
        box_temp._xmin = (data[0] + grid._x ) * stride; //center x
        box_temp._ymin = (data[1] + grid._y ) * stride; //center y
        box_temp._width = activation(data[2]) * stride;
        box_temp._height = activation(data[3]) * stride;
        dxapp::common::BBox result = {
            ._xmin=box_temp._xmin - box_temp._width / 2.f,
            ._ymin=box_temp._ymin - box_temp._height / 2.f,
            ._xmax=box_temp._xmin + box_temp._width / 2.f,
            ._ymax=box_temp._ymin + box_temp._height / 2.f,
            ._width = box_temp._width,
            ._height = box_temp._height,
        };
        return result;
    };

    dxapp::common::BBox yoloCustomDecode(std::function<float(float)> activation, float* data, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
    {
        /**
         * @brief adding your decode method
         * 
         * example code ..
         * 
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