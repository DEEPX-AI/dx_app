#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

typedef enum{
    CLASSIFICATION = 0,
    DETECTION,
    SEGMENTATION,
    YOLOPOSE,
    FACEID, 
}AppUsage;

typedef enum{
    OFFLINE = 0,
    REALTIME,
    NONE,
}AppType;

typedef enum {
    IMAGE = 0,
    VIDEO,
    CAMERA,
    ISP,
    ETHERNET,
    BINARY,
    CSV,
    MULTI,
}AppInputType;

typedef enum {
    IMAGE_BGR = 0,
    IMAGE_RGB,
    IMAGE_GRAY,
    INPUT_BINARY,
}AppInputFormat;

typedef enum {
    OUTPUT_ARGMAX = 0,
    OUTPUT_NONE_ARGMAX,
    OUTPUT_YOLO,
    OUTPUT_SSD,
    OUTPUT_FD,
}AppOutputType;

struct AppSourceInfo
{
    AppInputType inputType;
    std::string inputPath;
    int numOfFrames;
};

namespace dxapp
{
namespace common
{
    template<typename _T> struct Size_{
        _T _width;
        _T _height;

        bool operator==(const Size_& a)
        {
            if(_width == a._width && _height == a._height){return true;}
            else{return false;}
        };
        Size_<_T>(_T width, _T height)
        {
            this->_width = width;
            this->_height = height;
        };
        Size_<_T>()
        {
            this->_width = 0;
            this->_height = 0;
        };
    };

    typedef Size_<int> Size;
    typedef Size_<float> Size_f;
    
    template<typename _T> struct Point_{
        _T _x;
        _T _y;
        _T _z;

        bool operator==(const Point_& a)
        {
            if(_x == a._x && _y == a._y && _z == a._z){return true;}
            else{return false;}
        };
        Point_<_T>(_T x, _T y, _T z=0)
        {
            this->_x = x;
            this->_y = y;
            this->_z = z;
        };
        Point_<_T>()
        {
            this->_x = 0;
            this->_y = 0;
            this->_z = 0;
        };
    };

    typedef Point_<int> Point;
    typedef Point_<float> Point_f;

    struct BBox
    {
        float _xmin;
        float _ymin;
        float _xmax;
        float _ymax;
        float _width;
        float _height;
    };
    struct Object
    {
        BBox _bbox;
        float _conf;
        int _classId;
        std::string _name;
    };

    struct DetectObject
    {
        std::vector<Object> _detections;
        int _num_of_detections;
    };

    struct ClsObject
    {
        int _top1;
        std::vector<float> _scores;
        std::string _name;
    };
    
} // namespace common
} // namespace dxapp