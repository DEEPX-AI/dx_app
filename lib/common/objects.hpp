#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif

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
    RTSP,
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
    int numOfFrames = -1;
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
        Size_<_T>(const Size_<_T>& other) = default;
        Size_<_T>& operator=(const Size_<_T>& other) = default;
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
        Point_<_T>(const Point_<_T>& other) = default;
        Point_<_T>& operator=(const Point_<_T>& other) = default;
    };

    typedef Point_<int> Point;
    typedef Point_<float> Point_f;

    struct BBox
    {
        float _xmin = 0.f;
        float _ymin = 0.f;
        float _xmax = 0.f;
        float _ymax = 0.f;
        float _width = 0.f;
        float _height = 0.f;
        std::vector<Point_f> _kpts = {};
        friend std::ostream& operator<<(std::ostream& os, const BBox& a)
        {
            os << a._xmin << ", " << a._ymin << ", " << a._width << ", " << a._height;
            return os;
        };
    };
    struct Object
    {
        BBox _bbox;
        float _conf = 0.f;
        int _classId = 0;
        std::string _name = "";
        friend std::ostream& operator<<(std::ostream& os, const Object& a)
        {
            os << "obj info : " << a._classId << " : " << a._bbox ;
            return os;
        };
        Object() = default;
        Object(const Object& other) = default;
        Object& operator=(const Object& other) = default;
    };

    struct DetectObject
    {
        std::vector<Object> _detections = {};
        int _num_of_detections = 0;
        friend std::ostream& operator<<(std::ostream& os, const DetectObject& a)
        {
            os << "detected : " << a._num_of_detections ;
            return os;
        };
    };

    struct ClsObject
    {
        int _top1 = 0;
        std::vector<float> _scores = {};
        std::string _name = "";
    };
    
} // namespace common
} // namespace dxapp