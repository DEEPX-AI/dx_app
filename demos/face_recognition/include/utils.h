#pragma once

#include <opencv2/opencv.hpp>

double GetTimestamp(void);
void data_pre_processing(uint8_t* src, uint8_t* dst, int shape, int align);
cv::Mat preprocess(cv::Mat image, cv::Size size);

cv::Rect get_rect(float *box, int image_w, int image_h);
float get_iou(cv::Rect rect1, cv::Rect rect2);

std::vector<cv::Point2f> get_landmark(float *fl_data, int w_crop, int h_crop, float x, float y);
void visualize_landmark(cv::Mat image, std::vector<cv::Point2f> landmark);
cv::Mat warp(cv::Mat image, std::vector<cv::Point2f> landmark);

float cos_sim(float *a, float *b, int s);

class Instance
{
public:
    int id;
    cv::Rect box;
    Instance();
    Instance(int _id, cv::Rect _box);
};

class Tracker
{
public:
    int id;
    float iou_threshold;
    std::vector<Instance> T;
    Tracker(float _iou_threshold);

    void run(std::vector<cv::Rect> D);
};

class FaceData
{
public:
    int id;
    cv::Mat image;
    float feature_vector[512];
    FaceData();
    FaceData(int _id, cv::Mat _image, float *_feature_vector);
    ~FaceData();
};