/**
 * @file sfa3d_bev_preprocessor.hpp
 * @brief BEV preprocessor for SFA3D LiDAR models
 */

#ifndef SFA3D_BEV_PREPROCESSOR_HPP
#define SFA3D_BEV_PREPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <opencv2/opencv.hpp>

namespace dxapp {

class SFA3DBEVPreprocessor : public IPreprocessor {
public:
    SFA3DBEVPreprocessor(int input_width, int input_height)
        : input_width_(input_width), input_height_(input_height) {}

    void process(const cv::Mat& input, cv::Mat& output,
                 PreprocessContext& ctx) override {
        ctx.original_width = input.cols;
        ctx.original_height = input.rows;
        ctx.input_width = input_width_;
        ctx.input_height = input_height_;
        ctx.scale_x = static_cast<float>(input_width_) / std::max(input.cols, 1);
        ctx.scale_y = static_cast<float>(input_height_) / std::max(input.rows, 1);
        ctx.scale = std::min(ctx.scale_x, ctx.scale_y);
        ctx.source_image = input;

        cv::Mat resized;
        if (input.cols != input_width_ || input.rows != input_height_) {
            cv::resize(input, resized, cv::Size(input_width_, input_height_));
        } else {
            resized = input;
        }

        if (resized.type() != CV_8UC3) {
            cv::Mat u8;
            if (resized.type() == CV_32FC3) {
                resized.convertTo(u8, CV_8UC3, 255.0);
            } else {
                resized.convertTo(u8, CV_8UC3);
            }
            output = u8;
        } else {
            output = resized;
        }
    }

    int getInputWidth() const override { return input_width_; }
    int getInputHeight() const override { return input_height_; }
    int getColorConversion() const override { return -1; }
    std::string getModelName() const { return "sfa3d_bev"; }

private:
    int input_width_;
    int input_height_;
};

}  // namespace dxapp

#endif  // SFA3D_BEV_PREPROCESSOR_HPP
