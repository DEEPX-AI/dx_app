/**
 * @file colorspace.hpp
 * @brief BT.601 limited-range (studio swing) YCbCr conversions.
 *
 * OpenCV's COLOR_BGR2GRAY / COLOR_BGR2YCrCb use the **full-range** convention:
 * Y spans [0, 255] with no offset. MATLAB's rgb2ycbcr — the convention almost
 * every published super-resolution model is trained with — uses **limited
 * range**: Y spans [16, 235], Cb/Cr are centred on 128 with a +-112 swing.
 *
 * Feeding a full-range Y to a model trained on limited-range Y puts the input
 * outside its training domain: the error is +20 at white and -16 at black. The
 * two chains are each internally self-consistent, so no global colour cast
 * appears; the damage shows up only through the network's non-linearities, as
 * lost detail and sharpness.
 *
 * This is the C++ mirror of src/python_example/common/utility/colorspace.py —
 * keep the two in sync.
 *
 * Channel order note: dx_app follows OpenCV and stores the chroma planes as
 * **YCrCb** (Cr before Cb), so these helpers do too.
 *
 * References: ITU-R BT.601; MATLAB rgb2ycbcr / ycbcr2rgb.
 */

#ifndef DXAPP_COLORSPACE_HPP
#define DXAPP_COLORSPACE_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace dxapp {
namespace colorspace {

constexpr int kYLimitedMin = 16;
constexpr int kYLimitedMax = 235;

// Forward coefficients on the 0-255 scale (MATLAB rgb2ycbcr / 255).
constexpr float kKr = 65.481f / 255.0f;
constexpr float kKg = 128.553f / 255.0f;
constexpr float kKb = 24.966f / 255.0f;
constexpr float kCbR = -37.797f / 255.0f;
constexpr float kCbG = -74.203f / 255.0f;
constexpr float kCbB = 112.0f / 255.0f;
constexpr float kCrR = 112.0f / 255.0f;
constexpr float kCrG = -93.786f / 255.0f;
constexpr float kCrB = -18.214f / 255.0f;

// Inverse coefficients on the 0-255 scale.
constexpr float kIy = 255.0f / 219.0f;  // 1.164383
constexpr float kIrCr = 1.596027f;
constexpr float kIgCb = -0.391762f;
constexpr float kIgCr = -0.812968f;
constexpr float kIbCb = 2.017232f;

/// cv::saturate_cast<uchar>(float) already rounds to nearest and clamps to
/// [0, 255] — do NOT add 0.5f here, that double-rounds (235 -> 236).
inline uchar saturate(float v) {
    return cv::saturate_cast<uchar>(v);
}

/**
 * @brief BGR (CV_8UC3) -> limited-range Y plane (CV_8UC1) in [16, 235].
 *
 * Drop-in replacement for cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY) when the
 * consumer is a model trained on MATLAB rgb2ycbcr Y (e.g. ESPCN).
 */
inline void bgrToYLimited(const cv::Mat& bgr, cv::Mat& y_out) {
    if (bgr.channels() == 1) { y_out = bgr.clone(); return; }
    CV_Assert(bgr.type() == CV_8UC3);
    y_out.create(bgr.rows, bgr.cols, CV_8UC1);
    for (int r = 0; r < bgr.rows; ++r) {
        const cv::Vec3b* src = bgr.ptr<cv::Vec3b>(r);
        uchar* dst = y_out.ptr<uchar>(r);
        for (int c = 0; c < bgr.cols; ++c) {
            const float b = static_cast<float>(src[c][0]);
            const float g = static_cast<float>(src[c][1]);
            const float rr = static_cast<float>(src[c][2]);
            dst[c] = saturate(kYLimitedMin + kKr * rr + kKg * g + kKb * b);
        }
    }
}

/**
 * @brief BGR (CV_8UC3) -> limited-range CV_8UC3 in OpenCV **YCrCb** order.
 *
 * Drop-in replacement for cv::cvtColor(src, dst, cv::COLOR_BGR2YCrCb).
 */
inline void bgrToYCrCbLimited(const cv::Mat& bgr, cv::Mat& ycrcb_out) {
    CV_Assert(bgr.type() == CV_8UC3);
    ycrcb_out.create(bgr.rows, bgr.cols, CV_8UC3);
    for (int r = 0; r < bgr.rows; ++r) {
        const cv::Vec3b* src = bgr.ptr<cv::Vec3b>(r);
        cv::Vec3b* dst = ycrcb_out.ptr<cv::Vec3b>(r);
        for (int c = 0; c < bgr.cols; ++c) {
            const float b = static_cast<float>(src[c][0]);
            const float g = static_cast<float>(src[c][1]);
            const float rr = static_cast<float>(src[c][2]);
            dst[c][0] = saturate(kYLimitedMin + kKr * rr + kKg * g + kKb * b);
            dst[c][1] = saturate(128.0f + kCrR * rr + kCrG * g + kCrB * b);
            dst[c][2] = saturate(128.0f + kCbR * rr + kCbG * g + kCbB * b);
        }
    }
}

/**
 * @brief Limited-range YCrCb (CV_8UC3) -> BGR (CV_8UC3).
 *
 * Drop-in replacement for cv::cvtColor(src, dst, cv::COLOR_YCrCb2BGR).
 */
inline void ycrcbLimitedToBgr(const cv::Mat& ycrcb, cv::Mat& bgr_out) {
    CV_Assert(ycrcb.type() == CV_8UC3);
    bgr_out.create(ycrcb.rows, ycrcb.cols, CV_8UC3);
    for (int r = 0; r < ycrcb.rows; ++r) {
        const cv::Vec3b* src = ycrcb.ptr<cv::Vec3b>(r);
        cv::Vec3b* dst = bgr_out.ptr<cv::Vec3b>(r);
        for (int c = 0; c < ycrcb.cols; ++c) {
            const float y = static_cast<float>(src[c][0]) - kYLimitedMin;
            const float cr = static_cast<float>(src[c][1]) - 128.0f;
            const float cb = static_cast<float>(src[c][2]) - 128.0f;
            const float yy = kIy * y;
            dst[c][0] = saturate(yy + kIbCb * cb);
            dst[c][1] = saturate(yy + kIgCb * cb + kIgCr * cr);
            dst[c][2] = saturate(yy + kIrCr * cr);
        }
    }
}

}  // namespace colorspace
}  // namespace dxapp

#endif  // DXAPP_COLORSPACE_HPP
