/**
 * @file roi_instance_mask.hpp
 * @brief Box-ROI instance-mask generation for YOLO-family segmentation.
 *
 * Produces a per-instance binary mask at original-image resolution while doing
 * all the expensive work (sigmoid dot-product + resize) over only the object's
 * bounding-box region, instead of the naive "sigmoid the whole prototype ->
 * resize full prototype to model input -> resize full input to original" path
 * that upsamples a large area immediately thrown away.
 *
 * This mirrors the ROI approach the YOLOv8-Seg path already uses by default.
 * The result differs from the full-resolution path only at sub-pixel mask
 * boundaries (measured mask IoU ~0.99), which is why the YOLOv8-Seg path adopts
 * it as its normal path.
 *
 * Header-only, C++14, depends only on OpenCV + PreprocessContext.
 */

#ifndef DXAPP_ROI_INSTANCE_MASK_HPP
#define DXAPP_ROI_INSTANCE_MASK_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/base/i_processor.hpp"

namespace dxapp {

/**
 * @brief Build a bbox-ROI instance mask at original-image resolution.
 *
 * @param coefs       per-instance mask coefficients (length == proto_c).
 * @param proto_data  prototype tensor data, layout [proto_c, proto_h, proto_w].
 * @param proto_c/h/w prototype dimensions.
 * @param ix1..iy2    bounding box in *model input* coordinates (letterboxed).
 * @param input_w/h   model input size.
 * @param ctx         preprocessing context (scale = input/original gain, pads).
 * @return CV_8UC1 mask (ctx.original_height x ctx.original_width), 255 inside
 *         the instance and 0 elsewhere.
 */
inline cv::Mat roiInstanceMask(const std::vector<float>& coefs,
                               const float* proto_data,
                               int proto_c, int proto_h, int proto_w,
                               float ix1, float iy1, float ix2, float iy2,
                               int input_w, int input_h,
                               const PreprocessContext& ctx) {
    const int ow = ctx.original_width;
    const int oh = ctx.original_height;
    cv::Mat out = cv::Mat::zeros(oh, ow, CV_8UC1);
    if (ow <= 0 || oh <= 0 || proto_c <= 0) return out;

    const float gain = std::max(ctx.scale, 1e-6f);
    const float ratio_w = static_cast<float>(proto_w) / std::max(input_w, 1);
    const float ratio_h = static_cast<float>(proto_h) / std::max(input_h, 1);

    // Exact box in original-image space (tight crop applied at the end).
    int ox1 = static_cast<int>(std::floor((ix1 - ctx.pad_x) / gain));
    int oy1 = static_cast<int>(std::floor((iy1 - ctx.pad_y) / gain));
    int ox2 = static_cast<int>(std::ceil((ix2 - ctx.pad_x) / gain));
    int oy2 = static_cast<int>(std::ceil((iy2 - ctx.pad_y) / gain));
    ox1 = std::max(0, std::min(ox1, ow));
    ox2 = std::max(0, std::min(ox2, ow));
    oy1 = std::max(0, std::min(oy1, oh));
    oy2 = std::max(0, std::min(oy2, oh));
    if (ox2 <= ox1 || oy2 <= oy1) return out;

    // Prototype crop covering the box (expanded to whole prototype cells).
    int px1 = std::max(0, static_cast<int>(std::floor(ix1 * ratio_w)));
    int py1 = std::max(0, static_cast<int>(std::floor(iy1 * ratio_h)));
    int px2 = std::min(proto_w, static_cast<int>(std::ceil(ix2 * ratio_w)));
    int py2 = std::min(proto_h, static_cast<int>(std::ceil(iy2 * ratio_h)));
    if (px2 <= px1 || py2 <= py1) return out;

    // sigmoid(coefs . proto) over just the ROI.
    const int rw = px2 - px1;
    const int rh = py2 - py1;
    const int plane = proto_h * proto_w;
    cv::Mat roi(rh, rw, CV_32FC1);
    for (int ph = py1; ph < py2; ++ph) {
        float* rrow = roi.ptr<float>(ph - py1);
        for (int pw = px1; pw < px2; ++pw) {
            float val = 0.0f;
            const int base = ph * proto_w + pw;
            for (int c = 0; c < proto_c; ++c)
                val += coefs[c] * proto_data[c * plane + base];
            rrow[pw - px1] = 1.0f / (1.0f + std::exp(-val));
        }
    }

    // Original-space span the crop covers, and its size (aligned resize).
    const float cx0 = (px1 / ratio_w - ctx.pad_x) / gain;
    const float cy0 = (py1 / ratio_h - ctx.pad_y) / gain;
    const int dst_w = std::max(1, static_cast<int>(std::round((px2 - px1) / ratio_w / gain)));
    const int dst_h = std::max(1, static_cast<int>(std::round((py2 - py1) / ratio_h / gain)));
    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    // The resized crop lands at (dx0, dy0); write the intersection with the
    // bbox directly into `out` (thresholded), no full-frame scratch buffer.
    const int dx0 = static_cast<int>(std::round(cx0));
    const int dy0 = static_cast<int>(std::round(cy0));
    const int xa = std::max(dx0, ox1);
    const int xb = std::min(dx0 + dst_w, ox2);
    const int ya = std::max(dy0, oy1);
    const int yb = std::min(dy0 + dst_h, oy2);
    if (xb <= xa || yb <= ya) return out;
    for (int y = ya; y < yb; ++y) {
        const float* src = resized.ptr<float>(y - dy0);
        uchar* dst = out.ptr<uchar>(y);
        for (int x = xa; x < xb; ++x)
            dst[x] = (src[x - dx0] > 0.5f) ? 255 : 0;
    }
    return out;
}

}  // namespace dxapp

#endif  // DXAPP_ROI_INSTANCE_MASK_HPP
