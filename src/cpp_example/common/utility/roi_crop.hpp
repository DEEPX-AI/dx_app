/**
 * @file roi_crop.hpp
 * @brief The one rule that turns a detection box into a crop window.
 *
 * The graph engine uses it (RouteRois in common/graph/roi_router.cpp) so
 * every second-stage crop is cut from a detection box the same way.
 *
 * The rule:
 *   1. pad: grow the box by `pad` x its width (height) on each side, around
 *      its own centre; pad <= 0 leaves it as is;
 *   2. clip: clamp the corners to the frame in float, then truncate the left
 *      and top corner and the clamped width and height to int, once each.
 * A box with no area inside the frame gives an empty cv::Rect; a box less
 * than one pixel wide (high) inside the frame gives a zero width (height).
 * Callers skip a crop whose width or height is not positive.
 */

#ifndef DXAPP_COMMON_UTILITY_ROI_CROP_HPP
#define DXAPP_COMMON_UTILITY_ROI_CROP_HPP

#include <algorithm>

#include <opencv2/core.hpp>

namespace dxapp {

/// Symmetric pad around the box's own centre; angle-agnostic, so it is safe
/// to reuse for both axis-aligned boxes and the OBB un-rotate path.
inline cv::Rect2f PadBox(const cv::Rect2f& box, float pad) {
    if (pad <= 0.f) return box;
    const float dx = box.width * pad;
    const float dy = box.height * pad;
    return cv::Rect2f(box.x - dx, box.y - dy, box.width + dx * 2.f,
                      box.height + dy * 2.f);
}

/// Clamp in float to a cols x rows frame, then truncate the corner and the
/// size once. Empty when nothing of the box is inside the frame.
inline cv::Rect ClipBoxToFrame(const cv::Rect2f& box, int cols, int rows) {
    const float x1 = std::max(0.f, box.x);
    const float y1 = std::max(0.f, box.y);
    const float x2 = std::min(static_cast<float>(cols), box.x + box.width);
    const float y2 = std::min(static_cast<float>(rows), box.y + box.height);
    if (x2 <= x1 || y2 <= y1) return cv::Rect();
    return cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
}

/// PadBox, then ClipBoxToFrame: the crop window of `box` in a cols x rows
/// frame.
inline cv::Rect PaddedCropRect(const cv::Rect2f& box, float pad, int cols, int rows) {
    return ClipBoxToFrame(PadBox(box, pad), cols, rows);
}

}  // namespace dxapp

#endif  // DXAPP_COMMON_UTILITY_ROI_CROP_HPP
