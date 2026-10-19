#include "common/graph/graph_visualizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "common/graph/roi_router.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/visualizers/obb_visualizer.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

namespace dxapp {
namespace graph {

cv::Point2f RestorePointWarped(const cv::Point2f& roi_local, const RoiRef& ref) {
    // Bit-exact for a full-frame result on the source frame; a full-frame
    // result from a handed-off image carries its scale in inv_align.
    if (!ref.from_roi && IsIdentityAffine(ref.inv_align)) return roi_local;
    // ref.inv_align alone maps a crop-local coordinate to the source frame
    // - RouteRois (roi_router.cpp) folds the crop's own translation into
    // inv_align at construction time, so nothing here adds ref.src_box a
    // second time (that was the fix round 3 defect: composing this as
    // inv_align*local + src_box double-counted/misordered the translation
    // for every warped crop - see roi_router.cpp's RouteRois comment).
    const cv::Matx23f& m = ref.inv_align;
    return cv::Point2f(m(0, 0) * roi_local.x + m(0, 1) * roi_local.y + m(0, 2),
                       m(1, 0) * roi_local.x + m(1, 1) * roi_local.y + m(1, 2));
}

namespace {

// ---------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------

cv::Rect ClampRect(const cv::Rect2f& r, const cv::Size& bounds) {
    const float x1 = std::max(0.f, r.x);
    const float y1 = std::max(0.f, r.y);
    const float x2 = std::min(static_cast<float>(bounds.width), r.x + r.width);
    const float y2 = std::min(static_cast<float>(bounds.height), r.y + r.height);
    if (x2 <= x1 || y2 <= y1) return cv::Rect();
    return cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
}

/**
 * @brief The canvas region a stage result's own array-shaped data (a
 *        labelmap, densemap, image or per-instance mask) covers, for every
 *        origin that keeps a rectangle a rectangle (see NeedsWarp).
 *
 * A full-frame stage's data spans the whole canvas. An ROI stage's data
 * spans the window it was cut from, anchored at `ref.src_box`'s top-left
 * (already in source-frame coordinates by RoiRef's own contract - see
 * shape.hpp) and sized by `ref.crop_size`, falling back to `ref.src_box`'s
 * own size when `crop_size` is unset (0,0) - a hand-built `RoiRef` in a
 * test, most commonly - rather than collapsing the region to nothing.
 *
 * One exception, for a crop cut from a scaled (handed-off) image: its
 * `inv_align` is a pure per-axis scale + translation, so `crop_size`
 * pixels do NOT cover `crop_size` source pixels. Its region is the crop's
 * own rectangle mapped through `inv_align` (RestoreBox) - exact, since a
 * scale keeps a rectangle a rectangle. A pure translation (every plain
 * crop) keeps the anchor formula above, bit-exact. A full-frame result -
 * including a hand-off child's dense result, which covers the whole
 * source - still spans the canvas.
 *
 * A rotated or aligned crop (OBB, face5) is not a rectangle on the canvas
 * at all and never reaches this function: it is warped (WarpOntoCanvas).
 */
cv::Rect TargetRegion(const cv::Mat& canvas, const RoiRef& origin) {
    if (!origin.from_roi) return cv::Rect(0, 0, canvas.cols, canvas.rows);
    const bool have_crop_size = origin.crop_size.width > 0 && origin.crop_size.height > 0;
    const cv::Matx23f& m = origin.inv_align;
    const bool scaled_crop =
        m(0, 1) == 0.f && m(1, 0) == 0.f && (m(0, 0) != 1.f || m(1, 1) != 1.f);
    if (scaled_crop) {
        const cv::Rect2f local(
            0.f, 0.f,
            have_crop_size ? static_cast<float>(origin.crop_size.width) : origin.src_box.width,
            have_crop_size ? static_cast<float>(origin.crop_size.height) : origin.src_box.height);
        return ClampRect(RestoreBox(local, origin), canvas.size());
    }
    const cv::Rect2f box(
        origin.src_box.x, origin.src_box.y,
        have_crop_size ? static_cast<float>(origin.crop_size.width) : origin.src_box.width,
        have_crop_size ? static_cast<float>(origin.crop_size.height) : origin.src_box.height);
    return ClampRect(box, canvas.size());
}

/**
 * @brief True when an ROI result's own pixels must be warped onto the
 *        canvas rather than resized into a rectangle.
 *
 * That is every ROI origin whose `inv_align` carries rotation, shear or a
 * flip - an OBB crop, a face5 crop, either of them cut from a handed-off
 * view. A full-frame result, a plain crop (pure translation) and a crop
 * from a scaled view (positive per-axis scale) keep TargetRegion's
 * rectangle, pixel for pixel as before.
 */
bool NeedsWarp(const RoiRef& origin) {
    if (!origin.from_roi) return false;
    const cv::Matx23f& m = origin.inv_align;
    return !(m(0, 1) == 0.f && m(1, 0) == 0.f && m(0, 0) > 0.f && m(1, 1) > 0.f);
}

/// The pixel size an ROI result's own data stands for: `crop_size`, or
/// `src_box`'s size when a hand-built ref leaves `crop_size` unset.
cv::Size CropPixels(const RoiRef& origin) {
    if (origin.crop_size.width > 0 && origin.crop_size.height > 0) return origin.crop_size;
    return cv::Size(static_cast<int>(origin.src_box.width),
                    static_cast<int>(origin.src_box.height));
}

/**
 * @brief One ROI result's pixels, placed on the canvas through its
 *        origin's `inv_align`.
 *
 * `local` must already be `CropPixels(origin)` in size. `inv_align` maps a
 * crop-local pixel to the source frame, i.e. it is exactly warpAffine's
 * forward (src -> dst) matrix. Only the canvas rectangle the crop can
 * touch is computed: `region` is the clamped bounding box of the crop's
 * pixel footprint (grown by a pixel), and the warp's translation is shifted
 * by `region`'s corner. Inside `region` that gives what a canvas-sized warp
 * gives, up to warpAffine fixed-point rounding (the shifted translation can
 * round a coordinate one step differently); outside it a canvas-sized warp
 * has nothing.
 *
 * `content` is warped with `interpolation` and replicated borders (so a
 * linear warp does not darken the edge); `coverage` is an all-255 mask of
 * `local`'s size warped the same way with nearest-neighbour and a zero
 * border - 255 exactly where the crop's pixels land. Callers blend or copy
 * under `coverage` only. Returns false when the crop misses the canvas.
 */
bool WarpOntoCanvas(const cv::Mat& local, const RoiRef& origin, const cv::Size& canvas_size,
                    int interpolation, cv::Rect* region, cv::Mat* content,
                    cv::Mat* coverage) {
    if (local.empty()) return false;
    const std::vector<cv::Point2f> corners = RestoreBoxCorners(
        cv::Rect2f(-1.f, -1.f, static_cast<float>(local.cols) + 2.f,
                   static_cast<float>(local.rows) + 2.f),
        origin);
    float min_x = corners[0].x;
    float min_y = corners[0].y;
    float max_x = corners[0].x;
    float max_y = corners[0].y;
    for (std::size_t i = 1; i < corners.size(); ++i) {
        min_x = std::min(min_x, corners[i].x);
        min_y = std::min(min_y, corners[i].y);
        max_x = std::max(max_x, corners[i].x);
        max_y = std::max(max_y, corners[i].y);
    }
    const float x1 = std::floor(min_x);
    const float y1 = std::floor(min_y);
    *region = ClampRect(cv::Rect2f(x1, y1, std::ceil(max_x) + 1.f - x1,
                                   std::ceil(max_y) + 1.f - y1),
                        canvas_size);
    if (region->width <= 0 || region->height <= 0) return false;

    const cv::Matx23f& m = origin.inv_align;
    cv::Mat shifted(2, 3, CV_64F);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) shifted.at<double>(r, c) = m(r, c);
    }
    shifted.at<double>(0, 2) -= region->x;
    shifted.at<double>(1, 2) -= region->y;

    cv::warpAffine(local, *content, shifted, region->size(), interpolation,
                   cv::BORDER_REPLICATE);
    const cv::Mat all_on(local.size(), CV_8UC1, cv::Scalar(255));
    cv::warpAffine(all_on, *coverage, shifted, region->size(), cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    return true;
}

/**
 * @brief Blend (alpha > 0) or copy (alpha == 0) `local` - an ROI result's
 *        pixels in crop space, of any size - onto a warped origin's
 *        footprint: resized to the crop's pixel size first, then warped.
 */
void WarpBlend(cv::Mat* canvas, const cv::Mat& local, const RoiRef& origin,
               int interpolation, double alpha) {
    cv::Mat sized = local;
    const cv::Size crop = CropPixels(origin);
    if (crop.width <= 0 || crop.height <= 0) return;
    if (local.size() != crop) cv::resize(local, sized, crop, 0, 0, interpolation);
    cv::Rect region;
    cv::Mat content;
    cv::Mat coverage;
    if (!WarpOntoCanvas(sized, origin, canvas->size(), interpolation, &region, &content,
                        &coverage)) {
        return;
    }
    cv::Mat target = (*canvas)(region);
    if (alpha > 0.0) {
        cv::Mat blended;
        cv::addWeighted(target, 1.0 - alpha, content, alpha, 0, blended);
        blended.copyTo(target, coverage);
    } else {
        content.copyTo(target, coverage);
    }
}

cv::Point TextAnchor(const cv::Mat& canvas, const RoiRef& origin) {
    if (!origin.from_roi) return cv::Point(10, 10);
    int x = static_cast<int>(origin.src_box.x);
    int y = static_cast<int>(origin.src_box.y);
    x = std::max(0, std::min(x, canvas.cols - 1));
    y = std::max(0, std::min(y, canvas.rows - 1));
    return cv::Point(x, y);
}

/// The box DrawTextLine fills for line `line_index` of `text` under `anchor`.
cv::Rect TextLineBox(const cv::Size& canvas, const std::string& text, const cv::Point& anchor,
                     int line_index, cv::Point* baseline_at) {
    const int kLineHeight = 16;
    cv::Point pos(anchor.x, anchor.y + kLineHeight * (line_index + 1));
    pos.x = std::max(0, std::min(pos.x, canvas.width - 1));
    pos.y = std::max(kLineHeight, std::min(pos.y, canvas.height - 1));
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
    if (baseline_at != NULL) *baseline_at = pos;
    return cv::Rect(cv::Point(pos.x, pos.y - text_size.height - 2),
                    cv::Point(pos.x + text_size.width + 1, pos.y + baseline + 1));
}

void DrawTextLine(cv::Mat* canvas, const std::string& text, const cv::Point& anchor,
                  int line_index) {
    cv::Point pos;
    const cv::Rect box = TextLineBox(canvas->size(), text, anchor, line_index, &pos);
    cv::rectangle(*canvas, box.tl(), box.br() - cv::Point(1, 1), cv::Scalar(0, 0, 0),
                  cv::FILLED);
    cv::putText(*canvas, text, pos, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(255, 255, 255), 1);
}

// ---------------------------------------------------------------------
// Pass 1: dense layers - images as the backdrop, then label and dense
// maps blended over them; all under everything else.
// ---------------------------------------------------------------------

void DrawLabelMap(cv::Mat* canvas, const StageResult& result) {
    const LabelMapData* label_map =
        dynamic_cast<const LabelMapData*>(result.data.get());
    if (label_map == NULL || label_map->labels.empty()) return;

    if (label_map->binary_mask) {
        const double kAlpha = 0.5;
        const cv::Vec3b& c = label_map->mask_color;
        const cv::Mat colored(label_map->labels.size(), CV_8UC3, cv::Scalar(c[0], c[1], c[2]));
        if (NeedsWarp(result.origin)) {  // documented: the whole footprint
            WarpBlend(canvas, colored, result.origin, cv::INTER_NEAREST, kAlpha);
            return;
        }
        const cv::Rect region = TargetRegion(*canvas, result.origin);
        if (region.width <= 0 || region.height <= 0) return;
        cv::Mat on = label_map->labels != 0;
        cv::Mat resized_on, resized_color, blended;
        cv::resize(on, resized_on, region.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(colored, resized_color, region.size(), 0, 0, cv::INTER_NEAREST);
        cv::addWeighted((*canvas)(region), 1.0 - kAlpha, resized_color, kAlpha, 0, blended);
        blended.copyTo((*canvas)(region), resized_on);
        return;
    }

    const std::vector<cv::Vec3b>& palette = SEGMENTATION_COLORS;
    cv::Mat colored(label_map->labels.size(), CV_8UC3);
    for (int y = 0; y < colored.rows; ++y) {
        for (int x = 0; x < colored.cols; ++x) {
            const int class_id = (label_map->labels.type() == CV_32S)
                ? label_map->labels.at<int>(y, x)
                : static_cast<int>(label_map->labels.at<uchar>(y, x));
            const std::size_t key = static_cast<std::size_t>(std::max(0, class_id));
            colored.at<cv::Vec3b>(y, x) = palette[key % palette.size()];
        }
    }

    const double kAlpha = 0.5;
    if (NeedsWarp(result.origin)) {
        WarpBlend(canvas, colored, result.origin, cv::INTER_NEAREST, kAlpha);
        return;
    }
    const cv::Rect region = TargetRegion(*canvas, result.origin);
    if (region.width <= 0 || region.height <= 0) return;
    cv::Mat resized;
    cv::resize(colored, resized, region.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat blended;
    cv::addWeighted((*canvas)(region), 1.0 - kAlpha, resized, kAlpha, 0, blended);
    blended.copyTo((*canvas)(region));
}

void DrawDenseMap(cv::Mat* canvas, const StageResult& result) {
    const DenseMapData* dense = dynamic_cast<const DenseMapData*>(result.data.get());
    if (dense == NULL || dense->values.empty()) return;
    if (!dense->spatial) return;  // a per-item matrix, not a picture

    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(dense->values, &min_v, &max_v);
    cv::Mat normalized;
    if (max_v > min_v) {
        const double scale = 255.0 / (max_v - min_v);
        dense->values.convertTo(normalized, CV_8UC1, scale, -scale * min_v);
    } else {
        normalized = cv::Mat::zeros(dense->values.size(), CV_8UC1);
    }

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_MAGMA);
    const double kAlpha = 0.5;
    if (NeedsWarp(result.origin)) {
        WarpBlend(canvas, colored, result.origin, cv::INTER_LINEAR, kAlpha);
        return;
    }
    const cv::Rect region = TargetRegion(*canvas, result.origin);
    if (region.width <= 0 || region.height <= 0) return;
    cv::Mat resized;
    cv::resize(colored, resized, region.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat blended;
    cv::addWeighted((*canvas)(region), 1.0 - kAlpha, resized, kAlpha, 0, blended);
    blended.copyTo((*canvas)(region));
}

void DrawImage(cv::Mat* canvas, const StageResult& result) {
    const ImageData* image = dynamic_cast<const ImageData*>(result.data.get());
    if (image == NULL || image->image.empty()) return;

    cv::Mat source_bgr;
    if (image->image.channels() == 1) {
        cv::cvtColor(image->image, source_bgr, cv::COLOR_GRAY2BGR);
    } else {
        source_bgr = image->image;
    }
    if (NeedsWarp(result.origin)) {
        WarpBlend(canvas, source_bgr, result.origin, cv::INTER_LINEAR, 0.0);
        return;
    }
    const cv::Rect region = TargetRegion(*canvas, result.origin);
    if (region.width <= 0 || region.height <= 0) return;
    cv::Mat resized;
    cv::resize(source_bgr, resized, region.size(), 0, 0, cv::INTER_LINEAR);
    resized.copyTo((*canvas)(region));
}

// ---------------------------------------------------------------------
// Pass 2: instance masks. Each instance's box and label are drawn in
// pass 3, by DrawBoxes, so every outline sits above every mask.
// ---------------------------------------------------------------------

void DrawInstances(cv::Mat* canvas, const StageResult& result) {
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(result.data.get());
    if (boxes == NULL) return;
    const bool warp = NeedsWarp(result.origin);
    const cv::Rect region =
        warp ? cv::Rect() : TargetRegion(*canvas, result.origin);
    if (!warp && (region.width <= 0 || region.height <= 0)) return;

    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const BoxItem& item = boxes->items[i];
        if (item.mask.empty()) continue;

        cv::Mat mask_u8;
        if (item.mask.type() == CV_32FC1 || item.mask.type() == CV_64FC1) {
            item.mask.convertTo(mask_u8, CV_8UC1, 255.0);
            cv::threshold(mask_u8, mask_u8, 127, 255, cv::THRESH_BINARY);
        } else {
            item.mask.convertTo(mask_u8, CV_8UC1);
        }

        const std::size_t color_key = (item.track_id >= 0)
            ? static_cast<std::size_t>(item.track_id) : i;
        const cv::Vec3b& c = SEGMENTATION_COLORS[color_key % SEGMENTATION_COLORS.size()];
        const double kAlpha = 0.45;

        if (warp) {
            // The mask, at the crop's pixel size, warped onto the canvas;
            // the colour goes wherever the warped mask is on AND the crop
            // actually lands.
            const cv::Size crop = CropPixels(result.origin);
            if (crop.width <= 0 || crop.height <= 0) return;
            cv::Mat sized = mask_u8;
            if (mask_u8.size() != crop) {
                cv::resize(mask_u8, sized, crop, 0, 0, cv::INTER_NEAREST);
            }
            cv::Rect warped_region;
            cv::Mat warped_mask;
            cv::Mat coverage;
            if (!WarpOntoCanvas(sized, result.origin, canvas->size(), cv::INTER_NEAREST,
                                &warped_region, &warped_mask, &coverage)) {
                continue;
            }
            cv::bitwise_and(warped_mask, coverage, warped_mask);
            cv::Mat roi = (*canvas)(warped_region);
            cv::Mat color_mat(roi.size(), CV_8UC3, cv::Scalar(c[0], c[1], c[2]));
            cv::Mat blended;
            cv::addWeighted(roi, 1.0 - kAlpha, color_mat, kAlpha, 0, blended);
            blended.copyTo(roi, warped_mask);
            continue;
        }

        cv::Mat resized_mask = mask_u8;
        if (mask_u8.size() != region.size()) {
            cv::resize(mask_u8, resized_mask, region.size(), 0, 0, cv::INTER_NEAREST);
        }

        cv::Mat roi = (*canvas)(region);
        cv::Mat color_mat(roi.size(), CV_8UC3, cv::Scalar(c[0], c[1], c[2]));
        cv::Mat blended;
        cv::addWeighted(roi, 1.0 - kAlpha, color_mat, kAlpha, 0, blended);
        blended.copyTo(roi, resized_mask);
    }
}

// ---------------------------------------------------------------------
// Pass 3: boxes, obboxes, keypoints - restored to source coordinates.
// ---------------------------------------------------------------------

// A detector's boxes and an instance stage's boxes alike (kBoxes and
// kInstances share BoxesData): each box restored through its origin, drawn
// with its label by the shared detection drawer.
void DrawBoxes(cv::Mat* canvas, const StageResult& result) {
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(result.data.get());
    if (boxes == NULL || boxes->items.empty()) return;

    std::vector<DetectionResult> detections;
    detections.reserve(boxes->items.size());
    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const BoxItem& item = boxes->items[i];
        const cv::Rect2f restored = RestoreBox(item.box, result.origin);
        DetectionResult det;
        det.box.push_back(restored.x);
        det.box.push_back(restored.y);
        det.box.push_back(restored.x + restored.width);
        det.box.push_back(restored.y + restored.height);
        det.confidence = item.score;
        det.class_id = item.class_id;
        det.class_name = item.class_name;
        detections.push_back(det);
    }

    PreprocessContext ctx;
    ctx.original_width = canvas->cols;
    ctx.original_height = canvas->rows;
    DetectionVisualizer visualizer;
    *canvas = visualizer.draw(*canvas, detections, ctx);
}

// Reconstructing an OBBResult{cx, cy, width, height, angle} from
// RestoreBox's axis-aligned-bounding-box extent plus a separately
// restored angle draws a quad matching neither the true rotated OBB nor a
// conservative AABB the moment the crop's own warp carries rotation.
//
// `item.box` (shape.hpp) is the OBB's UN-rotated axis-aligned extent in
// the crop's own local frame - result_to_shape.hpp builds it as
// `(cx - w/2, cy - h/2, w, h)`, deliberately excluding `item.angle`. The
// box's real shape is that rect rotated by `item.angle` about its own
// centre, still in the crop's local frame - only THEN does `result.
// origin`'s `inv_align` (the crop's own warp) map it into source
// coordinates. Composing these the other way around (restore first,
// rotate second - or, as fix round 3 shipped, never rotate at all) is
// wrong the moment `item.angle != 0`: fix round 4's regression, caught by
// TestRenderReportRotatesObBoxByItemAngle below, was exactly this -
// `RestoreBoxCorners` alone only ever sees the un-rotated rect, so
// `item.angle` never appeared anywhere in this function.
void DrawObBoxes(cv::Mat* canvas, const StageResult& result) {
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(result.data.get());
    if (boxes == NULL || boxes->items.empty()) return;

    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const BoxItem& item = boxes->items[i];
        const cv::Point2f centre(item.box.x + item.box.width * 0.5f,
                                 item.box.y + item.box.height * 0.5f);
        const float cos_a = std::cos(item.angle);
        const float sin_a = std::sin(item.angle);
        const float local_xs[2] = {item.box.x, item.box.x + item.box.width};
        const float local_ys[2] = {item.box.y, item.box.y + item.box.height};

        std::vector<cv::Point> pts;
        pts.reserve(4);
        for (int ix = 0; ix < 2; ++ix) {
            for (int iy = 0; iy < 2; ++iy) {
                // Rotate this un-rotated corner about the box's own centre,
                // in the crop's local frame (matching OBBResult::getCorners
                // - i_processor.hpp - same rotation matrix convention:
                // R(angle) = [[cos,-sin],[sin,cos]] applied to the offset
                // from centre), THEN restore through inv_align.
                const float dx = local_xs[ix] - centre.x;
                const float dy = local_ys[iy] - centre.y;
                const cv::Point2f rotated_local(centre.x + dx * cos_a - dy * sin_a,
                                                centre.y + dx * sin_a + dy * cos_a);
                const cv::Point2f restored =
                    RestorePointWarped(rotated_local, result.origin);
                pts.push_back(cv::Point(static_cast<int>(restored.x),
                                        static_cast<int>(restored.y)));
            }
        }
        // pts is in (ix,iy) = (0,0),(0,1),(1,0),(1,1) order - swap the last
        // two so cv::polylines(..., isClosed=true) traces the quad's
        // actual perimeter, not a self-crossing bowtie.
        std::swap(pts[2], pts[3]);

        const std::size_t color_key =
            static_cast<std::size_t>(std::max(0, item.class_id)) % OBB_CLASS_COLORS.size();
        const cv::Scalar& color = OBB_CLASS_COLORS[color_key];
        cv::polylines(*canvas, pts, true, color, 2);

        const std::string label = item.class_name + ": " +
            std::to_string(static_cast<int>(item.score * 100)) + "%";
        int min_x = pts[0].x;
        int min_y = pts[0].y;
        for (std::size_t c = 1; c < pts.size(); ++c) {
            min_x = std::min(min_x, pts[c].x);
            min_y = std::min(min_y, pts[c].y);
        }
        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const cv::Point label_pt(min_x, min_y - 10);
        cv::rectangle(*canvas, cv::Point(label_pt.x, label_pt.y - text_size.height),
                      cv::Point(label_pt.x + text_size.width, label_pt.y + baseline),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(*canvas, label, label_pt, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}

void DrawKeypoints(cv::Mat* canvas, const StageResult& result) {
    const KeypointsData* kps = dynamic_cast<const KeypointsData*>(result.data.get());
    if (kps == NULL || kps->items.empty()) return;

    std::vector<PoseResult> restored;
    restored.reserve(kps->items.size());
    for (std::size_t i = 0; i < kps->items.size(); ++i) {
        PoseResult pose = kps->items[i];
        if (pose.box.size() >= 4) {
            const cv::Point2f p1 = RestorePointWarped(
                cv::Point2f(pose.box[0], pose.box[1]), result.origin);
            const cv::Point2f p2 = RestorePointWarped(
                cv::Point2f(pose.box[2], pose.box[3]), result.origin);
            pose.box[0] = std::min(p1.x, p2.x);
            pose.box[1] = std::min(p1.y, p2.y);
            pose.box[2] = std::max(p1.x, p2.x);
            pose.box[3] = std::max(p1.y, p2.y);
        }
        for (std::size_t k = 0; k < pose.keypoints.size(); ++k) {
            const cv::Point2f restored_pt = RestorePointWarped(
                cv::Point2f(pose.keypoints[k].x, pose.keypoints[k].y), result.origin);
            pose.keypoints[k].x = restored_pt.x;
            pose.keypoints[k].y = restored_pt.y;
        }
        restored.push_back(pose);
    }

    PreprocessContext ctx;
    ctx.original_width = canvas->cols;
    ctx.original_height = canvas->rows;
    PoseVisualizer visualizer;
    visualizer.setUnorderedKeypoints(kps->unordered);
    *canvas = visualizer.draw(*canvas, restored, ctx);
}

// ---------------------------------------------------------------------
// Pass 4: scores and vectors as text anchored to their parent box.
// ---------------------------------------------------------------------

void DrawScores(cv::Mat* canvas, const StageResult& result) {
    const ScoresData* scores = dynamic_cast<const ScoresData*>(result.data.get());
    if (scores == NULL || scores->items.empty()) return;
    const cv::Point anchor = TextAnchor(*canvas, result.origin);
    for (std::size_t i = 0; i < scores->items.size(); ++i) {
        const ClassificationResult& item = scores->items[i];
        const std::string label = item.class_name + ": " +
            std::to_string(static_cast<int>(item.confidence * 100)) + "%";
        DrawTextLine(canvas, label, anchor, static_cast<int>(i));
    }
}

void DrawVector(cv::Mat* canvas, const StageResult& result) {
    const VectorData* vec = dynamic_cast<const VectorData*>(result.data.get());
    if (vec == NULL) return;
    const cv::Point anchor = TextAnchor(*canvas, result.origin);
    const std::string label = "vec[" + std::to_string(vec->values.size()) + "]";
    DrawTextLine(canvas, label, anchor, 0);
}

// ---------------------------------------------------------------------
// Pass 5: logic nodes - zones, lines, highlighted boxes, labels.
// ---------------------------------------------------------------------

const cv::Scalar kLogicColor(255, 191, 0);  // BGR: light blue
const cv::Scalar kAlertColor(0, 0, 255);    // BGR: red
const int kPanelLineHeight = 20;

cv::Point Pixel(const cv::Point2f& point) {
    return cv::Point(cvRound(point.x), cvRound(point.y));
}

/// White text on a black (alerted: red) box, baseline at `at`, kept inside the canvas.
void DrawLabel(cv::Mat* canvas, const std::string& text, cv::Point at, bool alert) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    at.x = std::max(0, std::min(at.x, canvas->cols - size.width - 1));
    at.y = std::max(size.height + 2, std::min(at.y, canvas->rows - baseline - 1));
    cv::rectangle(*canvas, cv::Point(at.x, at.y - size.height - 2),
                  cv::Point(at.x + size.width, at.y + baseline),
                  alert ? kAlertColor : cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(*canvas, text, at, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

/// The box DrawLabel fills for `text` with its baseline at `at` (before clamping).
cv::Rect LabelBox(const std::string& text, cv::Point at) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    return cv::Rect(at.x, at.y - size.height - 2, size.width + 1, size.height + 2 + baseline + 1);
}

/// `box` kept inside the canvas the way DrawLabel keeps a label inside it.
cv::Rect InsideCanvas(cv::Rect box, const cv::Size& canvas) {
    box.x = std::max(0, std::min(box.x, canvas.width - box.width));
    box.y = std::max(0, std::min(box.y, canvas.height - box.height));
    return box;
}

/// `box` moved up (`up`) or down until it overlaps nothing in `taken`.
cv::Rect PastTaken(cv::Rect box, bool up, const cv::Size& canvas,
                   const std::vector<cv::Rect>& taken) {
    for (std::size_t tries = 0; tries <= taken.size(); ++tries) {
        const cv::Rect placed = InsideCanvas(box, canvas);
        bool moved = false;
        for (std::size_t i = 0; i < taken.size() && !moved; ++i) {
            if ((placed & taken[i]).area() == 0) continue;
            box.y = up ? taken[i].y - box.height - 1 : taken[i].y + taken[i].height + 1;
            moved = true;
        }
        if (!moved) break;
    }
    return InsideCanvas(box, canvas);
}

/// The box of a label laid beside `text.at` towards `text.away`
/// (OverlayText): `kGap` pixels off `at` on each axis that leans that way,
/// centred on it on an axis that does not, then moved on along `away`'s
/// vertical lean (down when it has none) past every box in `taken`.
cv::Rect BesideAnchor(const OverlayText& text, const cv::Size& canvas,
                      const std::vector<cv::Rect>& taken) {
    const int kGap = 4;
    const float kLean = 0.2f;
    cv::Rect box = LabelBox(text.text, cv::Point(0, 0));
    const cv::Point at = Pixel(text.at);
    box.x = at.x - box.width / 2;
    if (text.away.x > kLean) box.x = at.x + kGap;
    if (text.away.x < -kLean) box.x = at.x - kGap - box.width;
    box.y = at.y - box.height / 2;
    if (text.away.y > kLean) box.y = at.y + kGap;
    if (text.away.y < -kLean) box.y = at.y - kGap - box.height;
    return PastTaken(box, text.away.y < -kLean, canvas, taken);
}

/// Draw a label whose filled box is `box` (LabelBox's geometry).
void DrawLabelIn(cv::Mat* canvas, const std::string& text, const cv::Rect& box, bool alert) {
    const cv::Rect origin = LabelBox(text, cv::Point(0, 0));
    DrawLabel(canvas, text, cv::Point(box.x, box.y - origin.y), alert);
}

/// The label rows DrawBoxes fills above each box (drawDetection's own
/// geometry, visualization.hpp), so a logic label can keep off them.
void AddBoxLabelRows(const StageResult& result, const cv::Size& canvas,
                     std::vector<cv::Rect>* taken) {
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(result.data.get());
    if (boxes == NULL) return;
    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const BoxItem& item = boxes->items[i];
        const cv::Rect2f restored = RestoreBox(item.box, result.origin);
        const float x1 = std::max(0.f, std::min(restored.x, static_cast<float>(canvas.width)));
        const float y1 = std::max(0.f, std::min(restored.y, static_cast<float>(canvas.height)));
        const std::string label =
            item.class_name + ": " + std::to_string(static_cast<int>(item.score * 100)) + "%";
        int baseline = 0;
        const cv::Size text = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
        const int y = y1 - 10 > 10 ? static_cast<int>(y1 - 10)
                                   : static_cast<int>(y1 + text.height + 10);
        taken->push_back(cv::Rect(static_cast<int>(x1), y - text.height - 5, text.width + 1,
                                  text.height + 5 + baseline + 1));
    }
}

/// The text lines pass 4 draws for scores and vectors (DrawScores, DrawVector).
void AddScoreLines(const StageResult& result, const cv::Mat& canvas,
                   std::vector<cv::Rect>* taken) {
    const cv::Point anchor = TextAnchor(canvas, result.origin);
    const ScoresData* scores = dynamic_cast<const ScoresData*>(result.data.get());
    for (std::size_t i = 0; scores != NULL && i < scores->items.size(); ++i) {
        const ClassificationResult& item = scores->items[i];
        const std::string label = item.class_name + ": " +
            std::to_string(static_cast<int>(item.confidence * 100)) + "%";
        taken->push_back(
            TextLineBox(canvas.size(), label, anchor, static_cast<int>(i), NULL));
    }
    const VectorData* vec = dynamic_cast<const VectorData*>(result.data.get());
    if (vec != NULL) {
        const std::string label = "vec[" + std::to_string(vec->values.size()) + "]";
        taken->push_back(TextLineBox(canvas.size(), label, anchor, 0, NULL));
    }
}

/// Zones, lines with their "in" arrow and highlighted boxes; labels are
/// drawn after every overlay's shapes (DrawLogicLabels).
void DrawLogicShapes(cv::Mat* canvas, const LogicOverlay& overlay, cv::Mat* tint) {
    const cv::Scalar color = overlay.alert ? kAlertColor : kLogicColor;
    const cv::Rect frame(0, 0, canvas->cols, canvas->rows);
    for (std::size_t i = 0; i < overlay.polygons.size(); ++i) {
        if (overlay.polygons[i].size() < 3) continue;
        std::vector<std::vector<cv::Point> > contour(1);
        for (std::size_t k = 0; k < overlay.polygons[i].size(); ++k) {
            contour[0].push_back(Pixel(overlay.polygons[i][k]));
        }
        // Tint only the polygon's bounding box, in one reused buffer.
        const cv::Rect bounds = cv::boundingRect(contour[0]) & frame;
        if (bounds.area() > 0) {
            cv::Mat region = (*canvas)(bounds);
            region.copyTo(*tint);
            std::vector<std::vector<cv::Point> > local(1);
            for (std::size_t k = 0; k < contour[0].size(); ++k) {
                local[0].push_back(contour[0][k] - bounds.tl());
            }
            cv::fillPoly(*tint, local, color);
            cv::addWeighted(*tint, 0.2, region, 0.8, 0.0, region);
        }
        cv::polylines(*canvas, contour, true, color, 2);
    }
    for (std::size_t i = 0; i < overlay.lines.size(); ++i) {
        const cv::Point2f a = overlay.lines[i].first;
        const cv::Point2f b = overlay.lines[i].second;
        cv::line(*canvas, Pixel(a), Pixel(b), color, 2);
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length <= 0.f) continue;
        // The "in" side: the right of A->B on the image (y down) is (-dy, dx).
        const cv::Point2f mid = (a + b) * 0.5f;
        const cv::Point2f normal(-dy / length * 24.f, dx / length * 24.f);
        cv::arrowedLine(*canvas, Pixel(mid), Pixel(mid + normal), color, 2, cv::LINE_8, 0, 0.4);
    }
    for (std::size_t i = 0; i < overlay.highlights.size(); ++i) {
        const cv::Rect2f& box = overlay.highlights[i];
        cv::rectangle(*canvas,
                      cv::Rect(cvRound(box.x), cvRound(box.y), cvRound(box.width),
                               cvRound(box.height)),
                      color, 3);
    }
    for (std::size_t i = 0; i < overlay.axes.size(); ++i) {
        const OverlayAxes& axes = overlay.axes[i];
        // As thick as the head is large, so the axes read on a small preview.
        const float length = std::max(static_cast<float>(cv::norm(axes.x - axes.origin)),
                                      static_cast<float>(cv::norm(axes.y - axes.origin)));
        const int thickness = std::max(2, std::min(16, cvRound(length / 12.f)));
        const cv::Point2f ends[3] = {axes.x, axes.y, axes.z};
        const cv::Scalar colors[3] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 200, 0),
                                      cv::Scalar(255, 80, 0)};  // BGR: red, green, blue
        for (int k = 0; k < 3; ++k) {
            const double reach = cv::norm(ends[k] - axes.origin);
            if (reach < 1.0) continue;
            cv::arrowedLine(*canvas, Pixel(axes.origin), Pixel(ends[k]), colors[k], thickness,
                            cv::LINE_AA, 0, std::min(0.3, 3.0 * thickness / reach));
        }
    }
}

/// Which nodes one rendering lays out (`options.draw`) and which of them
/// it paints: all of them, or `only` (one RenderLayers layer).
struct Painting {
    const RenderOptions* options;
    const std::string* only;  ///< NULL: every node laid out is painted

    bool Lays(const std::string& node) const { return options->Draws(node); }
    bool Paints(const std::string& node) const {
        return Lays(node) && (only == NULL || *only == node);
    }
};

/// Every laid-out overlay's labels: first the top-left panel (one line
/// under the other, each moved down past the box labels in `taken`), then
/// the labels anchored in the picture, each kept off everything laid out
/// before. Only a painted overlay's labels are drawn; every laid-out one
/// takes its place, so a layer's labels land where the merged image's do.
void DrawLogicLabels(cv::Mat* canvas, const std::map<std::string, LogicOverlay>& overlays,
                     const Painting& painting, std::vector<cv::Rect>* taken) {
    const cv::Size size = canvas->size();
    std::vector<cv::Rect> panel;
    int next_baseline = 24;
    for (int pass = 0; pass < 2; ++pass) {
        for (std::map<std::string, LogicOverlay>::const_iterator it = overlays.begin();
             it != overlays.end(); ++it) {
            if (!painting.Lays(it->first)) continue;
            const bool paint = painting.Paints(it->first);
            for (std::size_t i = 0; i < it->second.texts.size(); ++i) {
                const OverlayText& text = it->second.texts[i];
                const bool in_panel = text.at.x < 0.f;
                if (in_panel != (pass == 0)) continue;
                cv::Rect box;
                if (in_panel) {
                    box = PastTaken(LabelBox(text.text, cv::Point(10, next_baseline)), false,
                                    size, *taken);
                    next_baseline =
                        box.y - LabelBox(text.text, cv::Point(0, 0)).y + kPanelLineHeight;
                    panel.push_back(box);
                } else if (text.away == cv::Point2f(0.f, 0.f)) {
                    box = InsideCanvas(LabelBox(text.text, Pixel(text.at)), size);
                } else {
                    box = BesideAnchor(text, size, *taken);
                }
                if (paint) DrawLabelIn(canvas, text.text, box, text.alert);
                if (!in_panel) taken->push_back(box);
            }
        }
        if (pass == 0) taken->insert(taken->end(), panel.begin(), panel.end());
    }
}

// ---------------------------------------------------------------------
// Driver: run one Draw* over every result of a given shape, in
// report.node_results then report.roi_results order (both std::map, so
// both iterate in a fixed, deterministic key order; roi_results' vectors
// already arrive sorted by ByOrigin, stage_graph.hpp's own contract).
// ---------------------------------------------------------------------

typedef void (*DrawFn)(cv::Mat*, const StageResult&);

/// Whether a port is drawn. A vector port is not: it rides on a primary
/// result that is drawn already (3DDFA's yaw, pitch and roll on its 68
/// landmarks), and DrawVector's "vec[N]" would say only its length, over
/// the face's own label. A node whose own result is a vector (an embedding)
/// still says "vec[N]".
bool DrawsPort(const StageDataPtr& port) {
    return port && port->shape() != Shape::kVector;
}

/// Draw `result`'s primary payload when it has `shape`, then each of its
/// ports of that shape (in port-name order) as if it were a result of its
/// own with the same `origin`: a port lands exactly where its primary does.
void DrawResultAndPorts(const StageResult& result, cv::Mat* canvas, Shape shape, DrawFn fn) {
    if (result.data && result.data->shape() == shape) {
        fn(canvas, result);
    }
    for (StagePorts::const_iterator it = result.ports.begin(); it != result.ports.end(); ++it) {
        if (DrawsPort(it->second) && it->second->shape() == shape) {
            StageResult port;
            port.data = it->second;
            port.origin = result.origin;
            fn(canvas, port);
        }
    }
}

/// True when `result`'s primary payload or one of its ports has `shape`.
bool HasShape(const StageResult& result, Shape shape) {
    if (result.data && result.data->shape() == shape) return true;
    for (StagePorts::const_iterator it = result.ports.begin(); it != result.ports.end(); ++it) {
        if (DrawsPort(it->second) && it->second->shape() == shape) return true;
    }
    return false;
}

/// Every result a shape pass draws, with the id of its node, in
/// report.node_results then report.roi_results order; a logic node's
/// payload is skipped (its overlay is pass 5), and so is a node `keep`
/// says no to.
template <typename Keep, typename Visit>
void ForEachDrawn(const FrameReport& report, Keep keep, Visit visit) {
    for (std::map<std::string, StageResult>::const_iterator it =
             report.node_results.begin();
         it != report.node_results.end(); ++it) {
        // A logic node: its overlay in pass 5, its payload is its input's.
        if (report.logic_nodes.count(it->first) != 0 || report.overlays.count(it->first) != 0) {
            continue;
        }
        if (!keep(it->first)) continue;
        visit(it->first, it->second);
    }
    for (std::map<std::string, std::vector<StageResult> >::const_iterator it =
             report.roi_results.begin();
         it != report.roi_results.end(); ++it) {
        if (!keep(it->first)) continue;
        const std::vector<StageResult>& results = it->second;
        for (std::size_t i = 0; i < results.size(); ++i) visit(it->first, results[i]);
    }
}

/// One shape pass of RenderReport: draw `fn` over every result of `shape`.
/// `text`: a scores/vector pass, which RenderOptions::quiet_text skips.
struct Step {
    Shape shape;
    DrawFn fn;
    bool text;
};

/// Passes 1-4 in order (pass 5, logic nodes, is RenderReport's own).
const Step kSteps[] = {
    // Pass 1: dense layers. Images first, as the backdrop: DrawImage copies
    // opaquely, and a hand-off graph always reports its image producer's
    // output, so drawing it after the maps would paint over the very
    // segmentation or depth that ran on it. Label and dense maps then blend
    // over it.
    {Shape::kImage, &DrawImage, false},
    {Shape::kLabelMap, &DrawLabelMap, false},
    {Shape::kDenseMap, &DrawDenseMap, false},
    // Pass 2: instance masks.
    {Shape::kInstances, &DrawInstances, false},
    // Pass 3: boxes (a detector's and each instance's), obboxes,
    // keypoints - restored to source coordinates.
    {Shape::kBoxes, &DrawBoxes, false},
    {Shape::kInstances, &DrawBoxes, false},
    {Shape::kObBoxes, &DrawObBoxes, false},
    {Shape::kKeypoints, &DrawKeypoints, false},
    // Pass 4: scores and vectors as text anchored to their parent box,
    // except a quiet node's (a gallery node names its crops instead).
    {Shape::kScores, &DrawScores, true},
    {Shape::kVector, &DrawVector, true},
};

/// Whether `step` draws `node` at all in a rendering laid out by `painting`.
bool StepKeeps(const Step& step, const Painting& painting, const std::string& node) {
    return painting.Lays(node) &&
           !(step.text && painting.options->quiet_text.count(node) != 0);
}

/// Every result ForEachDrawn visits, its drawn ports (DrawsPort) as results
/// of their own, in its order, each with the id of the node that produced it.
template <typename Visit>
void VisitResults(const FrameReport& report, Visit visit) {
    std::vector<std::pair<const std::string*, const StageResult*> > results;
    for (std::map<std::string, StageResult>::const_iterator it = report.node_results.begin();
         it != report.node_results.end(); ++it) {
        if (report.logic_nodes.count(it->first) != 0 || report.overlays.count(it->first) != 0) {
            continue;
        }
        results.push_back(std::make_pair(&it->first, &it->second));
    }
    for (std::map<std::string, std::vector<StageResult> >::const_iterator it =
             report.roi_results.begin();
         it != report.roi_results.end(); ++it) {
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            results.push_back(std::make_pair(&it->first, &it->second[i]));
        }
    }
    for (std::size_t i = 0; i < results.size(); ++i) {
        const StageResult& result = *results[i].second;
        visit(*results[i].first, result);
        for (StagePorts::const_iterator port = result.ports.begin(); port != result.ports.end();
             ++port) {
            if (!DrawsPort(port->second)) continue;
            StageResult as_result;
            as_result.data = port->second;
            as_result.origin = result.origin;
            visit(*results[i].first, as_result);
        }
    }
}

/// The boxes DrawEventLog's `lines` fill with the panel in `corner`
/// (0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left), and the
/// baseline each line is drawn at.
std::vector<cv::Rect> EventLogBoxes(const cv::Size& canvas, const std::vector<std::string>& lines,
                                    int corner, std::vector<cv::Point>* baselines) {
    const int kMargin = 10;
    const int kLineHeight = 22;
    const int kTopBaseline = 24;
    int widest = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        widest = std::max(widest, LabelBox(lines[i], cv::Point(0, 0)).width);
    }
    const bool right = corner == 1 || corner == 2;
    const bool bottom = corner == 0 || corner == 1;
    std::vector<cv::Rect> boxes;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const int from_bottom = static_cast<int>(lines.size() - 1 - i);
        cv::Point at(right ? canvas.width - kMargin - widest : kMargin,
                     bottom ? canvas.height - kMargin - kLineHeight * from_bottom
                            : kTopBaseline + kLineHeight * static_cast<int>(i));
        // Where DrawLabel puts it: inside the canvas.
        const cv::Rect origin = LabelBox(lines[i], cv::Point(0, 0));
        const cv::Rect box = InsideCanvas(LabelBox(lines[i], at), canvas);
        boxes.push_back(box);
        if (baselines != NULL) baselines->push_back(cv::Point(box.x, box.y - origin.y));
    }
    return boxes;
}

}  // namespace

RenderOptions RenderOptionsFor(const GraphSpec& spec) {
    RenderOptions options;
    for (std::size_t i = 0; i < spec.edges.size(); ++i) {
        const NodeSpec* to = spec.FindNode(spec.edges[i].to);
        if (to != NULL && to->is_logic && to->op == "gallery") {
            options.quiet_text.insert(spec.edges[i].from);
        }
    }
    return options;
}

namespace {

/// RenderReport, painting only what `painting` paints.
cv::Mat Paint(const cv::Mat& source, const FrameReport& report, const Painting& painting,
              std::vector<cv::Rect>* labels) {
    if (labels != NULL) labels->clear();
    cv::Mat canvas = source.clone();
    if (canvas.empty()) return canvas;

    // Passes 1-4 (kSteps).
    for (std::size_t s = 0; s < sizeof(kSteps) / sizeof(kSteps[0]); ++s) {
        const Step& step = kSteps[s];
        ForEachDrawn(report,
                     [&step, &painting](const std::string& node) {
                         return StepKeeps(step, painting, node) && painting.Paints(node);
                     },
                     [&canvas, &step](const std::string&, const StageResult& result) {
                         DrawResultAndPorts(result, &canvas, step.shape, step.fn);
                     });
    }

    // Pass 5: logic nodes, over everything else, in node-id order. A label
    // laid beside its anchor keeps off the box labels of pass 3 and off
    // the labels drawn before it - every laid-out node's, painted or not.
    std::vector<cv::Rect> taken;
    const RenderOptions& options = *painting.options;
    VisitResults(report, [&canvas, &taken, &options](const std::string& node,
                                                     const StageResult& result) {
        if (!result.data || !options.Draws(node)) return;
        const Shape shape = result.data->shape();
        if (shape == Shape::kBoxes || shape == Shape::kInstances) {
            AddBoxLabelRows(result, canvas.size(), &taken);
        }
        if (options.quiet_text.count(node) == 0) AddScoreLines(result, canvas, &taken);
    });
    cv::Mat tint;
    for (std::map<std::string, LogicOverlay>::const_iterator it = report.overlays.begin();
         it != report.overlays.end(); ++it) {
        if (painting.Paints(it->first)) DrawLogicShapes(&canvas, it->second, &tint);
    }
    DrawLogicLabels(&canvas, report.overlays, painting, &taken);
    if (labels != NULL) labels->swap(taken);
    return canvas;
}

/// The nodes RenderLayers gives a layer, in stacking order: each at the
/// first step it draws in, then the logic nodes in id order.
std::vector<std::string> LayerOrder(const FrameReport& report, const RenderOptions& options) {
    const Painting painting = {&options, NULL};
    std::vector<std::string> order;
    std::set<std::string> seen;
    for (std::size_t s = 0; s < sizeof(kSteps) / sizeof(kSteps[0]); ++s) {
        const Step& step = kSteps[s];
        ForEachDrawn(report,
                     [&step, &painting](const std::string& node) {
                         return StepKeeps(step, painting, node);
                     },
                     [&step, &order, &seen](const std::string& node, const StageResult& result) {
                         if (HasShape(result, step.shape) && seen.insert(node).second) {
                             order.push_back(node);
                         }
                     });
    }
    for (std::map<std::string, LogicOverlay>::const_iterator it = report.overlays.begin();
         it != report.overlays.end(); ++it) {
        if (painting.Lays(it->first) && seen.insert(it->first).second) order.push_back(it->first);
    }
    return order;
}

}  // namespace

cv::Mat RenderReport(const cv::Mat& source, const FrameReport& report,
                     const RenderOptions& options, std::vector<cv::Rect>* labels) {
    const Painting painting = {&options, NULL};
    return Paint(source, report, painting, labels);
}

std::vector<RenderLayer> RenderLayers(const cv::Size& size, const FrameReport& report,
                                      const RenderOptions& options) {
    std::vector<RenderLayer> layers;
    if (size.area() <= 0) return layers;
    const cv::Mat black = cv::Mat::zeros(size, CV_8UC3);
    const cv::Mat white(size, CV_8UC3, cv::Scalar(255, 255, 255));
    const std::vector<std::string> order = LayerOrder(report, options);
    for (std::size_t i = 0; i < order.size(); ++i) {
        const Painting painting = {&options, &order[i]};
        RenderLayer layer;
        layer.node = order[i];
        layer.rgba = LayerFromBackdrops(Paint(black, report, painting, NULL),
                                        Paint(white, report, painting, NULL));
        std::vector<cv::Mat> channels;
        cv::split(layer.rgba, channels);
        if (cv::countNonZero(channels[3]) != 0) layers.push_back(layer);
    }
    return layers;
}

cv::Mat LayerFromBackdrops(const cv::Mat& on_black, const cv::Mat& on_white) {
    CV_Assert(on_black.type() == CV_8UC3 && on_white.type() == CV_8UC3 &&
              on_black.size() == on_white.size());
    cv::Mat layer(on_black.size(), CV_8UC4);
    for (int y = 0; y < layer.rows; ++y) {
        const cv::Vec3b* b = on_black.ptr<cv::Vec3b>(y);
        const cv::Vec3b* w = on_white.ptr<cv::Vec3b>(y);
        cv::Vec4b* out = layer.ptr<cv::Vec4b>(y);
        for (int x = 0; x < layer.cols; ++x) {
            // Over black a pixel is a*c; over white a*c + (1-a)*255.
            int through = 0;
            for (int c = 0; c < 3; ++c) through += static_cast<int>(w[x][c]) - b[x][c];
            const int alpha = std::max(0, std::min(255, 255 - (through + 1) / 3));
            out[x][3] = static_cast<uchar>(alpha);
            for (int c = 0; c < 3; ++c) {
                out[x][c] = alpha == 0 ? 0
                    : static_cast<uchar>(std::min(255, (b[x][c] * 255 + alpha / 2) / alpha));
            }
        }
    }
    return layer;
}

cv::Mat CompositeLayers(const cv::Mat& base, const std::vector<cv::Mat>& layers) {
    cv::Mat out = base.clone();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const cv::Mat& layer = layers[i];
        CV_Assert(layer.type() == CV_8UC4 && layer.size() == out.size() &&
                  out.type() == CV_8UC3);
        for (int y = 0; y < out.rows; ++y) {
            const cv::Vec4b* l = layer.ptr<cv::Vec4b>(y);
            cv::Vec3b* o = out.ptr<cv::Vec3b>(y);
            for (int x = 0; x < out.cols; ++x) {
                const int a = l[x][3];
                if (a == 0) continue;
                for (int c = 0; c < 3; ++c) {
                    o[x][c] = static_cast<uchar>((l[x][c] * a + o[x][c] * (255 - a) + 127) / 255);
                }
            }
        }
    }
    return out;
}

void DrawEventLog(cv::Mat* canvas, const std::vector<std::string>& lines,
                  const std::vector<cv::Rect>& avoid) {
    if (canvas == NULL || canvas->empty() || lines.empty()) return;
    // The first corner that covers no label; failing that, the one that
    // covers the least of them (bottom-left first on a tie).
    int best_corner = 0;
    long long best_cover = -1;
    for (int corner = 0; corner < 4 && best_cover != 0; ++corner) {
        const std::vector<cv::Rect> boxes = EventLogBoxes(canvas->size(), lines, corner, NULL);
        long long cover = 0;
        for (std::size_t b = 0; b < boxes.size(); ++b) {
            for (std::size_t a = 0; a < avoid.size(); ++a) cover += (boxes[b] & avoid[a]).area();
        }
        if (best_cover < 0 || cover < best_cover) {
            best_cover = cover;
            best_corner = corner;
        }
    }
    std::vector<cv::Point> baselines;
    EventLogBoxes(canvas->size(), lines, best_corner, &baselines);
    for (std::size_t i = 0; i < lines.size(); ++i) DrawLabel(canvas, lines[i], baselines[i], false);
}

}  // namespace graph
}  // namespace dxapp
