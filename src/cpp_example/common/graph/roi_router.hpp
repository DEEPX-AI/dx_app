/**
 * @file roi_router.hpp
 * @brief The operator chain on a ROI edge, and the inverse transform back.
 *
 * Operator order is part of the specification, not an implementation detail:
 * applying pad after align yields different crops. The order is
 *
 *   filter -> max cut -> track -> pad+clip -> align/un-rotate -> crop
 *
 * and it is documented verbatim in multi_model_graph/README.md.
 */
#ifndef DXAPP_GRAPH_ROI_ROUTER_HPP
#define DXAPP_GRAPH_ROI_ROUTER_HPP

#include <string>
#include <vector>

#include "common/graph/graph_config.hpp"
#include "common/graph/shape.hpp"
#include "common/trackers/iou_tracker.hpp"

namespace dxapp {
namespace graph {

struct RoiCrop {
    cv::Mat image;
    RoiRef ref;
};

struct RouteStats {
    int filtered;       ///< dropped by classes / min_score / min_area / max
    int clipped_away;   ///< nothing left after clipping to the frame
    RouteStats() : filtered(0), clipped_away(0) {}
};

/**
 * @brief Assign stable ids to every box, in place.
 *
 * Must be called once per frame, in frame order. The tracker is stateful, so
 * feeding frame 3 before frame 2 breaks the ids - the async executor enforces
 * ordering around this call rather than inside it.
 */
void ApplyTrackIds(BoxesData* boxes, IouTracker* tracker);

/**
 * @brief Run the operator chain and cut the crops.
 *
 * @param tracker unused here; ids must already be on the boxes. Present for
 *        signature stability while the operator registry grows.
 */
std::vector<RoiCrop> RouteRois(const BoxesData& boxes,
                               const cv::Mat& source,
                               const RoiSpec& spec,
                               const std::string& parent_node,
                               IouTracker* tracker,
                               RouteStats* stats);

/**
 * @brief The image a full-frame node runs on, and the map from its pixel
 *        coordinates to the source frame's.
 *
 * The source frame's own view is `{frame, identity}`. A node fed by a
 * plain edge out of an image-producing node receives that node's output
 * image as its view instead (HandOffView), and `to_source` carries the
 * scale back to the source frame.
 */
struct FrameView {
    cv::Mat image;
    cv::Matx23f to_source;  ///< maps image pixels to source-frame pixels; identity for the source
    FrameView() : to_source(1.f, 0.f, 0.f, 0.f, 1.f, 0.f) {}
};

/// True only for the exact identity (no tolerance): the bit-exact fast
/// path every restore helper and the view overload of RouteRois keep.
bool IsIdentityAffine(const cv::Matx23f& m);

/**
 * @brief "" if `data` can be handed off on a plain edge, else why not
 *        ("handed off no image" / "handed off an empty image" /
 *        "handed off a non-BGR image").
 */
std::string HandOffProblem(const StageDataPtr& data);

/**
 * @brief The view a consumer of `producer_input`'s image-producing node
 *        receives.
 *
 * `image` is the produced image as is (never resized back); `to_source`
 * is `producer_input.to_source` composed with the per-axis scale
 * `in / out`, so chains compose naturally.
 * Precondition: HandOffProblem(data).empty().
 */
FrameView HandOffView(const ImageData& produced, const FrameView& producer_input);

/**
 * @brief RouteRois over a view: crops are cut from (and clipped against)
 *        `view.image`, and every crop's ref (src_box, inv_align) is
 *        expressed against the SOURCE frame.
 *
 * `crop_size` stays the crop's actual pixel size. For an identity view
 * this is exactly the frame overload above - byte-identical crops and refs.
 */
std::vector<RoiCrop> RouteRois(const BoxesData& boxes, const FrameView& view,
                               const RoiSpec& spec, const std::string& parent_node,
                               IouTracker* tracker, RouteStats* stats);

/**
 * @brief Map a box's four corners from crop-local coordinates to the source
 *        frame.
 *
 * Applies `ref.inv_align` alone to each of `roi_local`'s four corners -
 * `ref.inv_align` already maps a crop-local coordinate directly to the
 * source frame (RouteRois folds the crop's own translation into it at
 * construction time; see RouteRois's own comment for why the three crop
 * kinds need three different compositions to establish that). It applies
 * `ref.inv_align` for `from_roi == false` too; for a full-frame result that
 * is the identity (returned bit-exact, without arithmetic) unless the
 * result came from a handed-off image (see `FrameView`), whose scale back
 * to the source frame `inv_align` then carries. The four points are NOT
 * necessarily an axis-aligned rectangle's corners in that order once
 * `inv_align` carries rotation - callers that need the true (possibly
 * rotated) quadrilateral, e.g. to draw an OBB, should use these corners
 * directly rather than reducing them to a box.
 */
std::vector<cv::Point2f> RestoreBoxCorners(const cv::Rect2f& roi_local,
                                           const RoiRef& ref);

/**
 * @brief Map a box expressed in crop-local coordinates back to the source
 *        frame.
 *
 * The axis-aligned bounding box of RestoreBoxCorners's four transformed
 * corners - well-defined, and honest about being an approximation for a
 * rotated source box (its true shape is a quadrilateral, not a box), rather
 * than transforming two opposite corners and assuming the result is still
 * axis-aligned. Applies `ref.inv_align`; for `from_roi == false` that is
 * the identity (`roi_local` returned unchanged) unless the result came
 * from a handed-off image (see `FrameView`).
 */
cv::Rect2f RestoreBox(const cv::Rect2f& roi_local, const RoiRef& ref);

/**
 * @brief Map a keypoint expressed in crop-local coordinates back to the
 *        source frame.
 *
 * Applies `ref.inv_align` to (x, y) and copies `confidence`, exactly like
 * `graph_visualizer.hpp`'s `RestorePointWarped`. For a plain crop
 * (`inv_align` = translate(src_box)) that is bit-identical to adding
 * `ref.src_box.x/y`; for a face5-aligned, OBB-un-rotated or scaled crop it
 * is the full map. For `from_roi == false` it is the identity (`roi_local`
 * returned unchanged) unless the result came from a handed-off image (see
 * `FrameView`).
 */
Keypoint RestorePoint(const Keypoint& roi_local, const RoiRef& ref);

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_ROI_ROUTER_HPP
