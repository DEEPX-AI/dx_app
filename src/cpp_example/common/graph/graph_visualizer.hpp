/**
 * @file graph_visualizer.hpp
 * @brief Draw every stage's result of one frame report onto one canvas.
 *
 * `ref.inv_align` alone maps a crop-local coordinate directly to the
 * source frame — `roi_router.cpp`'s `RouteRois` folds the crop's own
 * `ref.src_box` translation into `inv_align` at construction time (a
 * different composition per crop kind; see `RouteRois`'s own comment), so
 * nothing downstream ever adds `ref.src_box` a second time. `RestorePoint-
 * Warped`, below, applies exactly `ref.inv_align` to a point. `roi_router.
 * hpp`'s `RestoreBoxCorners`/`RestoreBox` apply the same `ref.inv_align`
 * to a box's four corners, then (`RestoreBox` only) take their axis-
 * aligned bounding box. Both are correct for a plain crop too
 * (`ref.inv_align` reduces to plain translation there) and for a
 * full-frame stage (`ref.from_roi == false`: identity `inv_align` passes
 * the input straight through; a hand-off child's `inv_align` carries its
 * image's scale back to the source frame).
 *
 * Fix round 3 was this file's structural correction: rounds 1-2 composed
 * `inv_align`-applied-then-`+src_box` in every one of these functions,
 * which was wrong for both warped-crop producers (OBB applies the
 * translation *before* the rotation, not after; face5 has no `src_box`
 * relationship at all — see `roi_router.cpp`'s `RouteRois` comment) but
 * untestable by every fixture in this suite, since every non-identity-
 * `inv_align` fixture put `ref.src_box` at the origin, where both
 * compositions agree by coincidence.
 *
 * Fix round 4 removed `RestoreAngleWarped`, round 2's scalar-angle
 * composition: it had zero non-test callers (`DrawObBoxes` never
 * reconstructed a `cx/cy/w/h/angle` tuple from it after round 3 switched
 * to drawing real corners, and no other consumer - including Task 13's
 * `--report` JSON serializer, checked directly - ever restores an OBB's
 * angle to a scalar). `DrawObBoxes` now rotates a box's four corners by
 * `item.angle` in the crop's own local frame, then restores each corner
 * through `RestorePointWarped` - rotate first, warp second, exactly the
 * order `item.angle`'s own crop-local meaning requires.
 */
#ifndef DXAPP_GRAPH_GRAPH_VISUALIZER_HPP
#define DXAPP_GRAPH_GRAPH_VISUALIZER_HPP

#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/graph/shape.hpp"
#include "common/graph/stage_graph.hpp"

namespace dxapp {
namespace graph {

/**
 * @brief Undo a crop's alignment/un-rotation, then its translation.
 *
 * `ref.inv_align` is the inverse of the affine transform recorded when the
 * crop was produced (identity for a plain crop, so this is a strict
 * generalization of a translation-only restore). `roi_local` is a point
 * expressed in that crop's own pixel space; the result is the same point
 * in source-frame coordinates. `ref.from_roi == false` (a full-frame
 * stage) applies `ref.inv_align` too, same contract as RestoreBox/
 * RestorePoint: that is the identity (`roi_local` returned unchanged)
 * unless the result came from a handed-off image (see `roi_router.hpp`'s
 * `FrameView`).
 */
cv::Point2f RestorePointWarped(const cv::Point2f& roi_local, const RoiRef& ref);

/// What RenderReport takes from the graph rather than from a report.
/// Built once per graph (RenderOptionsFor); the same for every executor,
/// so sync and async reports still render to the same pixels.
struct RenderOptions {
    /// Nodes whose scores and vector text pass 4 leaves out (and whose
    /// lines are not label areas): each one feeds a gallery logic node,
    /// which writes the name it found under each crop itself, so the
    /// node's own "vec[N]" and top matches would only cover the person.
    std::set<std::string> quiet_text;
    /// The only nodes drawn (the CLI's --draw); empty draws every node. A
    /// logic node's id draws its overlay. A node left out still ran and
    /// still reports: it only draws nothing, and its labels are not areas
    /// the others keep off.
    std::set<std::string> draw;

    bool Draws(const std::string& node) const { return draw.empty() || draw.count(node) != 0; }
};

/// The options `spec`'s edges call for: every node an edge carries into a
/// node of op "gallery" is quiet.
RenderOptions RenderOptionsFor(const GraphSpec& spec);

/**
 * @brief Composite every stage's result in `report` onto `source`.
 *
 * Draw order is fixed so two calls with the same report produce identical
 * pixels: dense layers first - images, copied opaquely as the backdrop,
 * then labelmaps and densemaps blended over them - then instance masks,
 * then boxes (a detector's and each instance's, with its label)/obboxes/
 * keypoints restored to source coordinates, then scores/vectors as text
 * anchored to their parent box. Both `report.node_results`
 * (full-frame stages) and `report.roi_results` (per-crop stages) are
 * drawn — a cascade's second-stage results are exactly the point of this
 * function, not an afterthought.
 *
 * Array-shaped ROI results (an image, a labelmap, a densemap, an instance
 * mask) land where their crop came from. A plain crop, a crop from a
 * scaled view and a full-frame result are rectangles on the canvas and are
 * resized into them. A crop whose `origin.inv_align` rotates or shears (an
 * OBB or face5 crop, before or after a hand-off) is resized to its
 * `crop_size` and warped through `inv_align`, and only the pixels the
 * warped crop covers are blended or copied.
 *
 * A result's ports (`StageResult::ports`) are drawn in the pass of each
 * port's shape, right after that result's primary payload, with the
 * result's own `origin`. A binary mask (`LabelMapData::binary_mask`) blends
 * only its non-zero pixels, in `mask_color` at alpha 0.5 (on a rotated or
 * aligned crop it blends its whole footprint). A non-spatial dense map
 * (`DenseMapData::spatial` false) is never drawn.
 *
 * Logic nodes (FrameReport::overlays) are drawn by pass 5 only, after all
 * of the above: their polygons tinted and outlined, their lines with an
 * arrow to the "in" side (the right of A->B), highlighted boxes, then their
 * labels, at a point or stacked in a panel from the top left. A logic
 * node's own payload (FrameReport::logic_nodes, or a node with an overlay)
 * is not drawn by the shape passes: a zone's boxes are its detector's
 * boxes, already drawn.
 *
 * A node in `options.quiet_text` draws no scores or vector text. With
 * `options.draw` set, only the nodes in it draw anything (passes 1-5), and
 * only their labels are areas a later label keeps off.
 *
 * `labels`, when given, receives every text area drawn: box label rows,
 * scores and vector lines, logic labels and the top-left panel - what a
 * later overlay (DrawEventLog) should keep off.
 *
 * An empty `source` returns it unchanged; a default (empty) `report`
 * draws nothing and returns a clone of `source`.
 */
cv::Mat RenderReport(const cv::Mat& source, const FrameReport& report,
                     const RenderOptions& options = RenderOptions(),
                     std::vector<cv::Rect>* labels = NULL);

/// One node's drawing alone (RenderLayers): `rgba` is CV_8UC4 in OpenCV's
/// channel order (B, G, R, A - cv::imwrite stores it as an RGBA PNG), the
/// size of the source frame, transparent wherever the node drew nothing.
struct RenderLayer {
    std::string node;
    cv::Mat rgba;
};

/**
 * @brief Every drawn node's drawing on its own transparent layer.
 *
 * The layers come in RenderReport's pass order - each node at the first
 * pass it draws in, nodes in one pass in RenderReport's order, logic nodes
 * last in id order - so stacking them over the clean source frame
 * (CompositeLayers) gives RenderReport's image: within a pixel or two of
 * rounding, and exactly so unless one node draws in two passes over a
 * pixel another node drew in between (an instance node's masks, pass 2,
 * and its boxes, pass 3, over a detector's boxes). Each layer is laid out
 * as in RenderReport with `options` (labels keep off the same areas). A
 * node that drew no pixel has no layer; `options.draw` leaves nodes out.
 *
 * Each layer is RenderReport painting that node alone over black and over
 * white (LayerFromBackdrops): every draw blends a colour that does not
 * depend on the canvas at an alpha that does not either.
 */
std::vector<RenderLayer> RenderLayers(const cv::Size& size, const FrameReport& report,
                                      const RenderOptions& options = RenderOptions());

/// The RGBA layer (RenderLayer::rgba) of a drawing made once over black
/// (`on_black`) and once over white (`on_white`), both CV_8UC3.
cv::Mat LayerFromBackdrops(const cv::Mat& on_black, const cv::Mat& on_white);

/// `base` (CV_8UC3) with every layer (CV_8UC4, RenderLayer::rgba) blended
/// over it in order: what a front end does when it stacks the files.
cv::Mat CompositeLayers(const cv::Mat& base, const std::vector<cv::Mat>& layers);

/// The CLI's recent-events panel: `lines` oldest first, the newest at the
/// bottom, in the first corner of bottom-left, bottom-right, top-right and
/// top-left where it covers none of `avoid` (RenderReport's `labels`);
/// when every corner covers some, the one covering the least. Kept out of
/// RenderReport, which draws one report.
void DrawEventLog(cv::Mat* canvas, const std::vector<std::string>& lines,
                  const std::vector<cv::Rect>& avoid = std::vector<cv::Rect>());

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_GRAPH_VISUALIZER_HPP
