/**
 * @file cpu_reduce.hpp
 * @brief CPU reducers a graph runs after model nodes: head pose and volume.
 *
 * A cpu node does not load a .dxnn. Its inputs arrive on "carry": "result"
 * edges (the producer results, payload and origin, not crops), and both
 * executors call ReduceCpu once those parents of the frame have finished.
 */
#ifndef DXAPP_GRAPH_CPU_REDUCE_HPP
#define DXAPP_GRAPH_CPU_REDUCE_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/graph/i_registry.hpp"
#include "common/graph/logic_ops.hpp"
#include "common/graph/shape.hpp"

namespace dxapp {
namespace graph {

/// One cpu node's output for one frame. `error` is empty when the frame is
/// valid, including "no face" and "no package" (those are empty records).
struct CpuReduction {
    StageResult result;
    std::string error;
    /// What the node draws (headpose: the head's axes and its angles); the
    /// runners put it in FrameReport::overlays under the node's id.
    LogicOverlay overlay;
};

/**
 * @brief Run `op` ("headpose" or "volume") on `inputs`.
 *
 * `frame_size` is the stream frame. headpose builds a pinhole camera with
 * focal length equal to the frame width; volume counts mask areas in it. `inputs` are
 * matched by shape, not by order.
 *
 * headpose records, one per face in the boxes' order: numbers pitch, yaw,
 * roll (degrees), face_index (the face's index in the boxes input) and
 * track_id when the face is tracked. A face with fewer than five landmarks,
 * or whose solvePnP fails, has no record.
 *
 * volume records, one per package box, all in source-frame pixels: numbers
 * box_x, box_y, box_w, box_h, class_id, mask_area_px (fractional when the
 * mask is not the frame's size), and median_depth plus
 * volume_proxy (mask_area_px x median_depth x scale_factor) when the
 * overlapping instance mask has a finite depth sample; text class_name.
 * Boxes go through their origin (RestoreBox: a hand-off is mapped back). The
 * instance mask and the depth map are full-frame rasters of any size, both
 * covering the frame as the overlay places them: each mask pixel's centre is
 * read from the depth pixel at the same place of the frame (nearest), never
 * resized. An input whose origin does not cover the frame (a crop, or an
 * inv_align that is not a positive axis-aligned scale) and a depth map that is
 * not spatial are a frame error.
 *
 * Each input is a parent's result: its payload and its origin.
 */
CpuReduction ReduceCpu(const std::string& op, const StageParams& params,
                       const std::vector<StageResult>& inputs,
                       const cv::Size& frame_size);

/// volume's package classes when its "classes" / "class_ids" are left out
/// (ids as text, as a graph writes them), for --list-ops.
std::vector<std::string> CpuPackageClassNames();
std::vector<std::string> CpuPackageClassIds();

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_CPU_REDUCE_HPP
