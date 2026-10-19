/**
 * @file shape.hpp
 * @brief The engine's entire type vocabulary.
 *
 * The graph engine switches on Shape, never on task name. 22 registry tasks
 * and 15 factory interfaces collapse onto 12 shapes; a new task that lands on
 * an existing shape needs no engine change.
 *
 * C++14: no std::variant, so payloads are a polymorphic hierarchy held by
 * shared_ptr.
 */
#ifndef DXAPP_GRAPH_SHAPE_HPP
#define DXAPP_GRAPH_SHAPE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/base/i_processor.hpp"

namespace dxapp {
namespace graph {

enum class Shape {
    kFrame,
    kBoxes,
    kObBoxes,
    kInstances,
    kKeypoints,
    kLabelMap,
    kDenseMap,
    kImage,
    kScores,
    kVector,
    kBoxes3d,
    kRecords
};

/// Stable lower-case name used in JSON, error messages and generated docs.
const char* ToString(Shape shape);

/// True when boxes can be cut out of this shape and handed to another model.
bool ProducesRoi(Shape shape);

enum class InputContract { kFullFrame, kRoi, kEither };

const char* ToString(InputContract contract);

/// True when a model with this contract may be fed a cropped region.
bool AcceptsRoi(InputContract contract);

struct StageData {
    virtual ~StageData() {}
    virtual Shape shape() const = 0;
};
typedef std::shared_ptr<const StageData> StageDataPtr;

/// A node's outputs beyond its primary payload, by port name (U-08).
typedef std::map<std::string, StageDataPtr> StagePorts;

/**
 * @brief One detected region, in source-frame coordinates.
 *
 * track_id lives here rather than in DetectionResult: the shared result
 * structs are used by all 348 examples and are not modified by this project.
 */
struct BoxItem {
    cv::Rect2f box;
    float score;
    int class_id;
    std::string class_name;
    int track_id;                     ///< -1 when untracked
    std::vector<Keypoint> landmarks;  ///< face detectors only
    cv::Mat mask;                     ///< instances only
    float angle;                      ///< obboxes only, radians

    BoxItem() : score(0.f), class_id(0), track_id(-1), angle(0.f) {}
};

struct FrameData : StageData {
    cv::Mat image;
    Shape shape() const { return Shape::kFrame; }
};

struct BoxesData : StageData {
    std::vector<BoxItem> items;
    Shape box_shape;  ///< kBoxes, kObBoxes or kInstances

    explicit BoxesData(Shape shape) : box_shape(shape) {}
    Shape shape() const { return box_shape; }
};

struct KeypointsData : StageData {
    std::vector<PoseResult> items;
    /// The points have no order or connectivity (SuperPoint's): drawn as
    /// points only, never joined by a skeleton chosen from their count.
    bool unordered;
    KeypointsData() : unordered(false) {}
    Shape shape() const { return Shape::kKeypoints; }
};

struct LabelMapData : StageData {
    cv::Mat labels;         ///< CV_8U or CV_32S class indices
    bool binary_mask;       ///< label 0 undrawn, every other label in mask_color
    cv::Vec3b mask_color;   ///< BGR
    LabelMapData() : binary_mask(false), mask_color(0, 0, 0) {}
    Shape shape() const { return Shape::kLabelMap; }
};

struct DenseMapData : StageData {
    cv::Mat values;         ///< CV_32F
    bool spatial;           ///< false: a per-item matrix, never drawn
    DenseMapData() : spatial(true) {}
    Shape shape() const { return Shape::kDenseMap; }
};

struct ImageData : StageData {
    cv::Mat image;
    Shape shape() const { return Shape::kImage; }
};

struct ScoresData : StageData {
    std::vector<ClassificationResult> items;
    Shape shape() const { return Shape::kScores; }
};

struct VectorData : StageData {
    std::vector<float> values;
    Shape shape() const { return Shape::kVector; }
};

struct Boxes3dData : StageData {
    std::vector<Detection3DResult> items;
    Shape shape() const { return Shape::kBoxes3d; }
};

/// One named measurement from a cpu reducer (headpose, volume).
struct RecordItem {
    std::vector<std::pair<std::string, double> > numbers;
    std::vector<std::pair<std::string, std::string> > text;
};

struct RecordsData : StageData {
    std::vector<RecordItem> items;
    Shape shape() const { return Shape::kRecords; }
};

/**
 * @brief Where a stage result came from, so it can be drawn on the source.
 *
 * For a full-frame stage, `inv_align` maps the image it ran on to the
 * source frame: the identity on the source frame itself, and the hand-off
 * scale (e.g. 0.5 after x2 super-resolution) for a stage fed another
 * node's image on a plain edge. For ROI stages, `inv_align` alone maps a
 * crop-local coordinate directly to source-frame coordinates — RouteRois
 * (roi_router.cpp) folds the crop's own translation into `inv_align` at
 * construction time, so nothing downstream ever composes `src_box` back in
 * a second time. `src_box` remains the crop's own footprint in source
 * coordinates (its authority for position/size, per the invariant pinned
 * in UnrotateCrop's own doc comment) — used by consumers that need the
 * crop's on-canvas region (e.g. where to blend a dense map), not as a term
 * in the point/box restore itself.
 *
 * `crop_size` is the crop image's own pixel dimensions — (0,0) unset. It
 * is NOT always `src_box.size()`: a face5-aligned crop is a `side x side`
 * warp of the whole frame, not a sub-window, so its pixel size differs
 * from the clipped detection rect `src_box` records. Consumers that need
 * the crop's actual footprint size (not just its recorded detection rect)
 * should prefer this field, falling back to `src_box.size()` when unset
 * (e.g. a hand-built RoiRef in a test).
 */
struct RoiRef {
    bool from_roi;
    cv::Rect2f src_box;
    cv::Matx23f inv_align;
    cv::Size crop_size;
    int track_id;
    std::string parent_node;
    int parent_index;  ///< index of the producing BoxItem
    int roi_index;     ///< ordinal within this frame, for deterministic sorting

    RoiRef()
        : from_roi(false),
          inv_align(1.f, 0.f, 0.f, 0.f, 1.f, 0.f),
          crop_size(0, 0),
          track_id(-1),
          parent_index(-1),
          roi_index(-1) {}
};

struct StageResult {
    StageDataPtr data;   ///< the primary output
    RoiRef origin;
    StagePorts ports;    ///< the declared extra outputs; empty for most models
};

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_SHAPE_HPP
