/**
 * @file logic_ops.hpp
 * @brief Logic nodes ("type": "logic"): the decisions around the models.
 *
 * A logic node runs on the CPU once per frame, after the nodes it reads
 * ("carry": "result" edges). Its op is one of eight: filter, zone, line,
 * count, dwell, condition, gallery, user. LogicOps() is the one description
 * of them: ValidateGraph checks params and inputs against it and
 * --list-ops prints it, so the two cannot disagree. headpose and volume are
 * cpu nodes (cpu_reduce.hpp); CpuOpInfos() describes them the same way for
 * --list-ops only.
 *
 * Coordinates in params are fractions of the source frame (0..1), so one
 * graph fits any resolution.
 */
#ifndef DXAPP_GRAPH_LOGIC_OPS_HPP
#define DXAPP_GRAPH_LOGIC_OPS_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <memory>

#include <opencv2/core.hpp>

#include "common/graph/graph_config.hpp"
#include "common/graph/shape.hpp"

namespace dxapp {
struct RetrievalGallery;  // common/processors/gallery_retrieval_postprocessor.hpp
namespace graph {

class UserLogic;  // common/graph/user_logic.hpp

/// A number param with no bound on that side (--list-ops leaves the bound out).
const double kLogicUnbounded = 1e9;

/// One param of a logic op: what --list-ops prints and ValidateGraph enforces.
struct LogicParamInfo {
    enum Kind {
        kNumber,    ///< a JSON number in [low, high] ((low, high] when low_open)
        kText,      ///< a string
        kChoice,    ///< one of `choices`
        kTextList,  ///< an array of strings
        kPolygon,   ///< [x0, y0, x1, y1, ...], at least 3 points, each 0..1
        kLine       ///< [x0, y0, x1, y1], two distinct points, each 0..1
    };
    std::string name;
    Kind kind;
    bool required;
    double low;
    double high;
    bool low_open;
    double number_default;              ///< kNumber, when not required
    std::string text_default;           ///< kText / kChoice, when not required
    std::vector<std::string> choices;   ///< kChoice
    std::string help;
    // Hints for an inspector (--list-ops --json), each written only when set.
    double step;                        ///< kNumber: a sensible increment; 0: none
    std::string items;                  ///< kTextList: "string", or "integer" (written as strings)
    std::vector<std::string> when_omitted;  ///< kTextList: what leaving it out means
    std::string format;                 ///< kText: "file" for a path
    std::vector<std::string> extensions;    ///< format "file": the accepted endings
    std::string equals;                 ///< the param whose value this one must repeat
    /// The values this param may take, per input kind (condition's "value").
    struct InputValues {
        Shape shape;
        bool per_crop;
        std::vector<std::string> values;  ///< the values this input allows
        bool from_numbers;                ///< or: a number the input's op lists in "numbers"
        InputValues() : shape(Shape::kBoxes), per_crop(false), from_numbers(false) {}
    };
    std::vector<InputValues> values_by_input;

    LogicParamInfo()
        : kind(kNumber), required(false), low(0.0), high(0.0), low_open(false),
          number_default(0.0), step(0.0) {}
};

/// One input a logic op accepts: a shape, as a full-frame payload or as
/// the results of a node that runs once per crop.
struct LogicPort {
    Shape shape;
    bool per_crop;
    LogicPort() : shape(Shape::kBoxes), per_crop(false) {}
};

struct LogicOpInfo {
    std::string op;
    std::string summary;
    std::vector<LogicPort> accepts;
    int min_inputs;
    int max_inputs;
    bool same_shape_out;               ///< outputs its input's shape; else records
    bool needs_tracking;               ///< its box inputs must carry track ids
    std::vector<std::string> events;   ///< names it raises ("<event>": the "event" param)
    std::vector<std::string> numbers;  ///< the records numbers it writes
    std::vector<LogicParamInfo> params;

    LogicOpInfo()
        : min_inputs(1), max_inputs(1), same_shape_out(false), needs_tracking(false) {}
};

/// A label a logic node asks to be drawn.
struct OverlayText {
    cv::Point2f at;    ///< source-frame pixels; at.x < 0: a line in the top-left panel
    std::string text;
    bool alert;        ///< drawn on the alert colour
    /// (0, 0): the label's baseline starts at `at`. Else the label is laid
    /// beside `at`, wholly on this side of it (a direction; each axis whose
    /// part is small is centred on `at`), and moved further that way past
    /// the labels of the boxes drawn under it.
    cv::Point2f away;
    OverlayText() : at(-1.f, -1.f), alert(false), away(0.f, 0.f) {}
};

/// A head's pose: its x (red), y (green) and z (blue, where the face points)
/// axes, drawn from `origin` to each end. Source-frame pixels.
struct OverlayAxes {
    cv::Point2f origin;
    cv::Point2f x;
    cv::Point2f y;
    cv::Point2f z;
};

inline bool operator==(const OverlayAxes& a, const OverlayAxes& b) {
    return a.origin == b.origin && a.x == b.x && a.y == b.y && a.z == b.z;
}
inline bool operator!=(const OverlayAxes& a, const OverlayAxes& b) { return !(a == b); }

/// What one logic node (or a cpu node: headpose) asks the visualizer to draw
/// for one frame, in source-frame pixels (RenderReport's pass 5).
struct LogicOverlay {
    std::vector<std::vector<cv::Point2f> > polygons;
    std::vector<std::pair<cv::Point2f, cv::Point2f> > lines;  ///< A -> B; "in" is to its right
    std::vector<cv::Rect2f> highlights;                        ///< boxes outlined
    std::vector<OverlayAxes> axes;                             ///< in their own colours
    std::vector<OverlayText> texts;
    bool alert;  ///< polygons, lines and highlights in the alert colour
    LogicOverlay() : alert(false) {}
};

/// Nothing to draw: no polygon, line, highlight, axes or label.
inline bool IsEmpty(const LogicOverlay& overlay) {
    return overlay.polygons.empty() && overlay.lines.empty() && overlay.highlights.empty() &&
           overlay.axes.empty() && overlay.texts.empty();
}

/// One event a logic node raised in one frame.
struct GraphEvent {
    std::string node;    ///< the logic node's id
    std::string event;   ///< "enter", "cross", a condition's "event", ...
    int track_id;        ///< -1: not about one track
    double time_s;       ///< FrameReport::time_s of its frame
    RecordItem data;     ///< numbers and text, written flat as "data"
    GraphEvent() : track_id(-1), time_s(0.0) {}
};

/// filter, zone, line, count, dwell, condition, gallery, user - in that order.
const std::vector<LogicOpInfo>& LogicOps();
/// NULL when `op` is not a logic op.
const LogicOpInfo* FindLogicOp(const std::string& op);
/// headpose and volume, described the same way (--list-ops, condition values).
const std::vector<LogicOpInfo>& CpuOpInfos();
/// NULL when `op` is not a cpu op.
const LogicOpInfo* FindCpuOpInfo(const std::string& op);
/// "number", "text", "choice", "text_list", "polygon", "line".
const char* ToString(LogicParamInfo::Kind kind);

// ---------------------------------------------------------------------
// Running a logic node
// ---------------------------------------------------------------------

/// What one "carry": "result" edge brings a logic node in one frame.
struct LogicInput {
    std::size_t edge;      ///< index into the graph's edges: inputs are used in this order
    std::string from;      ///< the producer's id
    StageDataPtr data;     ///< a full-frame payload; null for per-crop results
    RoiRef origin;         ///< the producer's origin: maps its boxes to source pixels
    bool per_crop;         ///< the producer runs once per crop
    std::vector<StageResult> crops;  ///< per_crop: every crop's result, ByOrigin order
    LogicInput() : edge(0), per_crop(false) {}
};

/// A logic node's params, typed, with the catalog's defaults filled in.
/// Built once (StageGraph::Build) and shared by every stream.
struct LogicConfig {
    std::string id;
    std::string op;
    std::map<std::string, double> numbers;
    std::map<std::string, std::string> text;
    std::map<std::string, std::vector<std::string> > lists;
    std::map<std::string, std::vector<double> > arrays;
    std::shared_ptr<const RetrievalGallery> gallery;  ///< op "gallery"
};

/// What a logic node remembers about one track id.
struct TrackMemo {
    cv::Point2f point;   ///< its last anchor, source pixels
    int side;            ///< line: -1 left of A->B, +1 right, 0 not known yet
    bool inside;         ///< zone: in the selected region when last seen
    double first_s;      ///< dwell: when this stay began
    double last_s;       ///< when it was last seen
    bool reported;       ///< dwell: this stay's event has been raised
    std::string name;    ///< gallery: the name last announced
    TrackMemo() : side(0), inside(false), first_s(0.0), last_s(0.0), reported(false) {}
};

/// One logic node's memory in one stream. Each op uses its own fields.
struct LogicState {
    std::map<int, TrackMemo> tracks;
    long long count_in;   ///< line
    long long count_out;  ///< line
    bool was_true;        ///< condition: last frame's verdict
    bool fired;           ///< condition: has raised an event
    double last_fire_s;   ///< condition: when
    /// user: the object registered under its "name" (user_logic.hpp), made
    /// on its first frame; null when nothing is registered (pass-through).
    std::shared_ptr<UserLogic> user;
    bool user_looked_up;  ///< user: the registry was asked once
    LogicState()
        : count_in(0), count_out(0), was_true(false), fired(false), last_fire_s(0.0),
          user_looked_up(false) {}
};

/// The frame a logic node runs on.
struct LogicFrame {
    cv::Size frame_size;  ///< the stream's source frame
    double time_s;        ///< FrameReport::time_s
    LogicFrame() : time_s(0.0) {}
};

struct LogicOutput {
    StageResult result;              ///< boxes (filter, zone, user) or records
    std::vector<GraphEvent> events;
    LogicOverlay overlay;
    std::string error;               ///< "" when the frame is valid
};

/// The typed config of a validated logic node. Throws GraphError
/// (kGraphSchema) when a gallery file cannot be read.
LogicConfig MakeLogicConfig(const NodeSpec& node);

/**
 * @brief One frame of one logic node.
 *
 * `inputs` are sorted by edge index first, so their arrival order (async)
 * does not matter. `state` is this node's state in this stream; the caller
 * runs a stream's frames in order. Boxes are read in source pixels (a
 * hand-off producer's are mapped back through its origin) and written in
 * source pixels with the identity origin.
 */
LogicOutput RunLogic(const LogicConfig& config, std::vector<LogicInput> inputs,
                     const LogicFrame& frame, LogicState* state);

/// Keeps the first use of each key across `data`'s numbers, then its text
/// (an event's "data" is one JSON object, so a repeated key would silently
/// replace a value) and drops the others. Returns the first key dropped, or "".
std::string DropRepeatedKeys(RecordItem* data);

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_LOGIC_OPS_HPP
