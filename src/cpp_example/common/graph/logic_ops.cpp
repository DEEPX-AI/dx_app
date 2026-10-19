#include "common/graph/logic_ops.hpp"
#include "common/graph/user_logic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "common/graph/cpu_reduce.hpp"
#include "common/graph/graph_error.hpp"
#include "common/graph/roi_router.hpp"
#include "common/processors/gallery_retrieval_postprocessor.hpp"

namespace dxapp {
namespace graph {
namespace {

LogicParamInfo Param(const char* name, LogicParamInfo::Kind kind, const char* help) {
    LogicParamInfo param;
    param.name = name;
    param.kind = kind;
    param.help = help;
    return param;
}

LogicParamInfo Number(const char* name, double low, double high, double fallback,
                      const char* help) {
    LogicParamInfo param = Param(name, LogicParamInfo::kNumber, help);
    param.low = low;
    param.high = high;
    param.number_default = fallback;
    return param;
}

/// A number that must be greater than 0.
LogicParamInfo Positive(const char* name, double high, double fallback, const char* help) {
    LogicParamInfo param = Number(name, 0.0, high, fallback, help);
    param.low_open = true;
    return param;
}

LogicParamInfo Text(const char* name, const char* fallback, const char* help) {
    LogicParamInfo param = Param(name, LogicParamInfo::kText, help);
    param.text_default = fallback;
    return param;
}

LogicParamInfo Choice(const char* name, const char* fallback,
                      const std::vector<std::string>& choices, const char* help) {
    LogicParamInfo param = Param(name, LogicParamInfo::kChoice, help);
    param.text_default = fallback;
    param.choices = choices;
    return param;
}

LogicParamInfo Step(LogicParamInfo param, double step) {
    param.step = step;
    return param;
}

LogicParamInfo List(const char* name, const char* items, const char* help) {
    LogicParamInfo param = Param(name, LogicParamInfo::kTextList, help);
    param.items = items;
    return param;
}

LogicParamInfo::InputValues ValuesFor(Shape shape, bool per_crop,
                                      const std::vector<std::string>& values) {
    LogicParamInfo::InputValues allowed;
    allowed.shape = shape;
    allowed.per_crop = per_crop;
    allowed.values = values;
    return allowed;
}

LogicParamInfo Required(LogicParamInfo param) {
    param.required = true;
    param.text_default.clear();
    return param;
}

LogicPort Port(Shape shape, bool per_crop) {
    LogicPort port;
    port.shape = shape;
    port.per_crop = per_crop;
    return port;
}

/// The three box shapes, full frame.
std::vector<LogicPort> BoxPorts() {
    std::vector<LogicPort> ports;
    ports.push_back(Port(Shape::kBoxes, false));
    ports.push_back(Port(Shape::kObBoxes, false));
    ports.push_back(Port(Shape::kInstances, false));
    return ports;
}

std::vector<std::string> Names(const char* const* names, std::size_t count) {
    return std::vector<std::string>(names, names + count);
}

/// Every shape a graph node can hand on, as the user op's "input"/"output" choices.
const Shape kPassable[] = {Shape::kBoxes, Shape::kObBoxes, Shape::kInstances,
                           Shape::kKeypoints, Shape::kLabelMap, Shape::kDenseMap,
                           Shape::kImage, Shape::kScores, Shape::kVector,
                           Shape::kRecords};

LogicParamInfo AnchorParam() {
    const char* const anchors[] = {"bottom", "center"};
    return Choice("anchor", "bottom", Names(anchors, 2),
                  "the point of a box that is tested: its bottom centre (where a person "
                  "stands) or its centre");
}

std::vector<LogicOpInfo> BuildLogicOps() {
    std::vector<LogicOpInfo> ops;

    LogicOpInfo filter;
    filter.op = "filter";
    filter.summary = "keeps the boxes of the listed classes with at least a score and an area";
    filter.accepts = BoxPorts();
    filter.same_shape_out = true;
    filter.params.push_back(List("classes", "string",
        "class names to keep, as the detector names them; leave it out to keep every class"));
    filter.params.push_back(Step(Number("min_score", 0.0, 1.0, 0.0,
        "the smallest confidence kept (0.5, not 50)"), 0.01));
    filter.params.push_back(Step(Number("min_area", 0.0, kLogicUnbounded, 0.0,
        "the smallest box area kept, in source-frame pixels"), 1.0));
    ops.push_back(filter);

    LogicOpInfo zone;
    zone.op = "zone";
    zone.summary = "keeps the boxes whose anchor point is inside (or outside) a polygon; "
                   "with track ids, an event when a box enters or leaves";
    zone.accepts = BoxPorts();
    zone.same_shape_out = true;
    zone.events.push_back("enter");
    zone.events.push_back("exit");
    zone.params.push_back(Required(Param("polygon", LogicParamInfo::kPolygon,
        "x0, y0, x1, y1, ...: at least 3 points, each a fraction 0..1 of the frame's "
        "width and height")));
    const char* const modes[] = {"inside", "outside"};
    zone.params.push_back(Choice("mode", "inside", Names(modes, 2),
        "which boxes pass: those whose anchor is inside the polygon, or outside it"));
    zone.params.push_back(AnchorParam());
    ops.push_back(zone);

    LogicOpInfo line;
    line.op = "line";
    line.summary = "counts tracked boxes crossing a line, in each direction, with an event "
                   "per crossing";
    line.accepts = BoxPorts();
    line.needs_tracking = true;
    line.events.push_back("cross");
    line.numbers.push_back("in");
    line.numbers.push_back("out");
    line.numbers.push_back("total");
    line.params.push_back(Required(Param("line", LogicParamInfo::kLine,
        "x0, y0, x1, y1: the line from A to B, each a fraction 0..1 of the frame's width "
        "and height; \"in\" is a crossing from the left of A->B to its right")));
    line.params.push_back(AnchorParam());
    const char* const directions[] = {"both", "in", "out"};
    line.params.push_back(Choice("direction", "both", Names(directions, 3),
        "which crossings raise a \"cross\" event; the counts always keep both"));
    ops.push_back(line);

    LogicOpInfo count;
    count.op = "count";
    count.summary = "counts the boxes of each input (one record per input), in total and "
                    "per listed class";
    count.accepts = BoxPorts();
    count.max_inputs = 8;
    count.numbers.push_back("total");
    count.params.push_back(List("classes", "string",
        "count only these classes, one number each besides \"total\"; leave it out to "
        "count every box"));
    ops.push_back(count);

    LogicOpInfo dwell;
    dwell.op = "dwell";
    dwell.summary = "measures how long each tracked box has stayed, with an event when it "
                    "passes a threshold";
    dwell.accepts = BoxPorts();
    dwell.needs_tracking = true;
    dwell.events.push_back("dwell");
    dwell.numbers.push_back("track_id");
    dwell.numbers.push_back("seconds");
    dwell.params.push_back(Step(Positive("threshold_s", kLogicUnbounded, 10.0,
        "seconds a box must stay before its \"dwell\" event"), 0.1));
    dwell.params.push_back(Step(Number("grace_s", 0.0, 60.0, 1.0,
        "seconds a box may be missing (an occlusion) and still be the same stay"), 0.1));
    ops.push_back(dwell);

    LogicOpInfo condition;
    condition.op = "condition";
    condition.summary = "raises an event when a value passes a threshold: a box or crop "
                        "count, a records number, or the crops with or without a label";
    condition.accepts = BoxPorts();
    condition.accepts.push_back(Port(Shape::kRecords, false));
    condition.accepts.push_back(Port(Shape::kScores, true));
    condition.accepts.push_back(Port(Shape::kVector, true));
    condition.events.push_back("<event>");
    condition.numbers.push_back("value");
    condition.numbers.push_back("active");
    LogicParamInfo value = Text("value", "count",
        "\"count\" (boxes or crops), a number the input writes (records), or "
        "\"with:<label>\" / \"without:<label>\" (crops with scores)");
    const std::vector<std::string> just_count(1, "count");
    const char* const crop_values[] = {"count", "with:<label>", "without:<label>"};
    for (std::size_t i = 0; i < condition.accepts.size(); ++i) {
        const LogicPort& port = condition.accepts[i];
        LogicParamInfo::InputValues allowed = ValuesFor(port.shape, port.per_crop, just_count);
        if (port.shape == Shape::kRecords) {
            allowed.values.clear();
            allowed.from_numbers = true;
        }
        if (port.shape == Shape::kScores) allowed.values = Names(crop_values, 3);
        value.values_by_input.push_back(allowed);
    }
    condition.params.push_back(value);
    const char* const comparators[] = {">", ">=", "<", "<=", "==", "!="};
    condition.params.push_back(Choice("comparator", ">", Names(comparators, 6),
        "how the value is compared with \"threshold\""));
    condition.params.push_back(Number("threshold", -kLogicUnbounded, kLogicUnbounded, 0.0,
        "the number the value is compared with"));
    const char* const reductions[] = {"max", "min", "sum"};
    condition.params.push_back(Choice("reduce", "max", Names(reductions, 3),
        "how several records' values become one: their largest, smallest or sum"));
    condition.params.push_back(Step(Number("cooldown_s", 0.0, 86400.0, 0.0,
        "0: one event when the condition becomes true; more: an event at most this "
        "often while it holds"), 0.1));
    condition.params.push_back(Text("event", "",
        "the event name (letters, digits, _ . : -); leave it out for the node id"));
    ops.push_back(condition);

    LogicOpInfo gallery;
    gallery.op = "gallery";
    gallery.summary = "names each embedding after its nearest gallery entry, or \"unknown\" "
                      "under a threshold, with an event when a track's name changes";
    gallery.accepts.push_back(Port(Shape::kVector, true));
    gallery.accepts.push_back(Port(Shape::kVector, false));
    gallery.events.push_back("match");
    gallery.events.push_back("unknown");
    gallery.numbers.push_back("track_id");
    gallery.numbers.push_back("score");
    gallery.numbers.push_back("parent_index");
    LogicParamInfo file = Required(Param("gallery", LogicParamInfo::kText,
        "a DXGAL1 .bin built from an image folder: python3 scripts/build_gallery_database.py "
        "--variant <model> --images <folder> --out <file>.bin"));
    file.format = "file";
    file.extensions.push_back(".bin");
    gallery.params.push_back(file);
    gallery.params.push_back(Step(Number("threshold", 0.0, 1.0, 0.5,
        "the smallest cosine similarity that names a match"), 0.01));
    const char* const modes_of_events[] = {"known", "all", "none"};
    gallery.params.push_back(Choice("events", "known", Names(modes_of_events, 3),
        "\"match\" events only, \"unknown\" events too, or none"));
    ops.push_back(gallery);

    LogicOpInfo user;
    user.op = "user";
    user.summary = "your own logic: in a run it passes its input through; an exported "
                   "project gets a function stub to fill in";
    std::vector<std::string> shapes;
    for (std::size_t i = 0; i < sizeof(kPassable) / sizeof(kPassable[0]); ++i) {
        user.accepts.push_back(Port(kPassable[i], false));
        shapes.push_back(ToString(kPassable[i]));
    }
    user.same_shape_out = true;
    user.params.push_back(Required(Param("name", LogicParamInfo::kText,
        "the stub function's name in an exported project: letters, digits and _, not "
        "starting with a digit")));
    user.params.push_back(Required(Choice("input", "", shapes,
        "the kind of data the function receives")));
    LogicParamInfo returned = Required(Choice("output", "", shapes,
        "the kind it returns; in a run the input passes through, so it equals \"input\""));
    returned.equals = "input";
    user.params.push_back(returned);
    ops.push_back(user);
    return ops;
}

std::vector<LogicOpInfo> BuildCpuOps() {
    std::vector<LogicOpInfo> ops;

    LogicOpInfo headpose;
    headpose.op = "headpose";
    headpose.summary = "head pitch, yaw and roll (degrees) of each face from its five "
                       "landmarks, one record per face";
    headpose.accepts.push_back(Port(Shape::kBoxes, false));
    const char* const angles[] = {"pitch", "yaw", "roll", "face_index", "track_id"};
    headpose.numbers = Names(angles, 5);
    ops.push_back(headpose);

    LogicOpInfo volume;
    volume.op = "volume";
    volume.summary = "a volume proxy per package box from its instance mask and the depth "
                     "map under it";
    volume.accepts.push_back(Port(Shape::kBoxes, false));
    volume.accepts.push_back(Port(Shape::kInstances, false));
    volume.accepts.push_back(Port(Shape::kDenseMap, false));
    volume.min_inputs = 3;
    volume.max_inputs = 3;
    const char* const measures[] = {"box_x", "box_y", "box_w", "box_h", "class_id",
                                    "mask_area_px", "median_depth", "volume_proxy"};
    volume.numbers = Names(measures, 8);
    volume.params.push_back(Step(Number("scale_factor", 0.0, kLogicUnbounded, 1.0,
        "multiplies the proxy; 1.0 leaves it in mask-pixels times depth"), 0.1));
    LogicParamInfo classes = List("classes", "string",
        "package class names; leave it out for the built-in ones (when_omitted), [] to "
        "match none");
    classes.when_omitted = CpuPackageClassNames();
    volume.params.push_back(classes);
    LogicParamInfo class_ids = List("class_ids", "integer",
        "package class ids, each an integer written as a string (\"24\"); leave it out "
        "for the built-in ones (when_omitted), [] to match none");
    class_ids.when_omitted = CpuPackageClassIds();
    volume.params.push_back(class_ids);
    ops.push_back(volume);
    return ops;
}

const LogicOpInfo* FindIn(const std::vector<LogicOpInfo>& ops, const std::string& op) {
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].op == op) return &ops[i];
    }
    return NULL;
}

}  // namespace

const std::vector<LogicOpInfo>& LogicOps() {
    static const std::vector<LogicOpInfo> ops = BuildLogicOps();
    return ops;
}

const LogicOpInfo* FindLogicOp(const std::string& op) { return FindIn(LogicOps(), op); }

const std::vector<LogicOpInfo>& CpuOpInfos() {
    static const std::vector<LogicOpInfo> ops = BuildCpuOps();
    return ops;
}

const LogicOpInfo* FindCpuOpInfo(const std::string& op) { return FindIn(CpuOpInfos(), op); }

const char* ToString(LogicParamInfo::Kind kind) {
    switch (kind) {
        case LogicParamInfo::kNumber:   return "number";
        case LogicParamInfo::kText:     return "text";
        case LogicParamInfo::kChoice:   return "choice";
        case LogicParamInfo::kTextList: return "text_list";
        case LogicParamInfo::kPolygon:  return "polygon";
        case LogicParamInfo::kLine:     return "line";
    }
    return "unknown";
}

namespace {

/// A track not seen for this long (stream seconds) is forgotten.
const double kForgetAfterS = 5.0;
/// Slack for comparing stream times. A file's frame k is at k / fps, and
/// the difference of two such times can miss a whole number of frame
/// periods by a rounding error: 1 s at 30 fps would then hold one frame
/// late. Every time threshold below is compared with this much slack.
const double kTimeSlackS = 1e-6;

double Number(const LogicConfig& c, const char* key) {
    std::map<std::string, double>::const_iterator it = c.numbers.find(key);
    return it == c.numbers.end() ? 0.0 : it->second;
}

std::string Text(const LogicConfig& c, const char* key) {
    std::map<std::string, std::string>::const_iterator it = c.text.find(key);
    return it == c.text.end() ? std::string() : it->second;
}

const std::vector<std::string>& List(const LogicConfig& c, const char* key) {
    static const std::vector<std::string> kNone;
    std::map<std::string, std::vector<std::string> >::const_iterator it = c.lists.find(key);
    return it == c.lists.end() ? kNone : it->second;
}

/// A polygon or line param in source pixels.
std::vector<cv::Point2f> Points(const LogicConfig& c, const char* key, const cv::Size& size) {
    std::vector<cv::Point2f> points;
    std::map<std::string, std::vector<double> >::const_iterator it = c.arrays.find(key);
    if (it == c.arrays.end()) return points;
    for (std::size_t i = 0; i + 1 < it->second.size(); i += 2) {
        points.push_back(cv::Point2f(static_cast<float>(it->second[i] * size.width),
                                     static_cast<float>(it->second[i + 1] * size.height)));
    }
    return points;
}

cv::Point2f Anchor(const cv::Rect2f& box, const std::string& anchor) {
    if (anchor == "center") {
        return cv::Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
    }
    return cv::Point2f(box.x + box.width * 0.5f, box.y + box.height);  // "bottom"
}

bool Listed(const std::vector<std::string>& classes, const BoxItem& item) {
    return classes.empty() ||
           std::find(classes.begin(), classes.end(), item.class_name) != classes.end();
}

void AddNumber(RecordItem* item, const std::string& name, double value) {
    item->numbers.push_back(std::make_pair(name, value));
}

void AddText(RecordItem* item, const std::string& name, const std::string& value) {
    item->text.push_back(std::make_pair(name, value));
}

GraphEvent MakeEvent(const LogicConfig& c, const std::string& name, int track_id,
                     const LogicFrame& at) {
    GraphEvent event;
    event.node = c.id;
    event.event = name;
    event.track_id = track_id;
    event.time_s = at.time_s;
    return event;
}

/// "3" for a whole number, else two decimals.
std::string Short(double value) {
    char text[64];
    if (std::isfinite(value) && std::fabs(value - std::round(value)) < 1e-9) {
        std::snprintf(text, sizeof(text), "%.0f", value);
    } else {
        std::snprintf(text, sizeof(text), "%.2f", value);
    }
    return text;
}

cv::Point2f Apply(const cv::Matx23f& m, const cv::Point2f& p) {
    return cv::Point2f(m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2),
                       m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2));
}

/// The input's boxes in source pixels; null when it carries no boxes.
std::shared_ptr<BoxesData> SourceBoxes(const LogicInput& input) {
    const BoxesData* boxes =
        input.data ? dynamic_cast<const BoxesData*>(input.data.get()) : NULL;
    if (boxes == NULL) return std::shared_ptr<BoxesData>();
    std::shared_ptr<BoxesData> out(new BoxesData(*boxes));
    if (!input.origin.from_roi && IsIdentityAffine(input.origin.inv_align)) return out;
    for (std::size_t i = 0; i < out->items.size(); ++i) {
        BoxItem& item = out->items[i];
        item.box = RestoreBox(item.box, input.origin);
        for (std::size_t k = 0; k < item.landmarks.size(); ++k) {
            const cv::Point2f p = Apply(input.origin.inv_align,
                                        cv::Point2f(item.landmarks[k].x, item.landmarks[k].y));
            item.landmarks[k].x = p.x;
            item.landmarks[k].y = p.y;
        }
    }
    return out;
}

/// The first full-frame input that carries a payload, or NULL.
const LogicInput* FirstFullFrame(const std::vector<LogicInput>& inputs) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].per_crop && inputs[i].data) return &inputs[i];
    }
    return NULL;
}

/// Forget every track not in `seen` and not seen for `after` seconds;
/// `on_forget` hears each one first.
template <typename OnForget>
void Forget(LogicState* s, const std::set<int>& seen, double now, double after,
            OnForget on_forget) {
    for (std::map<int, TrackMemo>::iterator it = s->tracks.begin(); it != s->tracks.end();) {
        if (seen.count(it->first) == 0 && now - it->second.last_s > after + kTimeSlackS) {
            on_forget(it->first, it->second);
            it = s->tracks.erase(it);
        } else {
            ++it;
        }
    }
}

LogicOutput RunFilter(const LogicConfig& c, const std::vector<LogicInput>& inputs) {
    LogicOutput out;
    const LogicInput* input = FirstFullFrame(inputs);
    const std::shared_ptr<BoxesData> boxes =
        input == NULL ? std::shared_ptr<BoxesData>() : SourceBoxes(*input);
    if (!boxes) {
        out.result.data.reset(new BoxesData(Shape::kBoxes));
        out.error = "missing boxes input";
        return out;
    }
    const std::vector<std::string>& classes = List(c, "classes");
    const float min_score = static_cast<float>(Number(c, "min_score"));
    const double min_area = Number(c, "min_area");
    std::shared_ptr<BoxesData> kept(new BoxesData(boxes->box_shape));
    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const BoxItem& item = boxes->items[i];
        if (!Listed(classes, item) || item.score < min_score) continue;
        if (static_cast<double>(item.box.width) * item.box.height < min_area) continue;
        kept->items.push_back(item);
    }
    out.result.data = kept;
    return out;
}

LogicOutput RunZone(const LogicConfig& c, const std::vector<LogicInput>& inputs,
                    const LogicFrame& at, LogicState* s) {
    LogicOutput out;
    const std::vector<cv::Point2f> polygon = Points(c, "polygon", at.frame_size);
    out.overlay.polygons.push_back(polygon);
    const LogicInput* input = FirstFullFrame(inputs);
    const std::shared_ptr<BoxesData> boxes =
        input == NULL ? std::shared_ptr<BoxesData>() : SourceBoxes(*input);
    if (!boxes) out.error = "missing boxes input";
    std::shared_ptr<BoxesData> kept(new BoxesData(boxes ? boxes->box_shape : Shape::kBoxes));
    const bool want_inside = Text(c, "mode") != "outside";
    const std::string anchor = Text(c, "anchor");
    std::set<int> seen;
    std::vector<std::pair<std::string, int> > changes;  // (enter|exit, track), in box order
    if (boxes && polygon.size() >= 3) {
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            const BoxItem& item = boxes->items[i];
            const cv::Point2f p = Anchor(item.box, anchor);
            const bool inside = cv::pointPolygonTest(polygon, p, false) >= 0.0;
            const bool selected = inside == want_inside;
            if (selected) {
                kept->items.push_back(item);
                out.overlay.highlights.push_back(item.box);
            }
            if (item.track_id < 0) continue;
            seen.insert(item.track_id);
            TrackMemo& memo = s->tracks[item.track_id];
            if (selected && !memo.inside) changes.push_back(std::make_pair("enter", item.track_id));
            if (!selected && memo.inside) changes.push_back(std::make_pair("exit", item.track_id));
            memo.inside = selected;
            memo.point = p;
            memo.last_s = at.time_s;
        }
    }
    Forget(s, seen, at.time_s, kForgetAfterS, [&changes](int track, const TrackMemo& memo) {
        if (memo.inside) changes.push_back(std::make_pair("exit", track));
    });
    for (std::size_t i = 0; i < changes.size(); ++i) {
        GraphEvent event = MakeEvent(c, changes[i].first, changes[i].second, at);
        AddNumber(&event.data, "count", static_cast<double>(kept->items.size()));
        out.events.push_back(event);
    }
    out.overlay.alert = !kept->items.empty();
    OverlayText label;  // inside the zone, at its top-left corner
    if (!polygon.empty()) {
        label.at = polygon[0];
        for (std::size_t i = 1; i < polygon.size(); ++i) {
            if (polygon[i].x + polygon[i].y < label.at.x + label.at.y) label.at = polygon[i];
        }
        label.away = cv::Point2f(1.f, 1.f);
    }
    label.text = c.id + ": " + std::to_string(kept->items.size());
    label.alert = out.overlay.alert;
    out.overlay.texts.push_back(label);
    out.result.data = kept;
    return out;
}

/// +1 when p is right of A->B on the image (y down), -1 left, 0 on the line.
int Side(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& p) {
    const double cross = static_cast<double>(b.x - a.x) * (p.y - a.y) -
                         static_cast<double>(b.y - a.y) * (p.x - a.x);
    return cross > 0.0 ? 1 : (cross < 0.0 ? -1 : 0);
}

/// The line through a movement p1 -> p2 meets segment A-B (endpoints included).
bool MovementMeetsSegment(const cv::Point2f& p1, const cv::Point2f& p2,
                          const cv::Point2f& a, const cv::Point2f& b) {
    return Side(p1, p2, a) * Side(p1, p2, b) <= 0;
}

LogicOutput RunLine(const LogicConfig& c, const std::vector<LogicInput>& inputs,
                    const LogicFrame& at, LogicState* s) {
    LogicOutput out;
    const std::vector<cv::Point2f> line = Points(c, "line", at.frame_size);
    if (line.size() == 2) out.overlay.lines.push_back(std::make_pair(line[0], line[1]));
    const LogicInput* input = FirstFullFrame(inputs);
    const std::shared_ptr<BoxesData> boxes =
        input == NULL ? std::shared_ptr<BoxesData>() : SourceBoxes(*input);
    if (!boxes) out.error = "missing boxes input";
    const std::string anchor = Text(c, "anchor");
    const std::string direction = Text(c, "direction");
    std::set<int> seen;
    if (boxes && line.size() == 2) {
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            const BoxItem& item = boxes->items[i];
            if (item.track_id < 0) continue;
            seen.insert(item.track_id);
            const cv::Point2f p = Anchor(item.box, anchor);
            const int side = Side(line[0], line[1], p);
            std::map<int, TrackMemo>::iterator found = s->tracks.find(item.track_id);
            if (found == s->tracks.end()) {
                TrackMemo memo;
                memo.point = p;
                memo.side = side;
                memo.last_s = at.time_s;
                s->tracks[item.track_id] = memo;
                continue;
            }
            TrackMemo& memo = found->second;
            if (side != 0 && memo.side != 0 && side != memo.side &&
                MovementMeetsSegment(memo.point, p, line[0], line[1])) {
                const bool inward = memo.side < 0;  // from the left of A->B to its right
                if (inward) {
                    ++s->count_in;
                } else {
                    ++s->count_out;
                }
                if (direction == "both" || direction == (inward ? "in" : "out")) {
                    GraphEvent event = MakeEvent(c, "cross", item.track_id, at);
                    AddText(&event.data, "direction", inward ? "in" : "out");
                    AddNumber(&event.data, "in", static_cast<double>(s->count_in));
                    AddNumber(&event.data, "out", static_cast<double>(s->count_out));
                    out.events.push_back(event);
                }
            }
            if (side != 0) memo.side = side;  // on the line: keep the side it came from
            memo.point = p;
            memo.last_s = at.time_s;
        }
    }
    Forget(s, seen, at.time_s, kForgetAfterS, [](int, const TrackMemo&) {});
    std::shared_ptr<RecordsData> records(new RecordsData());
    RecordItem totals;
    AddNumber(&totals, "in", static_cast<double>(s->count_in));
    AddNumber(&totals, "out", static_cast<double>(s->count_out));
    AddNumber(&totals, "total", static_cast<double>(s->count_in + s->count_out));
    records->items.push_back(totals);
    out.result.data = records;
    if (line.size() == 2) {
        OverlayText label;  // beside the middle, left of A->B: the "in" arrow points right
        label.at = (line[0] + line[1]) * 0.5f;
        const cv::Point2f along = line[1] - line[0];
        const float length = std::sqrt(along.x * along.x + along.y * along.y);
        if (length > 0.f) label.away = cv::Point2f(along.y / length, -along.x / length);
        label.text = c.id + " in " + std::to_string(s->count_in) + " out " +
                     std::to_string(s->count_out);
        label.alert = !out.events.empty();
        out.overlay.texts.push_back(label);
        out.overlay.alert = !out.events.empty();
    }
    return out;
}

LogicOutput RunCount(const LogicConfig& c, const std::vector<LogicInput>& inputs) {
    LogicOutput out;
    const std::vector<std::string>& classes = List(c, "classes");
    std::shared_ptr<RecordsData> records(new RecordsData());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].per_crop) continue;
        const BoxesData* boxes =
            inputs[i].data ? dynamic_cast<const BoxesData*>(inputs[i].data.get()) : NULL;
        std::size_t total = 0;
        std::vector<std::size_t> per_class(classes.size(), 0);
        if (boxes != NULL) {
            for (std::size_t b = 0; b < boxes->items.size(); ++b) {
                if (!Listed(classes, boxes->items[b])) continue;
                ++total;
                for (std::size_t k = 0; k < classes.size(); ++k) {
                    if (classes[k] == boxes->items[b].class_name) ++per_class[k];
                }
            }
        }
        RecordItem item;
        AddNumber(&item, "total", static_cast<double>(total));
        for (std::size_t k = 0; k < classes.size(); ++k) {
            AddNumber(&item, classes[k], static_cast<double>(per_class[k]));
        }
        AddText(&item, "node", inputs[i].from);
        records->items.push_back(item);
        OverlayText label;  // a line in the top-left panel
        label.text = c.id + " " + inputs[i].from + ": " + std::to_string(total);
        out.overlay.texts.push_back(label);
    }
    if (inputs.empty()) out.error = "missing boxes input";
    out.result.data = records;
    return out;
}

LogicOutput RunDwell(const LogicConfig& c, const std::vector<LogicInput>& inputs,
                     const LogicFrame& at, LogicState* s) {
    LogicOutput out;
    const double threshold = Number(c, "threshold_s");
    const double grace = Number(c, "grace_s");
    const LogicInput* input = FirstFullFrame(inputs);
    const std::shared_ptr<BoxesData> boxes =
        input == NULL ? std::shared_ptr<BoxesData>() : SourceBoxes(*input);
    if (!boxes) out.error = "missing boxes input";
    std::shared_ptr<RecordsData> records(new RecordsData());
    std::set<int> seen;
    if (boxes) {
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            const BoxItem& item = boxes->items[i];
            if (item.track_id < 0) continue;
            seen.insert(item.track_id);
            std::map<int, TrackMemo>::iterator found = s->tracks.find(item.track_id);
            if (found == s->tracks.end() || at.time_s - found->second.last_s > grace + kTimeSlackS) {
                TrackMemo stay;  // a new stay: first seen now
                stay.first_s = at.time_s;
                s->tracks[item.track_id] = stay;
                found = s->tracks.find(item.track_id);
            }
            TrackMemo& memo = found->second;
            memo.last_s = at.time_s;
            memo.point = Anchor(item.box, "bottom");
            // To the nanosecond: 34/30 - 1/30 s is 1.1, not 1.0999999999999999.
            const double seconds = std::round((at.time_s - memo.first_s) * 1e9) / 1e9;
            RecordItem row;
            AddNumber(&row, "track_id", static_cast<double>(item.track_id));
            AddNumber(&row, "seconds", seconds);
            records->items.push_back(row);
            const bool over = seconds >= threshold - kTimeSlackS;
            if (over && !memo.reported) {
                memo.reported = true;
                GraphEvent event = MakeEvent(c, "dwell", item.track_id, at);
                AddNumber(&event.data, "seconds", seconds);
                out.events.push_back(event);
            }
            if (over) out.overlay.highlights.push_back(item.box);
            OverlayText label;
            label.at = cv::Point2f(item.box.x, item.box.y + item.box.height);
            char text[64];
            std::snprintf(text, sizeof(text), "#%d %.1fs", item.track_id, seconds);
            label.text = text;
            label.alert = over;
            out.overlay.texts.push_back(label);
        }
    }
    Forget(s, seen, at.time_s, std::max(kForgetAfterS, grace), [](int, const TrackMemo&) {});
    out.overlay.alert = !out.overlay.highlights.empty();
    out.result.data = records;
    return out;
}

/// Values within kTimeSlackS of the threshold count as equal to it: a
/// records number is often a stream time (dwell's seconds) or a sum of
/// fractions, and misses its threshold by a rounding error like times do.
bool Compare(double value, const std::string& comparator, double threshold) {
    const bool equal = std::fabs(value - threshold) <= kTimeSlackS;
    if (comparator == ">") return !equal && value > threshold;
    if (comparator == ">=") return equal || value > threshold;
    if (comparator == "<") return !equal && value < threshold;
    if (comparator == "<=") return equal || value < threshold;
    if (comparator == "==") return equal;
    if (comparator == "!=") return !equal;
    return false;
}

/// The value a condition reads this frame, and the tracks behind it; false
/// when the input holds none (records without that number).
bool ConditionValue(const LogicConfig& c, const LogicInput& input, double* value,
                    std::vector<int>* tracks) {
    const std::string name = Text(c, "value");
    if (input.per_crop) {
        const bool with = name.compare(0, 5, "with:") == 0;
        const bool without = name.compare(0, 8, "without:") == 0;
        const std::string label = with ? name.substr(5) : without ? name.substr(8) : std::string();
        std::size_t matched = 0;
        for (std::size_t i = 0; i < input.crops.size(); ++i) {
            bool counts = true;
            if (with || without) {
                const ScoresData* scores =
                    dynamic_cast<const ScoresData*>(input.crops[i].data.get());
                bool named = false;
                for (std::size_t k = 0; scores != NULL && k < scores->items.size(); ++k) {
                    named = named || scores->items[k].class_name == label;
                }
                counts = with ? named : !named;
            }
            if (!counts) continue;
            ++matched;
            tracks->push_back(input.crops[i].origin.track_id);
        }
        *value = static_cast<double>(matched);
        return true;
    }
    if (!input.data) return false;
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(input.data.get());
    if (boxes != NULL) {
        *value = static_cast<double>(boxes->items.size());
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            tracks->push_back(boxes->items[i].track_id);
        }
        return true;
    }
    const RecordsData* records = dynamic_cast<const RecordsData*>(input.data.get());
    if (records == NULL) return false;
    const std::string reduce = Text(c, "reduce");
    bool found = false;
    double result = 0.0;
    for (std::size_t i = 0; i < records->items.size(); ++i) {
        for (std::size_t n = 0; n < records->items[i].numbers.size(); ++n) {
            if (records->items[i].numbers[n].first != name) continue;
            const double v = records->items[i].numbers[n].second;
            if (!found) {
                result = v;
            } else if (reduce == "sum") {
                result += v;
            } else if (reduce == "min") {
                result = std::min(result, v);
            } else {
                result = std::max(result, v);
            }
            found = true;
        }
    }
    *value = result;
    return found;
}

LogicOutput RunCondition(const LogicConfig& c, const std::vector<LogicInput>& inputs,
                         const LogicFrame& at, LogicState* s) {
    LogicOutput out;
    double value = 0.0;
    std::vector<int> tracks;
    const bool has_value = !inputs.empty() && ConditionValue(c, inputs[0], &value, &tracks);
    if (inputs.empty()) out.error = "missing input";
    const double threshold = Number(c, "threshold");
    const std::string comparator = Text(c, "comparator");
    const bool active = has_value && Compare(value, comparator, threshold);
    const double cooldown = Number(c, "cooldown_s");
    // cooldown 0: once, when it becomes true; else at most every cooldown_s while true.
    const bool fire = active && (cooldown <= 0.0
                                     ? !s->was_true
                                     : (!s->fired ||
                                        at.time_s - s->last_fire_s >= cooldown - kTimeSlackS));
    s->was_true = active;
    const std::string name = Text(c, "event").empty() ? c.id : Text(c, "event");
    if (fire) {
        s->fired = true;
        s->last_fire_s = at.time_s;
        std::string ids;
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            if (tracks[i] < 0) continue;
            ids += (ids.empty() ? "" : ",") + std::to_string(tracks[i]);
        }
        GraphEvent event = MakeEvent(c, name, tracks.size() == 1 ? tracks[0] : -1, at);
        AddNumber(&event.data, "value", value);
        AddNumber(&event.data, "threshold", threshold);
        if (!ids.empty()) AddText(&event.data, "track_ids", ids);
        out.events.push_back(event);
    }
    std::shared_ptr<RecordsData> records(new RecordsData());
    RecordItem row;
    AddNumber(&row, "value", value);
    AddNumber(&row, "active", active ? 1.0 : 0.0);
    records->items.push_back(row);
    out.result.data = records;
    OverlayText label;  // a line in the top-left panel
    label.text = name + ": " + (has_value ? Short(value) : std::string("-")) + " " + comparator +
                 " " + Short(threshold);
    label.alert = active;
    out.overlay.texts.push_back(label);
    return out;
}

/// An entry's name: its label, else its image's file stem.
std::string GalleryName(const RetrievalGallery& gallery, std::size_t row) {
    if (row < gallery.labels.size() && !gallery.labels[row].empty()) return gallery.labels[row];
    const std::string path = row < gallery.paths.size() ? gallery.paths[row] : std::string();
    const std::size_t slash = path.find_last_of("/\\");
    std::string stem = slash == std::string::npos ? path : path.substr(slash + 1);
    const std::size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    return stem.empty() ? "entry_" + std::to_string(row) : stem;
}

/// BestMatch's row for a query with no direction (all zeros).
const std::size_t kNoMatch = static_cast<std::size_t>(-1);

/// The nearest gallery row by cosine similarity. False when the widths
/// differ. A zero vector matches no row (*row = kNoMatch, *score = 0).
bool BestMatch(const RetrievalGallery& gallery, const std::vector<float>& vector,
               std::size_t* row, double* score) {
    if (gallery.count <= 0 || static_cast<int>(vector.size()) != gallery.dim) return false;
    double norm = 0.0;
    for (std::size_t d = 0; d < vector.size(); ++d) {
        norm += static_cast<double>(vector[d]) * vector[d];
    }
    norm = std::sqrt(norm);
    *row = kNoMatch;
    *score = 0.0;
    if (norm <= 1e-12) return true;
    double best = -2.0;
    for (int r = 0; r < gallery.count; ++r) {
        const float* entry = gallery.vectors.data() + static_cast<std::size_t>(r) * gallery.dim;
        double dot = 0.0;
        for (int d = 0; d < gallery.dim; ++d) dot += static_cast<double>(entry[d]) * vector[d];
        dot /= norm;
        if (dot > best) {
            best = dot;
            *row = static_cast<std::size_t>(r);
        }
    }
    *score = best;
    return true;
}

LogicOutput RunGallery(const LogicConfig& c, const std::vector<LogicInput>& inputs,
                       const LogicFrame& at, LogicState* s) {
    LogicOutput out;
    std::shared_ptr<RecordsData> records(new RecordsData());
    out.result.data = records;
    if (!c.gallery) {
        out.error = "no gallery loaded";
        return out;
    }
    if (inputs.empty()) {
        out.error = "missing vector input";
        return out;
    }
    // One query per crop, or the one full-frame vector.
    std::vector<std::pair<const VectorData*, RoiRef> > queries;
    const LogicInput& input = inputs[0];
    if (input.per_crop) {
        for (std::size_t i = 0; i < input.crops.size(); ++i) {
            const VectorData* vector =
                dynamic_cast<const VectorData*>(input.crops[i].data.get());
            if (vector != NULL) queries.push_back(std::make_pair(vector, input.crops[i].origin));
        }
    } else if (input.data) {
        const VectorData* vector = dynamic_cast<const VectorData*>(input.data.get());
        if (vector != NULL) queries.push_back(std::make_pair(vector, input.origin));
    }
    const double threshold = Number(c, "threshold");
    const std::string mode = Text(c, "events");
    std::set<int> seen;
    for (std::size_t q = 0; q < queries.size(); ++q) {
        std::size_t row = 0;
        double score = 0.0;
        if (!BestMatch(*c.gallery, queries[q].first->values, &row, &score)) {
            out.error = "a vector of " + std::to_string(queries[q].first->values.size()) +
                        " values, but the gallery holds " + std::to_string(c.gallery->dim);
            continue;
        }
        const RoiRef& origin = queries[q].second;
        const bool known = row != kNoMatch && score >= threshold;
        const std::string name = known ? GalleryName(*c.gallery, row) : "unknown";
        RecordItem item;
        AddNumber(&item, "track_id", static_cast<double>(origin.track_id));
        AddNumber(&item, "score", score);
        AddNumber(&item, "parent_index", static_cast<double>(origin.parent_index));
        AddText(&item, "name", name);
        records->items.push_back(item);
        // A track announces a name once, when it gets it or it changes; an
        // untracked crop every frame.
        bool announce = true;
        if (origin.track_id >= 0) {
            seen.insert(origin.track_id);
            TrackMemo& memo = s->tracks[origin.track_id];
            announce = memo.name != name;
            memo.name = name;
            memo.last_s = at.time_s;
        }
        if (announce && ((known && mode != "none") || (!known && mode == "all"))) {
            GraphEvent event = MakeEvent(c, known ? "match" : "unknown", origin.track_id, at);
            AddNumber(&event.data, "score", score);
            if (known) AddText(&event.data, "name", name);
            out.events.push_back(event);
        }
        OverlayText label;
        if (origin.from_roi) {
            label.at = cv::Point2f(origin.src_box.x, origin.src_box.y + origin.src_box.height);
        }
        label.text = name + " " + Short(score);
        label.alert = !known;
        out.overlay.texts.push_back(label);
    }
    Forget(s, seen, at.time_s, kForgetAfterS, [](int, const TrackMemo&) {});
    return out;
}

/// The input passed through, unless a class is registered under the node's
/// "name" (user_logic.hpp): then that object's output, events and labels.
/// Anything the object does wrong is a frame error and the input passes on.
LogicOutput RunUser(const LogicConfig& config, const std::vector<LogicInput>& inputs,
                    const LogicFrame& frame, LogicState* s) {
    LogicOutput out;
    const LogicInput* input = FirstFullFrame(inputs);
    if (input == NULL) {
        out.result.data.reset(new RecordsData());
        out.error = "missing input";
        return out;
    }
    const std::shared_ptr<BoxesData> boxes = SourceBoxes(*input);
    const StageDataPtr given = boxes ? StageDataPtr(boxes) : input->data;
    out.result.data = given;
    // ValidateGraph requires "name"; a spec that skipped it passes through.
    const std::map<std::string, std::string>::const_iterator named = config.text.find("name");
    if (named == config.text.end()) return out;
    const std::string& name = named->second;
    if (!s->user_looked_up) {
        // Asked once per stream. A class that cannot be made says so on this
        // frame; the node then passes its input through, as with no class.
        s->user_looked_up = true;
        try {
            s->user = MakeUserLogic(name);  // unique_ptr -> shared_ptr; null: pass-through
        } catch (const std::exception& e) {
            out.error = "user logic \"" + name + "\" could not be made: " + e.what();
            return out;
        } catch (...) {
            out.error = "user logic \"" + name +
                        "\" could not be made: an exception that is not a std::exception";
            return out;
        }
    }
    if (!s->user || !given) return out;
    UserContext ctx(config.id, frame.time_s, frame.frame_size);
    StageDataPtr produced;
    try {
        produced = s->user->Run(given, ctx);
    } catch (const std::exception& e) {
        out.error = "user logic \"" + name + "\" failed: " + e.what();
        return out;
    } catch (...) {
        out.error = "user logic \"" + name + "\" failed: an exception that is not a std::exception";
        return out;
    }
    if (!produced || produced->shape() != given->shape()) {
        out.error = "user logic \"" + name + "\" returned " +
                    (produced ? std::string(ToString(produced->shape())) : std::string("nothing")) +
                    "; it must return " + ToString(given->shape());
        return out;
    }
    out.result.data = produced;
    for (std::size_t i = 0; i < ctx.events().size(); ++i) {
        if (!IsEventName(ctx.events()[i].event)) {
            if (out.error.empty()) {
                out.error = "user logic \"" + name + "\" raised \"" + ctx.events()[i].event +
                            "\", which is not an event name (1 to 64 letters, digits, _ . : -)";
            }
            continue;
        }
        out.events.push_back(ctx.events()[i]);
    }
    out.overlay.texts = ctx.labels();
    return out;
}

/// The op's own run; RunLogic sorts the inputs and checks the events.
LogicOutput RunOp(const LogicConfig& config, const std::vector<LogicInput>& inputs,
                  const LogicFrame& frame, LogicState* s) {
    if (config.op == "filter") return RunFilter(config, inputs);
    if (config.op == "zone") return RunZone(config, inputs, frame, s);
    if (config.op == "line") return RunLine(config, inputs, frame, s);
    if (config.op == "count") return RunCount(config, inputs);
    if (config.op == "dwell") return RunDwell(config, inputs, frame, s);
    if (config.op == "condition") return RunCondition(config, inputs, frame, s);
    if (config.op == "gallery") return RunGallery(config, inputs, frame, s);
    if (config.op == "user") return RunUser(config, inputs, frame, s);
    LogicOutput out;
    out.result.data.reset(new RecordsData());
    out.error = "unknown logic op \"" + config.op + "\"";
    return out;
}

}  // namespace

LogicConfig MakeLogicConfig(const NodeSpec& node) {
    const LogicOpInfo* info = FindLogicOp(node.op);
    if (info == NULL) {
        throw std::invalid_argument("MakeLogicConfig: \"" + node.op + "\" is not a logic op");
    }
    LogicConfig config;
    config.id = node.id;
    config.op = node.op;
    for (std::size_t p = 0; p < info->params.size(); ++p) {
        const LogicParamInfo& param = info->params[p];
        switch (param.kind) {
            case LogicParamInfo::kNumber: {
                std::map<std::string, double>::const_iterator it =
                    node.params.numeric.find(param.name);
                config.numbers[param.name] =
                    it != node.params.numeric.end() ? it->second : param.number_default;
                break;
            }
            case LogicParamInfo::kText:
            case LogicParamInfo::kChoice: {
                std::map<std::string, std::string>::const_iterator it =
                    node.params.text.find(param.name);
                config.text[param.name] =
                    it != node.params.text.end() ? it->second : param.text_default;
                break;
            }
            case LogicParamInfo::kTextList: {
                std::map<std::string, std::vector<std::string> >::const_iterator it =
                    node.params.lists.find(param.name);
                if (it != node.params.lists.end()) config.lists[param.name] = it->second;
                break;
            }
            case LogicParamInfo::kPolygon:
            case LogicParamInfo::kLine: {
                std::map<std::string, std::vector<double> >::const_iterator it =
                    node.arrays.find(param.name);
                if (it != node.arrays.end()) config.arrays[param.name] = it->second;
                break;
            }
        }
    }
    if (node.op == "gallery") {
        std::shared_ptr<RetrievalGallery> gallery(
            new RetrievalGallery(loadRetrievalGallery(config.text["gallery"])));
        if (!gallery->loaded) {
            throw GraphError(GraphErrorCode::kGraphSchema, "node \"" + node.id + "\" \"params\"",
                             gallery->error,
                             "build one from an image folder: python3 "
                             "scripts/build_gallery_database.py --variant <model> --images "
                             "<folder> --out <file>.bin");
        }
        config.gallery = gallery;
    }
    return config;
}

std::string DropRepeatedKeys(RecordItem* data) {
    std::set<std::string> seen;
    std::string first;
    std::vector<std::pair<std::string, double> > numbers;
    for (std::size_t i = 0; i < data->numbers.size(); ++i) {
        if (seen.insert(data->numbers[i].first).second) {
            numbers.push_back(data->numbers[i]);
        } else if (first.empty()) {
            first = data->numbers[i].first;
        }
    }
    std::vector<std::pair<std::string, std::string> > text;
    for (std::size_t i = 0; i < data->text.size(); ++i) {
        if (seen.insert(data->text[i].first).second) {
            text.push_back(data->text[i]);
        } else if (first.empty()) {
            first = data->text[i].first;
        }
    }
    data->numbers.swap(numbers);
    data->text.swap(text);
    return first;
}

LogicOutput RunLogic(const LogicConfig& config, std::vector<LogicInput> inputs,
                     const LogicFrame& frame, LogicState* state) {
    std::stable_sort(inputs.begin(), inputs.end(),
                     [](const LogicInput& a, const LogicInput& b) { return a.edge < b.edge; });
    LogicState scratch;
    LogicOutput out = RunOp(config, inputs, frame, state != NULL ? state : &scratch);
    for (std::size_t e = 0; e < out.events.size(); ++e) {
        const std::string repeated = DropRepeatedKeys(&out.events[e].data);
        if (!repeated.empty() && out.error.empty()) {
            out.error = "event \"" + out.events[e].event + "\" repeats the data key \"" +
                        repeated + "\"";
        }
    }
    return out;
}

}  // namespace graph
}  // namespace dxapp
