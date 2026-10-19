#include "multi_model_graph/graph_consumer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/graph/shape.hpp"
#include "common/inputs/input_factory.hpp"
#include "common/utility/repo_path.hpp"

namespace dxapp {
namespace graph {
namespace consumer {

using nlohmann::json;

std::string ProjectRoot() { return ::dxapp::RuntimeRoot(); }

bool FileExists(const std::string& path) {
    std::ifstream probe(path.c_str(), std::ios::binary);
    return probe.good();
}

std::string DefaultModelDir() { return ProjectRoot() + "/assets/models"; }

namespace {

// ---------------------------------------------------------------------
// The FrameReport as JSON
// ---------------------------------------------------------------------

/// Seconds are written to 0.001 s, the way a report writes "time".
double RoundedSeconds(double seconds) { return std::round(seconds * 1000.0) / 1000.0; }

/// A records or event number as written: a time ("seconds", or a name
/// ending in "_s") rounded like "time", so 0.30000000000000004 is 0.3;
/// any other number unchanged.
double WrittenNumber(const std::string& name, double value) {
    const bool seconds = name == "seconds" ||
                         (name.size() > 2 && name.compare(name.size() - 2, 2, "_s") == 0);
    return seconds && std::isfinite(value) ? RoundedSeconds(value) : value;
}

/// FNV-1a over a matrix's pixels, so a dense payload (a label map, a depth
/// map, a restored image) is comparable between the two executors without
/// writing megabytes of JSON. Walks row by row: cv::Mat rows need not be
/// contiguous.
std::string MatDigest(const cv::Mat& image) {
    unsigned long long hash = 1469598103934665603ULL;
    const std::size_t row_bytes =
        static_cast<std::size_t>(image.cols) * image.elemSize();
    for (int y = 0; y < image.rows; ++y) {
        const unsigned char* row = image.ptr<unsigned char>(y);
        for (std::size_t x = 0; x < row_bytes; ++x) {
            hash ^= static_cast<unsigned long long>(row[x]);
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return text.str();
}

json MatSummary(const cv::Mat& image) {
    json out = json::object();
    out["rows"] = image.rows;
    out["cols"] = image.cols;
    out["type"] = image.type();
    out["digest"] = MatDigest(image);
    // rows*cols only says the matrix was allocated - an all-zero label map
    // from a stage that decoded nothing passes that test. countNonZero is
    // the cheap property that distinguishes "produced something" from
    // "produced an empty canvas", and it is what
    // tests/cpp_example/test_graph_cli.py asserts on. Deterministic from
    // the payload alone, so it carries no parity risk.
    out["nonzero"] = image.empty()
                         ? 0
                         : cv::countNonZero(image.reshape(1));
    return out;
}

json KeypointsToJson(const std::vector<Keypoint>& points) {
    json out = json::array();
    for (std::size_t i = 0; i < points.size(); ++i) {
        json item = json::object();
        item["x"] = points[i].x;
        item["y"] = points[i].y;
        item["confidence"] = points[i].confidence;
        out.push_back(item);
    }
    return out;
}

json OriginToJson(const RoiRef& origin) {
    json out = json::object();
    out["from_roi"] = origin.from_roi;
    out["parent_node"] = origin.parent_node;
    out["parent_index"] = origin.parent_index;
    out["roi_index"] = origin.roi_index;
    out["track_id"] = origin.track_id;
    json box = json::array();
    box.push_back(origin.src_box.x);
    box.push_back(origin.src_box.y);
    box.push_back(origin.src_box.width);
    box.push_back(origin.src_box.height);
    out["src_box"] = box;
    // Task 12 added crop_size to RoiRef and to FrameReport's operator==,
    // but this serializer never learned of it - so the only parity check
    // that runs on real hardware could not see it, even though shape.hpp
    // tells consumers to PREFER it over src_box.size() for a face5 crop
    // (an aligned crop's pixel size is not its source box's size).
    json crop = json::array();
    crop.push_back(origin.crop_size.width);
    crop.push_back(origin.crop_size.height);
    out["crop_size"] = crop;
    json affine = json::array();
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) affine.push_back(origin.inv_align(r, c));
    }
    out["inv_align"] = affine;
    return out;
}

json PayloadToJson(const StageDataPtr& data) {
    json out = json::object();
    if (!data) {
        out["shape"] = "none";
        return out;
    }
    out["shape"] = ToString(data->shape());
    switch (data->shape()) {
        case Shape::kFrame: {
            const FrameData* frame = static_cast<const FrameData*>(data.get());
            out["image"] = MatSummary(frame->image);
            break;
        }
        case Shape::kBoxes:
        case Shape::kObBoxes:
        case Shape::kInstances: {
            const BoxesData* boxes = static_cast<const BoxesData*>(data.get());
            json items = json::array();
            for (std::size_t i = 0; i < boxes->items.size(); ++i) {
                const BoxItem& item = boxes->items[i];
                json entry = json::object();
                json box = json::array();
                box.push_back(item.box.x);
                box.push_back(item.box.y);
                box.push_back(item.box.width);
                box.push_back(item.box.height);
                entry["box"] = box;
                entry["score"] = item.score;
                entry["class_id"] = item.class_id;
                entry["class_name"] = item.class_name;
                entry["track_id"] = item.track_id;
                entry["angle"] = item.angle;
                if (!item.landmarks.empty()) {
                    entry["landmarks"] = KeypointsToJson(item.landmarks);
                }
                if (!item.mask.empty()) entry["mask"] = MatSummary(item.mask);
                items.push_back(entry);
            }
            out["items"] = items;
            break;
        }
        case Shape::kKeypoints: {
            const KeypointsData* poses =
                static_cast<const KeypointsData*>(data.get());
            json items = json::array();
            for (std::size_t i = 0; i < poses->items.size(); ++i) {
                json entry = json::object();
                entry["box"] = poses->items[i].box;
                entry["confidence"] = poses->items[i].confidence;
                entry["keypoints"] = KeypointsToJson(poses->items[i].keypoints);
                items.push_back(entry);
            }
            out["items"] = items;
            break;
        }
        case Shape::kLabelMap: {
            const LabelMapData* labels =
                static_cast<const LabelMapData*>(data.get());
            out["labels"] = MatSummary(labels->labels);
            break;
        }
        case Shape::kDenseMap: {
            const DenseMapData* values =
                static_cast<const DenseMapData*>(data.get());
            out["values"] = MatSummary(values->values);
            break;
        }
        case Shape::kImage: {
            const ImageData* image = static_cast<const ImageData*>(data.get());
            out["image"] = MatSummary(image->image);
            break;
        }
        case Shape::kScores: {
            const ScoresData* scores =
                static_cast<const ScoresData*>(data.get());
            json items = json::array();
            for (std::size_t i = 0; i < scores->items.size(); ++i) {
                const ClassificationResult& item = scores->items[i];
                json entry = json::object();
                entry["class_id"] = item.class_id;
                entry["class_name"] = item.class_name;
                entry["confidence"] = item.confidence;
                json top = json::array();
                for (std::size_t k = 0; k < item.top_k.size(); ++k) {
                    json pair = json::array();
                    pair.push_back(item.top_k[k].first);
                    pair.push_back(item.top_k[k].second);
                    top.push_back(pair);
                }
                entry["top_k"] = top;
                items.push_back(entry);
            }
            out["items"] = items;
            break;
        }
        case Shape::kVector: {
            const VectorData* vector =
                static_cast<const VectorData*>(data.get());
            out["values"] = vector->values;
            break;
        }
        case Shape::kBoxes3d: {
            // Unreachable: ValidateGraph rejects a boxes3d model in a
            // camera graph (kGraphEdge, "consumes LiDAR point clouds").
            // Recorded by count rather than silently omitted, so that if
            // the rejection is ever relaxed this line is visibly the next
            // thing to write.
            const Boxes3dData* boxes =
                static_cast<const Boxes3dData*>(data.get());
            out["count"] = static_cast<unsigned>(boxes->items.size());
            break;
        }
        case Shape::kRecords: {
            const RecordsData* records = static_cast<const RecordsData*>(data.get());
            json items = json::array();
            for (std::size_t i = 0; i < records->items.size(); ++i) {
                const RecordItem& item = records->items[i];
                json entry = json::object();
                json numbers = json::object();
                for (std::size_t n = 0; n < item.numbers.size(); ++n) {
                    numbers[item.numbers[n].first] =
                        WrittenNumber(item.numbers[n].first, item.numbers[n].second);
                }
                json text = json::object();
                for (std::size_t t = 0; t < item.text.size(); ++t) {
                    text[item.text[t].first] = item.text[t].second;
                }
                entry["numbers"] = numbers;
                entry["text"] = text;
                items.push_back(entry);
            }
            out["items"] = items;
            break;
        }
    }
    return out;
}

json ResultToJson(const StageResult& result) {
    json out = json::object();
    out["origin"] = OriginToJson(result.origin);
    out["payload"] = PayloadToJson(result.data);
    if (!result.ports.empty()) {
        json ports = json::object();
        for (StagePorts::const_iterator it = result.ports.begin(); it != result.ports.end(); ++it) {
            ports[it->first] = PayloadToJson(it->second);
        }
        out["ports"] = ports;
    }
    return out;
}

/// U-11. nlohmann writes NaN and +-Inf all as null, so a report could not
/// tell them apart. RFC 8259 has no such numbers and strict parsers
/// (JSON.parse, jq, nlohmann::json::parse) reject bare NaN/Infinity tokens,
/// so they become the strings "NaN", "Infinity" and "-Infinity": valid JSON
/// everywhere, distinct from each other and from null, and what Python's
/// float() and JavaScript's Number() turn back into numbers. Finite numbers
/// are untouched, so every finite report is byte-identical to before.
void EncodeNonFinite(json* value) {
    if (value->is_number_float()) {
        const double number = value->get<double>();
        if (std::isnan(number)) {
            *value = "NaN";
        } else if (std::isinf(number)) {
            *value = number > 0 ? "Infinity" : "-Infinity";
        }
        return;
    }
    if (value->is_object() || value->is_array()) {
        for (json::iterator it = value->begin(); it != value->end(); ++it) {
            EncodeNonFinite(&*it);
        }
    }
}

// ---------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------

/// The source node ids, in declaration order: StageGraph::streams() order.
std::vector<std::string> SourceIds(const GraphSpec& spec) {
    std::vector<std::string> ids;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (spec.nodes[i].is_source) ids.push_back(spec.nodes[i].id);
    }
    return ids;
}

/// "camA, camB".
std::string JoinNames(const std::vector<std::string>& names) {
    std::string text;
    for (std::size_t i = 0; i < names.size(); ++i) {
        text += (i == 0 ? "" : ", ") + names[i];
    }
    return text;
}

/// How a message names the graph: graph "<name>", or "the graph" unnamed.
std::string GraphName(const GraphSpec& spec) {
    return spec.name.empty() ? std::string("the graph") : "graph \"" + spec.name + "\"";
}

}  // namespace

std::vector<MissingArtifact> CollectMissing(const GraphSpec& spec,
                                            const IModelRegistry& registry,
                                            const std::string& model_dir) {
    std::vector<MissingArtifact> missing;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (!IsModelNode(node)) continue;
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;  // ValidateGraph already rejected this
        const std::string path = model_dir + "/" + info->dxnn_file;
        if (FileExists(path)) continue;
        MissingArtifact entry;
        entry.node = node.id;
        entry.model = node.model;
        entry.path = path;
        entry.download_name =
            info->download_name.empty() ? node.model : info->download_name;
        entry.variant = info->model_name;
        entry.published = info->published;
        missing.push_back(entry);
    }
    return missing;
}

namespace {

/// `word` as one shell word: unchanged when it is plain, else single-quoted.
std::string ShellWord(const std::string& word) {
    static const char kPlain[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-./:+,=@%";
    if (!word.empty() && word.find_first_not_of(kPlain) == std::string::npos) return word;
    std::string quoted = "'";
    for (std::size_t i = 0; i < word.size(); ++i) {
        if (word[i] == '\'') quoted += "'\\''";
        else quoted += word[i];
    }
    return quoted + "'";
}

}  // namespace

std::string DownloadCommand(const std::vector<std::string>& names,
                            const std::string& model_dir) {
    std::string command = "python3 scripts/download_models.py --models";
    for (std::size_t i = 0; i < names.size(); ++i) command += " " + ShellWord(names[i]);
    return command + " --output " + ShellWord(model_dir);
}

// The text is built from the graph file's own strings, and JSON may spell a
// NUL ("\u0000") inside a node id. The std::string keeps it, but every reader
// goes through a C string - MissingModelsError::what(), the CLI's fputs of
// it (a run) or of this text (--check), dx_graph's GraphError message - so
// each one ends the text at the first NUL. They end it at the same byte, so
// the CLI and dx_graph still agree; what follows the NUL is lost in both.
std::string FormatMissingList(const std::vector<MissingArtifact>& missing,
                              const std::string& model_dir) {
    std::string text;
    std::string unpublished;
    std::vector<std::string> names;
    for (std::size_t i = 0; i < missing.size(); ++i) {
        text += "  node \"" + missing[i].node + "\"  model " +
                missing[i].model + "  ->  " + missing[i].path + "\n";
        if (!missing[i].published) {
            // The manifest has no file for it ("pending"): the downloader
            // would fetch nothing, so it is not in the command.
            unpublished += "  node \"" + missing[i].node + "\": " + missing[i].variant +
                           " is not published by the model zoo yet; download_models.py "
                           "cannot download it\n";
            continue;
        }
        names.push_back(missing[i].download_name);
    }
    text += unpublished;
    // One command for every downloadable name, the form StageGraph::Build's
    // fallback line and the docs print too.
    if (!names.empty()) text += "  -> " + DownloadCommand(names, model_dir) + "\n";
    return text;
}

std::string MissingHeadline(std::size_t count, const std::string& model_dir) {
    std::ostringstream text;
    text << count << " model file" << (count == 1 ? " is" : "s are")
         << " not in " << model_dir;
    return text.str();
}

std::string PortsText(const ModelInfo& info) {
    std::string text;
    for (std::size_t i = 0; i < info.ports.size(); ++i) {
        if (i > 0) text += " ";
        text += info.ports[i].name + "=" + ToString(info.ports[i].shape);
    }
    return text;
}

std::string AliasNote(const std::string& name, const ModelInfo& info,
                      const std::map<std::string, ModelAlias>& aliases) {
    if (name == info.model_name) return std::string();
    std::map<std::string, ModelAlias>::const_iterator alias = aliases.find(name);
    const bool alias_of = alias != aliases.end() && alias->second.kind == ModelAlias::kAliasOf;
    return "\"" + name + "\" is " + (alias_of ? "an alias of" : "the old name of") + " \"" +
           info.model_name + "\"";
}

std::string AliasNotes(const GraphSpec& spec, const IModelRegistry& registry) {
    std::map<std::string, ModelAlias> alias_by_name;
    const std::vector<ModelAlias> aliases = registry.aliases();
    for (std::size_t a = 0; a < aliases.size(); ++a) alias_by_name[aliases[a].name] = aliases[a];
    std::string text;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (!IsModelNode(node)) continue;
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        const std::string note = AliasNote(node.model, *info, alias_by_name);
        if (!note.empty()) text += "note: node \"" + node.id + "\": " + note + "\n";
    }
    return text;
}

void PrepareGraph(const std::string& graph_path, const std::string& model_dir,
                  const IModelRegistry& registry, GraphSpec* spec,
                  StageGraph* graph) {
    *spec = ParseGraphFile(graph_path);
    ValidateGraph(*spec, registry);                             // T0
    // An old name or an alias runs its variant (R6). --check says so on
    // stdout; a run (the CLI's and dx_graph's) says it here, on stderr, so
    // reports and stdout stay as the variant name writes them.
    std::fputs(AliasNotes(*spec, registry).c_str(), stderr);

    // T1, fatal here. StageGraph::Build would raise kModelMissing one
    // model at a time; listing every absent file at once matters when
    // one graph names three to five models.
    const std::vector<MissingArtifact> missing =
        CollectMissing(*spec, registry, model_dir);
    if (!missing.empty()) {
        throw MissingModelsError(
            "ERROR [MODEL_MISSING] graph \"" + spec->name + "\": " +
            MissingHeadline(missing.size(), model_dir) + "\n" +
            FormatMissingList(missing, model_dir));
    }

    graph->Build(*spec, registry, model_dir);
}

namespace {

template <typename Json>
Json EventData(const RecordItem& data) {
    Json out = Json::object();
    for (std::size_t n = 0; n < data.numbers.size(); ++n) {
        out[data.numbers[n].first] =
            JsonNumber<Json>(WrittenNumber(data.numbers[n].first, data.numbers[n].second));
    }
    for (std::size_t t = 0; t < data.text.size(); ++t) {
        out[data.text[t].first] = data.text[t].second;
    }
    return out;
}

json EventToReportJson(const GraphEvent& event) {
    json out = json::object();
    out["node"] = event.node;
    out["event"] = event.event;
    out["track_id"] = event.track_id;
    out["time"] = JsonNumber<json>(RoundedSeconds(event.time_s));
    out["data"] = EventData<json>(event.data);
    return out;
}

}  // namespace

nlohmann::ordered_json EventLineJson(const GraphEvent& event, const FrameReport& report) {
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    out["frame"] = static_cast<unsigned long long>(report.frame_index);
    out["time"] = JsonNumber<nlohmann::ordered_json>(RoundedSeconds(event.time_s));
    out["stream"] = report.stream;
    out["node"] = event.node;
    out["event"] = event.event;
    out["track_id"] = event.track_id;
    out["data"] = EventData<nlohmann::ordered_json>(event.data);
    return out;
}

bool NamesStreams(const StageGraph& graph) { return graph.streams().size() > 1; }

json ReportToJson(const FrameReport& report, bool name_stream, bool with_timing) {
    json out = json::object();
    out["index"] = static_cast<unsigned>(report.frame_index);
    out["error"] = report.error;
    out["skipped_out_of_bounds"] = report.skipped_out_of_bounds;

    json nodes = json::object();
    for (std::map<std::string, StageResult>::const_iterator it =
             report.node_results.begin();
         it != report.node_results.end(); ++it) {
        nodes[it->first] = ResultToJson(it->second);
    }
    out["nodes"] = nodes;

    json roi_nodes = json::object();
    for (std::map<std::string, std::vector<StageResult> >::const_iterator it =
             report.roi_results.begin();
         it != report.roi_results.end(); ++it) {
        json crops = json::array();
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            crops.push_back(ResultToJson(it->second[i]));
        }
        roi_nodes[it->first] = crops;
    }
    out["roi_nodes"] = roi_nodes;
    if (name_stream) out["stream"] = report.stream;
    if (!report.events.empty()) {
        json events = json::array();
        for (std::size_t e = 0; e < report.events.size(); ++e) {
            events.push_back(EventToReportJson(report.events[e]));
        }
        out["events"] = events;
    }
    if (with_timing) {
        json timing = json::object();
        for (std::map<std::string, double>::const_iterator it = report.node_ms.begin();
             it != report.node_ms.end(); ++it) {
            timing[it->first] = std::round(it->second * 1000.0) / 1000.0;
        }
        out["node_ms"] = timing;
    }
    EncodeNonFinite(&out);
    return out;
}

std::string ResolvePath(const std::string& path) {
    if (path.empty() || path[0] == '/') return path;
    if (FileExists(path)) return path;
    const std::string rooted = ProjectRoot() + "/" + path;
    if (FileExists(rooted)) return rooted;
    return path;  // let the input factory produce the error message
}

InputSourcePtr OpenInput(const std::string& uri) {
    if (uri.compare(0, 7, "rtsp://") == 0) {
        return InputFactory::createFromRTSP(uri);
    }
    if (uri.compare(0, 7, "camera:") == 0) {
        return InputFactory::createFromCamera(std::atoi(uri.c_str() + 7));
    }
    return InputFactory::createFromFile(ResolvePath(uri));
}

std::string SourceUri(const GraphSpec& spec, const std::string& override_uri) {
    if (!override_uri.empty()) return override_uri;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (spec.nodes[i].is_source) return spec.nodes[i].uri;
    }
    return std::string();
}

InputSourcePtr OpenGraphSource(const GraphSpec& spec,
                               const std::string& override_uri,
                               std::string* uri) {
    *uri = SourceUri(spec, override_uri);
    StreamUri stream;
    const std::vector<std::string> sources = SourceIds(spec);
    if (!sources.empty()) stream.source = sources[0];
    stream.uri = *uri;
    return OpenStream(spec, stream);
}

namespace {

/// "<prefix>=..." whose prefix is not a source id: a misspelled id when the
/// prefix could not begin a uri (no '/', '\\' or ':'), so "camX=a.mp4" is
/// named as a wrong id, while "rtsp://h/s?a=1" and "dir/a=b.mp4" stay uris.
bool LooksLikeSourceId(const std::string& prefix) {
    return !prefix.empty() && prefix.find_first_of("/\\:") == std::string::npos;
}

/// One --input value into *overrides; "" or the usage error.
std::string BindInput(const GraphSpec& spec, const std::vector<std::string>& sources,
                      const std::string& value, std::map<std::string, std::string>* overrides) {
    std::string source;
    std::string uri = value;
    // "<id>=" only when <id> is a source: "rtsp://h/s?a=1" stays a uri. An id
    // may itself hold '=', so the longest id that begins the value wins:
    // with sources "back" and "back=yard", "back=yard=a.mp4" is back=yard's.
    bool named = false;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const std::string head = sources[i] + "=";
        if ((!named || sources[i].size() > source.size()) &&
            value.compare(0, head.size(), head) == 0) {
            source = sources[i];
            named = true;
        }
    }
    const std::size_t equals = value.find('=');
    const std::string prefix = equals == std::string::npos ? std::string()
                                                           : value.substr(0, equals);
    if (named) {
        uri = value.substr(source.size() + 1);
        if (uri.empty()) return "--input " + value + " has no uri after \"=\"";
    } else if (sources.size() == 1) {
        source = sources[0];
    } else {
        std::ostringstream text;
        text << "--input \"" << value << "\" does not say which source it replaces: ";
        if (equals != std::string::npos && LooksLikeSourceId(prefix)) {
            text << "\"" << prefix << "\" is not a source node of " << GraphName(spec)
                 << "; its source nodes: " << JoinNames(sources);
        } else {
            text << GraphName(spec) << " has " << sources.size() << " source nodes ("
                 << JoinNames(sources) << "); write --input <source_id>=<uri>, e.g. --input "
                 << sources[0] << "=" << value;
        }
        return text.str();
    }
    if (!overrides->insert(std::make_pair(source, uri)).second) {
        return "--input names source \"" + source + "\" twice";
    }
    return std::string();
}

}  // namespace

std::string ParseInputOverrides(const GraphSpec& spec, const std::vector<std::string>& inputs,
                                std::map<std::string, std::string>* overrides) {
    overrides->clear();
    const std::vector<std::string> sources = SourceIds(spec);
    if (sources.empty()) return std::string();  // ValidateGraph reports the graph
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::string problem = BindInput(spec, sources, inputs[i], overrides);
        if (!problem.empty()) {
            overrides->clear();  // an error binds nothing
            return problem;
        }
    }
    return std::string();
}

std::string ResolveStreams(const GraphSpec& spec,
                           const std::map<std::string, std::string>& overrides,
                           std::vector<StreamUri>* streams) {
    const std::vector<std::string> sources = SourceIds(spec);
    for (std::map<std::string, std::string>::const_iterator it = overrides.begin();
         it != overrides.end(); ++it) {
        if (std::find(sources.begin(), sources.end(), it->first) == sources.end()) {
            return "\"" + it->first + "\" is not a source node of " + GraphName(spec) +
                   "; its source nodes: " + JoinNames(sources);
        }
    }
    streams->clear();
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (!spec.nodes[i].is_source) continue;
        StreamUri stream;
        stream.source = spec.nodes[i].id;
        std::map<std::string, std::string>::const_iterator it = overrides.find(stream.source);
        stream.uri = it != overrides.end() ? it->second : spec.nodes[i].uri;
        streams->push_back(stream);
    }
    return std::string();
}

InputSourcePtr OpenStream(const GraphSpec& spec, const StreamUri& stream) {
    const bool several = SourceIds(spec).size() > 1;
    if (stream.uri.empty()) {
        if (!several) {
            throw GraphError(GraphErrorCode::kGraphSchema, "graph \"" + spec.name + "\"",
                             "the source node has no \"uri\"", "add \"uri\", or pass --input");
        }
        throw GraphError(GraphErrorCode::kGraphSchema, "node \"" + stream.source + "\"",
                         "the source node has no \"uri\"",
                         "add \"uri\" to it, or pass --input " + stream.source + "=<uri>");
    }
    try {
        return OpenInput(stream.uri);
    } catch (const std::exception& error) {
        throw GraphError(GraphErrorCode::kGraphSchema, "source \"" + stream.uri + "\"",
                         error.what(),
                         several ? "check the path, or pass --input " + stream.source + "=<file>"
                                 : std::string("check the path, or pass --input <file>"));
    }
}

StreamReader::StreamReader(std::vector<StreamInput> inputs, std::size_t frames_per_stream)
    : inputs_(std::move(inputs)), read_(inputs_.size(), 0), ended_(inputs_.size(), 0),
      limit_(frames_per_stream), turn_(0), loop_(false), restarts_(inputs_.size(), 0),
      read_since_start_(inputs_.size(), 0) {}

void StreamReader::SetLoop(bool loop) { loop_ = loop; }

std::size_t StreamReader::restarts(std::size_t stream) const { return restarts_.at(stream); }

bool StreamReader::Restart(std::size_t s) {
    if (inputs_[s].input->getType() != InputType::VIDEO) return false;
    try {
        InputSourcePtr reopened = OpenInput(inputs_[s].uri);
        if (!reopened || !reopened->isOpened()) return false;
        inputs_[s].input = std::move(reopened);
    } catch (const std::exception&) {
        return false;
    }
    ++restarts_[s];
    read_since_start_[s] = 0;
    return true;
}

bool StreamReader::Next(std::size_t* stream, cv::Mat* frame, std::size_t* index) {
    // At most one pass over the streams: each gets its turn once, starting
    // with the one after the stream that gave the last frame, so frames
    // come out one per stream still reading, in declaration order.
    for (std::size_t tries = 0; tries < inputs_.size(); ++tries) {
        const std::size_t s = turn_;
        turn_ = (turn_ + 1) % inputs_.size();
        if (ended_[s]) continue;
        if (!inputs_[s].input->getFrame(*frame)) {
            // --loop: a video that gave frames starts again, once per end.
            const bool again = loop_ && read_since_start_[s] != 0 && Restart(s) &&
                               inputs_[s].input->getFrame(*frame);
            if (!again) {
                ended_[s] = 1;  // its end: never read again
                continue;
            }
        }
        *stream = s;
        *index = read_[s]++;
        ++read_since_start_[s];
        if (limit_ != 0 && read_[s] >= limit_) ended_[s] = 1;  // no read past --frames
        return true;
    }
    return false;
}

std::size_t StreamReader::size() const { return inputs_.size(); }

const StreamInput& StreamReader::input(std::size_t stream) const { return inputs_.at(stream); }

std::size_t StreamReader::frames_read(std::size_t stream) const { return read_.at(stream); }

bool StreamReader::ended(std::size_t stream) const { return ended_.at(stream) != 0; }

GraphError NoFramesError(const std::string& uri) {
    return GraphError(GraphErrorCode::kGraphSchema, "source \"" + uri + "\"",
                      "produced no frames",
                      "check that the file is a readable image or video");
}

std::string StuckMessage(const std::vector<std::string>& ids) {
    std::string names;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        names += (i == 0 ? "" : ", ") + ids[i];
    }
    return "interrupted: still waiting on stage(s): " + names;
}

}  // namespace consumer
}  // namespace graph
}  // namespace dxapp
