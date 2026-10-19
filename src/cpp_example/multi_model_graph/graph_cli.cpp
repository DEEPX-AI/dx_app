#include "multi_model_graph/graph_cli.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#endif

#include <dxrt/dxrt_api.h>
#include <opencv2/imgcodecs.hpp>

#include "common/graph/graph_config.hpp"
#include "common/graph/graph_error.hpp"
#include "common/graph/graph_runner_async.hpp"
#include "common/graph/graph_runner_sync.hpp"
#include "common/graph/graph_visualizer.hpp"
#include "common/graph/i_registry.hpp"
#include "common/graph/logic_ops.hpp"
#include "common/graph/shape.hpp"
#include "common/graph/stage_graph.hpp"
#include "common/inputs/input_factory.hpp"
#include "common/registry/static_model_registry.hpp"
#include "common/third_party/nlohmann_json.hpp"
#include "common/utility/dxnn_container.hpp"
#include "common/utility/progress.hpp"
#include "common/utility/repo_path.hpp"
#include "multi_model_graph/graph_cli_interrupt.hpp"
#include "multi_model_graph/graph_cli_output.hpp"
#include "multi_model_graph/graph_consumer.hpp"
#include "multi_model_graph/graph_report_writer.hpp"

namespace dxapp {
namespace graph {
namespace cli {
namespace {

/// Frames in flight (spec 4.5). Set by the 1/2/4/8/16 sweep on DX-M1: the
/// smallest count within 3 % of the best throughput (16 reached 199.9 FPS,
/// 8 reached 191.9, 4 reached 179.8).
const std::size_t kDefaultMaxInflight = 16;

/// The largest value any numeric option takes: INT_MAX, the same on every
/// platform (a long is 32 bits on Windows). No count or timeout that a run
/// can use is anywhere near it - 2147483647 ms is 24 days - so a larger
/// value is a typo, and it would overflow on its way to a duration.
const long kMaxOptionValue = 2147483647L;

// The report schema, the path rule, the build sequence and the executor
// adapters are shared with the dx_graph Python module; see
// graph_consumer.hpp.
using consumer::AliasNotes;
using consumer::AsyncFrameExecutor;
using consumer::CollectMissing;
using consumer::DefaultModelDir;
using consumer::DownloadCommand;
using consumer::EventLineJson;
using consumer::FileExists;
using consumer::FormatMissingList;
using consumer::FrameExecutor;
using consumer::MissingArtifact;
using consumer::MissingHeadline;
using consumer::MissingModelsError;
using consumer::NamesStreams;
using consumer::NoFramesError;
using consumer::OpenStream;
using consumer::ParseInputOverrides;
using consumer::PortsText;
using consumer::PrepareGraph;
using consumer::ReportToJson;
using consumer::ResolveStreams;
using consumer::StreamInput;
using consumer::StreamReader;
using consumer::StreamUri;
using consumer::StuckMessage;
using consumer::SyncFrameExecutor;

// ---------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------

struct GraphCliArgs {
    std::string graph_path;
    std::vector<std::string> inputs;  ///< --input values: [<source>=]<uri>, in order
    std::string output;     ///< rendered result: one video, or images
    std::string report;     ///< FrameReport as JSON
    std::string events;     ///< --events: one JSON line per event
    std::vector<std::string> draw;  ///< --draw: the only nodes drawn (empty: all)
    std::string output_layers;  ///< --output-layers: per-node RGBA layers + layers.json
    std::string model_dir;
    std::string consumes;   ///< --list-models filter
    std::string produces;   ///< --list-models filter
    std::size_t max_inflight;
    std::size_t max_jobs_per_stage;  ///< 0 = no limit
    std::size_t stall_timeout_ms;    ///< 0 = wait forever
    std::size_t frames;     ///< 0 = every frame the source yields
    bool check_only;
    bool list_models;
    bool display;           ///< show each rendered frame in a window
    bool help;
    bool report_timing;   ///< --report-timing: node_ms in --report
    bool json;            ///< --check / --list-models: JSON on stdout
    bool loop;            ///< --loop: a video starts again at its end
    bool list_ops;        ///< --list-ops: the logic and cpu op catalog
    bool draw_given;      ///< --draw was passed (with no id it is a usage error)

    GraphCliArgs()
        : model_dir(DefaultModelDir()),
          max_inflight(kDefaultMaxInflight),
          max_jobs_per_stage(0),
          stall_timeout_ms(0),
          frames(0),
          check_only(false),
          list_models(false),
          display(false),
          help(false),
          report_timing(false),
          json(false),
          loop(false),
          list_ops(false),
          draw_given(false) {}
};

/// The usage text, on `out`: stdout for --help and for a usage error,
/// stderr for a usage error when --json was asked for, whose stdout holds
/// one JSON document or nothing.
void PrintUsage(std::FILE* out, const char* program) {
    std::fprintf(
        out,
        "Run a multi-model pipeline described entirely by a node-graph JSON\n"
        "file. Changing which models run is a text edit, never a rebuild.\n"
        "\n"
        "Usage: %s --graph <file.json> [options]\n"
        "       %s --check <file.json> [--model-dir <dir>] [--json]\n"
        "       %s --list-models [--consumes <roi|frame|either>]"
        " [--produces <shape>]\n"
        "                     [--model-dir <dir>] [--json]\n"
        "       %s --list-ops [--json]\n"
        "\n"
        "Options:\n"
        "  --graph <file>        the node-graph JSON to run\n"
        "  --input [<source>=]<uri>\n"
        "                        override a source node's \"uri\" (image/video"
        " path,\n"
        "                        camera:<N>, rtsp://...). Repeat it once per"
        " source;\n"
        "                        a graph with several sources needs <source>=\n"
        "  --output <file>       write the rendered result. .mp4, .avi or"
        " .mkv: one\n"
        "                        video. Any other extension: an image; a"
        " multi-frame\n"
        "                        source writes <stem>_<index><ext>, and a"
        " live source\n"
        "                        (camera:, rtsp://) then needs --frames N."
        " With\n"
        "                        several sources, one file per stream: the"
        " video\n"
        "                        <stem>_<source_id><ext>, or images, always"
        " numbered:\n"
        "                        <stem>_<source_id>_<index><ext>\n"
        "  --display             show each rendered frame in a window; q or"
        " ESC stops\n"
        "                        the run as Ctrl-C does. With several sources,"
        " one\n"
        "                        window: a grid of the streams, each cell"
        " labelled\n"
        "                        with its source id\n"
        "  --draw <id[,id...]>   draw only these nodes (a logic node's id draws its\n"
        "                        overlay) in the window, --output and"
        " --output-layers;\n"
        "                        every node still runs and reports. The events"
        " panel\n"
        "                        lists only these nodes' events\n"
        "  --output-layers <dir> per frame, write the clean frame\n"
        "                        (<stem>[_<source>]_<frame>_base.png), each drawn\n"
        "                        node's drawing alone on a transparent RGBA PNG\n"
        "                        (..._<frame>_<node>.png; ..._events.png for the"
        " events\n"
        "                        panel) and <dir>/layers.json, listing them in"
        " stacking\n"
        "                        order. <stem>: --output's, else the graph file's."
        " A\n"
        "                        video or live source needs --frames N\n"
        "  --report <file>       write every FrameReport as JSON, ordered by\n"
        "                        (node, parent_index, roi_index)\n"
        "  --report-timing       with --report: add each frame's \"node_ms\", the\n"
        "                        time per node. Off by default, so the sync and\n"
        "                        async reports stay byte-identical\n"
        "  --events <file>       write every event of the graph's logic nodes, one\n"
        "                        JSON object per line (JSONL); they are printed as\n"
        "                        [EVENT] {...} lines too\n"
        "  --model-dir <dir>     where the .dxnn files live\n"
        "                        (default: %s)\n"
        "  --max-inflight <N>    frames in flight at once (default: %u, at"
        " least 1).\n"
        "                        multi_model_graph_async only\n"
        "  --max-jobs-per-stage <N>\n"
        "                        jobs one stage may have outstanding across"
        " frames\n"
        "                        (default: 0 = no limit)."
        " multi_model_graph_async only\n"
        "  --stall-timeout-ms <N>\n"
        "                        fail when work is outstanding and nothing"
        " completes\n"
        "                        for N ms (default: 0 = wait forever).\n"
        "                        multi_model_graph_async only\n"
        "  --frames <N>          stop after N frames (default: all); with"
        " several\n"
        "                        sources, N frames of each stream\n"
        "  --loop                a video source starts again at its end, so the run\n"
        "                        goes on until Ctrl-C, q/ESC or --frames N (Live)\n"
        "  --check [<file>]      validate the graph and list the model files"
        " it\n"
        "                        needs, without opening the NPU, then exit.\n"
        "                        Exit code reflects the graph itself; a"
        " missing\n"
        "                        .dxnn is reported but is not an error here;"
        " a .dxnn\n"
        "                        that cannot load here (not a .dxnn, or v9"
        " before\n"
        "                        DX-RT 3.5.0) is one (exit 1)\n"
        "  --list-models         print every model this build can put in a"
        " graph, with\n"
        "                        its published flag and the container version"
        " of its\n"
        "                        .dxnn in --model-dir (v8, v9, missing; and"
        " when this\n"
        "                        DX-RT cannot load it, why), then"
        " the\n"
        "                        alias_of names of other models\n"
        "  --consumes <what>     with --list-models: roi | frame | either\n"
        "  --produces <shape>    with --list-models: boxes, obboxes,"
        " instances,\n"
        "                        keypoints, labelmap, densemap, image,"
        " scores,\n"
        "                        vector, boxes3d\n"
        "  --json                with --check, --list-models or --list-ops: one JSON\n"
        "                        document on stdout instead of the tables (README)\n"
        "  --list-ops            print every logic and cpu op a graph can use: its\n"
        "                        inputs, output, events and params (--json: the\n"
        "                        catalog a front end builds its palette from)\n"
        "  --help                this message\n"
        "\n"
        "Sample graphs: src/cpp_example/multi_model_graph/\n"
        "Model list:    docs/graph_models.md\n"
        "Schema:        src/cpp_example/multi_model_graph/README.md\n",
        program, program, program, program, DefaultModelDir().c_str(),
        static_cast<unsigned>(kDefaultMaxInflight));
}

/// True on success. On failure *error explains which argument was wrong;
/// the caller prints usage and exits 2.
bool ParseArgs(int argc, char** argv, ExecutorKind kind, GraphCliArgs* args,
               std::string* error) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        const bool has_value = (i + 1 < argc) && argv[i + 1][0] != '\0';
        // A value that itself looks like an option is a missing value, not
        // a value: "--graph --check x" must not silently run on a graph
        // named "--check".
        const bool value_is_option =
            has_value && std::strncmp(argv[i + 1], "--", 2) == 0;

        if (option == "--help" || option == "-h") {
            args->help = true;
            return true;
        }
        if (option == "--list-models") {
            args->list_models = true;
            continue;
        }
        if (option == "--list-ops") {
            args->list_ops = true;
            continue;
        }
        if (option == "--display") {
            args->display = true;
            continue;
        }
        if (option == "--report-timing") {
            args->report_timing = true;
            continue;
        }
        if (option == "--loop") {
            args->loop = true;
            continue;
        }
        if (option == "--json") {
            args->json = true;
            continue;
        }
        if (option == "--check") {
            args->check_only = true;
            // --check takes the graph path as an optional value so both
            // "--check g.json" and "--graph g.json --check" work.
            if (has_value && !value_is_option) args->graph_path = argv[++i];
            continue;
        }

        // Repeatable, once per source node (SP2); which source each value
        // replaces is decided in Main, once the graph is read.
        if (option == "--input") {
            if (!has_value || value_is_option) {
                *error = option + " needs a value";
                return false;
            }
            args->inputs.push_back(argv[++i]);
            continue;
        }

        // Repeatable: every value adds its comma-separated node ids.
        if (option == "--draw") {
            if (!has_value || value_is_option) {
                *error = option + " needs a value";
                return false;
            }
            args->draw_given = true;
            std::stringstream ids(argv[++i]);
            std::string id;
            while (std::getline(ids, id, ',')) {
                const std::size_t first = id.find_first_not_of(" \t");
                if (first == std::string::npos) continue;
                const std::size_t last = id.find_last_not_of(" \t");
                args->draw.push_back(id.substr(first, last - first + 1));
            }
            continue;
        }

        std::string* text = NULL;
        if (option == "--graph") text = &args->graph_path;
        else if (option == "--output") text = &args->output;
        else if (option == "--report") text = &args->report;
        else if (option == "--events") text = &args->events;
        else if (option == "--output-layers") text = &args->output_layers;
        else if (option == "--model-dir") text = &args->model_dir;
        else if (option == "--consumes") text = &args->consumes;
        else if (option == "--produces") text = &args->produces;
        if (text != NULL) {
            if (!has_value || value_is_option) {
                *error = option + " needs a value";
                return false;
            }
            *text = argv[++i];
            continue;
        }

        std::size_t* number = NULL;
        if (option == "--max-inflight") number = &args->max_inflight;
        else if (option == "--max-jobs-per-stage") number = &args->max_jobs_per_stage;
        else if (option == "--stall-timeout-ms") number = &args->stall_timeout_ms;
        else if (option == "--frames") number = &args->frames;
        if (number != NULL) {
            // Refused before its value is read: the sync executor runs one
            // frame at a time, so it would silently ignore these.
            if (kind == kSyncExecutor &&
                (option == "--max-inflight" || option == "--max-jobs-per-stage" ||
                 option == "--stall-timeout-ms")) {
                *error = option + " applies to multi_model_graph_async only: "
                                  "this binary runs one frame at a time";
                return false;
            }
            if (!has_value || value_is_option) {
                *error = option + " needs a value";
                return false;
            }
            const std::string raw = argv[++i];
            char* end = NULL;
            errno = 0;
            const long parsed = std::strtol(raw.c_str(), &end, 10);
            if (end == raw.c_str() || *end != '\0' || parsed < 0) {
                *error = option + " expects a non-negative integer, got \"" +
                         raw + "\"";
                return false;
            }
            // strtol saturates at LONG_MAX with ERANGE; either way the value
            // is past anything the option can mean.
            if (errno == ERANGE || parsed > kMaxOptionValue) {
                *error = option + " expects an integer from 0 to " +
                         std::to_string(kMaxOptionValue) + ", got \"" + raw + "\"";
                return false;
            }
            *number = static_cast<std::size_t>(parsed);
            if (option == "--max-inflight" && parsed == 0) {
                *error = "--max-inflight must be at least 1 (frames in flight)";
                return false;
            }
            continue;
        }

        *error = "unknown option \"" + option + "\"";
        return false;
    }

    if (!args->list_models && !args->list_ops && args->graph_path.empty()) {
        *error = "no graph: pass --graph <file.json>";
        return false;
    }
    if (!args->list_models && !args->consumes.empty()) {
        *error = "--consumes only applies to --list-models";
        return false;
    }
    if (!args->list_models && !args->produces.empty()) {
        *error = "--produces only applies to --list-models";
        return false;
    }
    if (args->report_timing && args->report.empty()) {
        *error = "--report-timing needs --report <file>";
        return false;
    }
    if (args->json && !args->list_models && !args->check_only && !args->list_ops) {
        *error = "--json applies to --check, --list-models and --list-ops";
        return false;
    }
    if (args->list_ops && (args->list_models || args->check_only || !args->graph_path.empty())) {
        *error = "--list-ops is a mode of its own: pass it without --graph, --check or "
                 "--list-models";
        return false;
    }
    if (args->list_models && args->check_only) {
        // Each prints its own report, and --list-models used to win
        // silently, leaving the graph unchecked.
        *error = "--list-models and --check are separate modes: pass one of them";
        return false;
    }
    if (!args->events.empty() && (args->check_only || args->list_models || args->list_ops)) {
        *error = "--events applies to a run (--graph), not to a listing or --check";
        return false;
    }
    if (args->loop && (args->check_only || args->list_models || args->list_ops)) {
        *error = "--loop applies to a run (--graph), not to a listing or --check";
        return false;
    }
    if (args->draw_given && (args->check_only || args->list_models || args->list_ops)) {
        *error = "--draw applies to a run (--graph), not to a listing or --check";
        return false;
    }
    if (!args->output_layers.empty() &&
        (args->check_only || args->list_models || args->list_ops)) {
        *error = "--output-layers applies to a run (--graph), not to a listing or --check";
        return false;
    }
    if (args->draw_given && args->draw.empty()) {
        *error = "--draw needs node ids, separated by commas: --draw od,seg";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// --list-models
// ---------------------------------------------------------------------

/// Every Shape, so --produces can be validated against the same names
/// ToString() prints. Listing the enumerators by hand is deliberate: a
/// new shape added to shape.hpp makes the -Wswitch warning in
/// PayloadToJson fire in graph_consumer.cpp, next to this file, which is
/// where the omission would be noticed.
const Shape kAllShapes[] = {
    Shape::kFrame,    Shape::kBoxes,    Shape::kObBoxes, Shape::kInstances,
    Shape::kKeypoints, Shape::kLabelMap, Shape::kDenseMap, Shape::kImage,
    Shape::kScores,   Shape::kVector,   Shape::kBoxes3d, Shape::kRecords};

/// Matches --consumes. Its value has already been validated.
bool ConsumesMatches(const std::string& wanted, InputContract contract) {
    if (wanted == "roi") return AcceptsRoi(contract);
    if (wanted == "frame" || wanted == "full_frame") {
        return contract == InputContract::kFullFrame ||
               contract == InputContract::kEither;
    }
    return contract == InputContract::kEither;  // "either"
}

/**
 * @brief Validate both --list-models filters before any row is examined.
 *
 * Up front, not inside the row loop, for two reasons. An unknown value
 * must be reported even when no row would have reached the test - an
 * in-loop check on an empty or heavily pre-filtered list reports nothing
 * and exits 0. And the two filters must behave the same way: a typo in
 * --produces used to print an empty table and exit 0, which reads as
 * "no model produces that" rather than "you misspelled a shape".
 */
bool ValidateFilters(const GraphCliArgs& args, std::string* error) {
    if (!args.consumes.empty() && args.consumes != "roi" &&
        args.consumes != "frame" && args.consumes != "full_frame" &&
        args.consumes != "either") {
        *error = "--consumes expects roi, frame or either, got \"" +
                 args.consumes + "\"";
        return false;
    }
    if (!args.produces.empty()) {
        const std::size_t count = sizeof(kAllShapes) / sizeof(kAllShapes[0]);
        std::string known;
        for (std::size_t i = 0; i < count; ++i) {
            if (args.produces == ToString(kAllShapes[i])) return true;
            if (i != 0) known += ", ";
            known += ToString(kAllShapes[i]);
        }
        *error = "--produces expects one of " + known + ", got \"" +
                 args.produces + "\"";
        return false;
    }
    return true;
}

/// This process's DX-RT version, the one every stage's engine would load on.
const std::string& RuntimeVersion() {
    static const std::string version = dxrt::Configuration::GetInstance().GetVersion();
    return version;
}

/// The file column of --list-models: the container version of the model's
/// .dxnn in model_dir ("v8", "v9"), "missing", or "invalid" for a file that
/// is not a .dxnn. A container this runtime cannot load says why:
/// "v9 (needs DX-RT >= 3.5.0)" (R12).
std::string ContainerColumn(const ModelInfo& info, const std::string& model_dir) {
    const std::string path = model_dir + "/" + info.dxnn_file;
    if (!FileExists(path)) return "missing";
    uint32_t version = 0;
    std::string error;
    if (!ReadDxnnContainerVersion(path, &version, &error)) return "invalid";
    const std::string requirement = ContainerRequirement(version, RuntimeVersion());
    return "v" + std::to_string(version) + (requirement.empty() ? "" : " (" + requirement + ")");
}

bool ListedByFilters(const ModelInfo& info, const GraphCliArgs& args) {
    if (!args.consumes.empty() && !ConsumesMatches(args.consumes, info.input_contract)) {
        return false;
    }
    return args.produces.empty() || args.produces == ToString(info.output_shape);
}

/// Exit code for PrintModels: 0 on success, 2 on a bad filter value.
int PrintModels(const IModelRegistry& registry, const GraphCliArgs& args) {
    // Every non-comment line starts with a model name, because
    // tests/cpp_example/test_graph_cli.py reads the first token of each
    // line back and requires docs/graph_models.md to carry it. Anything
    // that is not a model row is a header ("model ...") or starts with
    // "-"; nothing else is printed on stdout.
    std::string filter_error;
    if (!ValidateFilters(args, &filter_error)) {
        std::fprintf(stderr, "%s\n", filter_error.c_str());
        return 2;
    }

    // One row per model, then one per alias_of alias (R6), which a filter
    // keeps or drops with the model it names. A model's old names are
    // aliases too, but one per model would double the table; --check notes
    // them where a graph uses one.
    const std::vector<ModelInfo> models = registry.list();
    const std::vector<ModelAlias> aliases = registry.aliases();
    // Columns grow with the longest name and task; a space always follows.
    std::size_t model_width = 46;
    std::size_t task_width = 28;
    for (std::size_t i = 0; i < models.size(); ++i) {
        model_width = std::max(model_width, models[i].model_name.size() + 1);
        task_width = std::max(task_width, models[i].task.size() + 1);
    }
    for (std::size_t a = 0; a < aliases.size(); ++a) {
        model_width = std::max(model_width, aliases[a].name.size() + 1);
    }
    // The file column grows too, for "v9 (needs DX-RT >= 3.5.0)".
    std::vector<std::string> files(models.size());
    std::size_t file_width = 9;
    for (std::size_t i = 0; i < models.size(); ++i) {
        if (!ListedByFilters(models[i], args)) continue;
        files[i] = ContainerColumn(models[i], args.model_dir);
        file_width = std::max(file_width, files[i].size() + 1);
    }
    const int model_w = static_cast<int>(model_width);
    const int task_w = static_cast<int>(task_width);
    const int file_w = static_cast<int>(file_width);

    std::ostringstream out;
    out << std::left << std::setw(model_w) << "model" << std::setw(task_w) << "task"
        << std::setw(11) << "produces" << std::setw(11) << "consumes"
        << std::setw(11) << "input" << std::setw(10) << "published"
        << std::setw(file_w) << "file" << "ready\n";
    out << std::string(model_width + task_width + 11 * 3 + 10 + file_width + 5, '-') << "\n";
    for (std::size_t i = 0; i < models.size(); ++i) {
        const ModelInfo& info = models[i];
        if (!ListedByFilters(info, args)) continue;
        std::ostringstream size;
        size << info.input_width << "x" << info.input_height;
        out << std::left << std::setw(model_w) << info.model_name
            << std::setw(task_w) << (info.task.empty() ? "-" : info.task)
            << std::setw(11) << ToString(info.output_shape)
            << std::setw(11) << ToString(info.input_contract)
            << std::setw(11) << size.str()
            << std::setw(10) << (info.published ? "yes" : "no")
            << std::setw(file_w) << files[i]
            << (info.ready ? "yes" : "no  " + info.not_ready_reason)
            << (info.ports.empty() ? std::string() : "  ports: " + PortsText(info)) << "\n";
    }
    for (std::size_t a = 0; a < aliases.size(); ++a) {
        if (aliases[a].kind != ModelAlias::kAliasOf) continue;
        const ModelInfo* info = registry.find(aliases[a].variant);
        if (info == NULL || !ListedByFilters(*info, args)) continue;
        out << std::left << std::setw(model_w) << aliases[a].name << "alias of "
            << aliases[a].variant << "\n";
    }
    std::fputs(out.str().c_str(), stdout);
    return 0;
}

/// One --list-models --json row: the table's columns, typed.
nlohmann::json ModelRowJson(const ModelInfo& info, const std::string& model_dir) {
    nlohmann::json row = nlohmann::json::object();
    row["model"] = info.model_name;
    row["task"] = info.task;
    row["produces"] = ToString(info.output_shape);
    row["consumes"] = ToString(info.input_contract);
    nlohmann::json input = nlohmann::json::object();
    input["width"] = info.input_width;
    input["height"] = info.input_height;
    row["input"] = input;
    row["published"] = info.published;
    row["file"] = info.dxnn_file;
    row["container"] = ContainerColumn(info, model_dir);
    row["ready"] = info.ready;
    row["reason"] = info.not_ready_reason;
    nlohmann::json ports = nlohmann::json::array();
    for (std::size_t p = 0; p < info.ports.size(); ++p) {
        nlohmann::json port = nlohmann::json::object();
        port["name"] = info.ports[p].name;
        port["shape"] = ToString(info.ports[p].shape);
        ports.push_back(port);
    }
    row["ports"] = ports;
    row["alias_of"] = nullptr;
    return row;
}

// ---------------------------------------------------------------------
// --list-ops
// ---------------------------------------------------------------------

/// Keys in the order the README documents (op first), not sorted.
typedef nlohmann::ordered_json OrderedJson;

OrderedJson ParamJson(const LogicParamInfo& param) {
    using consumer::JsonNumber;
    OrderedJson out = OrderedJson::object();
    out["name"] = param.name;
    out["type"] = ToString(param.kind);
    out["required"] = param.required;
    switch (param.kind) {
        case LogicParamInfo::kNumber:
            if (!param.required) out["default"] = JsonNumber<OrderedJson>(param.number_default);
            if (param.low > -kLogicUnbounded) out["min"] = JsonNumber<OrderedJson>(param.low);
            if (param.low_open) out["min_exclusive"] = true;
            if (param.high < kLogicUnbounded) out["max"] = JsonNumber<OrderedJson>(param.high);
            break;
        case LogicParamInfo::kText:
            if (!param.required) out["default"] = param.text_default;
            break;
        case LogicParamInfo::kChoice:
            if (!param.required) out["default"] = param.text_default;
            out["choices"] = param.choices;
            break;
        case LogicParamInfo::kTextList:
            // No "default": leaving a list out is not [] (volume: the
            // built-in classes, "when_omitted"; filter, count: every class).
            out["items"] = param.items.empty() ? std::string("string") : param.items;
            if (!param.when_omitted.empty()) out["when_omitted"] = param.when_omitted;
            break;
        case LogicParamInfo::kPolygon:
            out["min_points"] = 3;
            break;
        case LogicParamInfo::kLine:
            out["points"] = 2;
            break;
    }
    if (param.step > 0.0) out["step"] = JsonNumber<OrderedJson>(param.step);
    if (!param.format.empty()) out["format"] = param.format;
    if (!param.extensions.empty()) out["extensions"] = param.extensions;
    if (!param.equals.empty()) out["equals"] = param.equals;
    if (!param.values_by_input.empty()) {
        OrderedJson by_input = OrderedJson::array();
        for (std::size_t i = 0; i < param.values_by_input.size(); ++i) {
            const LogicParamInfo::InputValues& allowed = param.values_by_input[i];
            OrderedJson row = OrderedJson::object();
            row["shape"] = ToString(allowed.shape);
            row["per_crop"] = allowed.per_crop;
            if (allowed.from_numbers) {
                row["values_from"] = "numbers";
            } else {
                row["values"] = allowed.values;
            }
            by_input.push_back(row);
        }
        out["values_by_input"] = by_input;
    }
    out["help"] = param.help;
    return out;
}

OrderedJson OpJson(const LogicOpInfo& info, const char* type) {
    OrderedJson out = OrderedJson::object();
    out["op"] = info.op;
    out["type"] = type;
    out["summary"] = info.summary;
    OrderedJson accepts = OrderedJson::array();
    for (std::size_t i = 0; i < info.accepts.size(); ++i) {
        OrderedJson port = OrderedJson::object();
        port["shape"] = ToString(info.accepts[i].shape);
        port["per_crop"] = info.accepts[i].per_crop;
        accepts.push_back(port);
    }
    OrderedJson inputs = OrderedJson::object();
    inputs["min"] = info.min_inputs;
    inputs["max"] = info.max_inputs;
    inputs["accepts"] = accepts;
    out["inputs"] = inputs;
    out["output"] = info.same_shape_out ? "same" : "records";
    out["tracking"] = info.needs_tracking;
    out["events"] = info.events;
    out["numbers"] = info.numbers;
    OrderedJson params = OrderedJson::array();
    for (std::size_t p = 0; p < info.params.size(); ++p) {
        params.push_back(ParamJson(info.params[p]));
    }
    out["params"] = params;
    return out;
}

/// --list-ops --json: logic ops, then cpu ops. Always 0.
int PrintOpsJson() {
    OrderedJson ops = OrderedJson::array();
    const std::vector<LogicOpInfo>& logic = LogicOps();
    const std::vector<LogicOpInfo>& cpu = CpuOpInfos();
    for (std::size_t i = 0; i < logic.size(); ++i) ops.push_back(OpJson(logic[i], "logic"));
    for (std::size_t i = 0; i < cpu.size(); ++i) ops.push_back(OpJson(cpu[i], "cpu"));
    std::printf("%s\n", ops.dump(2).c_str());
    return 0;
}

/// "boxes, obboxes, instances"; per-crop inputs end in "/crop".
std::string AcceptsColumn(const LogicOpInfo& info) {
    std::string text;
    for (std::size_t i = 0; i < info.accepts.size(); ++i) {
        if (i != 0) text += ",";
        text += ToString(info.accepts[i].shape);
        if (info.accepts[i].per_crop) text += "/crop";
    }
    return text;
}

/// --list-ops: one row per op. Always 0.
int PrintOps() {
    std::printf("%-10s %-6s %-6s %-8s %-8s %s\n", "op", "type", "inputs", "output", "tracking",
                "accepts");
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<LogicOpInfo>& ops = pass == 0 ? LogicOps() : CpuOpInfos();
        for (std::size_t i = 0; i < ops.size(); ++i) {
            const LogicOpInfo& info = ops[i];
            const std::string count =
                info.min_inputs == info.max_inputs
                    ? std::to_string(info.min_inputs)
                    : std::to_string(info.min_inputs) + "-" + std::to_string(info.max_inputs);
            std::printf("%-10s %-6s %-6s %-8s %-8s %s\n", info.op.c_str(),
                        pass == 0 ? "logic" : "cpu", count.c_str(),
                        info.same_shape_out ? "same" : "records",
                        info.needs_tracking ? "yes" : "no", AcceptsColumn(info).c_str());
        }
    }
    return 0;
}

/// --list-models --json: the table as one array. 0, or 2 on a bad filter.
int PrintModelsJson(const IModelRegistry& registry, const GraphCliArgs& args) {
    std::string filter_error;
    if (!ValidateFilters(args, &filter_error)) {
        std::fprintf(stderr, "%s\n", filter_error.c_str());
        return 2;
    }
    nlohmann::json rows = nlohmann::json::array();
    const std::vector<ModelInfo> models = registry.list();
    for (std::size_t i = 0; i < models.size(); ++i) {
        if (!ListedByFilters(models[i], args)) continue;
        rows.push_back(ModelRowJson(models[i], args.model_dir));
    }
    const std::vector<ModelAlias> aliases = registry.aliases();
    for (std::size_t a = 0; a < aliases.size(); ++a) {
        if (aliases[a].kind != ModelAlias::kAliasOf) continue;
        const ModelInfo* info = registry.find(aliases[a].variant);
        if (info == NULL || !ListedByFilters(*info, args)) continue;
        nlohmann::json row = ModelRowJson(*info, args.model_dir);
        row["model"] = aliases[a].name;
        row["alias_of"] = aliases[a].variant;
        rows.push_back(row);
    }
    std::printf("%s\n", rows.dump(2).c_str());
    return 0;
}

// ---------------------------------------------------------------------
// --check / the T1 resource stage
// ---------------------------------------------------------------------

/// The --check line of one resource a node needs (R9).
std::string ResourceLine(const NodeSpec& node, const ResourceInfo& resource,
                         const std::string& model_dir) {
    const std::string who = "node \"" + node.id + "\": ";
    if (resource.kind == ResourceInfo::kNote) return "note: " + who + resource.value;
    if (resource.kind == ResourceInfo::kCompanion) {
        return "resource: " + who + "companion " + resource.value +
               (FileExists(model_dir + "/" + resource.value) ? " [present]" : " [MISSING]");
    }
    // The gallery the run opens: the node's "gallery" param over the
    // registry's (config.json), read against the repository unless it is
    // absolute - the stage overlay's rule (GalleryAgainstRoot).
    const std::map<std::string, std::string>::const_iterator param =
        node.params.text.find("gallery");
    const std::string gallery = param != node.params.text.end() ? param->second : resource.value;
    const std::string path = ResolveRepoRelative(gallery);
    std::string status = "[MISSING]";
    if (FileExists(path)) {
        char magic[8] = {0};
        std::ifstream in(path.c_str(), std::ios::binary);
        in.read(magic, sizeof(magic));
        status = in.gcount() == static_cast<std::streamsize>(sizeof(magic)) &&
                         std::memcmp(magic, "DXGAL1", 6) == 0
                     ? "[present, DXGAL1]"
                     : "[present, but not a DXGAL1 gallery]";
    }
    return "resource: " + who + "gallery " + gallery + " " + status;
}

void PrintCheckSummary(const GraphSpec& spec, const IModelRegistry& registry,
                       const std::string& graph_path,
                       const std::string& model_dir) {
    std::size_t model_nodes = 0;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (IsModelNode(spec.nodes[i])) ++model_nodes;
    }
    std::printf("OK: graph \"%s\" is valid (%u nodes, %u edges, %u models)\n",
                spec.name.empty() ? graph_path.c_str() : spec.name.c_str(),
                static_cast<unsigned>(spec.nodes.size()),
                static_cast<unsigned>(spec.edges.size()),
                static_cast<unsigned>(model_nodes));

    // An old name or an alias runs its variant (R6): say so, once per node
    // (the same lines a run prints on stderr).
    std::fputs(AliasNotes(spec, registry).c_str(), stdout);

    // Columns grow with the longest id and model name; a space always
    // follows. Shorter ones keep the historical widths (10, 30). The model
    // column names the variant that runs.
    std::size_t id_width = 10;
    std::size_t model_width = 30;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        id_width = std::max(id_width, spec.nodes[i].id.size() + 1);
        if (spec.nodes[i].is_cpu || spec.nodes[i].is_logic) {
            const std::string kind = spec.nodes[i].is_cpu ? "(cpu " : "(logic ";
            model_width = std::max(model_width, kind.size() + spec.nodes[i].op.size() + 2);
            continue;
        }
        if (!spec.nodes[i].is_source) {
            const ModelInfo* info = registry.find(spec.nodes[i].model);
            const std::string& name = info != NULL ? info->model_name : spec.nodes[i].model;
            model_width = std::max(model_width, name.size() + 1);
        }
    }
    const int id_w = static_cast<int>(id_width);
    const int model_w = static_cast<int>(model_width);

    std::ostringstream table;
    table << "\n" << std::left << std::setw(id_w) << "node"
          << std::setw(model_w) << "model" << std::setw(11) << "produces"
          << std::setw(11) << "consumes" << "artifact\n";
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        table << std::left << std::setw(id_w) << node.id;
        if (node.is_source) {
            table << std::setw(model_w) << "(source)" << std::setw(11) << "frame"
                  << std::setw(11) << "-" << node.uri << "\n";
            continue;
        }
        if (node.is_cpu) {
            table << std::setw(model_w) << ("(cpu " + node.op + ")")
                  << std::setw(11) << "records" << std::setw(11) << "result" << "-\n";
            continue;
        }
        if (node.is_logic) {
            table << std::setw(model_w) << ("(logic " + node.op + ")")
                  << std::setw(11) << ToString(NodeOutputShape(spec, registry, node.id))
                  << std::setw(11) << "result" << "-\n";
            continue;
        }
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        // A present file names its container version, as --list-models
        // does ("[present, v8]"); one this runtime cannot load also says why
        // (R12), and the error itself follows on stderr (ModelFileProblems).
        const std::string column = ContainerColumn(*info, model_dir);
        const std::string status = column == "missing" ? "[MISSING]" : "[present, " + column + "]";
        table << std::setw(model_w) << info->model_name
              << std::setw(11) << ToString(info->output_shape)
              << std::setw(11) << ToString(info->input_contract)
              << info->dxnn_file << "  " << status << "\n";
    }
    std::fputs(table.str().c_str(), stdout);

    // What each node needs besides its .dxnn (R9): galleries, companion
    // engines, notes. Reported, like a missing .dxnn, not enforced.
    std::ostringstream resources;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (!IsModelNode(node)) continue;
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        for (std::size_t r = 0; r < info->resources.size(); ++r) {
            resources << ResourceLine(node, info->resources[r], model_dir) << "\n";
        }
    }
    if (!resources.str().empty()) std::printf("\n%s", resources.str().c_str());

    // SP2: with several sources, say which nodes each stream runs. A
    // one-source graph prints nothing here, so its --check is unchanged.
    const std::vector<StreamSpec> streams = ListStreams(spec);
    if (streams.size() > 1) {
        std::size_t source_width = 10;
        for (std::size_t s = 0; s < streams.size(); ++s) {
            source_width = std::max(source_width, streams[s].source.size() + 1);
        }
        const int source_w = static_cast<int>(source_width);
        std::ostringstream text;
        text << "\nstreams: " << streams.size()
             << " - one per source node, read in turn, one frame from each\n";
        for (std::size_t s = 0; s < streams.size(); ++s) {
            text << "  " << std::left << std::setw(source_w) << streams[s].source << "-> ";
            bool first = true;
            for (std::size_t k = 0; k < streams[s].nodes.size(); ++k) {
                if (streams[s].nodes[k] == streams[s].source) continue;
                text << (first ? "" : ", ") << streams[s].nodes[k];
                first = false;
            }
            text << "\n";
        }
        std::fputs(text.str().c_str(), stdout);
    }

    const std::vector<MissingArtifact> missing =
        CollectMissing(spec, registry, model_dir);
    if (!missing.empty()) {
        std::printf("\nMISSING: %s\n",
                    MissingHeadline(missing.size(), model_dir).c_str());
        std::fputs(FormatMissingList(missing, model_dir).c_str(), stdout);
    }
}

/**
 * @brief One MODEL_LOAD error per model node whose .dxnn is present but
 *        cannot load here: not a .dxnn at all (empty, short, no DXNN
 *        header), or a container this DX-RT cannot load (v9 before 3.5.0,
 *        R12).
 *
 * --check exits 1 on any of them: unlike a missing file, which a download
 * fixes and which --check only reports, the graph cannot run with these
 * files, and a run would stop on them when the engine opens.
 */
std::vector<GraphError> ModelFileProblems(const GraphSpec& spec,
                                          const IModelRegistry& registry,
                                          const std::string& model_dir) {
    std::vector<GraphError> problems;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (!IsModelNode(node)) continue;
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        const std::string path = model_dir + "/" + info->dxnn_file;
        if (!FileExists(path)) continue;  // missing: reported, not an error
        uint32_t version = 0;
        std::string header_error;
        if (!ReadDxnnContainerVersion(path, &version, &header_error)) {
            problems.push_back(GraphError(
                GraphErrorCode::kModelLoad, "node \"" + node.id + "\"",
                "model \"" + node.model + "\" (" + info->dxnn_file +
                    ") is not a loadable .dxnn: " + header_error,
                "download it again: " +
                    DownloadCommand(std::vector<std::string>(
                                        1, info->download_name.empty() ? node.model
                                                                       : info->download_name),
                                    model_dir)));
            continue;
        }
        const std::string error = ContainerLoadError(path, RuntimeVersion());
        if (error.empty()) continue;
        // The run's hint for the same file (typed_stage.hpp): downloading
        // again would fetch the same container.
        problems.push_back(GraphError(GraphErrorCode::kModelLoad, "node \"" + node.id + "\"",
                                      "model \"" + node.model + "\" (" + info->dxnn_file +
                                          ") cannot be loaded here: " + error,
                                      ContainerSupportHint(version)));
    }
    return problems;
}

// ---------------------------------------------------------------------
// --check --json
// ---------------------------------------------------------------------

/// The graph file as plain JSON, to name the node or edge of a parse error
/// by the index the error gives; null when it does not parse.
nlohmann::json RawGraph(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in.is_open()) return nlohmann::json();
    try {
        return nlohmann::json::parse(in);
    } catch (const std::exception&) {
        return nlohmann::json();
    }
}

/// The <i> of "<kind>[<i>]" in `where`, searched from `from`.
bool IndexAfter(const std::string& where, const std::string& kind, std::size_t from,
                std::size_t* index) {
    const std::string head = kind + "[";
    const std::size_t start = where.find(head, from);
    if (start == std::string::npos) return false;
    const char* digits = where.c_str() + start + head.size();
    char* end = NULL;
    const unsigned long value = std::strtoul(digits, &end, 10);
    if (end == digits || *end != ']') return false;
    *index = static_cast<std::size_t>(value);
    return true;
}

/// raw[list][index][key] when it is a string, else null.
nlohmann::json StringAt(const nlohmann::json& raw, const char* list, std::size_t index,
                        const char* key) {
    if (!raw.is_object()) return nullptr;
    nlohmann::json::const_iterator items = raw.find(list);
    if (items == raw.end() || !items->is_array() || index >= items->size()) return nullptr;
    const nlohmann::json& item = (*items)[index];
    if (!item.is_object()) return nullptr;
    nlohmann::json::const_iterator value = item.find(key);
    if (value == item.end() || !value->is_string()) return nullptr;
    return *value;
}

/// error["node"] / error["edge"] for a GraphError's where (see README).
void AddErrorTarget(const std::string& where, const GraphSpec* spec, const std::string& path,
                    nlohmann::json* error) {
    (*error)["node"] = nullptr;
    (*error)["edge"] = nullptr;
    if (spec != NULL) {
        for (std::size_t e = 0; e < spec->edges.size(); ++e) {
            const EdgeSpec& edge = spec->edges[e];
            if (where.find("edge \"" + edge.from + "\"->\"" + edge.to + "\"") == std::string::npos) {
                continue;
            }
            nlohmann::json target = nlohmann::json::object();
            target["from"] = edge.from;
            target["to"] = edge.to;
            (*error)["edge"] = target;
            return;
        }
        for (std::size_t n = 0; n < spec->nodes.size(); ++n) {
            if (where.find("node \"" + spec->nodes[n].id + "\"") == std::string::npos) continue;
            (*error)["node"] = spec->nodes[n].id;
            return;
        }
        return;
    }
    // A parse error: "<path> node[3] ..." / "<path> edge[2] ...".
    const std::size_t from = where.compare(0, path.size(), path) == 0 ? path.size() : 0;
    std::size_t index = 0;
    if (IndexAfter(where, "edge", from, &index)) {
        const nlohmann::json raw = RawGraph(path);
        nlohmann::json target = nlohmann::json::object();
        target["from"] = StringAt(raw, "edges", index, "from");
        target["to"] = StringAt(raw, "edges", index, "to");
        (*error)["edge"] = target;
    } else if (IndexAfter(where, "node", from, &index)) {
        (*error)["node"] = StringAt(RawGraph(path), "nodes", index, "id");
    }
}

nlohmann::json ErrorJson(const GraphError& error, const GraphSpec* spec,
                         const std::string& path) {
    nlohmann::json out = nlohmann::json::object();
    out["code"] = ToString(error.code());
    out["where"] = error.where();
    out["what"] = error.what_happened();
    out["how"] = error.how_to_fix();
    AddErrorTarget(error.where(), spec, path, &out);
    return out;
}

/// present | missing | invalid | unloadable, for the .dxnn at `path`.
std::string FileState(const std::string& path) {
    if (!FileExists(path)) return "missing";
    uint32_t version = 0;
    std::string error;
    if (!ReadDxnnContainerVersion(path, &version, &error)) return "invalid";
    return ContainerSupportError(version, RuntimeVersion()).empty() ? "present" : "unloadable";
}

/// One entry per model node, in declaration order (see README).
nlohmann::json ModelsJson(const GraphSpec& spec, const IModelRegistry& registry,
                          const std::string& model_dir) {
    const std::map<std::string, std::string> owners = SharedEngineOwners(spec, registry);
    nlohmann::json models = nlohmann::json::array();
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (!IsModelNode(node)) continue;
        nlohmann::json row = nlohmann::json::object();
        row["node"] = node.id;
        nlohmann::json engine = nlohmann::json::object();
        engine["buffers"] = node.engine.buffers;
        engine["share"] = node.engine.share;
        std::map<std::string, std::string>::const_iterator owner = owners.find(node.id);
        engine["shared_with"] = owner != owners.end() && owner->second != node.id
                                    ? nlohmann::json(owner->second)
                                    : nlohmann::json(nullptr);
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) {
            row["model"] = node.model;
            row["file"] = nullptr;
            row["state"] = "unknown";
            row["container"] = nullptr;
        } else {
            const std::string path = model_dir + "/" + info->dxnn_file;
            row["model"] = info->model_name;
            row["file"] = info->dxnn_file;
            row["state"] = FileState(path);
            uint32_t version = 0;
            std::string ignored;
            row["container"] = FileExists(path) && ReadDxnnContainerVersion(path, &version, &ignored)
                                   ? nlohmann::json("v" + std::to_string(version))
                                   : nlohmann::json(nullptr);
        }
        row["engine"] = engine;
        models.push_back(row);
    }
    return models;
}

/// --check --json: exit 0 when nothing is in "errors", else 1.
int PrintCheckJson(const GraphCliArgs& args, const IModelRegistry& registry) {
    nlohmann::json errors = nlohmann::json::array();
    nlohmann::json models = nlohmann::json::array();
    nlohmann::json notes = nlohmann::json::array();
    std::string graph_name = args.graph_path;
    GraphSpec spec;
    bool parsed = false;
    try {
        spec = ParseGraphFile(args.graph_path);
        parsed = true;
        if (!spec.name.empty()) graph_name = spec.name;
        ValidateGraph(spec, registry);
    } catch (const GraphError& error) {
        errors.push_back(ErrorJson(error, parsed ? &spec : NULL, args.graph_path));
    }
    if (parsed) {
        const std::vector<GraphError> files = ModelFileProblems(spec, registry, args.model_dir);
        for (std::size_t i = 0; i < files.size(); ++i) {
            errors.push_back(ErrorJson(files[i], &spec, args.graph_path));
        }
        models = ModelsJson(spec, registry, args.model_dir);
        std::istringstream lines(AliasNotes(spec, registry));
        std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty()) notes.push_back(line);
        }
    }
    nlohmann::json out = nlohmann::json::object();
    out["ok"] = errors.empty();
    out["graph"] = graph_name;
    out["errors"] = errors;
    out["models"] = models;
    out["notes"] = notes;
    std::printf("%s\n", out.dump(2).c_str());
    return errors.empty() ? 0 : 1;
}

// ---------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------

/**
 * @brief Warn when a ROI edge's filter matched nothing its producer found.
 *
 * A class-name typo is otherwise a silent no-op: "classes": ["persons"]
 * passes --check, runs, exits 0, reports "1 frame, 0 failed", and hands
 * the consumer zero crops. Nothing in the run says the filter was the
 * reason, and "persons" is a plausible spelling of a COCO class that is
 * actually "person".
 *
 * Computed here, in the CLI, from the report the executors already
 * produce - deliberately NOT a new FrameReport field. A field there would
 * have to be set identically by both executors and compared by
 * operator==, which is a sync/async parity risk taken on behalf of a
 * warning.
 *
 * Only fires when the producer actually emitted boxes: a detector that
 * found nothing is a legitimate empty frame, not a misconfigured filter.
 * Per-parent, because a node with two ROI parents has one results vector
 * for both, so a non-empty vector does not mean every parent contributed.
 * Once per edge per run, so a video does not repeat it every frame.
 */
void WarnAboutEmptyRoiEdges(const GraphSpec& spec, const FrameReport& report,
                            std::set<std::size_t>* warned) {
    for (std::size_t e = 0; e < spec.edges.size(); ++e) {
        const EdgeSpec& edge = spec.edges[e];
        if (!edge.roi.present) continue;
        if (warned->count(e) != 0) continue;

        std::map<std::string, std::vector<StageResult> >::const_iterator
            consumer = report.roi_results.find(edge.to);
        if (consumer == report.roi_results.end()) continue;
        std::size_t from_this_parent = 0;
        for (std::size_t i = 0; i < consumer->second.size(); ++i) {
            if (consumer->second[i].origin.parent_node == edge.from) {
                ++from_this_parent;
            }
        }
        if (from_this_parent != 0) continue;

        std::map<std::string, StageResult>::const_iterator producer =
            report.node_results.find(edge.from);
        if (producer == report.node_results.end()) continue;
        const BoxesData* boxes =
            dynamic_cast<const BoxesData*>(producer->second.data.get());
        if (boxes == NULL || boxes->items.empty()) continue;

        // The producer found things and the consumer got none of them.
        warned->insert(e);

        std::ostringstream filter;
        if (!edge.roi.classes.empty()) {
            filter << " \"classes\": [";
            for (std::size_t c = 0; c < edge.roi.classes.size(); ++c) {
                if (c != 0) filter << ", ";
                filter << "\"" << edge.roi.classes[c] << "\"";
            }
            filter << "]";
        }
        if (edge.roi.min_score >= 0.f) filter << " \"min_score\"";
        if (edge.roi.min_area > 0.f) filter << " \"min_area\"";
        if (edge.roi.max >= 0) filter << " \"max\"";

        std::set<std::string> emitted;
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            if (!boxes->items[i].class_name.empty()) {
                emitted.insert(boxes->items[i].class_name);
            }
        }
        std::ostringstream names;
        std::size_t shown = 0;
        for (std::set<std::string>::const_iterator it = emitted.begin();
             it != emitted.end(); ++it, ++shown) {
            if (shown == 12) {
                names << ", ... (" << (emitted.size() - shown) << " more)";
                break;
            }
            if (shown != 0) names << ", ";
            names << *it;
        }

        std::fprintf(stderr,
                     "warning: edge \"%s\"->\"%s\" produced no crops, "
                     "although \"%s\" found %u box%s\n",
                     edge.from.c_str(), edge.to.c_str(), edge.from.c_str(),
                     static_cast<unsigned>(boxes->items.size()),
                     boxes->items.size() == 1 ? "" : "es");
        if (!filter.str().empty()) {
            std::fprintf(stderr, "  the \"roi\" filter on this edge sets:%s\n",
                         filter.str().c_str());
        }
        std::fprintf(stderr, "  \"%s\" emitted these classes: %s\n",
                     edge.from.c_str(),
                     names.str().empty() ? "(unnamed)" : names.str().c_str());
        if (report.skipped_out_of_bounds != 0) {
            std::fprintf(stderr,
                         "  %d region(s) were dropped for falling outside "
                         "the frame\n", report.skipped_out_of_bounds);
        }
        std::fprintf(stderr,
                     "  -> check the class names against that list, or relax "
                     "\"min_score\"/\"min_area\"/\"max\"\n");
    }
}

std::string FormatNumber(double value) {
    char buffer[64];
    if (std::isfinite(value) && std::fabs(value) < 1e9 &&
        std::fabs(value - std::round(value)) < 1e-4) {
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.2f", value);
    }
    return buffer;
}

/// One payload reduced to a count, plus the measurements of a records payload.
std::string DescribePayload(const StageData* data) {
    if (data == NULL) return "none";
    switch (data->shape()) {
        case Shape::kBoxes:
        case Shape::kObBoxes:
        case Shape::kInstances: {
            const BoxesData* boxes = dynamic_cast<const BoxesData*>(data);
            if (boxes == NULL) return ToString(data->shape());
            return std::string(ToString(data->shape())) + "=" +
                   std::to_string(boxes->items.size());
        }
        case Shape::kKeypoints: {
            const KeypointsData* poses = dynamic_cast<const KeypointsData*>(data);
            if (poses == NULL) return "keypoints";
            return "keypoints=" + std::to_string(poses->items.size());
        }
        case Shape::kScores: {
            const ScoresData* scores = dynamic_cast<const ScoresData*>(data);
            if (scores == NULL) return "scores";
            return "scores=" + std::to_string(scores->items.size());
        }
        case Shape::kVector: {
            const VectorData* vector = dynamic_cast<const VectorData*>(data);
            if (vector == NULL) return "vector";
            return "vector=" + std::to_string(vector->values.size());
        }
        case Shape::kBoxes3d: {
            const Boxes3dData* boxes = dynamic_cast<const Boxes3dData*>(data);
            if (boxes == NULL) return "boxes3d";
            return "boxes3d=" + std::to_string(boxes->items.size());
        }
        case Shape::kRecords: {
            const RecordsData* records = dynamic_cast<const RecordsData*>(data);
            if (records == NULL) return "records";
            std::ostringstream text;
            text << "records=" << records->items.size();
            const std::size_t kMaxItems = 3;
            const std::size_t shown = std::min(records->items.size(), kMaxItems);
            for (std::size_t i = 0; i < shown; ++i) {
                text << (i == 0 ? "" : " |");
                const RecordItem& item = records->items[i];
                for (std::size_t n = 0; n < item.numbers.size(); ++n) {
                    text << " " << item.numbers[n].first << "="
                         << FormatNumber(item.numbers[n].second);
                }
                for (std::size_t n = 0; n < item.text.size(); ++n) {
                    text << " " << item.text[n].first << "=" << item.text[n].second;
                }
            }
            if (records->items.size() > kMaxItems) {
                text << " | +" << (records->items.size() - kMaxItems);
            }
            return text.str();
        }
        case Shape::kFrame:
        case Shape::kImage:
        case Shape::kLabelMap:
        case Shape::kDenseMap:
            return ToString(data->shape());
    }
    return ToString(data->shape());
}

/// Sum countable payloads that share one shape. A records node keeps each
/// item, because the numbers are the result a video run is watched for.
std::string DescribePayloads(const std::vector<const StageData*>& payloads) {
    std::vector<const StageData*> present;
    for (std::size_t i = 0; i < payloads.size(); ++i) {
        if (payloads[i] != NULL) present.push_back(payloads[i]);
    }
    if (present.empty()) return "none";
    const Shape shape = present[0]->shape();
    bool same = true;
    for (std::size_t i = 1; i < present.size(); ++i) {
        if (present[i]->shape() != shape) same = false;
    }
    if (!same || shape == Shape::kRecords || present.size() == 1) {
        if (present.size() == 1) return DescribePayload(present[0]);
        std::ostringstream text;
        text << "rois=" << present.size();
        for (std::size_t i = 0; i < present.size(); ++i) {
            text << " " << DescribePayload(present[i]);
        }
        return text.str();
    }
    std::size_t count = 0;
    for (std::size_t i = 0; i < present.size(); ++i) {
        const std::string one = DescribePayload(present[i]);
        const std::size_t eq = one.rfind('=');
        if (eq == std::string::npos) return "rois=" + std::to_string(present.size()) + " " + ToString(shape);
        count += static_cast<std::size_t>(std::atoi(one.c_str() + eq + 1));
    }
    return std::string(ToString(shape)) + "=" + std::to_string(count);
}

/// One stdout line per frame, so a video run is readable without a window.
void PrintFrameLine(const GraphSpec& spec, const FrameReport& report, bool several) {
    std::ostringstream line;
    if (several) line << "stream " << report.stream << " ";
    line << "frame " << report.frame_index << ":";
    if (!report.error.empty()) line << " error";
    for (std::size_t n = 0; n < spec.nodes.size(); ++n) {
        const NodeSpec& node = spec.nodes[n];
        if (node.is_source) continue;
        std::map<std::string, StageResult>::const_iterator full =
            report.node_results.find(node.id);
        std::map<std::string, std::vector<StageResult> >::const_iterator rois =
            report.roi_results.find(node.id);
        if (full == report.node_results.end() && rois == report.roi_results.end()) continue;
        std::vector<const StageData*> payloads;
        if (full != report.node_results.end()) {
            payloads.push_back(full->second.data.get());
        } else {
            for (std::size_t i = 0; i < rois->second.size(); ++i) {
                payloads.push_back(rois->second[i].data.get());
            }
        }
        line << " " << node.id << " " << DescribePayloads(payloads);
    }
    std::printf("%s\n", line.str().c_str());
    std::fflush(stdout);
}

/// Where one stream's rendered frames go.
struct StreamSink {
    std::unique_ptr<VideoOutput> video;
    std::unique_ptr<DisplayWindow> window;
    bool single_frame;
    int failed;
    StreamSink() : single_frame(false), failed(0) {}
};

/// This frame's FrameReport::node_ms in graph declaration order, for [PROGRESS].
NodeMs ProgressNodeMs(const GraphSpec& spec, const FrameReport& report) {
    NodeMs out;
    for (std::size_t n = 0; n < spec.nodes.size(); ++n) {
        std::map<std::string, double>::const_iterator ms = report.node_ms.find(spec.nodes[n].id);
        if (ms != report.node_ms.end()) out.push_back(std::make_pair(ms->first, ms->second));
    }
    return out;
}

/// Events the recent-events panel keeps per stream.
const std::size_t kRecentEvents = 5;

/// One line of that panel: "12.3s zone enter #4".
std::string EventPanelLine(const GraphEvent& event) {
    char time[32];
    std::snprintf(time, sizeof(time), "%.1fs", event.time_s);
    std::string line = std::string(time) + " " + event.node + " " + event.event;
    if (event.track_id >= 0) line += " #" + std::to_string(event.track_id);
    return line;
}

// ---------------------------------------------------------------------
// Results for front ends: --draw, --output-layers
// ---------------------------------------------------------------------

/// "" when every --draw id names a node of `spec` that can draw (any node
/// but a source); otherwise the usage error, listing those nodes.
std::string DrawProblem(const GraphSpec& spec, const std::vector<std::string>& ids) {
    std::string listed;
    for (std::size_t n = 0; n < spec.nodes.size(); ++n) {
        if (spec.nodes[n].is_source) continue;
        listed += (listed.empty() ? "" : ", ") + spec.nodes[n].id;
    }
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const NodeSpec* node = spec.FindNode(ids[i]);
        if (node == NULL || node->is_source) {
            return "--draw: \"" + ids[i] + "\" is not a node of this graph that draws; "
                   "its nodes: " + listed;
        }
    }
    return std::string();
}

/// "" when --output-layers can go with this stream: a still image, or any
/// source when --frames bounds it. Otherwise the usage error.
std::string LayersSourceProblem(const std::string& uri, std::size_t frames) {
    if (frames != 0) return std::string();
    const bool live = IsLiveUri(uri);
    if (!live && InputFactory::detectFileType(uri) == InputType::IMAGE) return std::string();
    return "--output-layers writes a base image and one layer per node for every frame, "
           "and \"" + uri + "\" is " +
           (live ? "a live source that never ends" : "not a still image") +
           ": stop after N frames (--frames N)";
}

/// "" when `dir` is a directory, made here when it does not exist yet (its
/// parent must); otherwise the error (exit 1, as an unwritable --report).
std::string LayersDirectoryProblem(const std::string& dir) {
    struct stat info;
    if (::stat(dir.c_str(), &info) == 0) {
        if ((info.st_mode & S_IFMT) == S_IFDIR) return std::string();
        return "could not write " + dir + ": it is not a directory";
    }
#ifdef _WIN32
    const int made = ::_mkdir(dir.c_str());
#else
    const int made = ::mkdir(dir.c_str(), 0777);
#endif
    if (made != 0) return "could not write " + dir + ": " + std::strerror(errno);
    return std::string();
}

/// --output-layers: per frame the clean frame, each drawn node's RGBA layer
/// (RenderLayers, in stacking order), the events panel's layer when it has
/// lines, and layers.json - rewritten after every frame (through a
/// temporary file), so a run stopped part way leaves a complete manifest.
class LayersWriter {
 public:
    LayersWriter(const std::string& dir, const std::string& stem, const GraphSpec& spec)
        : dir_(dir), stem_(stem), spec_(spec), frames_(OrderedJson::array()) {
        while (dir_.size() > 1 && (dir_.back() == '/' || dir_.back() == '\\')) dir_.pop_back();
    }

    std::string manifest_path() const { return Join("layers.json"); }

    /// False (with *problem set) when a file cannot be written.
    bool Write(const FrameReport& report, const cv::Mat& frame, const RenderOptions& options,
               const std::vector<std::string>& panel, const std::vector<cv::Rect>& labels,
               bool several, std::string* problem) {
        OrderedJson entry = OrderedJson::object();
        entry["stream"] = report.stream;
        entry["frame"] = report.frame_index;
        const std::string base = Name(report, several, "base");
        if (!Save(base, frame, problem)) return false;
        entry["base"] = base;
        OrderedJson layers = OrderedJson::array();
        const std::vector<RenderLayer> drawn = RenderLayers(frame.size(), report, options);
        for (std::size_t i = 0; i < drawn.size(); ++i) {
            const std::string file = Name(report, several, drawn[i].node);
            if (!Save(file, drawn[i].rgba, problem)) return false;
            OrderedJson layer = OrderedJson::object();
            layer["node"] = drawn[i].node;
            const NodeSpec* node = spec_.FindNode(drawn[i].node);
            if (node != NULL && IsModelNode(*node)) layer["model"] = node->model;
            else if (node != NULL) layer["op"] = node->op;
            layer["file"] = file;
            layers.push_back(layer);
        }
        if (!panel.empty()) {
            cv::Mat on_black = cv::Mat::zeros(frame.size(), CV_8UC3);
            cv::Mat on_white(frame.size(), CV_8UC3, cv::Scalar(255, 255, 255));
            DrawEventLog(&on_black, panel, labels);
            DrawEventLog(&on_white, panel, labels);
            const std::string file = Name(report, several, "events");
            if (!Save(file, LayerFromBackdrops(on_black, on_white), problem)) return false;
            OrderedJson layer = OrderedJson::object();
            layer["panel"] = "events";
            layer["file"] = file;
            layers.push_back(layer);
        }
        entry["layers"] = layers;
        frames_.push_back(entry);
        return WriteManifest(problem);
    }

 private:
    std::string Join(const std::string& name) const { return dir_ + "/" + name; }

    std::string Name(const FrameReport& report, bool several, const std::string& part) const {
        return LayerFileName(stem_, report.stream, several, report.frame_index, part);
    }

    bool Save(const std::string& name, const cv::Mat& image, std::string* problem) const {
        bool written = false;
        try {
            written = cv::imwrite(Join(name), image);
        } catch (const cv::Exception&) {
        }
        if (!written) *problem = "could not write " + Join(name);
        return written;
    }

    bool WriteManifest(std::string* problem) const {
        OrderedJson manifest = OrderedJson::object();
        manifest["frames"] = frames_;
        const std::string path = manifest_path();
        const std::string partial = path + ".part";
        {
            std::ofstream out(partial.c_str(), std::ios::out | std::ios::trunc);
            out << manifest.dump(2) << "\n";
            if (!out) {
                *problem = "could not write " + path;
                return false;
            }
        }
        if (std::rename(partial.c_str(), path.c_str()) != 0) {
            *problem = "could not write " + path;
            return false;
        }
        return true;
    }

    std::string dir_;
    std::string stem_;
    const GraphSpec& spec_;
    OrderedJson frames_;
};

int RunInputLoop(StageGraph& graph, FrameExecutor& executor, const GraphSpec& spec,
                 const GraphCliArgs& args, const std::vector<StreamUri>& uris) {
    // Every stream is opened before the report file: a uri that will not
    // open is a GraphError, printed by Main, and nothing has been written.
    std::vector<StreamInput> inputs;
    for (std::size_t s = 0; s < uris.size(); ++s) {
        StreamInput input;
        input.source = uris[s].source;
        input.uri = uris[s].uri;
        input.input = OpenStream(spec, uris[s]);
        inputs.push_back(std::move(input));
    }
    StreamReader reader(std::move(inputs), args.frames);
    reader.SetLoop(args.loop);
    // Logic nodes count seconds per stream: a file's frame index over its
    // frame rate, a camera's or an RTSP stream's wall clock.
    for (std::size_t s = 0; s < reader.size(); ++s) {
        const InputSourcePtr& input = reader.input(s).input;
        graph.SetStreamClock(s, input->getFPS(), input->isLiveSource());
    }
    // One source: every line, file name and window name is the one the loop
    // printed before streams existed (SP2 decision 3).
    const bool several = NamesStreams(graph);
    // Opened before the first frame: an unwritable path fails now, not after
    // the whole input, and every exit from this scope - normal, a stop
    // request, a failed frame write, an exception - finalizes the file.
    ReportFileWriter report_file;
    std::string problem;
    if (!args.report.empty() && !report_file.Open(args.report, spec.name, &problem)) {
        std::fprintf(stderr, "%s\n", problem.c_str());
        return 1;
    }
    EventFileWriter events_file;
    if (!args.events.empty() && !events_file.Open(args.events, &problem)) {
        std::fprintf(stderr, "%s\n", problem.c_str());
        return 1;
    }
    for (std::size_t s = 0; s < reader.size(); ++s) {
        const std::string description = reader.input(s).input->getDescription();
        if (several) {
            std::printf("input:  %s: %s\n", reader.input(s).source.c_str(), description.c_str());
        } else {
            std::printf("input:  %s\n", description.c_str());
        }
    }
    // Flushed even into a pipe: this line says the frame loop is starting,
    // which is what a wrapper waiting to send Ctrl-C needs to know.
    std::fflush(stdout);

    const OutputKind output_kind = ClassifyOutput(args.output);
    std::unique_ptr<LayersWriter> layers;
    if (!args.output_layers.empty()) {
        layers.reset(new LayersWriter(args.output_layers,
                                      LayersStem(args.output, args.graph_path), spec));
    }
    // --display with several sources: one window, a grid of every stream's
    // latest frame; its cells are the size of the first frame rendered.
    std::unique_ptr<DisplayWindow> grid_window;
    std::vector<cv::Mat> latest(reader.size());
    std::vector<std::string> stream_names;
    cv::Size grid_cell;
    if (args.display && several) {
        grid_window.reset(new DisplayWindow(WindowName(std::string(), false)));
        for (std::size_t s = 0; s < reader.size(); ++s) {
            stream_names.push_back(reader.input(s).source);
        }
    }
    std::vector<StreamSink> sinks(reader.size());
    std::vector<std::deque<std::string> > recent(reader.size());  // the event panel, per stream
    for (std::size_t s = 0; s < reader.size(); ++s) {
        const StreamInput& input = reader.input(s);
        sinks[s].single_frame = input.input->getTotalFrames() == 1;
        if (output_kind == OutputKind::kVideo) {
            sinks[s].video.reset(new VideoOutput(
                StreamVideoPath(args.output, input.source, several), input.input->getFPS()));
        }
        if (args.display && !several) {
            sinks[s].window.reset(new DisplayWindow(WindowName(input.source, several)));
        }
    }
    int failed_frames = 0;
    std::set<std::size_t> warned_edges;
    RenderOptions render_options = RenderOptionsFor(spec);
    render_options.draw.insert(args.draw.begin(), args.draw.end());

    // Under pipelining the report that comes out is not necessarily for the
    // frame just read, so every report is rendered onto ITS OWN frame, which
    // the executor hands back with it, and into ITS OWN stream's sink.
    auto handle = [&](const FrameReport& report, const cv::Mat& shown) -> bool {
        const std::size_t stream_index = graph.StreamIndex(report.stream);
        StreamSink& sink = sinks.at(stream_index);
        if (!report.error.empty()) {
            ++failed_frames;
            ++sink.failed;
            if (several) {
                std::fprintf(stderr, "stream %s frame %u: %s\n", report.stream.c_str(),
                             static_cast<unsigned>(report.frame_index), report.error.c_str());
            } else {
                std::fprintf(stderr, "frame %u: %s\n",
                             static_cast<unsigned>(report.frame_index), report.error.c_str());
            }
        }
        PrintFrameLine(spec, report, several);
        // Each event right after its frame's line: on stdout always (Studio
        // reads it), in --events when asked, and in the window's panel.
        for (std::size_t e = 0; e < report.events.size(); ++e) {
            const std::string line = EventLineJson(report.events[e], report).dump();
            std::printf("[EVENT] %s\n", line.c_str());
            if (events_file.is_open() && !events_file.Append(line, &problem)) {
                std::fprintf(stderr, "%s\n", problem.c_str());
                return false;
            }
            // --draw: the panel lists the drawn nodes' events only.
            if (!render_options.Draws(report.events[e].node)) continue;
            std::deque<std::string>& panel = recent.at(stream_index);
            panel.push_back(EventPanelLine(report.events[e]));
            if (panel.size() > kRecentEvents) panel.pop_front();
        }
        if (!report.events.empty()) std::fflush(stdout);
        progress().tickNodes(ProgressNodeMs(spec, report));  // DXAPP_PROGRESS=1 only
        WarnAboutEmptyRoiEdges(spec, report, &warned_edges);
        if (report_file.is_open() &&
            !report_file.Append(ReportToJson(report, several, args.report_timing), &problem)) {
            std::fprintf(stderr, "%s\n", problem.c_str());
            return false;
        }
        if (output_kind != OutputKind::kNone || sink.window || grid_window || layers) {
            std::vector<cv::Rect> labels;
            cv::Mat rendered = RenderReport(shown, report, render_options, &labels);
            const std::vector<std::string> panel(recent.at(stream_index).begin(),
                                                 recent.at(stream_index).end());
            DrawEventLog(&rendered, panel, labels);
            if (layers && !layers->Write(report, shown, render_options, panel, labels, several,
                                         &problem)) {
                std::fprintf(stderr, "%s\n", problem.c_str());
                return false;
            }
            if (sink.video) {
                if (!sink.video->Write(rendered, &problem)) {
                    std::fprintf(stderr, "%s\n", problem.c_str());
                    return false;
                }
            } else if (output_kind == OutputKind::kImagePerFrame) {
                const std::string path = StreamImagePath(args.output, report.stream, several,
                                                         sink.single_frame, report.frame_index);
                if (!cv::imwrite(path, rendered)) {
                    std::fprintf(stderr, "could not write %s\n", path.c_str());
                    return false;
                }
                std::printf("output: %s\n", path.c_str());
            }
            if (sink.window && sink.window->Show(rendered)) RequestStop();  // q / ESC
            if (grid_window) {
                if (grid_cell.area() == 0) grid_cell = rendered.size();
                latest.at(stream_index) = rendered;
                if (grid_window->Show(TileFrames(latest, stream_names, grid_cell))) {
                    RequestStop();
                }
            }
        }
        return true;
    };

    std::size_t stream = 0;
    std::size_t index = 0;
    cv::Mat frame;
    FrameReport report;
    cv::Mat shown;
    // avg_fps counts from here (model load is out, pipeline fill is in): an input
    // shorter than --max-inflight returns every report in one burst at Finish().
    progress().start();
    std::vector<std::size_t> restarts(reader.size(), 0);
    while (!Interrupted() && reader.Next(&stream, &frame, &index)) {
        // --loop: a video that started again starts its tracks and its logic
        // nodes' state again too (from this frame on, in frame order).
        if (reader.restarts(stream) != restarts[stream]) {
            restarts[stream] = reader.restarts(stream);
            graph.RestartStream(stream);
        }
        executor.Submit(graph, stream, frame, index);
        while (executor.TryNext(&report, &shown)) {
            if (!handle(report, shown)) {
                executor.Finish(graph);
                return 1;
            }
        }
    }
    executor.Finish(graph);
    while (executor.TryNext(&report, &shown)) {
        if (!handle(report, shown)) return 1;
    }

    std::size_t total = 0;
    for (std::size_t s = 0; s < reader.size(); ++s) total += reader.frames_read(s);
    // A stream that ended without a frame (spec R9): the others ran to their
    // end first, and their outputs, report and summary lines follow as in
    // any finished run; the exit code is 1. With one source there is nothing
    // else to finish, so the run ends here, as before streams existed.
    bool no_frames = false;
    if (total == 0 && Interrupted()) {
        // A stop request that came while the models loaded ends the run
        // before the loop reads anything: not any input's fault, and the
        // same graceful stop as one mid-run (the report is "frames": []).
        std::fprintf(stderr, "interrupted before the first frame\n");
    } else {
        // One that a stop request kept from its turn has not ended, and is
        // not blamed.
        for (std::size_t s = 0; s < reader.size(); ++s) {
            if (reader.frames_read(s) != 0 || !reader.ended(s)) continue;
            std::fprintf(stderr, "%s\n", NoFramesError(reader.input(s).uri).what());
            no_frames = true;
        }
        if (no_frames && !several) return 1;
    }

    for (std::size_t s = 0; s < reader.size(); ++s) {
        if (!sinks[s].video) continue;
        sinks[s].video->Close();  // finalizes the container
        if (sinks[s].video->frames_written() != 0) {
            std::printf("output: %s\n",
                        StreamVideoPath(args.output, reader.input(s).source, several).c_str());
        }
    }

    if (report_file.is_open()) {
        if (!report_file.Finish(&problem)) {
            std::fprintf(stderr, "%s\n", problem.c_str());
            return 1;
        }
        std::printf("report: %s\n", args.report.c_str());
    }
    if (layers) std::printf("layers: %s\n", layers->manifest_path().c_str());

    if (several) {
        for (std::size_t s = 0; s < reader.size(); ++s) {
            const std::size_t read = reader.frames_read(s);
            std::printf("stream %s: %u frame%s, %d failed\n", reader.input(s).source.c_str(),
                        static_cast<unsigned>(read), read == 1 ? "" : "s", sinks[s].failed);
        }
    }
    progress().finish();  // the run averages, before the summary line
    std::printf("%u frame%s, %d failed\n", static_cast<unsigned>(total),
                total == 1 ? "" : "s", failed_frames);
    return failed_frames == 0 && !no_frames ? 0 : 1;
}

}  // namespace

int Main(int argc, char** argv, ExecutorKind kind) {
    GraphCliArgs args;
    std::string error;
    if (!ParseArgs(argc, argv, kind, &args, &error)) {
        // Searched in argv, not args.json: ParseArgs may stop before it
        // reads a later --json.
        bool json = false;
        for (int i = 1; i < argc; ++i) json = json || std::strcmp(argv[i], "--json") == 0;
        std::fprintf(stderr, "%s\n\n", error.c_str());
        PrintUsage(json ? stderr : stdout, argv[0]);
        return 2;
    }
    if (args.help) {
        PrintUsage(stdout, argv[0]);
        return 0;
    }
    if (args.list_ops) return args.json ? PrintOpsJson() : PrintOps();

    // The constructor populates itself from the generated tables; this
    // one line is the entire cost of naming a concrete registry.
    StaticModelRegistry registry;

    if (args.list_models) {
        return args.json ? PrintModelsJson(registry, args) : PrintModels(registry, args);
    }

    // First Ctrl-C or SIGTERM: stop reading, finish in-flight frames, write
    // the report. A later Ctrl-C terminates. See graph_cli_interrupt.hpp.
    InstallInterruptHandler();

    try {
        if (args.check_only && args.json) return PrintCheckJson(args, registry);
        if (args.check_only) {
            GraphSpec spec = ParseGraphFile(args.graph_path);
            ValidateGraph(spec, registry);                      // T0
            // T1 is reported, not enforced: the exit code answers "is this
            // graph valid?", which is a question a checkout with no models
            // downloaded must still be able to ask. A present file that
            // cannot load here (not a .dxnn, or R12) is the exception: exit 1.
            PrintCheckSummary(spec, registry, args.graph_path, args.model_dir);
            std::fflush(stdout);
            const std::vector<GraphError> problems =
                ModelFileProblems(spec, registry, args.model_dir);
            for (std::size_t i = 0; i < problems.size(); ++i) {
                std::fprintf(stderr, "%s\n", problems[i].what());
            }
            return problems.empty() ? 0 : 1;
        }

        // Usage errors that need the graph's uri or the environment, found
        // before any model loads or any device opens.
        if (args.display && !DisplayAvailable()) {
            std::fprintf(stderr, "--display needs a window system, and neither DISPLAY "
                                 "nor WAYLAND_DISPLAY is set: run it in a desktop session, "
                                 "or save the result with --output\n\n");
            PrintUsage(stdout, argv[0]);
            return 2;
        }
        if (args.display && !DisplayOpens()) {
            std::fprintf(stderr, "--display could not open a window with %s: no window "
                                 "system answered there; run it in a desktop session, or "
                                 "save the result with --output\n\n",
                         DisplayEnvironment().c_str());
            PrintUsage(stdout, argv[0]);
            return 2;
        }
        // Which uri every stream reads, and the usage errors that depend on
        // it, before any model loads or any device opens (SP2). A graph file
        // that does not parse throws here the GraphError PrepareGraph threw
        // before: same message, same exit 1.
        const GraphSpec early = ParseGraphFile(args.graph_path);
        std::map<std::string, std::string> overrides;
        const std::string binding = ParseInputOverrides(early, args.inputs, &overrides);
        if (!binding.empty()) {
            std::fprintf(stderr, "%s\n\n", binding.c_str());
            PrintUsage(stdout, argv[0]);
            return 2;
        }
        std::vector<StreamUri> streams;
        const std::string unresolved = ResolveStreams(early, overrides, &streams);
        if (!unresolved.empty()) {
            // ParseInputOverrides binds only source ids, so this is not
            // expected; if it ever happens it is a usage error all the same.
            std::fprintf(stderr, "%s\n\n", unresolved.c_str());
            PrintUsage(stdout, argv[0]);
            return 2;
        }
        const std::string undrawable = DrawProblem(early, args.draw);
        if (!undrawable.empty()) {
            std::fprintf(stderr, "%s\n\n", undrawable.c_str());
            PrintUsage(stdout, argv[0]);
            return 2;
        }
        if (!args.output_layers.empty()) {
            for (std::size_t s = 0; s < streams.size(); ++s) {
                const std::string problem = LayersSourceProblem(streams[s].uri, args.frames);
                if (!problem.empty()) {
                    std::fprintf(stderr, "%s\n\n", problem.c_str());
                    PrintUsage(stdout, argv[0]);
                    return 2;
                }
            }
        }
        if (!args.output.empty()) {
            // SP1's live-source refusal, for each stream's own uri.
            for (std::size_t s = 0; s < streams.size(); ++s) {
                const std::string problem =
                    OutputProblem(args.output, streams[s].uri, args.frames);
                if (!problem.empty()) {
                    std::fprintf(stderr, "%s\n\n", problem.c_str());
                    PrintUsage(stdout, argv[0]);
                    return 2;
                }
            }
            // Only a video file loops: images, cameras and RTSP end (or not) as without --loop.
            bool loops = false;
            for (std::size_t s = 0; s < streams.size(); ++s) {
                loops = loops || (!IsLiveUri(streams[s].uri) &&
                                  InputFactory::detectFileType(streams[s].uri) == InputType::VIDEO);
            }
            if (args.loop && loops && args.frames == 0 &&
                ClassifyOutput(args.output) == OutputKind::kImagePerFrame) {
                std::fprintf(stderr,
                             "--loop never ends a video, and --output %s writes one image per "
                             "frame: write a video instead (--output <stem>.mkv, .avi or .mp4) "
                             "or stop after N frames (--frames N)\n\n",
                             args.output.c_str());
                PrintUsage(stdout, argv[0]);
                return 2;
            }
            // Not a usage error: exit 1, as an unwritable --report. Checked
            // on the first file each stream writes; with one source that is
            // --output itself, as before streams existed.
            const bool several = streams.size() > 1;
            const bool video = ClassifyOutput(args.output) == OutputKind::kVideo;
            for (std::size_t s = 0; s < streams.size(); ++s) {
                const std::string path =
                    !several ? args.output
                    : video  ? StreamVideoPath(args.output, streams[s].source, several)
                             : StreamImagePath(args.output, streams[s].source, several,
                                               false, 0);
                const std::string directory = OutputDirectoryProblem(path);
                if (!directory.empty()) {
                    std::fprintf(stderr, "%s\n", directory.c_str());
                    return 1;
                }
            }
        }

        if (!args.output_layers.empty()) {
            const std::string directory = LayersDirectoryProblem(args.output_layers);
            if (!directory.empty()) {
                std::fprintf(stderr, "%s\n", directory.c_str());
                return 1;
            }
        }

        // T0, then T1 (fatal here), then Build - the same sequence the
        // dx_graph module runs.
        GraphSpec spec;
        StageGraph graph;
        PrepareGraph(args.graph_path, args.model_dir, registry, &spec, &graph);
        std::printf("graph:  %s (%u nodes, %u edges)\n",
                    spec.name.empty() ? args.graph_path.c_str()
                                      : spec.name.c_str(),
                    static_cast<unsigned>(spec.nodes.size()),
                    static_cast<unsigned>(spec.edges.size()));
        std::fflush(stdout);  // the graph is built: visible through a pipe

        std::unique_ptr<FrameExecutor> executor;
        if (kind == kAsyncExecutor) {
            AsyncOptions options;
            options.max_frames_in_flight = args.max_inflight;
            options.max_jobs_per_stage = args.max_jobs_per_stage;
            options.stall_timeout_ms = args.stall_timeout_ms;
            options.interrupted = [] { return Interrupted(); };
            options.on_stuck = [](const std::vector<std::string>& ids) {
                std::fprintf(stderr, "%s\n", StuckMessage(ids).c_str());
            };
            executor.reset(new AsyncFrameExecutor(options));
        } else {
            executor.reset(new SyncFrameExecutor());
        }
        return RunInputLoop(graph, *executor, spec, args, streams);  // T2
    } catch (const MissingModelsError& missing) {
        // Already the complete message, list and trailing newline included.
        std::fputs(missing.what(), stderr);
        return 1;
    } catch (const GraphError& graph_error) {
        std::fprintf(stderr, "%s\n", graph_error.what());
        return 1;
    } catch (const std::exception& other) {
        std::fprintf(stderr, "ERROR: %s\n", other.what());
        return 1;
    }
}

ProjectCall ProjectArgs(const std::vector<std::string>& argv, const ProjectDefaults& defaults,
                        const std::string& root) {
    ProjectCall call;
    bool graph_given = false;
    bool model_dir_given = false;
    bool listing = false;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        const std::string& word = argv[i];
        if (i > 0 && word == "--sync") {
            call.kind = kSyncExecutor;
            continue;
        }
        call.args.push_back(word);
        if (i == 0) continue;
        if (word == "--graph") {
            graph_given = true;
        } else if (word == "--model-dir") {
            model_dir_given = true;
        } else if (word == "--check") {
            // "--check g.json" names the graph; "--check --json" does not.
            graph_given = graph_given || (i + 1 < argv.size() && !argv[i + 1].empty() &&
                                          argv[i + 1].compare(0, 2, "--") != 0);
        } else if (word == "--list-models" || word == "--list-ops" || word == "--help" ||
                   word == "-h") {
            listing = true;
        }
    }
    if (call.args.empty()) call.args.push_back("graph_project");
    if (!model_dir_given && !defaults.model_dir.empty()) {
        call.args.push_back("--model-dir");
        call.args.push_back(::dxapp::ResolveAgainstRoot(root, defaults.model_dir));
    }
    if (!graph_given && !listing && !defaults.graph.empty()) {
        call.args.push_back("--graph");
        call.args.push_back(::dxapp::ResolveAgainstRoot(root, defaults.graph));
    }
    return call;
}

int ProjectMain(int argc, char** argv, const ProjectDefaults& defaults) {
    const ProjectCall call =
        ProjectArgs(std::vector<std::string>(argv, argv + argc), defaults, ::dxapp::RuntimeRoot());
    std::vector<std::string> args = call.args;  // owns the text the pointers below point at
    std::vector<char*> pointers;
    for (std::size_t i = 0; i < args.size(); ++i) pointers.push_back(&args[i][0]);
    pointers.push_back(NULL);
    return Main(static_cast<int>(args.size()), pointers.data(), call.kind);
}

}  // namespace cli
}  // namespace graph
}  // namespace dxapp
