/**
 * @file graph_config.hpp
 * @brief Graph file structs, parsing and validation.
 *
 * Four node kinds (source, model, cpu, logic) and three edge kinds (frame,
 * ROI, result) express every pipeline this example supports. A result edge
 * is accepted only by a cpu or a logic node. Keeping that surface small is
 * the point: the schema is the only interface a customer sees.
 */
#ifndef DXAPP_GRAPH_GRAPH_CONFIG_HPP
#define DXAPP_GRAPH_GRAPH_CONFIG_HPP

#include <map>
#include <string>
#include <vector>

#include "common/graph/graph_error.hpp"
#include "common/graph/i_registry.hpp"

namespace dxapp {
namespace graph {

/// Options on an edge that carries cropped regions. All optional.
struct RoiSpec {
    bool present;
    std::vector<std::string> classes;
    float min_score;  ///< < 0 means "inherit the producer's threshold"
    float min_area;
    float pad;
    int max;          ///< < 0 means unlimited
    std::string align;  ///< "" or "face5"

    RoiSpec() : present(false), min_score(-1.f), min_area(0.f), pad(0.f), max(-1) {}
};

/// Tracking declared on a producer node, so every consumer shares one id space.
struct TrackSpec {
    bool present;
    std::string algo;
    float iou;
    int max_age;

    TrackSpec() : present(false), algo("iou"), iou(0.3f), max_age(30) {}
};

/// NPU engine settings of a model node ("engine"). All optional.
struct EngineSpec {
    bool present;
    int buffers;  ///< DX-RT I/O buffer count, 1..100; 0 = DX-RT's default (6)
    bool share;   ///< one engine for every share node of the same model variant

    EngineSpec() : present(false), buffers(0), share(false) {}
};

struct NodeSpec {
    std::string id;
    bool is_source;
    bool is_cpu;
    bool is_logic;
    std::string uri;    ///< source only
    std::string model;  ///< model only
    std::string op;     ///< cpu: "headpose" or "volume"; logic: a LogicOps() name
    StageParams params;
    /// Logic only: params that are arrays of numbers (a zone's "polygon", a
    /// line's "line"), fractions 0..1 of the source frame.
    std::map<std::string, std::vector<double> > arrays;
    TrackSpec track;
    EngineSpec engine;

    NodeSpec() : is_source(false), is_cpu(false), is_logic(false) {}
};

/// True for a model node: not a source, a cpu node or a logic node.
inline bool IsModelNode(const NodeSpec& node) {
    return !node.is_source && !node.is_cpu && !node.is_logic;
}

struct EdgeSpec {
    std::string from;
    std::string to;
    RoiSpec roi;
    std::string port;  ///< "" = primary
    bool carry_result;  ///< true: the producer payload, consumed only by a cpu node

    EdgeSpec() : carry_result(false) {}
};

struct GraphSpec {
    int version;
    std::string name;
    std::vector<NodeSpec> nodes;
    std::vector<EdgeSpec> edges;

    GraphSpec() : version(0) {}

    /// NULL when no node carries this id.
    const NodeSpec* FindNode(const std::string& id) const;
};

/// One stream (SP2): a source node and every node reachable from it.
struct StreamSpec {
    std::string source;              ///< the source node's id
    std::vector<std::string> nodes;  ///< its members, in node declaration order; `source` is one
};

/// One stream per source node, in node declaration order. Edges whose
/// endpoints name no node are ignored, so the result is meaningful once
/// ValidateGraph's step 2 has passed, and on every validated spec. The only
/// reachability the stream model uses: ValidateGraph, --check and
/// StageGraph::Build all call it.
std::vector<StreamSpec> ListStreams(const GraphSpec& spec);

/// Node id -> the id of the node whose engine it uses, for every model node
/// with "engine": {"share": true} whose model resolves: the first such node
/// (declaration order) of the same model variant, itself included. Nodes
/// that do not share are absent. StageGraph::Build and --check --json use it.
std::map<std::string, std::string> SharedEngineOwners(const GraphSpec& spec,
                                                      const IModelRegistry& registry);

/// The model node whose boxes reach node `id` through filter, zone and user
/// nodes (their "carry": "result" input): `id` itself for a model node;
/// NULL for a source, a cpu node, a logic node of another op, or an unknown
/// id. Bounded by the node count, so a cyclic graph cannot loop it.
const NodeSpec* BoxSourceNode(const GraphSpec& spec, const std::string& id);

/// What node `id` hands on: a source a frame, a model its registry shape,
/// a cpu node records, a logic node records, or (filter, zone, user) the
/// shape of its input. kRecords for an unknown id or an unresolved model.
Shape NodeOutputShape(const GraphSpec& spec, const IModelRegistry& registry,
                      const std::string& id);

/// Syntax only. Throws GraphError with kGraphSchema, kGraphVersion or
/// kGraphReserved. origin appears in the message so the reader knows the file.
GraphSpec ParseGraphText(const std::string& json_text, const std::string& origin);

/// Reads the file, then ParseGraphText. Throws kGraphSchema when unreadable.
GraphSpec ParseGraphFile(const std::string& path);

/// Semantics. Throws GraphError. Implemented in Task 7.
void ValidateGraph(const GraphSpec& spec, const IModelRegistry& registry);

/// An event name: 1 to 64 letters, digits, _ . : - (a condition's "event",
/// an event a user node's code raises).
bool IsEventName(const std::string& text);

/// A "user" node's "name": a C identifier of at most 64 characters that is
/// not a C++ keyword and not reserved ("__" anywhere, "_" then a capital).
/// An exported project (scripts/export_graph_project.py) makes it a class
/// name; CppKeyword's list there is CPP_KEYWORDS in that script.
bool IsUserLogicName(const std::string& text);

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_GRAPH_CONFIG_HPP
