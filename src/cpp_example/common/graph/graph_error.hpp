/**
 * @file graph_error.hpp
 * @brief One error type with a fixed three-part message.
 *
 * Every message names WHERE (node or edge id), WHAT (the fact), and HOW (the
 * recovery command). That format is the practical condition for a customer to
 * edit a graph file without help.
 */
#ifndef DXAPP_GRAPH_GRAPH_ERROR_HPP
#define DXAPP_GRAPH_GRAPH_ERROR_HPP

#include <stdexcept>
#include <string>

namespace dxapp {
namespace graph {

enum class GraphErrorCode {
    kGraphVersion,    ///< schema version this build does not support
    kGraphSchema,     ///< malformed JSON, wrong type, missing or duplicate key
    kGraphEdge,       ///< shape or input-contract mismatch across an edge
    kGraphAlign,      ///< align requested from a producer without landmarks
    kGraphCycle,      ///< the node graph is not acyclic
    kGraphOrphan,     ///< a node unreachable from any source
    kGraphReserved,   ///< a key reserved for a later release
    kModelUnknown,    ///< not present in the model registry config file
    kModelNoTask,     ///< registered without a task
    kModelNotReady,   ///< registered but no factory or no postprocessor
    kModelMissing,    ///< the .dxnn artifact is absent from --model-dir
    kModelLoad        ///< the .dxnn is there but the runtime could not load it
                      ///< (device memory, device state, an incompatible .dxnn)
};

const char* ToString(GraphErrorCode code);

class GraphError : public std::runtime_error {
 public:
    GraphError(GraphErrorCode code, const std::string& where,
               const std::string& what_happened, const std::string& how_to_fix)
        : std::runtime_error(Format(code, where, what_happened, how_to_fix)),
          code_(code),
          where_(where),
          what_happened_(what_happened),
          how_to_fix_(how_to_fix) {}

    GraphErrorCode code() const { return code_; }
    /// The three parts what() joins, for a caller that reports them apart
    /// (--check --json).
    const std::string& where() const { return where_; }
    const std::string& what_happened() const { return what_happened_; }
    const std::string& how_to_fix() const { return how_to_fix_; }

 private:
    static std::string Format(GraphErrorCode code, const std::string& where,
                              const std::string& what_happened,
                              const std::string& how_to_fix) {
        std::string message = "ERROR [";
        message += ToString(code);
        message += "] ";
        message += where;
        message += ": ";
        message += what_happened;
        if (!how_to_fix.empty()) {
            message += "\n  -> ";
            message += how_to_fix;
        }
        return message;
    }

    GraphErrorCode code_;
    std::string where_;
    std::string what_happened_;
    std::string how_to_fix_;
};

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_GRAPH_ERROR_HPP
