/**
 * @file graph_cli.hpp
 * @brief Everything both multi-model-graph binaries share.
 *
 * WHY THIS FILE IS NOT UNDER common/graph/
 * ----------------------------------------
 * This is the engine's CONSUMER, not the engine. It constructs a concrete
 * registry (common/registry/static_model_registry.hpp) and that is exactly
 * what scripts/check_graph_boundary.py exists to keep out of
 * common/graph/. The guard walks every file under that directory, so a
 * graph_cli.cpp placed there would either fail the guard or force
 * static_model_registry.hpp onto ENGINE_INCLUDE_ALLOWLIST, which would
 * open the allowlist to every engine file and defeat the guard entirely.
 * Living here, next to the two mains it serves, the CLI can name the
 * concrete registry the way it is meant to: once, at the top of main().
 *
 * The two binaries differ in exactly one value - which executor runs a
 * frame - so everything else lives in graph_cli.cpp and each main is six
 * lines.
 */
#ifndef DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_HPP
#define DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_HPP

#include <string>
#include <vector>

namespace dxapp {
namespace graph {
namespace cli {

/// Which executor Main() constructs. The only difference between the two
/// shipped binaries.
enum ExecutorKind { kSyncExecutor, kAsyncExecutor };

/**
 * @brief Parse arguments, build the graph, run it. The whole program.
 *
 * Exit codes, which scripts depend on:
 *   0  success, or --help / --list-models / a --check whose graph is valid
 *   1  a GraphError: the graph, or the models it names, cannot run
 *   2  a usage error: an unknown option, a missing value, no graph
 */
int Main(int argc, char** argv, ExecutorKind kind);

/// What an exported graph project's main() fixes
/// (scripts/export_graph_project.py): the graph and the model folder used
/// when the command line names none. Relative paths are read against
/// RuntimeRoot() (common/utility/repo_path.hpp).
struct ProjectDefaults {
    std::string graph;      ///< "graph.json"
    std::string model_dir;  ///< "models"
};

/// The executor and the arguments ProjectMain hands to Main.
struct ProjectCall {
    ExecutorKind kind;
    std::vector<std::string> args;  ///< args[0] is the program
    ProjectCall() : kind(kAsyncExecutor) {}
};

/**
 * @brief An exported project's command line as Main reads it.
 *
 * `--sync` is taken out and picks the sync executor (async otherwise).
 * `--model-dir <root>/<model_dir>` is added unless one is given;
 * `--graph <root>/<graph>` too, unless a graph is given (`--graph`, or a
 * value after `--check`) or the call only lists (`--list-models`,
 * `--list-ops`, `--help`).
 */
ProjectCall ProjectArgs(const std::vector<std::string>& argv, const ProjectDefaults& defaults,
                        const std::string& root);

/// main() of an exported project: ProjectArgs against RuntimeRoot(), then
/// Main. Exit codes as Main.
int ProjectMain(int argc, char** argv, const ProjectDefaults& defaults);

}  // namespace cli
}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_HPP
