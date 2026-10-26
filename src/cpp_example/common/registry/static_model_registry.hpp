/**
 * @file static_model_registry.hpp
 * @brief IModelRegistry backed by the generated tables.
 *
 * Swapping this for a dlopen-based registry changes nothing above it: the
 * graph schema, the executors, the router and the visualizer all name only
 * i_registry.hpp.
 *
 * WHY THIS FILE IS NOT UNDER common/graph/
 * ----------------------------------------
 * This is a CONCRETE registry, and common/graph/ is the engine.
 * scripts/check_graph_boundary.py's whole purpose is to stop engine code
 * from naming or including one, so the concrete registry lives outside the
 * directory the guard polices, not inside it with an exemption. In
 * particular it is deliberately NOT added to ENGINE_INCLUDE_ALLOWLIST:
 * allowlisting it would make every engine file free to include it and would
 * defeat the guard entirely. Where it is now, two independent checks fire if
 * an engine file ever includes it - it is outside the engine's allowed
 * include surface, and its path names a registry - which is strictly
 * stronger than the allowlist entry the boundary guard was written to
 * prevent.
 *
 * Only two kinds of translation unit include this header: the generated
 * build/generated/graph_registry_*.cpp tables, and the test binary under
 * common/graph/test/ (which the guard exempts because fixtures legitimately
 * construct a concrete registry).
 */
#ifndef DXAPP_REGISTRY_STATIC_MODEL_REGISTRY_HPP
#define DXAPP_REGISTRY_STATIC_MODEL_REGISTRY_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/graph/i_registry.hpp"

namespace dxapp {
namespace graph {

/// How the generated tables hand a model's constructor to the registry.
/// Deliberately a plain function pointer, not a std::function: the generated
/// tables hold one per model as static data, and a function pointer costs
/// nothing to store.
typedef std::unique_ptr<IStage> (*StageMaker)(const std::string& model_path,
                                              const ModelInfo& info,
                                              const StageParams& params);

class StaticModelRegistry : public IModelRegistry {
 public:
    /// Populates itself from the generated tables.
    StaticModelRegistry();

    /// Called once per model by the generated tables. maker may be NULL for
    /// a model registered without a usable factory (info.ready is false and
    /// info.not_ready_reason says why).
    void Add(const ModelInfo& info, StageMaker maker);

    /// Called by the generated tables after every Add() (R6): `name` resolves
    /// to the model whose key is `variant`. A model's own key always wins
    /// over an alias of the same spelling, and an alias of a variant that
    /// was never added resolves to nothing; scripts/gen_model_registry.py
    /// refuses both, so either means the tables were edited by hand.
    void AddAlias(const std::string& name, const std::string& variant, ModelAlias::Kind kind);

    const ModelInfo* find(const std::string& model_name) const;
    std::vector<ModelInfo> list() const;
    std::vector<ModelAlias> aliases() const;
    std::unique_ptr<IStage> createStage(const std::string& model_name,
                                        const std::string& model_path,
                                        const StageParams& params) const;
    /// createStage with the node's buffer count in effect while the engine opens.
    std::unique_ptr<IStage> createStageWith(const std::string& model_name,
                                            const std::string& model_path,
                                            const StageParams& params,
                                            const EngineOptions& engine) const;

 private:
    std::map<std::string, ModelInfo> infos_;
    std::map<std::string, StageMaker> makers_;
    std::vector<std::string> order_;  ///< registration order, for list()
    std::map<std::string, std::string> alias_variant_;  ///< alias -> variant
    std::vector<ModelAlias> aliases_;                   ///< registration order

    /// The model key `name` stands for: itself, or the variant it aliases.
    const std::string& Resolve(const std::string& name) const;
};

/// Defined in the generated graph_registry_all.cpp.
void RegisterAllGraphModels(StaticModelRegistry* registry);

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_REGISTRY_STATIC_MODEL_REGISTRY_HPP
