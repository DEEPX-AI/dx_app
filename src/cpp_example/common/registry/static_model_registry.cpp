#include "common/registry/static_model_registry.hpp"
#include "common/registry/engine_options.hpp"

#include <cstddef>
#include <stdexcept>

namespace dxapp {
namespace graph {

StaticModelRegistry::StaticModelRegistry() {
    RegisterAllGraphModels(this);
}

void StaticModelRegistry::Add(const ModelInfo& info, StageMaker maker) {
    // Last registration wins, but the name is listed once. The generator
    // refuses to emit a duplicate model_name, so reaching the second branch
    // means the tables were edited by hand.
    if (infos_.find(info.model_name) == infos_.end()) {
        order_.push_back(info.model_name);
    }
    infos_[info.model_name] = info;
    makers_[info.model_name] = maker;
}

void StaticModelRegistry::AddAlias(const std::string& name, const std::string& variant,
                                   ModelAlias::Kind kind) {
    if (alias_variant_.find(name) == alias_variant_.end()) {
        aliases_.push_back(ModelAlias(name, variant, kind));
    } else {
        for (std::size_t i = 0; i < aliases_.size(); ++i) {
            if (aliases_[i].name == name) aliases_[i] = ModelAlias(name, variant, kind);
        }
    }
    alias_variant_[name] = variant;
}

const std::string& StaticModelRegistry::Resolve(const std::string& name) const {
    if (infos_.find(name) != infos_.end()) return name;
    std::map<std::string, std::string>::const_iterator alias = alias_variant_.find(name);
    return alias == alias_variant_.end() ? name : alias->second;
}

const ModelInfo* StaticModelRegistry::find(const std::string& model_name) const {
    std::map<std::string, ModelInfo>::const_iterator it =
        infos_.find(Resolve(model_name));
    if (it == infos_.end()) return NULL;
    return &it->second;
}

std::vector<ModelInfo> StaticModelRegistry::list() const {
    std::vector<ModelInfo> all;
    all.reserve(order_.size());
    for (std::size_t i = 0; i < order_.size(); ++i) {
        std::map<std::string, ModelInfo>::const_iterator it =
            infos_.find(order_[i]);
        if (it != infos_.end()) all.push_back(it->second);
    }
    return all;
}

std::vector<ModelAlias> StaticModelRegistry::aliases() const {
    return aliases_;
}

std::unique_ptr<IStage> StaticModelRegistry::createStage(
    const std::string& model_name, const std::string& model_path,
    const StageParams& params) const {
    // An alias builds its variant's stage; the messages keep the name the
    // caller used.
    const std::string& key = Resolve(model_name);
    std::map<std::string, ModelInfo>::const_iterator info =
        infos_.find(key);
    if (info == infos_.end()) {
        throw std::runtime_error("unknown model \"" + model_name + "\"");
    }
    if (!info->second.ready) {
        throw std::runtime_error("model \"" + model_name +
                                 "\" is not usable in a graph: " +
                                 info->second.not_ready_reason);
    }
    std::map<std::string, StageMaker>::const_iterator maker =
        makers_.find(key);
    if (maker == makers_.end() || maker->second == NULL) {
        throw std::runtime_error("model \"" + model_name +
                                 "\" is marked ready but has no constructor");
    }
    // The runtime reports a missing artifact, an incompatible .dxnn and a
    // device failure as dxrt::Exception (a std::runtime_error) - but a
    // factory or a postprocessor may throw anything. IModelRegistry
    // documents this as std::runtime_error, so everything is normalised
    // here. The text is the runtime's own, unprefixed: the caller's
    // MODEL_LOAD already names the node, the model and its .dxnn, so a
    // prefix here printed the model twice.
    // A container this DX-RT cannot load keeps its type (and hint): Build
    // must not offer to download the same file again.
    try {
        return maker->second(model_path, info->second, params);
    } catch (const ModelContainerError&) {
        throw;
    } catch (const std::exception& error) {
        throw std::runtime_error(error.what());
    } catch (...) {
        throw std::runtime_error("unknown error");
    }
}

std::unique_ptr<IStage> StaticModelRegistry::createStageWith(
    const std::string& model_name, const std::string& model_path,
    const StageParams& params, const EngineOptions& engine) const {
    detail::EngineBufferScope scope(engine.buffers);
    return createStage(model_name, model_path, params);
}

}  // namespace graph
}  // namespace dxapp
