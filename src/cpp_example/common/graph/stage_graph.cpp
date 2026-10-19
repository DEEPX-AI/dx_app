#include "common/graph/stage_graph.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace dxapp {
namespace graph {
namespace {

bool FileExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

bool SameKeypoint(const Keypoint& a, const Keypoint& b) {
    return a.x == b.x && a.y == b.y && a.confidence == b.confidence;
}

/**
 * @brief Exact, element-by-element equality for any dense cv::Mat payload.
 *
 * Shared by every Mat-shaped payload (kFrame's image, kLabelMap's label
 * indices, kDenseMap's float values) so those three cannot drift apart on
 * what "the same pixels" means.
 *
 * No epsilon, and none wanted: sync and async run the identical stage on
 * the identical input, so any difference is a scheduling bug. NaN inherits
 * the rule the file documents elsewhere - absdiff(NaN, NaN) is NaN, which
 * countNonZero counts, so a NaN anywhere can never satisfy parity. That
 * matters most for kDenseMap, the one CV_32F payload here.
 */
bool SameMat(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size()) return false;
    if (a.type() != b.type()) return false;
    if (a.empty()) return true;  // both empty: size() equality above
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(diff.reshape(1)) == 0;
}

bool SameFrameData(const FrameData& a, const FrameData& b) {
    return SameMat(a.image, b.image);
}

bool SameBoxItem(const BoxItem& a, const BoxItem& b) {
    if (a.box != b.box) return false;
    if (a.score != b.score) return false;
    if (a.class_id != b.class_id) return false;
    if (a.class_name != b.class_name) return false;
    if (a.track_id != b.track_id) return false;
    if (a.angle != b.angle) return false;
    if (a.landmarks.size() != b.landmarks.size()) return false;
    for (std::size_t i = 0; i < a.landmarks.size(); ++i) {
        if (!SameKeypoint(a.landmarks[i], b.landmarks[i])) return false;
    }
    // Instance masks (kInstances): one with a mask and one without never
    // match; two masks are compared pixel for pixel, exactly, with the same
    // SameMat every other Mat-shaped payload uses.
    if (a.mask.empty() != b.mask.empty()) return false;
    if (!a.mask.empty() && !SameMat(a.mask, b.mask)) return false;
    return true;
}

bool SameBoxesData(const BoxesData& a, const BoxesData& b) {
    if (a.items.size() != b.items.size()) return false;
    for (std::size_t i = 0; i < a.items.size(); ++i) {
        if (!SameBoxItem(a.items[i], b.items[i])) return false;
    }
    return true;
}

/// Every one of Detection3DResult's 14 fields, exactly.
bool SameDetection3D(const Detection3DResult& a, const Detection3DResult& b) {
    if (a.class_id != b.class_id) return false;
    if (a.class_name != b.class_name) return false;
    if (a.confidence != b.confidence) return false;
    if (a.bev_x != b.bev_x) return false;
    if (a.bev_y != b.bev_y) return false;
    if (a.bev_w != b.bev_w) return false;
    if (a.bev_h != b.bev_h) return false;
    if (a.x3d != b.x3d) return false;
    if (a.y3d != b.y3d) return false;
    if (a.z3d != b.z3d) return false;
    if (a.dim_h != b.dim_h) return false;
    if (a.dim_w != b.dim_w) return false;
    if (a.dim_l != b.dim_l) return false;
    if (a.yaw != b.yaw) return false;
    return true;
}

bool SameBoxes3dData(const Boxes3dData& a, const Boxes3dData& b) {
    if (a.items.size() != b.items.size()) return false;
    for (std::size_t i = 0; i < a.items.size(); ++i) {
        if (!SameDetection3D(a.items[i], b.items[i])) return false;
    }
    return true;
}

bool SameRecords(const RecordsData& a, const RecordsData& b) {
    if (a.items.size() != b.items.size()) return false;
    for (std::size_t i = 0; i < a.items.size(); ++i) {
        if (a.items[i].numbers != b.items[i].numbers) return false;
        if (a.items[i].text != b.items[i].text) return false;
    }
    return true;
}

bool SameVectorData(const VectorData& a, const VectorData& b) {
    return a.values == b.values;  // vector<float>::operator==, exact
}

/**
 * @brief Exact equality for a classifier's scores, top-k included.
 *
 * Added in Task 11, when the first parity fixture running a REAL
 * classification stage (a resnet50 ROI consumer behind a real detector)
 * made kScores reachable. Before that, kScores sat in the fail-closed
 * group below and any parity test producing one would have failed
 * regardless of what the two executors did - which is the correct default,
 * and is exactly what happened the first time this fixture ran.
 *
 * top_k is compared, not skipped: it is the part of a ClassificationResult
 * a downstream consumer is most likely to read, and a comparator that
 * ignored it would call two differently-ordered top-k lists equal.
 */
bool SameScoresData(const ScoresData& a, const ScoresData& b) {
    if (a.items.size() != b.items.size()) return false;
    for (std::size_t i = 0; i < a.items.size(); ++i) {
        const ClassificationResult& x = a.items[i];
        const ClassificationResult& y = b.items[i];
        if (x.class_id != y.class_id) return false;
        if (x.class_name != y.class_name) return false;
        if (x.confidence != y.confidence) return false;  // exact, per above
        if (x.top_k != y.top_k) return false;  // vector<pair<int,float>>
    }
    return true;
}

bool SameKeypointsData(const KeypointsData& a, const KeypointsData& b) {
    if (a.unordered != b.unordered) return false;
    if (a.items.size() != b.items.size()) return false;
    for (std::size_t i = 0; i < a.items.size(); ++i) {
        const PoseResult& x = a.items[i];
        const PoseResult& y = b.items[i];
        if (x.confidence != y.confidence) return false;
        if (x.box != y.box) return false;  // vector<float>, exact
        if (x.keypoints.size() != y.keypoints.size()) return false;
        for (std::size_t k = 0; k < x.keypoints.size(); ++k) {
            if (!SameKeypoint(x.keypoints[k], y.keypoints[k])) return false;
        }
    }
    return true;
}

/**
 * @brief Dispatch on shape and compare the actual payload. Every Shape has
 * its own exact comparator (pinned by
 * TestOperatorEqualityEveryShapeComparesIdenticalPayloadsEqual). Fail-closed:
 * a payload of the wrong type for its shape, or a shape outside the enum,
 * returns false, never true — a parity test must never silently pass
 * because the comparator didn't know how to read the payload it was handed.
 * A new Shape value needs a case here (with a same-type dynamic_cast guard,
 * as the existing cases do); -Wswitch flags the missing case.
 *
 * Floats are compared exactly, not with an epsilon: sync and async run the
 * identical stage on the identical input, so they can only differ in
 * scheduling, and an exact-compare mismatch is itself the finding a parity
 * test exists to catch, not something to paper over with a tolerance.
 *
 * NaN (review round 3, item 3): exact comparison means a NaN anywhere in a
 * compared field can never satisfy parity, because IEEE 754 defines
 * NaN != NaN — SameFrameData's pixel diff, SameBoxItem's score/box/angle
 * checks and SameVectorData's std::vector<float>::operator== all inherit
 * this. That is deliberate, not an oversight: a stage that emits NaN is
 * itself a bug, and a parity comparator that let two NaN-bearing results
 * compare equal would hide it. If a real parity run in Task 10 fails here,
 * the fix is in the stage producing the NaN, not in this comparator.
 */
bool SamePayload(const StageData& a, const StageData& b) {
    switch (a.shape()) {
        case Shape::kFrame: {
            const FrameData* fa = dynamic_cast<const FrameData*>(&a);
            const FrameData* fb = dynamic_cast<const FrameData*>(&b);
            if (fa == NULL || fb == NULL) return false;
            return SameFrameData(*fa, *fb);
        }
        // kObBoxes and kInstances share BoxesData with kBoxes: SameBoxItem
        // compares the OBB angle and the instance mask pixels too.
        case Shape::kBoxes:
        case Shape::kObBoxes:
        case Shape::kInstances: {
            const BoxesData* ba = dynamic_cast<const BoxesData*>(&a);
            const BoxesData* bb = dynamic_cast<const BoxesData*>(&b);
            if (ba == NULL || bb == NULL) return false;
            return SameBoxesData(*ba, *bb);
        }
        case Shape::kVector: {
            const VectorData* va = dynamic_cast<const VectorData*>(&a);
            const VectorData* vb = dynamic_cast<const VectorData*>(&b);
            if (va == NULL || vb == NULL) return false;
            return SameVectorData(*va, *vb);
        }
        // The other two branches of the project's headline graph
        // (od + seg + pose). Added in Task 10 fix round 1: until a fixture
        // could run a segmentation or depth branch to SUCCESS and have it
        // compared, sync/async parity was proven only for box-shaped
        // payloads, which is not the guarantee this engine advertises.
        case Shape::kLabelMap: {
            const LabelMapData* la = dynamic_cast<const LabelMapData*>(&a);
            const LabelMapData* lb = dynamic_cast<const LabelMapData*>(&b);
            if (la == NULL || lb == NULL) return false;
            if (la->binary_mask != lb->binary_mask) return false;
            if (la->mask_color != lb->mask_color) return false;
            return SameMat(la->labels, lb->labels);
        }
        case Shape::kDenseMap: {
            const DenseMapData* da = dynamic_cast<const DenseMapData*>(&a);
            const DenseMapData* db = dynamic_cast<const DenseMapData*>(&b);
            if (da == NULL || db == NULL) return false;
            if (da->spatial != db->spatial) return false;
            return SameMat(da->values, db->values);
        }
        case Shape::kKeypoints: {
            const KeypointsData* ka = dynamic_cast<const KeypointsData*>(&a);
            const KeypointsData* kb = dynamic_cast<const KeypointsData*>(&b);
            if (ka == NULL || kb == NULL) return false;
            return SameKeypointsData(*ka, *kb);
        }
        case Shape::kScores: {
            const ScoresData* sa = dynamic_cast<const ScoresData*>(&a);
            const ScoresData* sb = dynamic_cast<const ScoresData*>(&b);
            if (sa == NULL || sb == NULL) return false;
            return SameScoresData(*sa, *sb);
        }
        // Payload hand-off makes an image output load-bearing: it becomes
        // the next node's input, so parity on it is compared exactly.
        case Shape::kImage: {
            const ImageData* ia = dynamic_cast<const ImageData*>(&a);
            const ImageData* ib = dynamic_cast<const ImageData*>(&b);
            if (ia == NULL || ib == NULL) return false;
            return SameMat(ia->image, ib->image);
        }
        case Shape::kBoxes3d: {
            const Boxes3dData* ta = dynamic_cast<const Boxes3dData*>(&a);
            const Boxes3dData* tb = dynamic_cast<const Boxes3dData*>(&b);
            if (ta == NULL || tb == NULL) return false;
            return SameBoxes3dData(*ta, *tb);
        }
        case Shape::kRecords: {
            const RecordsData* ra = dynamic_cast<const RecordsData*>(&a);
            const RecordsData* rb = dynamic_cast<const RecordsData*>(&b);
            if (ra == NULL || rb == NULL) return false;
            return SameRecords(*ra, *rb);
        }
    }
    // Every Shape has a case above, so this is reached only for a value
    // outside the enum. Fail closed: never report parity on a payload this
    // function does not know how to read.
    return false;
}

/// Every RoiRef field, exactly. Shared by the node_results and roi_results
/// loops of operator== so a coordinate mismatch cannot hide in either.
bool SameOrigin(const RoiRef& x, const RoiRef& y) {
    if (x.from_roi != y.from_roi) return false;
    if (x.parent_index != y.parent_index) return false;
    if (x.roi_index != y.roi_index) return false;
    if (x.track_id != y.track_id) return false;
    if (x.parent_node != y.parent_node) return false;
    if (x.src_box != y.src_box) return false;
    if (x.inv_align != y.inv_align) return false;
    if (x.crop_size != y.crop_size) return false;
    return true;
}

/// The rule operator== applies to `data`: the same pointer (both null
/// included) is equal, one null is not, else the shapes must match and
/// SamePayload decides.
bool SameData(const StageDataPtr& x, const StageDataPtr& y) {
    if (x.get() == y.get()) return true;
    if (!x || !y) return false;
    if (x->shape() != y->shape()) return false;
    return SamePayload(*x, *y);
}

/// The same port names, each with SameData payloads.
bool SamePorts(const StagePorts& x, const StagePorts& y) {
    if (x.size() != y.size()) return false;
    StagePorts::const_iterator a = x.begin();
    StagePorts::const_iterator b = y.begin();
    for (; a != x.end(); ++a, ++b) {
        if (a->first != b->first) return false;
        if (!SameData(a->second, b->second)) return false;
    }
    return true;
}

bool SameEvents(const std::vector<GraphEvent>& a, const std::vector<GraphEvent>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].node != b[i].node || a[i].event != b[i].event) return false;
        if (a[i].track_id != b[i].track_id || a[i].time_s != b[i].time_s) return false;
        if (a[i].data.numbers != b[i].data.numbers || a[i].data.text != b[i].data.text) {
            return false;
        }
    }
    return true;
}

bool SameOverlay(const LogicOverlay& a, const LogicOverlay& b) {
    if (a.alert != b.alert || a.polygons != b.polygons || a.lines != b.lines) return false;
    if (a.highlights != b.highlights || a.axes != b.axes || a.texts.size() != b.texts.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.texts.size(); ++i) {
        if (a.texts[i].at != b.texts[i].at || a.texts[i].text != b.texts[i].text ||
            a.texts[i].alert != b.texts[i].alert || a.texts[i].away != b.texts[i].away) {
            return false;
        }
    }
    return true;
}

/**
 * @brief One node's handle on a stage several nodes share ("engine":
 *        {"share": true}).
 *
 * Every call forwards. The stage is owned jointly by its handles, so it is
 * destroyed (and flushes, clause (5)) with the last of them, whatever order
 * the nodes are torn down in. Safe because both executors call into stages
 * from one driver thread only (clause (1) still holds for the shared
 * instance) and route every completion by the callback passed to that one
 * submit(), never by stage; see the rule in the Graph Composer plan and in
 * README "Device memory".
 */
class SharedStage : public IStage {
 public:
    explicit SharedStage(const std::shared_ptr<IStage>& stage) : stage_(stage) {}
    StageResult run(const StageInput& input) { return stage_->run(input); }
    void submit(const StageInput& input, StageCallback callback) {
        stage_->submit(input, callback);
    }
    void flush() { stage_->flush(); }
    void poll() { stage_->poll(); }
    Shape outputShape() const { return stage_->outputShape(); }
    InputContract inputContract() const { return stage_->inputContract(); }

 private:
    std::shared_ptr<IStage> stage_;
};

/// The same StageParams, as ValidateGraph compares sharers' "params".
bool SameStageParams(const StageParams& a, const StageParams& b) {
    return a.numeric == b.numeric && a.text == b.text && a.lists == b.lists;
}

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

/// The command that fetches one model into model_dir, as the CLI's
/// consumer::DownloadCommand prints it (multi_model_graph/graph_consumer.hpp;
/// the engine does not depend on that file). download_models.py, never
/// "./setup.sh --models": setup.sh also re-fetches the sample videos, and
/// under its default --force scripts/get_resource.sh rm -rf's an
/// assets/videos symlink.
std::string DownloadCommand(const std::string& name, const std::string& model_dir) {
    return "python3 scripts/download_models.py --models " + ShellWord(name) + " --output " +
           ShellWord(model_dir);
}

}  // namespace

EdgeCargo SelectEdgeCargo(const EdgeRuntime& edge, const StageDataPtr& produced,
                          const BoxesData* boxes, const StagePorts& ports) {
    EdgeCargo cargo;
    if (edge.port.empty()) {
        cargo.payload = produced;
        cargo.boxes = boxes;
        return cargo;
    }
    StagePorts::const_iterator it = ports.find(edge.port);
    if (it == ports.end() || !it->second) return cargo;
    cargo.payload = it->second;
    if (ProducesRoi(cargo.payload->shape())) {
        cargo.boxes = dynamic_cast<const BoxesData*>(cargo.payload.get());
    }
    return cargo;
}

void StageGraph::Build(const GraphSpec& spec, const IModelRegistry& registry,
                       const std::string& model_dir, bool require_artifacts) {
    nodes_.clear();
    edges_.clear();
    order_.clear();
    sources_.clear();
    streams_.clear();

    std::map<std::string, std::size_t> index_of;
    nodes_.resize(spec.nodes.size());
    std::size_t engines_opened = 0;
    // "engine": {"share": true}: node id -> owner id (validated: same
    // variant, params and buffers), and owner id -> the one stage.
    const std::map<std::string, std::string> owners = SharedEngineOwners(spec, registry);
    std::map<std::string, std::shared_ptr<IStage> > shared;
    try {
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        NodeRuntime& runtime = nodes_[i];
        runtime.id = node.id;
        runtime.is_source = node.is_source;
        runtime.is_cpu = node.is_cpu;
        runtime.is_logic = node.is_logic;
        runtime.uri = node.uri;
        runtime.model_name = node.model;
        runtime.op = node.op;
        runtime.params = node.params;
        index_of[node.id] = i;

        if (node.is_source) {
            runtime.output_shape = Shape::kFrame;
            sources_.push_back(i);
            continue;
        }
        if (node.is_cpu) {
            runtime.output_shape = Shape::kRecords;
            continue;
        }
        if (node.is_logic) {
            runtime.output_shape = NodeOutputShape(spec, registry, node.id);
            runtime.logic.reset(new LogicConfig(MakeLogicConfig(node)));
            continue;
        }

        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) {
            // A validated GraphSpec never reaches this in practice:
            // ValidateGraph (Task 7) already rejects an unknown model with
            // kModelUnknown. This is a defensive guard against Build() being
            // called directly on an unvalidated spec (pinned by
            // TestStageGraphBuildThrowsModelUnknownWhenRegistryLookupMisses,
            // review round 1, fix 4), so the guard itself doesn't silently
            // dereference a NULL ModelInfo* if that ever happens.
            throw GraphError(GraphErrorCode::kModelUnknown,
                             "node \"" + node.id + "\"",
                             "model \"" + node.model + "\" is not registered",
                             "run --list-models, or see docs/graph_models.md");
        }

        const std::string path = model_dir + "/" + info->dxnn_file;
        if (require_artifacts && !FileExists(path) && !info->published) {
            // The model zoo has no file for it yet: no downloader can help.
            throw GraphError(GraphErrorCode::kModelMissing, "node \"" + node.id + "\"",
                             info->dxnn_file + " not found in " + model_dir,
                             info->model_name +
                                 " is not published by the model zoo yet; "
                                 "download_models.py cannot download it");
        }
        if (require_artifacts && !FileExists(path)) {
            throw GraphError(
                GraphErrorCode::kModelMissing, "node \"" + node.id + "\"",
                info->dxnn_file + " not found in " + model_dir,
                // The name is the model zoo's (ModelInfo::download_name,
                // which is not always the registry's), the same one the
                // CLI's PrepareGraph prints in its richer version of this
                // line (every absent model at once) before Build() is
                // reached; this is the fallback for a caller that builds
                // directly.
                DownloadCommand(info->download_name.empty() ? node.model
                                                            : info->download_name,
                                model_dir));
        }
        std::map<std::string, std::string>::const_iterator owner = owners.find(node.id);
        if (owner != owners.end() && owner->second != node.id) {
            // The owner comes first in declaration order, so it is built.
            std::map<std::string, std::shared_ptr<IStage> >::const_iterator stage =
                shared.find(owner->second);
            if (stage == shared.end()) {
                throw std::logic_error("StageGraph::Build: node \"" + node.id +
                                       "\" shares an engine that was not built");
            }
            // ValidateGraph rejects sharers whose params or buffers differ
            // (the one stage runs the owner's configuration); this guards a
            // Build() on an unvalidated spec, like the unknown-model guard.
            const NodeSpec* owner_spec = spec.FindNode(owner->second);
            if (owner_spec == NULL || !SameStageParams(node.params, owner_spec->params) ||
                node.engine.buffers != owner_spec->engine.buffers) {
                throw std::logic_error("StageGraph::Build: node \"" + node.id +
                                       "\" shares the engine of node \"" + owner->second +
                                       "\" but its params or buffers differ "
                                       "(the spec was not validated)");
            }
            runtime.stage.reset(new SharedStage(stage->second));
            runtime.output_shape = info->output_shape;
            runtime.tracked = node.track.present;
            continue;
        }
        try {
            EngineOptions engine;
            engine.buffers = node.engine.buffers;
            runtime.stage = registry.createStageWith(node.model, path, node.params, engine);
        } catch (const ModelContainerError& error) {
            // A container this DX-RT cannot load (R12): no engine was
            // opened, and downloading it again fetches the same file.
            throw GraphError(GraphErrorCode::kModelLoad, "node \"" + node.id + "\"",
                             "model \"" + node.model + "\" (" + info->dxnn_file +
                                 ") could not be loaded: " + error.what(),
                             error.hint());
        } catch (const std::exception& error) {
            // The runtime's own text is kept whole: it is the only place the
            // reason (memory, device, model format) is spelled out.
            std::ostringstream how;
            if (engines_opened == 0) {
                how << "check the NPU with dxrt-cli -s, and that " << info->dxnn_file
                    << " was compiled for this DX-RT version; to download it again: "
                    << DownloadCommand(
                           info->download_name.empty() ? node.model : info->download_name,
                           model_dir);
            } else {
                how << "the NPU's device memory may be full: this graph loaded "
                    << engines_opened << " model(s) before this one, and every node "
                    << "loads its own copy, even of the same model, unless the nodes "
                    << "set \"engine\": {\"share\": true} (another process on the NPU "
                    << "uses memory too); share an engine, lower a node's \"engine\": "
                    << "{\"buffers\": N} (DX-RT's default is 6), remove a node or use a "
                    << "smaller model - DX-M1 cannot hold a second realesrgan-x2_192x192 "
                    << "next to yolov8-n_640x640 and resnet50_224x224";
            }
            throw GraphError(GraphErrorCode::kModelLoad, "node \"" + node.id + "\"",
                             "model \"" + node.model + "\" (" + info->dxnn_file +
                                 ") could not be loaded: " + error.what(),
                             how.str());
        }
        ++engines_opened;
        if (owner != owners.end()) {
            // The first sharer: the stage becomes jointly owned.
            shared[node.id] = std::shared_ptr<IStage>(runtime.stage.release());
            runtime.stage.reset(new SharedStage(shared[node.id]));
        }
        runtime.output_shape = info->output_shape;

        runtime.tracked = node.track.present;
    }
    } catch (...) {
        // Close every engine this Build opened before the error leaves it,
        // so the device memory is free at once - for a caller that reuses
        // this StageGraph or retries with a smaller graph - and
        // the graph is left empty rather than half built.
        shared.clear();
        nodes_.clear();
        sources_.clear();
        throw;
    }

    for (std::size_t i = 0; i < spec.edges.size(); ++i) {
        EdgeRuntime edge;
        edge.from = index_of[spec.edges[i].from];
        edge.to = index_of[spec.edges[i].to];
        edge.roi = spec.edges[i].roi;
        edge.port = spec.edges[i].port;
        edge.carry_result = spec.edges[i].carry_result;
        if (edge.port == ToString(nodes_[edge.from].output_shape)) edge.port.clear();
        edges_.push_back(edge);
    }

    // Kahn, lowest index first so the order is deterministic.
    std::vector<int> indegree(nodes_.size(), 0);
    for (std::size_t i = 0; i < edges_.size(); ++i) indegree[edges_[i].to] += 1;

    std::vector<std::size_t> ready;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (indegree[i] == 0) ready.push_back(i);
    }
    std::sort(ready.begin(), ready.end());

    while (!ready.empty()) {
        const std::size_t current = ready.front();
        ready.erase(ready.begin());
        order_.push_back(current);
        std::vector<std::size_t> unlocked;
        for (std::size_t i = 0; i < edges_.size(); ++i) {
            if (edges_[i].from != current) continue;
            if (--indegree[edges_[i].to] == 0) unlocked.push_back(edges_[i].to);
        }
        std::sort(unlocked.begin(), unlocked.end());
        ready.insert(ready.end(), unlocked.begin(), unlocked.end());
        std::sort(ready.begin(), ready.end());
    }

    // Streams (SP2): one per source node, in declaration order - the order
    // of sources_. ListStreams is the reachability ValidateGraph and --check
    // use, so the three cannot disagree about who is in a stream.
    const std::vector<StreamSpec> specs = ListStreams(spec);
    for (std::size_t s = 0; s < specs.size(); ++s) {
        StreamPlan plan;
        plan.source = index_of[specs[s].source];
        plan.source_id = specs[s].source;
        plan.member.assign(nodes_.size(), 0);
        for (std::size_t k = 0; k < specs[s].nodes.size(); ++k) {
            plan.member[index_of[specs[s].nodes[k]]] = 1;
        }
        // Every edge out of a member ends at a member (reachability is
        // closed under successors), so these are exactly the inputs a frame
        // of this stream brings each node.
        plan.in_edges.assign(nodes_.size(), 0);
        for (std::size_t e = 0; e < edges_.size(); ++e) {
            if (plan.member[edges_[e].from]) plan.in_edges[edges_[e].to] += 1;
        }
        for (std::size_t o = 0; o < order_.size(); ++o) {
            if (plan.member[order_[o]]) plan.order.push_back(order_[o]);
        }
        streams_.push_back(plan);
    }
    // One tracker per (tracked node, stream that runs it): track ids never
    // cross streams (SP2 decision 5).
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].tracked) continue;
        const TrackSpec& track = spec.nodes[i].track;
        nodes_[i].track = track;
        nodes_[i].state_generation.assign(streams_.size(), 0);
        nodes_[i].trackers.assign(streams_.size(), std::shared_ptr<IouTracker>());
        for (std::size_t s = 0; s < streams_.size(); ++s) {
            if (streams_[s].member[i]) {
                nodes_[i].trackers[s].reset(new IouTracker(track.iou, track.max_age));
            }
        }
    }
    // One state per (logic node, stream that runs it), as for trackers: a
    // zone's tracks, a line's counts, a condition's cooldown never cross
    // streams.
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].is_logic) continue;
        nodes_[i].state_generation.assign(streams_.size(), 0);
        nodes_[i].logic_states.assign(streams_.size(), std::shared_ptr<LogicState>());
        for (std::size_t s = 0; s < streams_.size(); ++s) {
            if (streams_[s].member[i]) nodes_[i].logic_states[s].reset(new LogicState());
        }
    }
}

std::size_t StageGraph::StreamIndex(const std::string& source_id) const {
    for (std::size_t s = 0; s < streams_.size(); ++s) {
        if (streams_[s].source_id == source_id) return s;
    }
    return streams_.size();
}

std::size_t StageGraph::OnlyStream(const char* caller) const {
    if (streams_.size() == 1) return 0;
    std::ostringstream text;
    text << caller << ": the graph has ";
    if (streams_.empty()) {
        text << "no source node";
    } else {
        text << streams_.size() << " source nodes (";
        for (std::size_t s = 0; s < streams_.size(); ++s) {
            text << (s == 0 ? "" : ", ") << streams_[s].source_id;
        }
        text << "), so each frame must name its stream (an index into "
                "StageGraph::streams())";
    }
    throw std::invalid_argument(text.str());
}

void StageGraph::SetStreamClock(std::size_t stream, double fps, bool wall_clock) {
    StreamPlan& plan = streams_.at(stream);
    plan.fps = std::isfinite(fps) && fps > 0.0 ? fps : kDefaultStreamFps;
    plan.wall_clock = wall_clock;
    plan.clock_started = false;
}

double StageGraph::FrameTime(std::size_t stream, std::size_t frame_index) {
    StreamPlan& plan = streams_.at(stream);
    if (!plan.wall_clock) return static_cast<double>(frame_index) / plan.fps;
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    if (!plan.clock_started) {
        plan.clock_started = true;
        plan.clock_start = now;
    }
    return std::chrono::duration<double>(now - plan.clock_start).count();
}

void StageGraph::RestartStream(std::size_t stream) { ++streams_.at(stream).generation; }

void StageGraph::FreshenStreamState(std::size_t index, std::size_t stream,
                                    std::size_t generation) {
    NodeRuntime& node = nodes_.at(index);
    if (stream >= node.state_generation.size() || node.state_generation[stream] == generation) {
        return;
    }
    node.state_generation[stream] = generation;
    if (stream < node.trackers.size() && node.trackers[stream]) {
        node.trackers[stream].reset(new IouTracker(node.track.iou, node.track.max_age));
    }
    if (stream < node.logic_states.size() && node.logic_states[stream]) {
        node.logic_states[stream].reset(new LogicState());
    }
}

std::vector<std::size_t> StageGraph::OutgoingEdges(std::size_t node) const {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < edges_.size(); ++i) {
        if (edges_[i].from == node) out.push_back(i);
    }
    return out;
}

/**
 * @brief Structural equality for FrameReport. See SamePayload's doc comment
 * for the NaN direction this inherits.
 *
 * Both-null data (review round 3, item 3): every payload comparison below
 * is guarded by `x.data.get() != y.data.get()`, so when both sides are null
 * (a StageResult with no data — a source or full-frame node whose stage was
 * never populated, or a default-constructed StageResult in a test) the
 * pointers compare equal and the guard skips the payload check entirely,
 * treating "no data" as equal to "no data" without asking SamePayload
 * anything. This is deliberate: there is no payload to disagree about, so
 * asking would only mean handling a `nullptr` case in every SamePayload
 * branch for no discriminating power. It is not the same claim as "false
 * for everything is safe" — a null StageDataPtr is a distinct, meaningful
 * state (absence), not a placeholder for content that might differ.
 *
 * node_ms is measured, not produced, and is deliberately not compared.
 */
bool operator==(const FrameReport& a, const FrameReport& b) {
    if (a.frame_index != b.frame_index) return false;
    if (a.stream != b.stream) return false;
    if (a.skipped_out_of_bounds != b.skipped_out_of_bounds) return false;
    if (a.error != b.error) return false;
    if (a.node_results.size() != b.node_results.size()) return false;
    if (a.roi_results.size() != b.roi_results.size()) return false;

    for (std::map<std::string, StageResult>::const_iterator it =
             a.node_results.begin(); it != a.node_results.end(); ++it) {
        std::map<std::string, StageResult>::const_iterator other =
            b.node_results.find(it->first);
        if (other == b.node_results.end()) return false;
        if (!SameOrigin(it->second.origin, other->second.origin)) return false;
        if (!SameData(it->second.data, other->second.data)) return false;
        if (!SamePorts(it->second.ports, other->second.ports)) return false;
    }
    // roi_results: compare both the origin (where the crop came from) and
    // the payload (what the stage produced from it). The origin comparison
    // covers every RoiRef field (review round 3, item 2 closed the
    // remaining two: from_roi and inv_align — a result whose crop came
    // through a different alignment path, or wasn't from_roi at all, must
    // not compare equal to one that was; Task 12 fix round 3 added
    // crop_size to RoiRef and to this comparison at the same time, so this
    // claim stays true rather than going stale the moment a field is
    // added). The vectors are compared
    // element-by-element at matching indices, which is only a meaningful
    // check because both executors are required to store this vector
    // pre-sorted into ByOrigin's canonical order — an unsorted vector would
    // make this comparison order-sensitive for no reason.
    for (std::map<std::string, std::vector<StageResult> >::const_iterator it =
             a.roi_results.begin(); it != a.roi_results.end(); ++it) {
        std::map<std::string, std::vector<StageResult> >::const_iterator other =
            b.roi_results.find(it->first);
        if (other == b.roi_results.end()) return false;
        if (it->second.size() != other->second.size()) return false;
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            const StageResult& x = it->second[i];
            const StageResult& y = other->second[i];
            if (!SameOrigin(x.origin, y.origin)) return false;
            if (!SameData(x.data, y.data)) return false;
            if (!SamePorts(x.ports, y.ports)) return false;
        }
    }
    // Logic nodes: the frame's time, its events and what is drawn, exactly
    // (both executors compute them from the same inputs in stream order).
    if (a.time_s != b.time_s) return false;
    if (!SameEvents(a.events, b.events)) return false;
    if (a.logic_nodes != b.logic_nodes) return false;
    if (a.overlays.size() != b.overlays.size()) return false;
    for (std::map<std::string, LogicOverlay>::const_iterator it = a.overlays.begin();
         it != a.overlays.end(); ++it) {
        std::map<std::string, LogicOverlay>::const_iterator other = b.overlays.find(it->first);
        if (other == b.overlays.end() || !SameOverlay(it->second, other->second)) return false;
    }
    return true;
}

bool ByOrigin(const StageResult& a, const StageResult& b) {
    if (a.origin.parent_node != b.origin.parent_node) {
        return a.origin.parent_node < b.origin.parent_node;
    }
    if (a.origin.parent_index != b.origin.parent_index) {
        return a.origin.parent_index < b.origin.parent_index;
    }
    return a.origin.roi_index < b.origin.roi_index;
}

}  // namespace graph
}  // namespace dxapp
