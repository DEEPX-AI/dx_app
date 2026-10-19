/**
 * @file typed_stage.hpp
 * @brief The bridge from IFactory to the engine's type-erased IStage.
 *
 * One template covers every model. IFactory is consumed, never bypassed:
 * the preprocessor and postprocessor both come from the factory, and the
 * factory is kept alive for the stage's lifetime so a later task can ask it
 * for its visualizer without a second construction path.
 *
 * WHY THIS FILE IS NOT UNDER common/graph/
 * ----------------------------------------
 * common/graph/ is the engine, and scripts/check_graph_boundary.py enforces
 * that the engine never names a concrete registry or a concrete factory.
 * This file names the concrete factories by template parameter and is
 * compiled only into the generated registry translation units; it belongs on
 * the zoo side of the i_registry.hpp boundary, next to the concrete registry
 * that hands it out. Nothing under common/graph/ includes it, and the
 * boundary guard rejects it if anything ever tries (it is neither on the
 * engine's sibling allowlist nor under an allowed prefix) - which is the
 * point, and is strictly stronger than adding it to that allowlist would be.
 *
 * HARDWARE FACTS THIS IMPLEMENTATION IS BUILT AROUND
 * --------------------------------------------------
 * All three were measured on dxrt::InferenceEngine (Task 3 spike, confirmed
 * 4 runs out of 4) and are restated in i_registry.hpp:
 *
 *  1. RegisterCallback takes std::function<int(TensorPtrs&, void*)> - the
 *     tensors come in BY REFERENCE and the data they wrap dangles the moment
 *     the callback returns. Everything needed is therefore copied out inside
 *     the callback: the postprocessor runs there, not afterwards.
 *  2. Wait(jobId) returns an EMPTY output once a callback is registered on
 *     that engine. No path in this file reads a Wait() return value.
 *  3. A blocking Run() ALSO fires the registered callback (5 Run() calls,
 *     zero RunAsync, produced exactly 5 callback fires). This file therefore
 *     does not call Run() at all - run() and submit() both go through
 *     RunAsync and take their result from the one callback - and it still
 *     carries a per-job delivered flag so that a second fire for one job,
 *     from any source, is dropped instead of being reported twice.
 *
 *     The identity a fire is matched on is a MONOTONIC COUNTER, not the
 *     address of the job object. An earlier draft keyed the pending map on
 *     job.get(); once a job completed and its shared_ptrs dropped, the
 *     allocator was free to hand that same address to the next new Job(),
 *     so a late fire for the old job would match a LIVE, UNDELIVERED one
 *     and pour stale outputs into it - strictly worse than the duplicate
 *     delivery the guard exists to prevent, and it made the sentence above
 *     false. A counter value is never recycled, so a late fire now finds
 *     nothing and is dropped, which is what that sentence claims.
 *
 * THROW CONTRACT (see IStage in i_registry.hpp)
 * ---------------------------------------------
 *  - run() throws std::runtime_error on any failure, as IStage documents.
 *  - submit() and flush() do NOT throw. A submission that fails before it
 *    reaches the hardware is reported through the StageCallback's `error`
 *    argument, exactly as a runtime failure is, so the asynchronous path has
 *    one error channel rather than two. That is what makes AsyncExecutor's
 *    per-job error handling equivalent to SyncExecutor's try/catch around
 *    run(), which is what sync/async parity on the error path means.
 */
#ifndef DXAPP_REGISTRY_TYPED_STAGE_HPP
#define DXAPP_REGISTRY_TYPED_STAGE_HPP

#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/base/i_factory.hpp"
#include "common/registry/engine_options.hpp"
#include "common/base/i_processor.hpp"
#include "common/config/model_config.hpp"
#include "common/graph/i_registry.hpp"
#include "common/graph/result_to_shape.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/dxnn_container.hpp"
#include "common/utility/repo_path.hpp"

namespace dxapp {
namespace graph {

namespace detail {

/**
 * @brief `model_path`, once its .dxnn container is one this DX-RT loads;
 *        otherwise throws ModelContainerError with ContainerLoadError's text
 *        and ContainerSupportHint's hint.
 *
 * Called on the path an engine is about to be created from (TypedStage,
 * MakeRestorationStage), so a v9 file on DX-RT < 3.5.0 fails with one
 * clear sentence before dxrt opens anything (spec section 4, R12).
 *
 * `runtime` is the DX-RT version to check against; the one-argument form
 * passes the running one (dxrt::Configuration::GetVersion()).
 */
inline const std::string& LoadableModelPath(const std::string& model_path,
                                            const std::string& runtime) {
    const std::string error = ContainerLoadError(model_path, runtime);
    if (!error.empty()) {
        uint32_t version = 0;
        std::string unread;
        ReadDxnnContainerVersion(model_path, &version, &unread);
        throw ModelContainerError(error, ContainerSupportHint(version));
    }
    return model_path;
}

inline const std::string& LoadableModelPath(const std::string& model_path) {
    return LoadableModelPath(model_path, dxrt::Configuration::GetInstance().GetVersion());
}

/**
 * @brief The config.json a stage reads: <task>/<family>/<variant>/config.json
 *        under src/cpp_example/, where the variant's factory lives.
 *
 * scripts/gen_model_registry.py refuses a registry row whose task or family
 * disagrees with its factory's directory (TASK MISMATCH), so the three
 * ModelInfo fields used here name a real directory. The file's nested
 * "config" object is read through ModelConfig's overlay (C3). Throws
 * std::runtime_error when family or variant is empty.
 */
inline std::string StageConfigPath(const ModelInfo& info) {
    // Without them the path would be "<task>//config.json", which does not
    // exist, and the stage would run on factory defaults without a word.
    if (info.family.empty() || info.variant.empty()) {
        throw std::runtime_error("ModelInfo for \"" + info.model_name +
                                 "\" has no family/variant: cannot locate its config.json");
    }
    return ::dxapp::RuntimeRoot() + "/src/cpp_example/" + info.task + "/" +
           info.family + "/" + info.variant + "/config.json";
}

/// The factory's graphInput(), or kDefault when it declares none.
template <class F>
constexpr auto DeclaredGraphInput(int) -> decltype(F::graphInput()) { return F::graphInput(); }
template <class F>
constexpr GraphInput DeclaredGraphInput(long) { return GraphInput::kDefault; }

/// The factory's graphPorts(), or "" when it declares none.
template <class F>
constexpr auto DeclaredGraphPorts(int) -> decltype(F::graphPorts()) { return F::graphPorts(); }
template <class F>
constexpr const char* DeclaredGraphPorts(long) { return ""; }

/// strcmp() == 0, usable in a static_assert (C++14 constexpr recursion).
constexpr bool SameText(const char* a, const char* b) {
    return *a == *b && (*a == '\0' || SameText(a + 1, b + 1));
}

/**
 * @brief Turn a postprocessor's std::vector<ResultT> into a StageDataPtr.
 *
 * result_to_shape.hpp takes a vector for the eight multi-item result types
 * and a single value for the five whole-frame ones (segmentation, depth,
 * restoration, embedding). C++14 has no `if constexpr`, so the choice is
 * made by ordinary overload ranking: the int overload is preferred whenever
 * a vector overload of ToStageData exists for ResultT, and SFINAE removes
 * it when one does not, leaving the long overload to unwrap the first
 * element. A missing result means the model produced nothing, which is the
 * default-constructed payload - not an error.
 */
template <class ResultT>
auto StageDataFromResults(const std::vector<ResultT>& results, int)
    -> decltype(ToStageData(results)) {
    return ToStageData(results);
}

template <class ResultT>
StageDataPtr StageDataFromResults(const std::vector<ResultT>& results, long) {
    if (results.empty()) return ToStageData(ResultT());
    return ToStageData(results[0]);
}

/**
 * @brief factory->createPostprocessor(w, h[, is_ort_configured]).
 *
 * i_factory.hpp declares SIXTEEN factory interfaces: eight take
 * is_ort_configured (with a default), eight do not declare it at all.
 * Calling the two-argument form everywhere compiles for all sixteen - and
 * is WRONG for the eight that take it, which
 * is how this was first written: with is_ort_configured defaulted to false
 * a YOLOv5 postprocessor looks for raw NPU tensors that an ORT-configured
 * model does not emit, prints "Failed to align output tensors" and decodes
 * whatever it was handed. Every runner in common/runner/ passes
 * ie.IsOrtConfigured(); this does the same, through expression SFINAE
 * because C++14 has no `if constexpr`. The int overload wins whenever the
 * three-argument call is valid.
 */
template <class FactoryT>
auto MakePostprocessor(FactoryT* factory, int width, int height, bool ort, int)
    -> decltype(factory->createPostprocessor(width, height, ort)) {
    return factory->createPostprocessor(width, height, ort);
}

template <class FactoryT>
auto MakePostprocessor(FactoryT* factory, int width, int height, bool, long)
    -> decltype(factory->createPostprocessor(width, height)) {
    return factory->createPostprocessor(width, height);
}

/// The postprocessor a graph stage decodes with: the factory's
/// createPostprocessor(), unless a specialization below says otherwise.
template <class ResultT>
struct PostprocessorFor {
    template <class FactoryT>
    static IPostprocessor<ResultT>* Make(FactoryT* factory, int width, int height, bool ort) {
        return MakePostprocessor(factory, width, height, ort, 0).release();
    }
};
/// YOLOPv2 in a graph uses the per-frame panoptic postprocessor (SP3): the
/// same DecodeYOLOPv2 as the boxes-only one, plus that frame's masks.
template <>
struct PostprocessorFor<PanopticResult> {
    template <class FactoryT>
    static IPostprocessor<PanopticResult>* Make(FactoryT* factory, int width, int height, bool) {
        return factory->createPanopticPostprocessor(width, height).release();
    }
};

/**
 * @brief Every extra output port the result type's converter builds (U-08).
 *
 * Same overload ranking as StageDataFromResults: the int overload exists
 * only when result_to_shape.hpp has a ToStagePorts for ResultT; otherwise
 * the long overload says the type has no ports.
 */
template <class ResultT>
auto PortsFromResults(const std::vector<ResultT>& results, int)
    -> decltype(ToStagePorts(std::declval<const std::vector<ResultT>&>())) {
    return ToStagePorts(results);
}

template <class ResultT>
StagePorts PortsFromResults(const std::vector<ResultT>&, long) {
    return StagePorts();
}

/// The ports the model declares, and only those. A converter may know more
/// ports than one model declares; the others are dropped here, so a model
/// that shares a converter (yolov8s_pose and SuperPoint) gains nothing.
template <class ResultT>
StagePorts DeclaredPortsFrom(const std::vector<ResultT>& results,
                             const std::vector<PortInfo>& declared) {
    const StagePorts all = PortsFromResults(results, 0);
    StagePorts kept;
    for (std::size_t i = 0; i < declared.size(); ++i) {
        const StagePorts::const_iterator it = all.find(declared[i].name);
        if (it != all.end()) kept[it->first] = it->second;
    }
    return kept;
}

/// A decoded frame as a StageResult: payload, declared ports and origin.
/// Both are built before `*result` is touched, so a converter that throws
/// (ragged SuperPoint descriptors) leaves it as it was: a failed result
/// carries neither data nor ports.
template <class ResultT>
void FillStageResult(const std::vector<ResultT>& results,
                     const std::vector<PortInfo>& declared, const RoiRef& origin,
                     StageResult* result) {
    const StageDataPtr data = StageDataFromResults(results, 0);
    StagePorts ports;
    if (!declared.empty()) ports = DeclaredPortsFrom(results, declared);
    result->data = data;
    result->ports.swap(ports);
    result->origin = origin;
}

/**
 * @brief Why ResultT cannot fill the declared ports, or "" when it can.
 *
 * Asked once, at stage construction, of a conversion of zero results: the
 * converters fill every port even then (the PORTS CONTRACT), so a missing
 * or null port here is a port the result type never produces.
 */
template <class ResultT>
std::string PortProblem(const std::vector<PortInfo>& declared) {
    const StagePorts all = PortsFromResults(std::vector<ResultT>(), 0);
    for (std::size_t i = 0; i < declared.size(); ++i) {
        const StagePorts::const_iterator it = all.find(declared[i].name);
        if (it == all.end() || !it->second) {
            return "output port \"" + declared[i].name +
                   "\" is declared, but this result type has no converter for it";
        }
        if (it->second->shape() != declared[i].shape) {
            return "output port \"" + declared[i].name + "\" is declared as " +
                   ToString(declared[i].shape) + " but its converter produces " +
                   ToString(it->second->shape());
        }
    }
    return std::string();
}

/// Only IDepthEstimationFactory declares getInputNormalization(); the other
/// fifteen interfaces do not, so this is an expression-SFINAE probe rather
/// than a virtual on a base they do not share.
template <class FactoryT>
auto InputNormalizationOf(FactoryT* factory, int)
    -> decltype(factory->getInputNormalization()) {
    return factory->getInputNormalization();
}

template <class FactoryT>
InputNormalizationParams InputNormalizationOf(FactoryT*, long) {
    return InputNormalizationParams();
}

/// A key with no name cannot be written.
inline bool EmptyParamKey(const std::string& key) {
    return key.empty();
}

/// `text` as a JSON string literal: '"', '\\' and control characters
/// escaped, which ModelConfig decodes back to `text` exactly (U-77).
inline std::string JsonQuote(const std::string& text) {
    std::string out = "\"";
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char escaped[8];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += escaped;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out + "\"";
}

/**
 * @brief Serialize a graph node's params as the JSON ModelConfig parses.
 *
 * Numbers are written exactly as before U-62 (setprecision(17), key order),
 * so a numeric-only overlay is byte-identical to the old one. Strings become
 * "k": "v" and lists "k": ["a", "b"]; keys, text values and list entries
 * are JSON-escaped (JsonQuote), so every character arrives as written. A
 * key with no name is dropped.
 *
 * file_arrays is config.json's arrays (ModelConfig::rawArrays()). Each one
 * the node does not replace is copied verbatim, because a factory that reads
 * get_string_list("k") on the overlay would otherwise reset it to empty.
 * They are copied only when the node wrote a key: no node keys, no overlay.
 */
inline std::string ParamsToJson(const StageParams& params,
                                const std::map<std::string, std::string>& file_arrays) {
    std::map<std::string, std::string> fields;  // key -> JSON text of the value
    for (std::map<std::string, double>::const_iterator it = params.numeric.begin();
         it != params.numeric.end(); ++it) {
        if (EmptyParamKey(it->first)) continue;
        std::ostringstream number;
        number << std::setprecision(17) << it->second;
        fields[it->first] = number.str();
    }
    for (std::map<std::string, std::string>::const_iterator it = params.text.begin();
         it != params.text.end(); ++it) {
        if (EmptyParamKey(it->first)) continue;
        fields[it->first] = JsonQuote(it->second);
    }
    for (std::map<std::string, std::vector<std::string> >::const_iterator it =
             params.lists.begin();
         it != params.lists.end(); ++it) {
        if (EmptyParamKey(it->first)) continue;
        std::string list = "[";
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            if (i != 0) list += ", ";
            list += JsonQuote(it->second[i]);
        }
        fields[it->first] = list + "]";
    }
    if (fields.empty()) return std::string();
    for (std::map<std::string, std::string>::const_iterator it = file_arrays.begin();
         it != file_arrays.end(); ++it) {
        if (EmptyParamKey(it->first)) continue;
        fields.insert(*it);  // a node key already written wins
    }

    std::string out = "{";
    for (std::map<std::string, std::string>::const_iterator it = fields.begin();
         it != fields.end(); ++it) {
        if (it != fields.begin()) out += ", ";
        out += JsonQuote(it->first) + ": " + it->second;
    }
    return out + "}";
}

/// The numeric-only form, kept for callers that have no text or lists.
inline std::string ParamsToJson(const std::map<std::string, double>& numeric) {
    StageParams params;
    params.numeric = numeric;
    return ParamsToJson(params, std::map<std::string, std::string>());
}

using ::dxapp::IsAbsolutePath;  // common/utility/repo_path.hpp

/**
 * @brief `params`, with a relative "gallery" read against `root`.
 *
 * --check looks a gallery up under the repository (RuntimeRoot()), while
 * the postprocessor opens whatever path it is given, relative to the
 * working directory. So a relative node param is made absolute here, and a
 * relative config.json value a node does not override is added as one, so
 * a run from any directory opens the gallery --check found. An absolute
 * path is kept as given. The rule is ResolveAgainstRoot (repo_path.hpp),
 * the one --check reads a gallery by.
 */
inline StageParams GalleryAgainstRoot(const StageParams& params, const ModelConfig& file_config,
                                      const std::string& root) {
    StageParams out = params;
    std::map<std::string, std::string>::iterator node = out.text.find("gallery");
    if (node != out.text.end()) {
        node->second = ResolveAgainstRoot(root, node->second);
        return out;
    }
    const std::string from_file = file_config.get<std::string>("gallery", std::string());
    if (!from_file.empty() && !IsAbsolutePath(from_file)) {
        out.text["gallery"] = ResolveAgainstRoot(root, from_file);
    }
    return out;
}

/**
 * @brief factory defaults < config.json < node params, on `factory`, with
 *        a relative gallery read against the repository (GalleryAgainstRoot).
 *
 * What a stage does to its factory before it builds anything from it,
 * without an engine, so a test can configure a real factory.
 */
template <class F>
void ConfigureFactory(F* factory, const ModelInfo& info, const StageParams& params) {
    const ModelConfig file_config = LoadOptionalConfig(StageConfigPath(info));
    if (file_config.isLoaded()) factory->loadConfig(file_config);

    const std::string overlay = ParamsToJson(
        GalleryAgainstRoot(params, file_config, ::dxapp::RuntimeRoot()), file_config.rawArrays());
    if (overlay.empty()) return;
    // loadConfig reads every key with the factory's CURRENT value as the
    // default, so a second call carrying only the overridden keys leaves
    // everything else exactly as config.json left it. That is the
    // overlay, not a reload.
    ModelConfig node_config(overlay, ConfigSource::kText);
    factory->loadConfig(node_config);
}

/**
 * @brief TypedStage's pending jobs, keyed by a monotonic id (U-36).
 *
 * The rules the double-delivery guard relies on, in one place a test can
 * reach without an engine: ids start at 1 (a null userArg never matches),
 * rise by one per Add and are never reissued, even after the job holding
 * one is gone; a callback fire claims a job at most once. Not thread-safe:
 * TypedStage calls it under its mutex_.
 */
template <class JobT>
class PendingJobs {
 public:
    typedef std::uintptr_t Id;

    PendingJobs() : next_id_(1) {}

    /// Give `job` the next id and register it. @return that id.
    Id Add(const std::shared_ptr<JobT>& job) {
        job->id = next_id_++;
        jobs_[job->id] = job;
        return job->id;
    }

    /// The job a callback fire names, now marked delivered - or null when
    /// the id is unknown, already finished, or already claimed.
    std::shared_ptr<JobT> Claim(Id id) {
        typename std::map<Id, std::shared_ptr<JobT> >::iterator it = jobs_.find(id);
        if (it == jobs_.end() || it->second->delivered) return std::shared_ptr<JobT>();
        it->second->delivered = true;
        return it->second;
    }

    /// Remove a job no fire has claimed. @return false when one has.
    bool Abandon(Id id) {
        typename std::map<Id, std::shared_ptr<JobT> >::iterator it = jobs_.find(id);
        if (it == jobs_.end() || it->second->delivered) return false;
        jobs_.erase(it);
        return true;
    }

    /// Forget a finished job.
    void Erase(Id id) { jobs_.erase(id); }

    std::size_t size() const { return jobs_.size(); }

 private:
    std::map<Id, std::shared_ptr<JobT> > jobs_;
    Id next_id_;
};

}  // namespace detail

/**
 * @brief One model, wired to its factory, presented as an IStage.
 *
 * @tparam FactoryT  the concrete I...Factory implementation.
 * @tparam ResultT   the result type that factory's postprocessor produces.
 */
template <class FactoryT, class ResultT>
class TypedStage : public IStage {
 public:
    /**
     * @brief Identity of one submitted job, as carried in the runtime's
     *        `void* userArg`.
     *
     * std::uintptr_t exactly - it is by definition the integer type a void*
     * round-trips through, so the counter is never truncated on the way out
     * and back. Values are issued monotonically from 1 and never reused, so
     * unlike a heap address an id cannot come back attached to a different
     * job. It wraps only after 2^(8*sizeof(void*)) submissions to ONE stage:
     * unreachable on the 64-bit targets this project builds, and ~4.3e9 on a
     * 32-bit build, at which point the guard would degrade to the old
     * behaviour for one instant rather than from the second inference
     * onwards.
     */
    typedef std::uintptr_t JobId;

    TypedStage(std::unique_ptr<FactoryT> factory, const std::string& model_path,
               const ModelInfo& info, const StageParams& params)
        : TypedStage(std::move(factory),
                     detail::OpenEngine(detail::LoadableModelPath(model_path)),
                     info, params) {}

    /// Over an engine the caller already opened (MakeRestorationStage probes
    /// it first). The engine must have no callback registered yet.
    TypedStage(std::unique_ptr<FactoryT> factory, std::unique_ptr<dxrt::InferenceEngine> engine,
               const ModelInfo& info, const StageParams& params)
        : factory_(std::move(factory)),
          info_(info),
          engine_(std::move(engine)),
          input_width_(0),
          input_height_(0),
          is_float_input_(false),
          is_nhwc_(false),
          input_bytes_(0),
          submitted_(0),
          completed_(0) {
        // GetInputs() returns by value; binding a reference into it would
        // dangle at the end of this statement (-Wdangling-reference).
        const dxrt::Tensors inputs = engine_->GetInputs();
        if (inputs.empty()) {
            throw std::runtime_error("model reports no input tensor");
        }
        const std::vector<int64_t> shape = inputs.front().shape();
        // The model self-describes its input; the registry's numbers are a
        // fallback for a model whose shape the runtime does not report.
        parseInputShape(shape, input_width_, input_height_);
        if (input_width_ <= 0) input_width_ = info_.input_width;
        if (input_height_ <= 0) input_height_ = info_.input_height;
        is_float_input_ = (inputs.front().type() == dxrt::DataType::FLOAT);
        is_nhwc_ = isInputNHWC(shape);
        input_bytes_ = static_cast<std::size_t>(engine_->GetInputSize());

        ApplyParams(params);

        preprocessor_ = factory_->createPreprocessor(input_width_, input_height_);
        postprocessor_.reset(detail::PostprocessorFor<ResultT>::Make(
            factory_.get(), input_width_, input_height_, engine_->IsOrtConfigured()));
        const std::string port_problem = detail::PortProblem<ResultT>(info_.ports);
        if (!port_problem.empty()) {
            throw std::runtime_error("model \"" + info_.model_name + "\": " + port_problem);
        }
        normalization_ = detail::InputNormalizationOf(factory_.get(), 0);

        engine_->RegisterCallback(
            [this](dxrt::TensorPtrs& outputs, void* user) -> int {
                return this->OnComplete(outputs, user);
            });
    }

    ~TypedStage() {
        // Nothing may still be in flight when the engine - and the `this`
        // the registered callback captured - goes away.
        flush();
        // Then tear the engine down FIRST, inside the destructor body. Its
        // registered callback holds `this` and touches mutex_, pending_ and
        // done_, all of which are destroyed before engine_ would be by the
        // implicit member teardown (reverse declaration order). Resetting it
        // here means no callback can possibly fire against a half-destroyed
        // stage, even if the runtime were to deliver a late or spurious one.
        engine_.reset();
    }

    /// Blocking execution. Throws std::runtime_error on inference failure.
    StageResult run(const StageInput& input) {
        std::shared_ptr<Job> job(new Job());
        job->origin = input.origin;
        Prepare(input, job.get());  // throws; run() is allowed to

        void* const user_arg = Enqueue(job);
        try {
            engine_->RunAsync(job->buffer.data(), user_arg, NULL);
        } catch (const std::exception& error) {
            // Abandon() returns false when a callback already claimed this
            // job, i.e. the submission DID reach the hardware and then threw
            // on the way out. The completion path owns the job in that case,
            // so fall through and wait for it - exactly what submit() does,
            // rather than throwing over a result that is about to arrive.
            if (Abandon(job)) {
                throw std::runtime_error(
                    std::string("inference submit failed: ") + error.what());
            }
        } catch (...) {
            if (Abandon(job)) {
                throw std::runtime_error("unknown inference failure");
            }
        }

        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!job->finished) done_.wait(lock);
        }
        if (!job->error.empty()) throw std::runtime_error(job->error);
        return job->result;
    }

    /// Non-blocking submission. Never throws; see the throw contract above.
    void submit(const StageInput& input, StageCallback callback) {
        std::shared_ptr<Job> job(new Job());
        job->origin = input.origin;
        job->callback = callback;

        try {
            Prepare(input, job.get());
        } catch (const std::exception& error) {
            Report(callback, error.what());
            return;
        } catch (...) {
            Report(callback, "unknown preprocessing failure");
            return;
        }

        void* const user_arg = Enqueue(job);
        try {
            engine_->RunAsync(job->buffer.data(), user_arg, NULL);
        } catch (const std::exception& error) {
            if (Abandon(job)) {
                Report(callback, std::string("inference submit failed: ") +
                                     error.what());
            }
        } catch (...) {
            if (Abandon(job)) Report(callback, "unknown inference failure");
        }
    }

    /**
     * @brief Block until every submitted job's callback has run.
     *
     * Genuinely blocks: it waits on a condition variable until the completed
     * counter reaches the submitted counter, and OnComplete increments that
     * counter only AFTER the user's StageCallback has returned. That is
     * clause (4)'s promise. AsyncExecutor no longer harvests on flush() - it
     * harvests each node as that node's completions reach the executor's
     * own completion queue - so a premature return here would not make it
     * read a node early. It would break the two callers that still rely on
     * it: this stage's destructor, which calls flush() before tearing the
     * runtime down and would then free state a callback can still touch
     * (clause (5)); and AsyncExecutor::Finish / DrainPending, which flush
     * every stage so that no job - including one no frame submitted - is
     * still outstanding when they return. Neither failure shows with a
     * fake, which delivers on the driver thread; both are real on hardware.
     */
    void flush() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (completed_ < submitted_) done_.wait(lock);
    }

    Shape outputShape() const { return info_.output_shape; }
    InputContract inputContract() const { return info_.input_contract; }

 private:
    /**
     * @brief One logical call.
     *
     * Identified by `id`, a value from a monotonic counter - NOT by this
     * object's address, which the allocator may recycle the moment the job
     * is destroyed (see the note at the top of this file).
     */
    struct Job {
        JobId id;                     ///< never 0, never reused
        std::vector<uint8_t> buffer;  ///< owns the input until completion
        PreprocessContext ctx;
        RoiRef origin;
        StageCallback callback;  ///< empty on the run() path
        StageResult result;
        std::string error;
        bool delivered;  ///< claimed by a callback fire
        bool finished;   ///< callback has fully run

        Job() : id(0), delivered(false), finished(false) {}
    };

    /**
     * @brief factory defaults < config.json < node params.
     *
     * The config path is <task>/<family>/<variant>/config.json under
     * src/cpp_example/ (detail::StageConfigPath), which is exactly why
     * scripts/gen_model_registry.py refuses to generate a registry whose task
     * or family disagrees with the factory's directory: the three fields of
     * ModelInfo used here have to name a real directory.
     */
    void ApplyParams(const StageParams& params) {
        detail::ConfigureFactory(factory_.get(), info_, params);
    }

    void Prepare(const StageInput& input, Job* job) {
        if (input.image.empty()) {
            throw std::runtime_error("stage received an empty image");
        }
        cv::Mat preprocessed;
        preprocessor_->process(input.image, preprocessed, job->ctx);
        if (preprocessed.empty()) {
            throw std::runtime_error("preprocessor produced an empty buffer");
        }

        job->buffer.assign(input_bytes_, 0);
        if (is_float_input_) {
            const std::vector<float> floats =
                normalization_.apply_mean_std
                    ? convertToFloatBufferNormalized(preprocessed, is_nhwc_,
                                                     normalization_.mean,
                                                     normalization_.std)
                    : convertToFloatBuffer(preprocessed, is_nhwc_);
            const std::size_t bytes = floats.size() * sizeof(float);
            if (bytes > job->buffer.size()) job->buffer.resize(bytes);
            std::memcpy(job->buffer.data(), floats.data(), bytes);
        } else {
            const std::size_t bytes =
                preprocessed.total() * preprocessed.elemSize();
            if (bytes > job->buffer.size()) job->buffer.resize(bytes);
            std::memcpy(job->buffer.data(), preprocessed.data, bytes);
        }
    }

    /**
     * @brief Enter the stage's one shared postprocessor, serialized.
     *
     * A named function rather than a locked block inside OnComplete, so the
     * critical section is STRUCTURAL and cannot silently widen: the only way
     * to add work under this lock is to edit this three-line body, where the
     * reason is in front of you. The same move that put the concrete
     * registry outside common/graph/ - arrange things so the question cannot
     * arise, instead of relying on everyone remembering the rule.
     *
     * WHY THE LOCK AT ALL. There is ONE postprocessor_ per stage, shared by
     * every job, and this runs on the runtime's completion thread. IStage's
     * THREADING clause (3) says two completions from one stage may run
     * concurrently, and dxrt promises nothing to the contrary, so without
     * this one IPostprocessor would be entered twice at once. That is a live
     * race, not a hypothetical: anchor_detection_postprocessor.hpp's
     * align_tensors() WRITES num_classes_ on the process() path, and yolov5n
     * uses it.
     *
     * WHY NOT mutex_. Reusing mutex_ would put a lock around the user's
     * StageCallback further down OnComplete, breaking the last paragraph of
     * IStage's contract (no internal lock across the user callback) and the
     * flush() ordering that depends on that callback running lock-free. The
     * two locks are siblings and are never held together.
     *
     * WHAT IT COSTS. Per-stage decoding serializes, which bites exactly on
     * the ROI fan-out case - several crops through one model. That is the
     * right trade and should not be "optimized" away by a later reader who
     * has not priced it: decoding is CPU work done while the NPU is busy
     * with other stages, and SyncExecutor already serializes it, so the
     * async path pays nothing the sync path does not.
     *
     * WHAT IT PROTECTS BEYOND MEMORY SAFETY. Two completions interleaved
     * inside one postprocessor could make the RESULT depend on the order
     * they arrived in, which would surface as a sync/async parity failure
     * whose actual cause lives in a factory rather than in either executor.
     * Serializing here keeps parity a property of the executors, not of the
     * runtime's scheduling.
     *
     * Throws whatever the postprocessor throws; OnComplete catches. The
     * lock_guard unwinds first, so no exception path leaves it held.
     */
    std::vector<ResultT> Decode(dxrt::TensorPtrs& outputs,
                                const PreprocessContext& ctx) {
        std::lock_guard<std::mutex> lock(postprocess_mutex_);
        return postprocessor_->process(outputs, ctx);
    }

    static void* ToUserArg(JobId id) {
        return reinterpret_cast<void*>(id);
    }
    static JobId FromUserArg(void* user) {
        return reinterpret_cast<JobId>(user);
    }

    /// Issue this job's identity and register it. @return the userArg to
    /// hand the runtime.
    void* Enqueue(const std::shared_ptr<Job>& job) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.Add(job);
        ++submitted_;
        return ToUserArg(job->id);
    }

    /// Undo an Enqueue whose RunAsync never reached the hardware.
    /// @return false when a callback already claimed the job (so the
    ///         completion path owns it and must not be reported twice).
    bool Abandon(const std::shared_ptr<Job>& job) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pending_.Abandon(job->id)) return false;
        --submitted_;
        return true;
    }

    static void Report(const StageCallback& callback, const std::string& error) {
        if (!callback) return;
        StageResult empty;
        callback(empty, error);
    }

    /**
     * @brief The one completion path, shared by run() and submit().
     *
     * `outputs` is engine-owned and dangles once this returns, so the
     * postprocessor runs here and the payload it builds is what leaves.
     */
    int OnComplete(dxrt::TensorPtrs& outputs, void* user) {
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Double-delivery guard. The id is monotonic and never
            // reused, so a fire for a job that has already completed finds
            // nothing here and cannot be mistaken for a later job that
            // happened to land on the same heap address; a second fire for
            // a job still in the map is caught by its delivered flag.
            // Either way: dropped, never reported twice.
            job = pending_.Claim(FromUserArg(user));
            if (!job) return 0;
        }

        std::string error;
        StageResult result;
        try {
            // The ONE entry into postprocessor_, and the only thing under
            // postprocess_mutex_ - see Decode(). Everything below this line
            // runs unlocked, which is what the user callback requires.
            const std::vector<ResultT> results = Decode(outputs, job->ctx);
            detail::FillStageResult(results, info_.ports, job->origin, &result);
        } catch (const std::exception& failure) {
            error = failure.what();
        } catch (...) {
            error = "unknown postprocessing failure";
        }

        job->result = result;
        job->error = error;

        // The user's callback runs BEFORE completed_ moves, so flush()
        // cannot return while a delivery is still in progress.
        if (job->callback) {
            try {
                job->callback(result, error);
            } catch (...) {
                // A stage must not let a consumer's exception escape onto
                // the runtime's callback thread, where nothing can catch it.
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_.Erase(job->id);
            job->finished = true;
            ++completed_;
        }
        done_.notify_all();
        return 0;
    }

    std::unique_ptr<FactoryT> factory_;
    ModelInfo info_;
    std::unique_ptr<dxrt::InferenceEngine> engine_;
    PreprocessorPtr preprocessor_;
    std::shared_ptr<IPostprocessor<ResultT> > postprocessor_;
    InputNormalizationParams normalization_;

    int input_width_;
    int input_height_;
    bool is_float_input_;
    bool is_nhwc_;
    std::size_t input_bytes_;

    mutable std::mutex mutex_;
    /// Guards the single shared postprocessor_ against concurrent
    /// completions. Taken in exactly one place, Decode(); never held across
    /// the user callback, and never held at the same time as mutex_.
    std::mutex postprocess_mutex_;
    std::condition_variable done_;
    detail::PendingJobs<Job> pending_;
    std::size_t submitted_;
    std::size_t completed_;
};

/// StageMaker for one (factory, result) pair. The generated registry takes
/// the address of one instantiation per model.
template <class FactoryT, class ResultT>
std::unique_ptr<IStage> MakeTypedStage(const std::string& model_path,
                                       const ModelInfo& info,
                                       const StageParams& params) {
    std::unique_ptr<FactoryT> factory(new FactoryT());
    return std::unique_ptr<IStage>(new TypedStage<FactoryT, ResultT>(
        std::move(factory), model_path, info, params));
}

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_REGISTRY_TYPED_STAGE_HPP
