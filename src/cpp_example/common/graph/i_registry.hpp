/**
 * @file i_registry.hpp
 * @brief The replaceable boundary between the engine and the model zoo.
 *
 * NORMATIVE. Three rules hold for every file under common/graph/:
 *
 *  1. The engine never includes a concrete registry. graph_config,
 *     graph_runner_sync, graph_runner_async, roi_router and graph_visualizer
 *     include only this header and take the interface by constructor
 *     injection.
 *  2. The registry knows nothing about graph semantics. It has no notion of
 *     nodes, edges or validation rules. It answers "name -> info" and
 *     "name -> stage".
 *  3. ModelInfo exposes value types only. No pointer containers, no
 *     templates, no handles.
 *  4. A ModelInfo* handed out by find() stays valid for the registry's
 *     whole lifetime, and NO other call on that registry invalidates it -
 *     createStage() included. See find()'s own comment for why this is the
 *     rule that most needs an implementer's attention.
 *
 * A later plugin registry replaces the implementation and nothing else:
 * the graph schema, the executors, the router, the visualizer and this
 * header all stay as they are.
 *
 * Limitation: std::string crossing a .so boundary is safe only within one
 * toolchain. A third-party plugin built with a different compiler needs a C
 * ABI shim. This interface has five methods, so that shim is a mechanical
 * 1:1 wrapper.
 */
#ifndef DXAPP_GRAPH_I_REGISTRY_HPP
#define DXAPP_GRAPH_I_REGISTRY_HPP

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/graph/shape.hpp"

namespace dxapp {
namespace graph {

/// One unit of work handed to a stage: a full frame or a crop, plus its origin.
struct StageInput {
    cv::Mat image;
    RoiRef origin;
};

/// Completion notification. error is empty on success; data is null on failure.
typedef std::function<void(const StageResult&, const std::string& error)> StageCallback;

/**
 * @brief Runtime parameters from the graph node's "params".
 *
 * "Overriding config.json" describes what THIS project's registry does with
 * them, not something this interface mandates. That precedence - factory
 * defaults < config.json < node params - is a zoo-side convention: the
 * config.json it names is resolved from the source tree, at
 * <source>/<task>/<family>/<variant>/config.json, so a plugin registry shipped as a
 * .so with no source tree beside it cannot reproduce the middle layer and
 * should not pretend to.
 *
 * What the interface does guarantee is the top layer: whatever is in here
 * is the caller's explicit, per-node intent and must win over whatever
 * defaults the implementation applies underneath. A plugin that has no
 * config.json layer simply has two layers instead of three; that is
 * conforming, and --list-models output is unaffected either way.
 *
 * Text and lists (U-62) travel to loadConfig in the same overlay as the
 * numbers. A list from config.json survives an overlay that does not name
 * it: the overlay carries it over, because a factory that reads
 * get_string_list("k") on the overlay would otherwise reset it to empty.
 * Which keys a model reads as text is ModelInfo::text_params; ValidateGraph
 * refuses a value of the wrong kind before any stage is built.
 */
struct StageParams {
    std::map<std::string, double> numeric;
    std::map<std::string, std::string> text;                  ///< string values
    std::map<std::string, std::vector<std::string> > lists;   ///< arrays of strings
};

/// Per-node engine settings from the graph's "engine" key, handed to the
/// registry with the stage request.
struct EngineOptions {
    int buffers;  ///< DX-RT I/O buffer count; 0 = the runtime's default

    EngineOptions() : buffers(0) {}
};

/**
 * @brief A model, type-erased. The only model abstraction the engine sees.
 *
 * Hardware facts (Task 3 spike, dxrt::InferenceEngine, confirmed 4/4 runs
 * across 3 models) that a real implementation of this interface must honour
 * at the point of use:
 *
 *  - Wait(jobId) returns empty output once a callback is registered on that
 *    engine. A real stage must take its result from the callback, never
 *    from a Wait() return value, even on the synchronous run() path.
 *  - A blocking Run() also fires the registered per-engine callback (5
 *    Run() calls with zero RunAsync produced exactly 5 callback fires). A
 *    real stage that offers both run() and submit() against one underlying
 *    engine with one registered callback must not let the synchronous path
 *    re-enter or double-fire the asynchronous completion path — run() and
 *    submit() completions share the same callback, so the implementation
 *    has to disambiguate which logical call a given callback fire belongs
 *    to (e.g. by job id), not assume submit()'s callback is the only source
 *    of completions.
 *
 * THROW CONTRACT (NORMATIVE). The two execution paths report failure
 * differently, and an implementation that mixes them up breaks sync/async
 * parity on the error path - a divergence no fake can demonstrate, because
 * a fake chooses which path it fails on:
 *
 *  - run() THROWS std::runtime_error on any failure. The synchronous
 *    executor wraps every run() in try/catch and turns the message into
 *    report.error.
 *  - submit() DOES NOT THROW. Every failure - an unusable input, a
 *    preprocessing error, a runtime that refuses the submission, a
 *    postprocessing error - is reported through the StageCallback's
 *    `error` argument, carrying the same message run() would have thrown.
 *    One error channel, not two: a job that fails before it reaches the
 *    hardware must be indistinguishable, to the caller, from one that
 *    fails on it. Without this rule the asynchronous executor would have
 *    to handle failures two ways and would still not agree with the
 *    synchronous one about which node's message survives.
 *  - flush() DOES NOT THROW, and must genuinely block until every
 *    submitted job's callback has RETURNED. Returning while a delivery is
 *    still in progress is not a race an in-process fake can expose (a fake
 *    delivers inline, so its counter is always already settled) but it
 *    silently loses results on real hardware, where the callback runs on
 *    the runtime's own thread.
 *
 * The asynchronous executor nonetheless catches around submit() and
 * flush(), so a future implementation that violates this contract degrades
 * to a reported node error instead of terminating the process.
 *
 * ORIGIN CONTRACT (NORMATIVE). Every result a stage returns - from run(),
 * and in every submit() callback - must carry `result.origin =
 * input.origin`, copied verbatim. The executors do not stamp it: they hand
 * the origin in and read it back out. `origin.inv_align` maps the result's
 * coordinates back to the source frame - the identity on the source frame,
 * a scale after a hand-off, the crop's own mapping for a ROI job - so a
 * stage that drops or rebuilds it reports, serializes and draws a detector
 * after super-resolution at the wrong place. `parent_node`, `parent_index`
 * and `roi_index` are the key both executors sort a node's ROI results by
 * (ByOrigin) before storing them, so a stage that loses them breaks the ROI
 * ordering that sync/async parity relies on.
 *
 * PORTS CONTRACT (NORMATIVE). A model may declare extra outputs in
 * ModelInfo::ports; its primary output stays `result.data` and is named
 * ToString(outputShape()). On success `result.ports` holds exactly one
 * non-null payload per declared port, of the declared shape, and nothing
 * else; a failed result carries no ports. The executors copy ports verbatim
 * and hand a port's payload along an edge whose "port" names it. A model
 * that declares no ports leaves `ports` empty and is reported exactly as
 * before ports existed.
 *
 * THREADING (NORMATIVE). Callers were relying on folklore here - on what
 * common/runner/'s idiom happens to do - and folklore is not a basis for
 * memory safety, so the rules are written down. This section is written for
 * someone with no access to this project's implementation: a plugin author
 * behind the dlopen registry, and a factory author whose preprocessor and
 * postprocessor a stage will call.
 *
 * (1) WHICH CALLS THE CALLER SERIALIZES.
 *     The caller serializes every call it makes INTO one IStage instance:
 *     run(), submit(), flush() and poll() on a single stage are never invoked
 *     concurrently, with each other or with themselves. That is a promise,
 *     not an accident of how the executors happen to be written, and an
 *     implementation may rely on it - per-stage scratch used only inside
 *     those calls (a preprocessor, an input buffer, a shape cache) needs no
 *     lock. Two things are NOT covered by it: different IStage instances
 *     are driven concurrently, so anything shared BETWEEN stages must be
 *     synchronized; and a completion callback can run at any time during
 *     any of these calls, so state touched by BOTH a method and a callback
 *     is still shared state (see (3)).
 *
 * (2) WHICH THREAD RUNS WHAT. Three separate questions with three
 *     different answers. A factory author who collapses them into one will
 *     write a postprocessor that is unsafe in exactly the way that is
 *     hardest to reproduce:
 *
 *      - The PREPROCESSOR runs on the thread that called run() or submit(),
 *        inside that call, and is therefore serialized by (1).
 *      - The POSTPROCESSOR need not. A stage backed by an asynchronous
 *        runtime decodes the output where the runtime hands it over - the
 *        runtime's completion thread - and it does so for run() JUST AS
 *        MUCH AS for submit(), because the result reaches such a stage only
 *        through the completion path in both cases. A postprocessor must
 *        therefore be safe to run on a thread that is not the caller's, and
 *        must not assume it is the same thread that ran the preprocessor.
 *      - DELIVERY is the third question, and "run() delivers on the calling
 *        thread" answers only that one. run() returns its result to its own
 *        caller and never reports through a StageCallback. A StageCallback
 *        passed to submit() may be invoked on ANY thread, typically one the
 *        runtime owns and the caller never created, and may be invoked
 *        INLINE, before submit() returns, when the submission fails early
 *        enough.
 *
 * (3) CALLBACK CONCURRENCY.
 *     Callbacks are NOT guaranteed to be serialized with each other. Two
 *     completions from one stage may run CONCURRENTLY. Whatever a callback
 *     writes into must therefore be safe against itself: the underlying
 *     runtime here (dxrt::InferenceEngine::RegisterCallback) documents no
 *     threading guarantee at all, and assuming one costs memory corruption
 *     rather than a clean failure when it turns out to be wrong.
 *
 * (4) FLUSH IS THE ORDERING POINT.
 *     flush() returns only after every callback submitted to that stage has
 *     RETURNED, and it establishes a happens-before edge, so everything
 *     those callbacks wrote is visible to the thread that called flush().
 *     That is the ONLY ordering guarantee this interface gives.
 *
 * (5) DESTRUCTION.
 *     Destroying a stage with work outstanding is the IMPLEMENTATION's
 *     problem, not the caller's: ~IStage must not return while a callback
 *     can still fire. It waits for every outstanding completion and detaches
 *     itself from the runtime BEFORE any member a callback touches is
 *     destroyed - note that members are otherwise torn down in reverse
 *     declaration order, which will free a mutex the callback still needs
 *     while the runtime handle that can still invoke it is alive. Two
 *     mirror-image obligations fall on the caller: it must not destroy a
 *     stage from inside that stage's own callback (the destructor's wait
 *     would deadlock against the callback it is waiting for), and it must
 *     not read destruction as cancellation - destruction BLOCKS on
 *     outstanding work. Calling flush() first is not required; it only
 *     moves that cost to a line where it is visible.
 *
 *     DESTRUCTION JOINS CLAUSE (1)'s SERIALIZED SET. ~IStage is not an
 *     extra concurrent entry point: the caller must not destroy a stage
 *     while any call into it is in progress, on its own thread or another.
 *     A flush() running on a second thread when the destructor starts is
 *     undefined, and undefined in the worst way - the destructor tears down
 *     the very mutex and condition variable that call is blocked on. The
 *     ONE thing that may legitimately still be outstanding when the
 *     destructor is entered is a completion callback, because waiting for
 *     those is exactly the destructor's job.
 *
 * (6) PROGRESS.
 *     Every job accepted by submit() must eventually have its callback
 *     invoked, either on the stage's own initiative (a runtime completion
 *     thread, as this project's hardware stage does) or during a call to
 *     poll(). A stage must NOT require flush() for a callback to fire: the
 *     asynchronous executor keeps several frames' jobs outstanding on one
 *     stage and harvests them as they complete, and flush() waits for ALL
 *     of them, so a flush-driven harvest would serialize every frame
 *     behind the slowest. poll() never blocks, is called only by the
 *     driver thread (clause (1)), and may run callbacks inline. A stage
 *     that pushes its own callbacks leaves poll() as the no-op default; a
 *     batching stage that runs deferred work when asked is conforming as
 *     long as poll() is what asks.
 *
 * What an IMPLEMENTATION must provide: exactly the above. In particular it
 * must not serialize callbacks on the caller's behalf (a caller that needs
 * serialization must do its own locking), and it must not hold an internal
 * lock across the user callback, which would let a callback that calls
 * back into the stage deadlock.
 *
 * Those two rules pull in opposite directions, and the resolution is worth
 * stating because getting it wrong is silent. Clause (3) means a stage that
 * decodes on the completion path must protect whatever that decoding
 * touches - typically ONE postprocessor instance shared by every job, which
 * the framework's IPostprocessor does not promise is reentrant. The lock
 * that protects it must be a SEPARATE lock, scoped to the decode alone, not
 * the lock the stage uses for its own bookkeeping: reuse it and the stage
 * ends up holding it across the user callback, which is the very thing the
 * paragraph above forbids. Serializing the decode also keeps a result from
 * depending on the order completions happened to arrive in, which would
 * otherwise surface as an executor parity failure whose cause is in a
 * factory.
 */
class IStage {
 public:
    virtual ~IStage() {}

    /// Blocking execution. Throws std::runtime_error on inference failure.
    /// The result carries `origin = input.origin` (ORIGIN CONTRACT above).
    virtual StageResult run(const StageInput& input) = 0;

    /**
     * @brief Non-blocking submission. The callback runs on the runtime's
     *        thread. Does not throw: a failure arrives through the
     *        callback's error string. See the THROW CONTRACT above. A
     *        result passed to the callback carries `origin = input.origin`
     *        (ORIGIN CONTRACT above).
     *
     * INPUT LIFETIME (NORMATIVE). submit() must take everything it needs
     * from `input` BEFORE it returns. The StageInput is a temporary the
     * caller may destroy the instant submit() returns - both executors
     * build it as a loop local - so an implementation must not retain a
     * reference or pointer to it, or to input.image.
     *
     * It may retain a COPY of input.image: cv::Mat is refcounted, so a
     * header copy taken inside submit() is cheap and keeps the pixels alive
     * for as long as the implementation holds it. That is the supported way
     * to defer preprocessing. What is NOT supported is reading through
     * `input` after the call, which is what a plugin that stored
     * `const StageInput&` would do.
     *
     * (This project's stage sidesteps the question by preprocessing
     * synchronously and keeping only a byte buffer, a PreprocessContext and
     * a RoiRef - all values. A plugin is free to make the other choice, as
     * long as it makes it before returning.)
     */
    virtual void submit(const StageInput& input, StageCallback callback) = 0;

    /// Block until every submitted job's callback has RETURNED. No throw.
    virtual void flush() = 0;

    /// Deliver any completions this stage is holding (clause (6)). Must not
    /// block and does not throw. Called only by the driver thread; callbacks
    /// may fire inline during this call. Default: nothing is being held.
    virtual void poll() {}

    virtual Shape outputShape() const = 0;
    virtual InputContract inputContract() const = 0;
};

/// One declared extra output of a model (U-08): an edge selects it with
/// "port": "<name>".
struct PortInfo {
    std::string name;
    Shape shape;
    PortInfo() : shape(Shape::kBoxes) {}
    PortInfo(const std::string& port_name, Shape port_shape) : name(port_name), shape(port_shape) {}
};

/**
 * @brief Something besides its own .dxnn that a model needs or says (R9).
 *
 *  - kGallery: a gallery .bin, relative to the repository root. --check
 *    verifies it exists and starts with the DXGAL1 magic.
 *  - kCompanion: another .dxnn the model runs with (EfficientAD's teacher,
 *    student, autoencoder), looked up in --model-dir.
 *  - kNote: text --check prints for the node (the CLIP prompt bank).
 */
struct ResourceInfo {
    enum Kind { kGallery, kCompanion, kNote };
    Kind kind;
    std::string value;   ///< the path, the .dxnn file name, or the note
    ResourceInfo() : kind(kNote) {}
    ResourceInfo(Kind resource_kind, const std::string& resource_value)
        : kind(resource_kind), value(resource_value) {}
};

/**
 * @brief A second name for a model (R6). The model key is the variant;
 *        find() resolves an alias to the variant's ModelInfo.
 *
 *  - kLegacyName: a registry row's own model_name where it differs from its
 *    variant ("yolov8n" is the old name of "yolov8-n_640x640").
 *  - kAliasOf: a registry row that names another row with alias_of
 *    ("deit_base384_distilled" is an alias of "deit-b_384x384_distilled").
 *    --list-models prints these as rows of their own.
 */
struct ModelAlias {
    enum Kind { kLegacyName, kAliasOf };
    std::string name;
    std::string variant;
    Kind kind;
    ModelAlias() : kind(kLegacyName) {}
    ModelAlias(const std::string& alias_name, const std::string& alias_variant, Kind alias_kind)
        : name(alias_name), variant(alias_variant), kind(alias_kind) {}
};

/// One config key a model reads as text (U-62). Every key a model does not
/// declare here is read as a number.
struct ParamInfo {
    enum Kind { kText, kTextList };
    std::string name;
    Kind kind;
    ParamInfo() : kind(kText) {}
    ParamInfo(const std::string& key, Kind key_kind) : name(key), kind(key_kind) {}
};

/**
 * @brief Everything the engine needs to know about one model.
 *
 * ready is false when the model is registered but cannot run: no factory, or
 * no postprocessor. not_ready_reason carries which, for the error message and
 * for --list-models.
 */
struct ModelInfo {
    std::string model_name;   ///< the model key: the variant (R6)
    /// The variant, and the family directory it lives in: the factory and
    /// its config.json are <task>/<family>/<variant>/ under src/cpp_example/.
    /// Empty in a registry without a source tree (a plugin, a test fake).
    std::string variant;
    std::string family;
    std::string task;
    std::string dxnn_file;
    int input_width;
    int input_height;
    Shape output_shape;
    InputContract input_contract;
    bool produces_landmarks;
    bool ready;
    std::string not_ready_reason;
    /// The model zoo's name for this model: what `scripts/download_models.py --models` accepts
    /// (scripts/modelzoo_manifest.json, joined on the .dxnn filename at build
    /// time by scripts/gen_model_registry.py). Empty when the manifest has no
    /// entry for dxnn_file; callers then fall back to model_name.
    ///
    /// 119 of the 348 registry names differ from the manifest's own spelling
    /// (`yolo26l_obb` is `yolo26l-obb` there, `fastsam_s` is `FastSAM-s`).
    /// scripts/download_models.py matches case-insensitively on the manifest
    /// name, the .dxnn file name, or a registry model_name - the last through
    /// aliases it reads from the registry JSON under config/, best effort, so
    /// only where that file is readable next to the script. The manifest
    /// name is the one spelling that matches wherever the script runs, so it
    /// is the one printed. The stable join between the two files is the
    /// .dxnn filename, the basename of the manifest's own download URL. It is
    /// compiled in rather than read at run time, so a binary deployed without
    /// a source tree still prints the right name.
    std::string download_name;
    std::vector<PortInfo> ports;   ///< extra outputs; the primary's name is ToString(output_shape)
    std::vector<ParamInfo> text_params;   ///< keys this model reads as text; every other key
                                          ///< is a number
    /// The registry row's "published" flag: the model zoo publishes this
    /// variant's .dxnn. --list-models prints it.
    bool published;
    std::vector<ResourceInfo> resources;  ///< galleries, companion engines, notes (R9)

    ModelInfo()
        : input_width(0),
          input_height(0),
          output_shape(Shape::kBoxes),
          input_contract(InputContract::kFullFrame),
          produces_landmarks(false),
          ready(false),
          published(false) {}
};

class IModelRegistry {
 public:
    virtual ~IModelRegistry() {}

    /**
     * @brief NULL when the name is not registered.
     *
     * A name is registered when it is a model's key (its variant) or one of
     * its aliases (R6); an alias returns the variant's ModelInfo, whose
     * model_name is the variant, not the name asked for.
     *
     * POINTER LIFETIME (NORMATIVE, and the one obligation this interface
     * places on an implementation that is easy to miss and impossible to
     * catch). The returned pointer:
     *
     *   - is valid for the whole lifetime of the registry, and
     *   - is NOT invalidated by any other call on that registry,
     *     INCLUDING createStage().
     *
     * The engine relies on both halves today and would keep doing so after
     * a .so migration: stage_graph.cpp reads info->output_shape AFTER
     * calling createStage() on the same registry, and graph_config.cpp
     * caches the pointer in a ResolvedNode for the length of a whole
     * validation pass.
     *
     * So an implementation must keep ModelInfo in stable-address storage
     * and must not rehash, reallocate or relocate it - a std::map or a
     * deque that only grows, never a std::vector it push_backs into and
     * never an unordered_map it may rehash. A dlopen registry that
     * materializes entries lazily, or that inserts into its table while
     * createStage() loads a plugin, invalidates the caller's pointer and is
     * undefined behaviour. No test and no guard in this project can detect
     * that, which is exactly why it is written here.
     *
     * Returning by value instead would make the rule unnecessary, and was
     * considered and rejected: the constraint is cheap for an implementer
     * to honour, and the signature is load-bearing for callers that hold a
     * ModelInfo across a scope.
     */
    virtual const ModelInfo* find(const std::string& model_name) const = 0;

    /// Every registered model, ready or not, for --list-models and docs.
    /// By value: the caller owns the copy, so no lifetime rule applies.
    virtual std::vector<ModelInfo> list() const = 0;

    /// Every alias find() resolves, in registration order (R6). By value.
    /// A registry without aliases keeps the default.
    virtual std::vector<ModelAlias> aliases() const { return std::vector<ModelAlias>(); }

    /// Throws std::runtime_error when the model is unknown or not ready,
    /// and ModelContainerError, unchanged, when the .dxnn is a container
    /// this DX-RT cannot load.
    /// Must not invalidate anything find() returned - see find() above.
    virtual std::unique_ptr<IStage> createStage(const std::string& model_name,
                                                const std::string& model_path,
                                                const StageParams& params) const = 0;

    /// createStage with this node's engine settings (StageGraph::Build calls
    /// this one). The default serves registries that predate EngineOptions:
    /// it creates the stage as createStage does when nothing is asked of the
    /// engine, and throws std::runtime_error rather than silently ignoring a
    /// buffer count it cannot apply. Same lifetime and threading rules as
    /// createStage.
    virtual std::unique_ptr<IStage> createStageWith(const std::string& model_name,
                                                    const std::string& model_path,
                                                    const StageParams& params,
                                                    const EngineOptions& engine) const {
        if (engine.buffers != 0) {
            throw std::runtime_error(
                "this model registry cannot set an engine's buffer count "
                "(\"engine\": {\"buffers\": ...}); remove it from the node");
        }
        return createStage(model_name, model_path, params);
    }
};

/**
 * @brief A model's .dxnn is a container this DX-RT cannot load (R12): a v9
 *        file on DX-RT older than 3.5.0.
 *
 * A stage maker throws it before it opens an engine, and createStage()
 * passes it through as this type, so StageGraph::Build can tell it from a
 * device or memory failure: its hint() replaces the "download it again"
 * line, which would fetch the same file.
 */
class ModelContainerError : public std::runtime_error {
 public:
    ModelContainerError(const std::string& what, const std::string& hint)
        : std::runtime_error(what), hint_(hint) {}

    /// What to do instead, e.g. "use the v8 file (dxnn/2_4_0) or upgrade
    /// DX-RT to >= 3.5.0".
    const std::string& hint() const { return hint_; }

 private:
    std::string hint_;
};

/**
 * @brief THREADING for the registry's own surface (NORMATIVE).
 *
 * IStage's clause (1) covers a stage's methods; this covers the registry's,
 * which is a separate question and was unstated.
 *
 *  - find(), list(), and a stage's outputShape()/inputContract() are const
 *    observers and MUST be safe to call concurrently, with each other and
 *    with themselves. An implementation that memoizes inside find() needs
 *    its own lock; "const" is not "thread-safe" by itself.
 *  - createStage() is serialized BY THE CALLER against other createStage()
 *    calls on the same registry. Graph construction is single-threaded, so
 *    an implementation that loads a plugin here does not need its own lock
 *    for that - but see find(): it still must not invalidate a ModelInfo*
 *    while doing it.
 */

/// A non-model step on an edge: crop, track, align, filter.
class IOperator {
 public:
    virtual ~IOperator() {}
    virtual const char* name() const = 0;
};

struct OpParams {
    std::map<std::string, double> numeric;
    std::map<std::string, std::string> text;
};

class IOperatorRegistry {
 public:
    virtual ~IOperatorRegistry() {}
    virtual bool has(const std::string& op) const = 0;
    virtual std::unique_ptr<IOperator> create(const std::string& op,
                                              const OpParams& params) const = 0;
};

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_I_REGISTRY_HPP
