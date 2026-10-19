/**
 * @file fake_model_registry.hpp
 * @brief Scripted IModelRegistry for hardware-free tests.
 *
 * Because the engine takes IModelRegistry by injection, the entire engine —
 * graph semantics, ROI routing, coordinate restoration, sync/async parity,
 * drain, error propagation, back-pressure — is testable with no NPU.
 *
 * Out-of-order callback arrival is the point: it is hard to reproduce on real
 * hardware and is exactly the class of bug that silently corrupts results.
 * FakeStage's delivery-order mode (forward or reverse) lets a test force that
 * arrival pattern deterministically instead of hoping a real engine races the
 * right way.
 */
#ifndef DXAPP_GRAPH_TEST_FAKE_MODEL_REGISTRY_HPP
#define DXAPP_GRAPH_TEST_FAKE_MODEL_REGISTRY_HPP

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/graph/i_registry.hpp"

namespace dxapp {
namespace graph {

/**
 * @brief Jobs outstanding across EVERY stage one registry handed out.
 *
 * A per-stage high-water mark cannot tell "od, then seg, then dep, one at a
 * time" from "od, seg and dep all in flight together" — both leave every
 * stage with a peak of one. The spec requires the second (design spec,
 * "Dispatch on ready, not waves": a node is submitted the moment its counter
 * reaches zero), so the fake has to observe the property across stages, in
 * one shared counter that every stage of a registry increments on submit and
 * decrements on delivery.
 */
struct InflightMeter {
    int outstanding;
    int peak;
    InflightMeter() : outstanding(0), peak(0) {}

    void Enter() {
        ++outstanding;
        if (outstanding > peak) peak = outstanding;
    }
    void Leave() { --outstanding; }
};
typedef std::shared_ptr<InflightMeter> InflightMeterPtr;

class FakeStage : public IStage {
 public:
    /// The order flush()/DeliverAll() hand pending completions to their
    /// callbacks. kReverse is what Task 10 uses to force out-of-order
    /// arrival deterministically. kShuffled draws from a seeded RNG (see
    /// SetShuffleSeed).
    enum DeliveryOrder { kForward, kReverse, kShuffled };

    FakeStage(const ModelInfo& info, StageDataPtr scripted,
              int latency_ms, const std::string& failure,
              InflightMeterPtr meter)
        : info_(info), scripted_(scripted),
          latency_ms_(latency_ms), failure_(failure), order_(kForward),
          double_fire_(false), max_pending_(0), meter_(meter),
          flush_delivers_(true), poll_delivers_(true) {}

    /**
     * @brief The failure message this stage reports for this input.
     *
     * Derived from the input, never from a call counter, so run() and
     * submit() report the identical message for the identical crop — a
     * message that depended on arrival order would make the two executors
     * disagree on report.error for no reason other than scheduling.
     *
     * Distinct per crop (Task 10 fix round 1): with one shared string for
     * every crop of a node, "keep the last crop's message" and "keep the
     * first crop's message" are indistinguishable, and SyncExecutor keeps
     * the last. Full-frame inputs keep the bare message.
     */
    std::string FailureFor(const StageInput& input) const {
        if (failure_.empty() || !input.origin.from_roi) return failure_;
        std::ostringstream text;
        text << failure_ << " [roi " << input.origin.roi_index << "]";
        return text.str();
    }

    /// The payload for this input: scripted per frame when a script is set,
    /// keyed by pixel (0,0) channel 0 so a test can give each frame its own
    /// detections; the model's single scripted payload otherwise.
    StageDataPtr PayloadFor(const StageInput& input) const {
        if (script_.empty() || input.image.empty() ||
            input.image.type() != CV_8UC3) {
            return scripted_;
        }
        const int key = input.image.at<cv::Vec3b>(0, 0)[0];
        std::map<int, StageDataPtr>::const_iterator it = script_.find(key);
        return it == script_.end() ? scripted_ : it->second;
    }

    StageResult run(const StageInput& input) {
        if (latency_ms_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(latency_ms_));
        seen_.push_back(input);
        if (!failure_.empty()) throw std::runtime_error(FailureFor(input));
        StageResult result;
        result.data = PayloadFor(input);
        result.ports = ports_;
        result.origin = input.origin;
        return result;
    }

    void submit(const StageInput& input, StageCallback callback) {
        seen_.push_back(input);
        pending_.push_back(std::make_pair(input, callback));
        if (pending_.size() > max_pending_) max_pending_ = pending_.size();
        if (meter_) meter_->Enter();
    }

    /// Block until every submitted job's callback has run - unless this
    /// stage was told to withhold completions from flush(), which is how a
    /// test stands in for a plugin that only delivers when poll()ed.
    void flush() {
        if (flush_delivers_) DeliverAll();
    }

    /// Clause (6): deliver whatever this stage is holding, in its
    /// delivery-order mode. Holding everything until asked is exactly the
    /// behaviour the progress clause exists for.
    void poll() {
        if (poll_delivers_) DeliverAll();
    }

    /// Deliver one pending completion, by index — lets a test force
    /// out-of-order arrival deterministically.
    ///
    /// Fires the callback twice when SetDoubleDelivery(true) is set. That is
    /// not a hypothetical: on real hardware a blocking Run() also fires the
    /// engine's registered callback (i_registry.hpp, Task 3 spike, 5 Run()
    /// calls with zero RunAsync produced exactly 5 callback fires), so a
    /// stage that mixes run() and submit() against one engine can hand the
    /// executor the same logical completion twice. An executor that appends
    /// blindly would then report one crop twice and drop none, which is a
    /// silent result corruption rather than a crash.
    void Deliver(std::size_t index) {
        StageInput input = pending_[index].first;
        StageCallback callback = pending_[index].second;
        pending_.erase(pending_.begin() + index);
        if (meter_) meter_->Leave();
        if (latency_ms_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(latency_ms_));
        if (!failure_.empty()) {
            const std::string message = FailureFor(input);
            callback(StageResult(), message);
            if (double_fire_) callback(StageResult(), message);
            return;
        }
        StageResult result;
        result.data = PayloadFor(input);
        result.ports = ports_;
        result.origin = input.origin;
        callback(result, std::string());
        if (double_fire_) callback(result, std::string());
    }

    /// Deliver every pending completion, honouring the delivery order.
    void DeliverAll() {
        while (!pending_.empty()) Deliver(NextIndex());
    }

    void SetDeliveryOrder(DeliveryOrder order) { order_ = order; }

    void SetDoubleDelivery(bool on) { double_fire_ = on; }

    /// kShuffled draws from a seeded std::mt19937, whose sequence the
    /// standard fixes, so a shuffled delivery is identical on every run
    /// and every platform.
    void SetShuffleSeed(unsigned seed) {
        order_ = kShuffled;
        rng_.seed(seed);
    }
    void SetFlushDelivers(bool on) { flush_delivers_ = on; }
    void SetPollDelivers(bool on) { poll_delivers_ = on; }
    void SetScript(int key, StageDataPtr payload) { script_[key] = payload; }
    /// Every successful result carries these ports (U-08).
    void SetPorts(const StagePorts& ports) { ports_ = ports; }

    std::size_t pending() const { return pending_.size(); }

    /// Count this stage in `live` until it is destroyed.
    void SetLiveCounter(const std::shared_ptr<int>& live) {
        live_ = live;
        if (live_) ++*live_;
    }
    ~FakeStage() {
        if (live_) --*live_;
    }

    /// The most jobs this stage ever held at once. An executor honouring a
    /// back-pressure limit of N never lets this exceed N; without the
    /// limit the whole batch is outstanding at once and this reaches the
    /// batch size. Nothing else in a FrameReport distinguishes the two,
    /// which is why the fake has to observe it directly.
    std::size_t max_pending() const { return max_pending_; }
    int latency_ms() const { return latency_ms_; }

    Shape outputShape() const { return info_.output_shape; }
    InputContract inputContract() const { return info_.input_contract; }

    /// Every input passed to run() or submit(), in call order. cv::Mat is
    /// refcounted, so each copy shares pixels with the original - tests
    /// compare `data` pointers rather than pixel contents to confirm which
    /// image a stage actually received.
    const std::vector<StageInput>& seen() const { return seen_; }

 private:
    std::size_t NextIndex() {
        switch (order_) {
            case kForward:  return 0;
            case kReverse:  return pending_.size() - 1;
            case kShuffled: return static_cast<std::size_t>(rng_() % pending_.size());
        }
        return 0;
    }

    ModelInfo info_;
    StageDataPtr scripted_;
    int latency_ms_;
    std::string failure_;
    DeliveryOrder order_;
    bool double_fire_;
    std::size_t max_pending_;
    InflightMeterPtr meter_;
    bool flush_delivers_;
    bool poll_delivers_;
    std::mt19937 rng_;
    std::map<int, StageDataPtr> script_;
    std::vector<std::pair<StageInput, StageCallback> > pending_;
    std::vector<StageInput> seen_;
    StagePorts ports_;
    std::shared_ptr<int> live_;
};

class FakeModelRegistry : public IModelRegistry {
 public:
    FakeModelRegistry() : meter_(new InflightMeter()), live_(new int(0)) {}

    /// The most jobs that were outstanding at one instant across every
    /// stage this registry created. An executor that submits each node and
    /// waits for it before submitting the next never exceeds that node's own
    /// batch size here, however many siblings were ready at the same moment.
    int peak_inflight() const { return meter_->peak; }

    void AddModel(const ModelInfo& info, StageDataPtr scripted) {
        infos_[info.model_name] = info;
        scripted_[info.model_name] = scripted;
        order_.push_back(info.model_name);
    }

    /// run() and every delivery of stages created for this model sleep this long first. Must be set before StageGraph::Build.
    void SetLatency(const std::string& model_name, int latency_ms) {
        latency_[model_name] = latency_ms;
    }

    /// Mirrors FakeStage::DeliveryOrder. Declared separately rather than
    /// aliased so a test reads FakeModelRegistry::kReverse without having
    /// to know which class owns the enum; createStage() maps it across, so
    /// the two cannot drift apart silently — a value added on one side and
    /// not the other fails to compile in the mapping below.
    enum DeliveryOrder { kForward, kReverse, kShuffled };

    /// The order in which stages created for this model hand pending
    /// completions to their callbacks. Must be set before StageGraph::Build
    /// creates the stage.
    void SetDeliveryOrder(const std::string& model_name, DeliveryOrder order) {
        order_mode_[model_name] = order;
    }

    /// Make stages created for this model fire every completion callback
    /// twice (see FakeStage::Deliver). Must be set before StageGraph::Build.
    void SetDoubleDelivery(const std::string& model_name, bool on) {
        double_fire_[model_name] = on;
    }

    void SetFailure(const std::string& model_name, const std::string& message) {
        failure_[model_name] = message;
    }

    /// createStage() for this model throws std::runtime_error(message) once
    /// `succeed_first` stages of it have been created - a device that loads
    /// the first copy of a model and runs out of memory on the next. Must be
    /// set before StageGraph::Build.
    void SetCreateFailure(const std::string& model_name, const std::string& message,
                          int succeed_first = 0) {
        create_failure_[model_name] = std::make_pair(succeed_first, message);
    }

    /// Stages created for this model deliver in a fixed-seed shuffled order.
    /// Must be set before StageGraph::Build.
    void SetShuffleSeed(const std::string& model_name, unsigned seed) {
        shuffle_seed_[model_name] = seed;
    }
    void SetFlushDelivers(const std::string& model_name, bool on) {
        flush_delivers_[model_name] = on;
    }
    void SetPollDelivers(const std::string& model_name, bool on) {
        poll_delivers_[model_name] = on;
    }
    /// Stages created for this model return `payload` for an input whose
    /// pixel (0,0) channel 0 equals `key`. Must be set before
    /// StageGraph::Build.
    void SetScriptByFirstPixel(const std::string& model_name, int key,
                               StageDataPtr payload) {
        script_[model_name][key] = payload;
    }
    /// Every successful result of stages created for this model carries
    /// `ports` (U-08). Must be set before StageGraph::Build.
    void SetPorts(const std::string& model_name, const StagePorts& ports) {
        ports_[model_name] = ports;
    }

    /// The last stage handed out for this model, so a test can drive delivery.
    FakeStage* last_stage(const std::string& model_name) const {
        std::map<std::string, FakeStage*>::const_iterator it =
            last_.find(model_name);
        return it == last_.end() ? NULL : it->second;
    }

    /// The params the last stage for this model was created with (U-62), or
    /// NULL when this model never got a stage.
    const StageParams* last_params(const std::string& model_name) const {
        std::map<std::string, StageParams>::const_iterator it =
            params_.find(model_name);
        return it == params_.end() ? NULL : &it->second;
    }

    /// The EngineOptions the last stage for this model was created with
    /// through createStageWith, or NULL when it never was.
    const EngineOptions* last_engine(const std::string& model_name) const {
        std::map<std::string, EngineOptions>::const_iterator it = engine_.find(model_name);
        return it == engine_.end() ? NULL : &it->second;
    }

    /// Records the options, then creates the stage as createStage does.
    std::unique_ptr<IStage> createStageWith(const std::string& model_name,
                                            const std::string& model_path,
                                            const StageParams& params,
                                            const EngineOptions& engine) const {
        std::unique_ptr<IStage> stage = createStage(model_name, model_path, params);
        engine_[model_name] = engine;
        return stage;
    }

    /// How many stages createStage() has handed out for this model.
    int stages_made(const std::string& model_name) const {
        std::map<std::string, int>::const_iterator it = made_.find(model_name);
        return it == made_.end() ? 0 : it->second;
    }

    /// Stages this registry created that are not destroyed yet.
    int live_stages() const { return *live_; }

    const ModelInfo* find(const std::string& model_name) const {
        std::map<std::string, ModelInfo>::const_iterator it =
            infos_.find(model_name);
        return it == infos_.end() ? NULL : &it->second;
    }

    std::vector<ModelInfo> list() const {
        std::vector<ModelInfo> out;
        for (std::size_t i = 0; i < order_.size(); ++i) {
            out.push_back(infos_.find(order_[i])->second);
        }
        return out;
    }

    std::unique_ptr<IStage> createStage(const std::string& model_name,
                                        const std::string& /*model_path*/,
                                        const StageParams& params) const {
        const ModelInfo* info = find(model_name);
        if (info == NULL) {
            throw std::runtime_error("unknown model " + model_name);
        }
        if (!info->ready) {
            throw std::runtime_error("model not ready: " + info->not_ready_reason);
        }
        std::map<std::string, std::pair<int, std::string> >::const_iterator refuse =
            create_failure_.find(model_name);
        if (refuse != create_failure_.end() &&
            created_[model_name]++ >= refuse->second.first) {
            throw std::runtime_error(refuse->second.second);
        }
        int latency = 0;
        std::map<std::string, int>::const_iterator lat = latency_.find(model_name);
        if (lat != latency_.end()) latency = lat->second;

        std::string failure;
        std::map<std::string, std::string>::const_iterator fail =
            failure_.find(model_name);
        if (fail != failure_.end()) failure = fail->second;

        FakeStage* stage = new FakeStage(
            *info, scripted_.find(model_name)->second, latency, failure,
            meter_);

        std::map<std::string, DeliveryOrder>::const_iterator ord =
            order_mode_.find(model_name);
        if (ord != order_mode_.end()) {
            switch (ord->second) {
                case kForward:  stage->SetDeliveryOrder(FakeStage::kForward); break;
                case kReverse:  stage->SetDeliveryOrder(FakeStage::kReverse); break;
                case kShuffled: stage->SetDeliveryOrder(FakeStage::kShuffled); break;
            }
        }
        std::map<std::string, bool>::const_iterator dbl =
            double_fire_.find(model_name);
        if (dbl != double_fire_.end()) stage->SetDoubleDelivery(dbl->second);

        std::map<std::string, unsigned>::const_iterator seed =
            shuffle_seed_.find(model_name);
        if (seed != shuffle_seed_.end()) stage->SetShuffleSeed(seed->second);
        std::map<std::string, bool>::const_iterator fl =
            flush_delivers_.find(model_name);
        if (fl != flush_delivers_.end()) stage->SetFlushDelivers(fl->second);
        std::map<std::string, bool>::const_iterator pl =
            poll_delivers_.find(model_name);
        if (pl != poll_delivers_.end()) stage->SetPollDelivers(pl->second);
        std::map<std::string, std::map<int, StageDataPtr> >::const_iterator sc =
            script_.find(model_name);
        if (sc != script_.end()) {
            for (std::map<int, StageDataPtr>::const_iterator s = sc->second.begin();
                 s != sc->second.end(); ++s) {
                stage->SetScript(s->first, s->second);
            }
        }
        if (ports_.count(model_name) != 0) stage->SetPorts(ports_.find(model_name)->second);

        stage->SetLiveCounter(live_);
        ++made_[model_name];
        last_[model_name] = stage;
        params_[model_name] = params;
        return std::unique_ptr<IStage>(stage);
    }

 private:
    std::map<std::string, ModelInfo> infos_;
    std::map<std::string, StageDataPtr> scripted_;
    std::map<std::string, int> latency_;
    std::map<std::string, std::string> failure_;
    std::map<std::string, std::pair<int, std::string> > create_failure_;
    mutable std::map<std::string, int> created_;
    std::map<std::string, DeliveryOrder> order_mode_;
    std::map<std::string, bool> double_fire_;
    std::map<std::string, unsigned> shuffle_seed_;
    std::map<std::string, bool> flush_delivers_;
    std::map<std::string, bool> poll_delivers_;
    std::map<std::string, std::map<int, StageDataPtr> > script_;
    std::map<std::string, StagePorts> ports_;
    std::vector<std::string> order_;
    InflightMeterPtr meter_;
    std::shared_ptr<int> live_;
    mutable std::map<std::string, int> made_;
    mutable std::map<std::string, FakeStage*> last_;
    mutable std::map<std::string, StageParams> params_;
    mutable std::map<std::string, EngineOptions> engine_;
};

/**
 * @brief A stage that delivers from its OWN worker thread, out of order.
 *
 * FakeStage delivers on the driver thread, inside poll() or flush(), so
 * everything the executor does on top of it is single-threaded. Real
 * hardware is not: callbacks arrive on the runtime's thread while the driver
 * is somewhere else - waiting on its completion queue, admitting a frame,
 * polling another stage. This wraps any stage: submit() queues the job, and
 * a worker picks a queued job at random, sleeps a bounded random delay, runs
 * the wrapped stage's run() and fires the callback. The picks and delays come
 * from a fixed-seed generator; the thread interleaving does not, so a run is
 * not bit-reproducible - which is the point.
 *
 * Conforms to IStage's threading contract (i_registry.hpp): poll() stays the
 * no-op default, because this stage delivers on its own initiative (6);
 * flush() blocks until every accepted job's callback has RETURNED (4); no
 * lock is held while a callback runs; and the destructor lets the worker
 * deliver every job still queued, then joins it, before any member it
 * touches is destroyed (5). The worker thread is declared - and therefore
 * started - last, after everything it reads.
 */
class ThreadedStage : public IStage {
 public:
    ThreadedStage(std::unique_ptr<IStage> inner, unsigned seed, int min_delay_ms,
                  int max_delay_ms)
        : inner_(std::move(inner)), rng_(seed), min_delay_ms_(min_delay_ms),
          max_delay_ms_(max_delay_ms < min_delay_ms ? min_delay_ms : max_delay_ms),
          stop_(false), busy_(0), worker_(&ThreadedStage::Loop, this) {}

    ~ThreadedStage() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        worker_.join();  // after the worker has delivered everything queued
    }

    StageResult run(const StageInput& input) {
        std::lock_guard<std::mutex> lock(inner_mutex_);
        return inner_->run(input);
    }

    void submit(const StageInput& input, StageCallback callback) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queued_.push_back(std::make_pair(input, callback));
        }
        wake_.notify_all();
    }

    void flush() {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [this] { return queued_.empty() && busy_ == 0; });
    }

    Shape outputShape() const { return inner_->outputShape(); }
    InputContract inputContract() const { return inner_->inputContract(); }

 private:
    void Loop() {
        for (;;) {
            std::pair<StageInput, StageCallback> job;
            int delay_ms = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || !queued_.empty(); });
                if (queued_.empty()) return;  // stopping, nothing left to deliver
                const std::size_t pick =
                    static_cast<std::size_t>(rng_() % queued_.size());
                job = queued_[pick];
                queued_.erase(queued_.begin() + static_cast<std::ptrdiff_t>(pick));
                delay_ms = min_delay_ms_ + static_cast<int>(
                    rng_() % static_cast<unsigned>(max_delay_ms_ - min_delay_ms_ + 1));
                ++busy_;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            StageResult result;
            std::string error;
            try {
                std::lock_guard<std::mutex> lock(inner_mutex_);
                result = inner_->run(job.first);
            } catch (const std::exception& failure) {
                error = failure.what();
                result = StageResult();
            }
            result.origin = job.first.origin;
            job.second(result, error);  // no lock held
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --busy_;
            }
            idle_.notify_all();
        }
    }

    std::unique_ptr<IStage> inner_;
    std::mutex inner_mutex_;  ///< run() may be called from the driver too
    std::mt19937 rng_;        ///< touched only under mutex_
    const int min_delay_ms_;
    const int max_delay_ms_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable idle_;
    std::deque<std::pair<StageInput, StageCallback> > queued_;
    bool stop_;
    int busy_;
    std::thread worker_;  ///< last: started after every member it reads
};

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_TEST_FAKE_MODEL_REGISTRY_HPP
