/**
 * @file tiled_sr_stage.hpp
 * @brief Tiled super-resolution as an IStage, tiled exactly as the runner tiles it.
 *
 * Spec §1 U-66. ESPCN is compiled at a fixed 17x17 input, so a plain
 * TypedStage resizes the whole frame to one 17x17 tile and hands off a 68x68
 * image. This stage cuts the frame into halo-overlapped tiles instead and
 * returns the frame at input x scale (275x150 -> 1100x600 at x4). Every step is
 * one of the runner's own functions in common/utility/sr_tiling.hpp -
 * planTiles, prepareLowRes, runAndAssembleTiles, mergeSrLuma -
 * so the graph's image equals the runner's _output_only image by
 * construction, not by a second implementation kept in step.
 *
 * MakeRestorationStage is the StageMaker the codegen emits for every
 * IRestorationFactory row. It opens the engine once and applies the runner's
 * rule: a 1-channel input whose probe output is larger than its input is tiled
 * SR; everything else (the 3-channel DnCNN and Real-ESRGAN models, and
 * 1-channel DnCNN at scale 1) stays on the unchanged TypedStage, through the
 * constructor that takes the already-open engine.
 *
 * WHY THIS FILE IS NOT UNDER common/graph/
 * ----------------------------------------
 * The same reason as typed_stage.hpp: it names concrete factories by template
 * parameter and is compiled only into the generated registry translation
 * units, so it lives on the zoo side of the i_registry.hpp boundary.
 *
 * THE ISTAGE CLAUSES IT RELIES ON (i_registry.hpp, THREADING)
 * -----------------------------------------------------------
 *  - One worker thread owns the engine and runs jobs in submission order. The
 *    engine has NO callback registered, ever, so RunAsync + Wait inside
 *    runAndAssembleTiles gets real outputs (hardware fact 2).
 *  - (1) run(), submit() and flush() are serialized by the caller; the queue
 *    they share with the worker is under mutex_.
 *  - (2)/(3) submit()'s callback runs on the worker, with no lock held, so a
 *    callback may call back into the stage.
 *  - (4) flush() returns once the queue is empty and no job is running; the
 *    worker clears busy_ only after the callback has returned.
 *  - (5) The destructor finishes every queued job, delivers its callback, and
 *    joins before any member is destroyed.
 *  - (6) The worker delivers on its own initiative; poll() stays the no-op.
 *  - run() and submit() share one execution path: run() enqueues a job with
 *    no callback and waits for it. THROW CONTRACT: run() throws the job's
 *    error; submit() never throws and reports it through the callback.
 *  - INPUT LIFETIME: submit() keeps a header copy of input.image, nothing else
 *    from `input`.
 */
#ifndef DXAPP_REGISTRY_TILED_SR_STAGE_HPP
#define DXAPP_REGISTRY_TILED_SR_STAGE_HPP

#include <dxrt/dxrt_api.h>

#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/config/model_config.hpp"
#include "common/graph/i_registry.hpp"
#include "common/graph/result_to_shape.hpp"
#include "common/registry/typed_stage.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/sr_tiling.hpp"

namespace dxapp {
namespace graph {

template <class FactoryT>
class TiledSrStage : public IStage {
 public:
    TiledSrStage(std::unique_ptr<FactoryT> factory, std::unique_ptr<dxrt::InferenceEngine> engine,
                 const ModelInfo& info, const StageParams& params, int scale_x, int scale_y)
        : factory_(std::move(factory)),
          info_(info),
          engine_(std::move(engine)),
          tile_w_(0),
          tile_h_(0),
          scale_x_(scale_x),
          scale_y_(scale_y),
          halo_(srtiling::kDefaultHalo),
          busy_(false),
          stop_(false) {
        // GetInputs() returns by value; keep the copy alive while reading it.
        const dxrt::Tensors inputs = engine_->GetInputs();
        if (inputs.empty()) {
            throw std::runtime_error("model reports no input tensor");
        }
        parseInputShape(inputs.front().shape(), tile_w_, tile_h_);
        if (tile_w_ <= 0 || tile_h_ <= 0) {
            throw std::runtime_error("super-resolution: the model reports no input size");
        }

        // node params > config.json > DXAPP_SR_TILE_HALO > default, through
        // the runner's own resolver.
        int cli = -1;
        const std::map<std::string, double>::const_iterator given =
            params.numeric.find("sr_tile_halo");
        if (given != params.numeric.end()) {
            const double value = given->second;
            if (!std::isfinite(value) || value != std::floor(value) ||
                value < static_cast<double>(INT_MIN) || value > static_cast<double>(INT_MAX)) {
                std::ostringstream text;
                text << "\"sr_tile_halo\" must be a whole number, got " << value;
                throw std::runtime_error(text.str());
            }
            cli = static_cast<int>(value);
        }
        int cfg = -1;
        const ModelConfig file_config = LoadOptionalConfig(detail::StageConfigPath(info_));
        if (file_config.isLoaded()) cfg = file_config.get<int>("sr_tile_halo", -1);
        std::string source, error;
        halo_ = srtiling::resolveHaloFrom(cli, cfg, tile_h_, tile_w_, source, error);
        if (!error.empty()) throw std::runtime_error(error);

        worker_ = std::thread(&TiledSrStage::Loop, this);
    }

    /// Finishes and delivers every queued job, then joins.
    ~TiledSrStage() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    /// Blocking execution. Throws the job's error as std::runtime_error.
    StageResult run(const StageInput& input) {
        std::shared_ptr<Job> job(new Job());
        job->image = input.image;
        job->origin = input.origin;
        Enqueue(job);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!job->done) idle_.wait(lock);
        }
        if (!job->error.empty()) throw std::runtime_error(job->error);
        return job->result;
    }

    /// Non-blocking submission. Never throws; a failure reaches the callback.
    void submit(const StageInput& input, StageCallback callback) {
        try {
            std::shared_ptr<Job> job(new Job());
            job->image = input.image;  // header copy: INPUT LIFETIME
            job->origin = input.origin;
            job->callback = callback;
            Enqueue(job);
        } catch (const std::exception& error) {
            Report(callback, std::string("super-resolution submit failed: ") + error.what());
        } catch (...) {
            Report(callback, "super-resolution submit failed");
        }
    }

    /// Block until the queue is empty and no job is running. No throw.
    void flush() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty() || busy_) idle_.wait(lock);
    }

    Shape outputShape() const { return info_.output_shape; }
    InputContract inputContract() const { return info_.input_contract; }

 private:
    struct Job {
        cv::Mat image;
        RoiRef origin;
        StageCallback callback;  ///< empty on the run() path
        bool done;
        StageResult result;
        std::string error;

        Job() : done(false) {}
    };

    void Enqueue(const std::shared_ptr<Job>& job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(job);
        }
        wake_.notify_one();
    }

    static void Report(const StageCallback& callback, const std::string& error) {
        if (!callback) return;
        try {
            callback(StageResult(), error);
        } catch (...) {
        }
    }

    void Loop() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                while (queue_.empty() && !stop_) wake_.wait(lock);
                if (queue_.empty()) return;  // stop_ is set and nothing is left
                job = queue_.front();
                queue_.pop_front();
                busy_ = true;
            }

            try {
                job->result = Process(job->image, job->origin);
            } catch (const std::exception& failure) {
                job->error = failure.what();
            } catch (...) {
                job->error = "unknown super-resolution failure";
            }
            job->image.release();

            // The callback returns before busy_ clears, so flush() cannot
            // return while a delivery is still in progress (clause (4)).
            if (job->callback) {
                try {
                    job->callback(job->result, job->error);  // result is empty on failure
                } catch (...) {
                    // A consumer's exception must not end the worker thread.
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                job->done = true;
                busy_ = false;
            }
            idle_.notify_all();
        }
    }

    StageResult Process(const cv::Mat& image, const RoiRef& origin) {
        if (image.empty()) {
            throw std::runtime_error("stage received an empty image");
        }
        const int rows = image.rows;
        const int cols = image.cols;
        int padded_h = 0, padded_w = 0;
        std::vector<srtiling::TilePlan> plans;
        srtiling::planTiles(rows, cols, tile_h_, tile_w_, halo_, padded_h, padded_w, plans);

        cv::Mat lr_bgr, lr_gray;
        srtiling::prepareLowRes(image, padded_h, padded_w, lr_bgr, lr_gray);

        cv::Mat sr_y_padded;
        const int tiles_done = srtiling::runAndAssembleTiles(
            *engine_, lr_gray, plans, tile_h_, tile_w_, padded_h * scale_y_,
            padded_w * scale_x_, scale_y_, scale_x_, tile_w_ * scale_x_, sr_y_padded);
        // The graph reports errors, so a missing tile fails the job instead
        // of leaving a black patch.
        if (tiles_done < static_cast<int>(plans.size())) {
            throw std::runtime_error("super-resolution: " +
                                     std::to_string(plans.size() - tiles_done) + " of " +
                                     std::to_string(plans.size()) + " tiles failed");
        }

        const cv::Mat sr_y =
            sr_y_padded(cv::Rect(0, 0, cols * scale_x_, rows * scale_y_)).clone();
        const cv::Mat sr_bgr = srtiling::mergeSrLuma(lr_bgr(cv::Rect(0, 0, cols, rows)), sr_y);

        RestorationResult restored;
        restored.restored_image = sr_bgr;
        StageResult result;
        result.data = ToStageData(restored);
        result.origin = origin;
        return result;
    }

    std::unique_ptr<FactoryT> factory_;
    ModelInfo info_;
    std::unique_ptr<dxrt::InferenceEngine> engine_;  ///< no callback registered, ever
    int tile_w_, tile_h_, scale_x_, scale_y_, halo_;
    std::mutex mutex_;
    std::condition_variable wake_, idle_;
    std::deque<std::shared_ptr<Job> > queue_;
    bool busy_, stop_;
    /// Declared last, so it starts after everything it touches exists.
    std::thread worker_;
};

/**
 * @brief StageMaker for every IRestorationFactory row.
 *
 * The runner's rule: input channels <= 1 and a probe output larger than the
 * input on either axis selects tiled SR. The probe runs before any callback
 * is registered, so it is a plain blocking Run().
 */
template <class FactoryT>
std::unique_ptr<IStage> MakeRestorationStage(const std::string& model_path, const ModelInfo& info,
                                             const StageParams& params) {
    std::unique_ptr<dxrt::InferenceEngine> engine =
        detail::OpenEngine(detail::LoadableModelPath(model_path));
    int scale_x = 1, scale_y = 1;
    const dxrt::Tensors inputs = engine->GetInputs();
    if (!inputs.empty()) {
        const std::vector<int64_t> shape = inputs.front().shape();
        const int channels =
            shape.size() >= 4 ? static_cast<int>(isInputNHWC(shape) ? shape[3] : shape[1]) : 1;
        if (channels <= 1) {
            int width = 0, height = 0;
            parseInputShape(shape, width, height);
            if (width > 0 && height > 0) {
                const std::pair<int, int> scale =
                    srtiling::probeOutputScale(*engine, width, height);
                scale_x = scale.first;
                scale_y = scale.second;
            }
        }
    }
    std::unique_ptr<FactoryT> factory(new FactoryT());
    if (scale_x > 1 || scale_y > 1) {
        return std::unique_ptr<IStage>(new TiledSrStage<FactoryT>(
            std::move(factory), std::move(engine), info, params, scale_x, scale_y));
    }
    return std::unique_ptr<IStage>(new TypedStage<FactoryT, RestorationResult>(
        std::move(factory), std::move(engine), info, params));
}

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_REGISTRY_TILED_SR_STAGE_HPP
