/**
 * @file common_unit_test.cpp
 * @brief Plain-assert test binary for the shared code under common/ that is
 *        not the graph engine (processors, runners' helpers, utilities).
 *
 * Same conventions as common/graph/test/graph_engine_test.cpp: this
 * repository has no gtest and no CTest, so C++ behaviour is exercised by
 * pytest (tests/cpp_example/test_common_unit.py) driving this binary.
 *
 * HARNESS
 *   COMMON_CHECK(cond)   counts a check; on failure prints FAIL file:line
 *                        and CONTINUES, so one run reports every failure.
 *   Skip(test, reason)   counts a skipped case and prints why - a skip is
 *                        loud, never silent.
 *   main() prints "N checks, F failures, S skipped" as its last line and
 *   returns non-zero when F > 0.
 *
 * ADDING TESTS: write a void TestXxx() in its own section below and call it
 * from main(). Keep every test independent of the others, except for the
 * one ordering note on the DXAPP_DEBUG test (see there).
 */
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define DXAPP_TEST_DUP _dup
#define DXAPP_TEST_DUP2 _dup2
#define DXAPP_TEST_FILENO _fileno
#define DXAPP_TEST_CLOSE _close
#else
#include <poll.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#define DXAPP_TEST_DUP dup
#define DXAPP_TEST_DUP2 dup2
#define DXAPP_TEST_FILENO fileno
#define DXAPP_TEST_CLOSE close
#endif

#include <dxrt/dxrt_api.h>

#include "common/base/i_processor.hpp"
#include "common/config/model_config.hpp"
#include "common/processors/anchor_face_postprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/processors/hand_landmark_postprocessor.hpp"
#include "common/processors/picodet_postprocessor.hpp"
#include "common/processors/pose_postprocessor.hpp"
#include "common/processors/serialized_postprocessor.hpp"
#include "common/third_party/nlohmann_json.hpp"
#include "common/utility/camera.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/frame_reorder.hpp"
#include "common/utility/labels.hpp"
#include "common/utility/ordered_queue.hpp"
#include "common/utility/roi_crop.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/sr_tiling.hpp"
#include "common/utility/verify_serialize.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/visualizers/retrieval_visualizer.hpp"
// header-only so this binary tests the CLI's own handler (see its file comment)
#include "multi_model_graph/graph_cli_interrupt.hpp"
#include "panoptic_driving_perception/yolopv2/yolopv2_384x640/factory/yolopv2_384x640_factory.hpp"
#include "pose_estimation/dark_hrnet/dark-hrnet-w32_256x192/factory/dark-hrnet-w32_256x192_factory.hpp"
#include "pose_estimation/vitpose/vitpose-s_256x192/factory/vitpose-s_256x192_factory.hpp"
#include "pose_estimation/yolov5_pose/yolov5-s6-pose_640x640/factory/yolov5-s6-pose_640x640_factory.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;
int g_skipped = 0;

void Check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL %s:%d  %s\n", file, line, expr);
    }
}

/// Loud, counted, and never silent: a skipped case prints why and is
/// reported in the summary line.
void Skip(const char* test, const std::string& reason) {
    ++g_skipped;
    std::printf("SKIP %s: %s\n", test, reason.c_str());
}

}  // namespace

#define COMMON_CHECK(cond) Check((cond), #cond, __FILE__, __LINE__)

namespace {

// =====================================================================
// Helpers
// =====================================================================

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void UnsetEnv(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

/// Redirects the process's stderr (fd 2, so fprintf(stderr) from headers is
/// caught too) into a temp file for the lifetime of the object; Stop()
/// restores stderr and returns what was written.
class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file_ = std::tmpfile();
        if (file_ == NULL) throw std::runtime_error("tmpfile failed");
        saved_fd_ = DXAPP_TEST_DUP(DXAPP_TEST_FILENO(stderr));
        DXAPP_TEST_DUP2(DXAPP_TEST_FILENO(file_), DXAPP_TEST_FILENO(stderr));
    }
    ~StderrCapture() {
        Restore();
        if (file_ != NULL) std::fclose(file_);
    }
    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    std::string Stop() {
        Restore();
        std::string text;
        std::rewind(file_);
        char buffer[4096];
        std::size_t n = 0;
        while ((n = std::fread(buffer, 1, sizeof(buffer), file_)) > 0) {
            text.append(buffer, n);
        }
        return text;
    }

private:
    void Restore() {
        if (saved_fd_ < 0) return;
        std::fflush(stderr);
        DXAPP_TEST_DUP2(saved_fd_, DXAPP_TEST_FILENO(stderr));
        DXAPP_TEST_CLOSE(saved_fd_);
        saved_fd_ = -1;
    }

    std::FILE* file_ = NULL;
    int saved_fd_ = -1;
};

int CountOccurrences(const std::string& text, const std::string& needle) {
    int count = 0;
    std::string::size_type pos = text.find(needle);
    while (pos != std::string::npos) {
        ++count;
        pos = text.find(needle, pos + needle.size());
    }
    return count;
}

// =====================================================================
// A1: SerializedPostprocessor - the async runners' one shared
// postprocessor is entered by every completion callback, and dxrt runs
// those on a pool of output threads.
// =====================================================================

/// A postprocessor that measures how many threads are inside process() at
/// once. It sleeps inside the call so that, without serialization, 8
/// threads reliably overlap.
class ConcurrencyProbe : public dxapp::IPostprocessor<int> {
public:
    std::vector<int> process(const dxrt::TensorPtrs& /*outputs*/,
                             const dxapp::PreprocessContext& ctx) override {
        const int now = ++inside_;
        int seen = max_inside_.load();
        while (now > seen && !max_inside_.compare_exchange_weak(seen, now)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        --inside_;
        ++calls_;
        // Echo an input-derived value so the caller can check that every
        // call's own result came back to it.
        return std::vector<int>(1, ctx.pad_x);
    }
    std::string getModelName() const override { return "ConcurrencyProbe"; }

    int max_inside() const { return max_inside_.load(); }
    int calls() const { return calls_.load(); }

private:
    std::atomic<int> inside_{0};
    std::atomic<int> max_inside_{0};
    std::atomic<int> calls_{0};
};

const int kThreads = 8;
const int kCallsPerThread = 50;

/// Runs kThreads x kCallsPerThread process() calls on `pp`; returns how
/// many calls got back exactly the value they passed in.
int Hammer(dxapp::IPostprocessor<int>& pp) {
    std::atomic<int> correct{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&pp, &correct, t]() {
            dxrt::TensorPtrs outputs;
            for (int i = 0; i < kCallsPerThread; ++i) {
                dxapp::PreprocessContext ctx;
                ctx.pad_x = t * 1000 + i;
                std::vector<int> result = pp.process(outputs, ctx);
                if (result.size() == 1 && result[0] == ctx.pad_x) ++correct;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    return correct.load();
}

void TestUnserializedProbeSeesOverlap() {
    // Proves the harness can see a race at all: through the bare probe,
    // 8 threads are inside process() together. If this ever reads 1 the
    // serialization test below proves nothing.
    ConcurrencyProbe probe;
    const int correct = Hammer(probe);
    COMMON_CHECK(probe.max_inside() > 1);
    COMMON_CHECK(probe.calls() == kThreads * kCallsPerThread);
    COMMON_CHECK(correct == kThreads * kCallsPerThread);
}

void TestSerializedPostprocessorAdmitsOneCallAtATime() {
    std::unique_ptr<ConcurrencyProbe> owned(new ConcurrencyProbe());
    ConcurrencyProbe* probe = owned.get();
    std::shared_ptr<dxapp::IPostprocessor<int>> serialized =
        dxapp::Serialize<int>(std::move(owned));
    COMMON_CHECK(serialized != nullptr);

    const int correct = Hammer(*serialized);
    COMMON_CHECK(probe->max_inside() == 1);
    COMMON_CHECK(probe->calls() == kThreads * kCallsPerThread);
    // All 400 results returned, each to the call that made it.
    COMMON_CHECK(correct == kThreads * kCallsPerThread);
}

/// Records every virtual that reaches it, to prove the decorator forwards
/// each one rather than answering itself.
class ForwardingProbe : public dxapp::IPostprocessor<int> {
public:
    std::vector<int> process(const dxrt::TensorPtrs& outputs,
                             const dxapp::PreprocessContext& ctx) override {
        ++process_calls;
        last_output_count = outputs.size();
        last_pad_y = ctx.pad_y;
        return std::vector<int>{7, 8, 9};
    }
    std::string getModelName() const override {
        ++name_calls;
        return "ForwardingProbe-name";
    }

    int process_calls = 0;
    mutable int name_calls = 0;
    std::size_t last_output_count = 0;
    int last_pad_y = 0;
};

void TestSerializedPostprocessorForwardsEveryVirtual() {
    std::unique_ptr<ForwardingProbe> owned(new ForwardingProbe());
    ForwardingProbe* probe = owned.get();
    std::shared_ptr<dxapp::IPostprocessor<int>> serialized =
        dxapp::Serialize<int>(std::move(owned));

    // getModelName(): forwarded, not re-answered by the decorator.
    COMMON_CHECK(serialized->getModelName() == "ForwardingProbe-name");
    COMMON_CHECK(probe->name_calls == 1);

    // process(): forwarded with the caller's own arguments and result.
    dxrt::TensorPtrs outputs(2);
    dxapp::PreprocessContext ctx;
    ctx.pad_y = 42;
    std::vector<int> result = serialized->process(outputs, ctx);
    COMMON_CHECK(probe->process_calls == 1);
    COMMON_CHECK(probe->last_output_count == 2);
    COMMON_CHECK(probe->last_pad_y == 42);
    COMMON_CHECK(result == std::vector<int>({7, 8, 9}));
}

/// Throws from process(); the decorator must release its lock on the way
/// out, or the next call (on any thread) would deadlock.
class ThrowingProbe : public dxapp::IPostprocessor<int> {
public:
    std::vector<int> process(const dxrt::TensorPtrs&,
                             const dxapp::PreprocessContext&) override {
        if (calls_++ == 0) throw std::runtime_error("first call throws");
        return std::vector<int>(1, calls_);
    }
    std::string getModelName() const override { return "ThrowingProbe"; }

private:
    int calls_ = 0;
};

void TestSerializedPostprocessorReleasesLockOnThrow() {
    std::shared_ptr<dxapp::IPostprocessor<int>> serialized =
        dxapp::Serialize<int>(std::unique_ptr<dxapp::IPostprocessor<int>>(new ThrowingProbe()));
    dxrt::TensorPtrs outputs;
    dxapp::PreprocessContext ctx;
    bool threw = false;
    try {
        serialized->process(outputs, ctx);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    COMMON_CHECK(threw);

    // Second call from ANOTHER thread: a leaked lock would hang here, so
    // bound the wait instead of joining blindly. The thread owns what it
    // touches (shared_ptrs), so detaching it on failure is safe.
    struct Outcome {
        std::atomic<bool> done{false};
        std::vector<int> result;
    };
    std::shared_ptr<Outcome> outcome = std::make_shared<Outcome>();
    std::thread other([serialized, outcome]() {
        dxrt::TensorPtrs no_outputs;
        dxapp::PreprocessContext no_ctx;
        outcome->result = serialized->process(no_outputs, no_ctx);
        outcome->done = true;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!outcome->done && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    COMMON_CHECK(outcome->done.load());
    if (outcome->done) {
        other.join();
        COMMON_CHECK(outcome->result == std::vector<int>(1, 2));
    } else {
        other.detach();  // the run is failing anyway; do not hang it
    }
}

void TestSerializeRejectsNull() {
    bool threw = false;
    try {
        dxapp::Serialize<int>(std::unique_ptr<dxapp::IPostprocessor<int>>());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    COMMON_CHECK(threw);
}

// =====================================================================
// A2: YOLOv5FacePostProcess's one-time DXAPP_DEBUG header is per
// instance and race-free.
// =====================================================================

/// One decoded-format output ([1, 0, 16]: zero candidate rows) - enough
/// to run postprocess() end to end without a model.
dxrt::TensorPtrs EmptyDecodedFaceOutput(std::vector<float>& storage) {
    storage.assign(16, 0.f);
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(0);
    shape.push_back(16);
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "decoded", shape, dxrt::DataType::FLOAT, storage.data()));
    return outputs;
}

const char kFaceHeader[] = "[DEBUG] is_ort_configured=";
const char kFacePerFrame[] = " post-NMS=";

// ORDER NOTE: postprocess() reads DXAPP_DEBUG once per process (a
// function-local static), so this test must run before anything else in
// this binary calls YOLOv5FacePostProcess::postprocess(). main() calls it
// first. It also leaves DXAPP_DEBUG's cached value (`s_debug`) true for the
// rest of this binary: every later YOLOv5FacePostProcess call here prints
// its debug lines to stderr, which no check reads.
void TestAnchorFaceDebugHeaderOncePerInstance() {
    SetEnv("DXAPP_DEBUG", "1");
    std::vector<float> storage;
    dxrt::TensorPtrs outputs = EmptyDecodedFaceOutput(storage);

    StderrCapture capture;
    YOLOv5FacePostProcess first;
    YOLOv5FacePostProcess second;
    std::size_t detections = 0;
    detections += first.postprocess(outputs).size();
    detections += first.postprocess(outputs).size();
    detections += second.postprocess(outputs).size();
    detections += second.postprocess(outputs).size();

    // One instance entered from 8 threads at once still logs its header
    // exactly once.
    YOLOv5FacePostProcess shared;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&shared, &outputs]() { shared.postprocess(outputs); });
    }
    for (auto& thread : threads) thread.join();
    const std::string text = capture.Stop();
    UnsetEnv("DXAPP_DEBUG");

    COMMON_CHECK(detections == 0);
    // first + second + shared: one header each.
    COMMON_CHECK(CountOccurrences(text, kFaceHeader) == 3);
    // The per-frame line still prints on every call (4 + 8).
    COMMON_CHECK(CountOccurrences(text, kFacePerFrame) == 4 + kThreads);
    if (CountOccurrences(text, kFaceHeader) != 3) {
        std::printf("captured stderr:\n%s\n", text.c_str());
    }
}

// =====================================================================
// A3: EfficientNetPostprocessor reports softmax probabilities and
// ImageNet class names (all 111 classification factories share this).
// =====================================================================

/// Wraps `logits` (not copied - caller keeps it alive) as the one FLOAT32
/// output tensor EfficientNetPostprocessor::process() reads.
dxrt::TensorPtrs MakeFloatClassificationOutput(std::vector<float>& logits) {
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(static_cast<int64_t>(logits.size()));
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "logits", shape, dxrt::DataType::FLOAT, logits.data()));
    return outputs;
}

void TestClassificationSoftmaxProbabilitiesInRangeSumToOneOrderPreserved() {
    // A deterministic random permutation of 0..999 gives every class a
    // distinct logit - no ties, so sorting by raw logit is an unambiguous
    // oracle for the expected order (with ties, partial_sort's and
    // std::sort's tie-breaks need not agree, which is not a bug).
    std::vector<int> permutation(1000);
    std::iota(permutation.begin(), permutation.end(), 0);
    unsigned int state = 12345u;
    for (std::size_t i = permutation.size(); i > 1; --i) {
        state = state * 1103515245u + 12345u;
        std::size_t j = (state >> 16) % i;
        std::swap(permutation[i - 1], permutation[j]);
    }
    std::vector<float> logits(1000);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<float>(permutation[i]) / 100.0f - 5.0f;  // [-5, 4.99]
    }

    // top_k == num_classes: process() returns the full softmax distribution
    // so its sum can be checked directly.
    dxapp::EfficientNetPostprocessor pp(1000, 1000);
    dxapp::PreprocessContext ctx;
    std::vector<dxapp::ClassificationResult> results =
        pp.process(MakeFloatClassificationOutput(logits), ctx);
    COMMON_CHECK(results.size() == 1000);

    double sum = 0.0;
    bool all_in_range = true;
    for (const auto& r : results) {
        if (r.confidence < 0.0f || r.confidence > 1.0f + 1e-6f) all_in_range = false;
        sum += r.confidence;
    }
    COMMON_CHECK(all_in_range);
    COMMON_CHECK(std::fabs(sum - 1.0) < 1e-4);

    // Oracle order: argsort the raw logits descending (no ties, per above).
    std::vector<int> expected(1000);
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(expected.begin(), expected.end(),
              [&logits](int a, int b) { return logits[a] > logits[b]; });

    bool order_matches = (results.size() == expected.size());
    for (std::size_t i = 0; order_matches && i < expected.size(); ++i) {
        if (results[i].class_id != expected[i]) order_matches = false;
    }
    COMMON_CHECK(order_matches);
}

void TestClassificationTopClassNameMatchesImageNetLabelZero() {
    std::vector<float> logits(1000, -1.0f);
    logits[0] = 100.0f;  // unambiguous top-1
    dxapp::EfficientNetPostprocessor pp(1000, 5);
    dxapp::PreprocessContext ctx;
    std::vector<dxapp::ClassificationResult> results =
        pp.process(MakeFloatClassificationOutput(logits), ctx);
    COMMON_CHECK(!results.empty());
    if (!results.empty()) {
        COMMON_CHECK(results[0].class_id == 0);
        COMMON_CHECK(results[0].class_name == dxapp::getImageNetClassName(0));
    }
}

void TestClassificationDistributionGuardAvoidsDoubleSoftmax() {
    // Already a valid probability distribution: non-negative, sums to 1.
    // A model whose network already ends in softmax hands us exactly this,
    // and it must come back through unchanged (no double softmax).
    std::vector<float> dist(1000, 0.0f);
    const int peak = 10;
    dist[peak] = 0.6f;
    const float remainder = 0.4f / 999.0f;
    for (std::size_t i = 0; i < dist.size(); ++i) {
        if (static_cast<int>(i) != peak) dist[i] = remainder;
    }

    dxapp::EfficientNetPostprocessor pp(1000, 1);
    dxapp::PreprocessContext ctx;
    std::vector<dxapp::ClassificationResult> results =
        pp.process(MakeFloatClassificationOutput(dist), ctx);
    COMMON_CHECK(results.size() == 1);
    if (!results.empty()) {
        COMMON_CHECK(results[0].class_id == peak);
        // Bit-for-bit (within float epsilon) the same value handed in - a
        // second softmax pass would move it away from 0.6.
        COMMON_CHECK(std::fabs(results[0].confidence - 0.6f) < 1e-5f);

        std::vector<float> double_softmaxed = dxapp::softmax(dist);
        COMMON_CHECK(std::fabs(results[0].confidence - double_softmaxed[peak]) > 1e-3f);
    }
}

void TestClassificationNonImageNetClassCountLeavesNameEmpty() {
    std::vector<float> logits(10, 0.0f);
    logits[3] = 5.0f;
    dxapp::EfficientNetPostprocessor pp(10, 3);
    dxapp::PreprocessContext ctx;
    std::vector<dxapp::ClassificationResult> results =
        pp.process(MakeFloatClassificationOutput(logits), ctx);
    COMMON_CHECK(!results.empty());
    for (const auto& r : results) {
        COMMON_CHECK(r.class_name.empty());
    }
}

void TestClassificationTopKFieldFilled() {
    std::vector<float> logits(1000, -1.0f);
    logits[0] = 100.0f;
    logits[7] = 50.0f;
    dxapp::EfficientNetPostprocessor pp(1000, 5);
    dxapp::PreprocessContext ctx;
    std::vector<dxapp::ClassificationResult> results =
        pp.process(MakeFloatClassificationOutput(logits), ctx);
    COMMON_CHECK(results.size() == 5);
    if (!results.empty()) {
        COMMON_CHECK(results[0].top_k.size() == 5);
        if (results[0].top_k.size() == 5) {
            COMMON_CHECK(results[0].top_k[0].first == results[0].class_id);
            COMMON_CHECK(std::fabs(results[0].top_k[0].second - results[0].confidence) < 1e-6f);
            COMMON_CHECK(results[0].top_k[1].first == 7);
        }
    }
}

void TestClassificationShortTensorNeverReadPast() {
    // A 10-element output while the postprocessor is configured for 1000
    // classes. The tensor is the head of a larger buffer whose element 500
    // is a huge logit: a read past the tensor ranks it first.
    std::vector<float> storage(1000, 0.0f);
    storage[3] = 5.0f;
    storage[500] = 1000.0f;
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(10);
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "logits", shape, dxrt::DataType::FLOAT, storage.data()));

    dxapp::EfficientNetPostprocessor pp(1000, 1000);
    dxapp::PreprocessContext ctx;
    std::vector<dxapp::ClassificationResult> results = pp.process(outputs, ctx);
    COMMON_CHECK(results.size() == 10);
    double sum = 0.0;
    bool in_tensor = true;
    for (const auto& r : results) {
        if (r.class_id < 0 || r.class_id >= 10) in_tensor = false;
        sum += r.confidence;
    }
    COMMON_CHECK(in_tensor);
    COMMON_CHECK(std::fabs(sum - 1.0) < 1e-4);
    if (!results.empty()) COMMON_CHECK(results[0].class_id == 3);

    // A zero-sized output gives no result rather than a read.
    std::vector<int64_t> empty_shape;
    empty_shape.push_back(1);
    empty_shape.push_back(0);
    dxrt::TensorPtrs empty_outputs;
    empty_outputs.push_back(std::make_shared<dxrt::Tensor>(
        "logits", empty_shape, dxrt::DataType::FLOAT, storage.data()));
    COMMON_CHECK(pp.process(empty_outputs, ctx).empty());
}

void TestClassificationIntegerOutputReadAtItsWidth() {
    // An argmax output of one UINT8 element followed by a byte that must not
    // be read: at a 2-byte read the class id would be 0xAB07, not 7.
    uint8_t storage[2] = {7, 0xAB};
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(1);
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "argmax", shape, dxrt::DataType::UINT8, storage));
    dxapp::EfficientNetPostprocessor pp(1000, 5);
    dxapp::PreprocessContext ctx;
    std::vector<dxapp::ClassificationResult> results = pp.process(outputs, ctx);
    COMMON_CHECK(results.size() == 1);
    if (!results.empty()) COMMON_CHECK(results[0].class_id == 7);
    if (!results.empty()) COMMON_CHECK(results[0].class_name == "cock");

    // INT32 argmax is read at 4 bytes.
    int32_t wide = 281;
    dxrt::TensorPtrs wide_outputs;
    wide_outputs.push_back(std::make_shared<dxrt::Tensor>(
        "argmax", shape, dxrt::DataType::INT32, &wide));
    results = pp.process(wide_outputs, ctx);
    COMMON_CHECK(results.size() == 1);
    if (!results.empty()) COMMON_CHECK(results[0].class_id == 281);
    if (!results.empty()) COMMON_CHECK(results[0].class_name == "tabby cat");

    // Not a 1000-class head: an integer output has no name, like a float one.
    dxapp::EfficientNetPostprocessor ten(10, 5);
    results = ten.process(wide_outputs, ctx);
    COMMON_CHECK(results.size() == 1);
    if (!results.empty()) COMMON_CHECK(results[0].class_name.empty());
}

// =====================================================================
// Interrupts (C5, R5).
//   * The first SIGINT/SIGTERM is graceful. Copies of it arriving within
//     kInterruptCoalesceMs (one Ctrl-C forwarded by `timeout`) are the same
//     request; a SIGINT later than that terminates. SIGTERM never
//     escalates.
//   * The image-mode "hold the window open" wait honours the first request.
//
// Every case runs in a fork()ed child, because the thing under test is
// process-wide signal disposition and, for an escalation, the death of the
// process. The child reports what it saw through a pipe as one byte, then
// does the final step; the parent checks both. Every wait is bounded: a
// child that neither reports nor exits in time is SIGKILLed and the case
// fails, with the child's stderr printed.
// =====================================================================

#ifndef _WIN32

struct ChildOutcome {
    bool forked = false;
    bool reported = false;          // the child wrote its report byte
    unsigned char report = 0;
    bool report_timed_out = false;  // no report within the bound
    bool timed_out = false;         // did not end within the bound
    int status = 0;                 // waitpid() status
    std::string child_stderr;       // everything the child wrote to fd 2
};

/// True when `sig`'s current disposition is SIG_DFL.
bool IsDefaultDisposition(int sig) {
    struct sigaction current;
    if (sigaction(sig, nullptr, &current) != 0) return false;
    return current.sa_handler == SIG_DFL;
}

std::string ReadWholeFd(int fd) {
    std::string text;
    if (lseek(fd, 0, SEEK_SET) != 0) return text;
    char buffer[4096];
    for (;;) {
        ssize_t got = read(fd, buffer, sizeof(buffer));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) break;
        text.append(buffer, static_cast<std::size_t>(got));
    }
    return text;
}

/// Wait up to `timeout_ms` for `pid` to end; SIGKILL it if it does not.
void ReapWithin(pid_t pid, int timeout_ms, ChildOutcome* outcome) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (waitpid(pid, &outcome->status, WNOHANG) == pid) return;
        if (std::chrono::steady_clock::now() > deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    outcome->timed_out = true;
    kill(pid, SIGKILL);
    waitpid(pid, &outcome->status, 0);
}

/// Fork; the child runs `child(write_fd)` and must _exit() or die. The
/// parent waits at most `timeout_ms` for one report byte, calls
/// `after_report(pid, byte)`, then waits at most `timeout_ms` for the child
/// to end. The child's stderr goes to an unlinked temporary file and comes
/// back in ChildOutcome::child_stderr.
template <typename Child, typename AfterReport>
ChildOutcome RunInChild(Child child, AfterReport after_report, int timeout_ms) {
    ChildOutcome outcome;
    int fds[2];
    if (pipe(fds) != 0) return outcome;
    char err_path[] = "/tmp/common_unit_test_child_XXXXXX";
    int err_fd = mkstemp(err_path);
    if (err_fd >= 0) unlink(err_path);
    std::fflush(stdout);
    std::fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        if (err_fd >= 0) close(err_fd);
        return outcome;
    }
    if (pid == 0) {
        close(fds[0]);
        if (err_fd >= 0) dup2(err_fd, 2);
        child(fds[1]);
        _exit(98);  // child bodies never return; this is a harness error
    }
    outcome.forked = true;
    close(fds[1]);

    struct pollfd ready = {};
    ready.fd = fds[0];
    ready.events = POLLIN;
    int polled;
    do { polled = poll(&ready, 1, timeout_ms); } while (polled < 0 && errno == EINTR);
    unsigned char byte = 0;
    if (polled > 0) {
        ssize_t got;
        do { got = read(fds[0], &byte, 1); } while (got < 0 && errno == EINTR);
        outcome.reported = got == 1;  // 0: the child closed the pipe unread
    } else {
        outcome.report_timed_out = true;
    }
    close(fds[0]);

    if (outcome.report_timed_out) {
        outcome.timed_out = true;
        kill(pid, SIGKILL);
        waitpid(pid, &outcome.status, 0);
    } else {
        if (outcome.reported) {
            outcome.report = byte;
            after_report(pid, byte);
        }
        ReapWithin(pid, timeout_ms, &outcome);
    }
    if (err_fd >= 0) {
        outcome.child_stderr = ReadWholeFd(err_fd);
        close(err_fd);
    }
    return outcome;
}

/// COMMON_CHECK that, on failure, also says why the child is not
/// trustworthy and shows its stderr.
void CheckChild(const ChildOutcome& outcome, bool condition, const char* expr,
                const char* file, int line) {
    Check(condition, expr, file, line);
    if (condition) return;
    if (outcome.report_timed_out) {
        std::printf("  child sent no report within the bound (SIGKILLed)\n");
    } else if (outcome.timed_out) {
        std::printf("  child did not end within the bound (SIGKILLed)\n");
    }
    std::printf("  child stderr: %s\n", outcome.child_stderr.empty()
                                            ? "(empty)"
                                            : outcome.child_stderr.c_str());
}

#define CHILD_CHECK(outcome, cond) \
    CheckChild((outcome), (cond), #cond, __FILE__, __LINE__)

void WriteByte(int fd, unsigned char byte) {
    ssize_t n;
    do { n = write(fd, &byte, 1); } while (n < 0 && errno == EINTR);
}

void NoAction(pid_t, unsigned char) {}

void SleepMs(long long ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

/// Longer than the coalescing window: a repeat this late is a new press.
const long long kLaterMs = dxapp::kInterruptCoalesceMs + 150;

bool TerminatedBy(const ChildOutcome& outcome, int sig) {
    return outcome.forked && !outcome.timed_out &&
           WIFSIGNALED(outcome.status) && WTERMSIG(outcome.status) == sig;
}

bool ExitedCleanly(const ChildOutcome& outcome) {
    return outcome.forked && !outcome.timed_out &&
           WIFEXITED(outcome.status) && WEXITSTATUS(outcome.status) == 0;
}

/// The two handlers under test: run_dir.hpp's (every standalone runner)
/// and graph_cli_interrupt.hpp's (the multi-model-graph CLI).
struct InterruptHandler {
    const char* name;
    void (*install)();
    bool (*flag)();
};

void InstallRunnerHandlers() { dxapp::installSignalHandlers(); }
bool RunnerFlag() { return dxapp::g_interrupted().load(); }
void InstallCliHandler() { dxapp::graph::cli::InstallInterruptHandler(); }
bool CliFlag() { return dxapp::graph::cli::Interrupted(); }

const InterruptHandler kRunnerHandler = {"runner", InstallRunnerHandlers, RunnerFlag};
const InterruptHandler kCliHandler = {"cli", InstallCliHandler, CliFlag};

// Report bits.
const unsigned char kFlagSet = 1;
const unsigned char kSigintStillHandled = 2;

/// One Ctrl-C under `timeout` (terminal + timeout's two forwards) is three
/// SIGINTs within microseconds: one request, the process lives.
void CheckSigintBurstIsOneRequest(const InterruptHandler& handler) {
    ChildOutcome outcome = RunInChild(
        [&handler](int fd) {
            handler.install();
            raise(SIGINT);
            raise(SIGINT);
            raise(SIGINT);
            unsigned char bits = 0;
            if (handler.flag()) bits |= kFlagSet;
            if (!IsDefaultDisposition(SIGINT)) bits |= kSigintStillHandled;
            WriteByte(fd, bits);
            _exit(0);
        },
        NoAction, 5000);
    std::printf("  [%s] SIGINT burst\n", handler.name);
    CHILD_CHECK(outcome, outcome.reported);
    CHILD_CHECK(outcome, outcome.report == (kFlagSet | kSigintStillHandled));
    CHILD_CHECK(outcome, ExitedCleanly(outcome));
}

/// A SIGINT more than kInterruptCoalesceMs after the first request (`first`)
/// is the user pressing Ctrl-C again: killed by SIGINT.
void CheckLaterSigintTerminates(const InterruptHandler& handler, int first) {
    ChildOutcome outcome = RunInChild(
        [&handler, first](int fd) {
            handler.install();
            raise(first);  // graceful: the handler runs, the process lives
            WriteByte(fd, handler.flag() ? kFlagSet : 0);
            SleepMs(kLaterMs);
            raise(SIGINT);  // must terminate
            _exit(99);
        },
        NoAction, 5000);
    std::printf("  [%s] signal %d, then SIGINT %lld ms later\n", handler.name,
                first, kLaterMs);
    CHILD_CHECK(outcome, outcome.reported && outcome.report == kFlagSet);
    CHILD_CHECK(outcome, TerminatedBy(outcome, SIGINT));
}

void TestSigintBurstIsOneRequest() {
    CheckSigintBurstIsOneRequest(kRunnerHandler);
    CheckSigintBurstIsOneRequest(kCliHandler);
}

void TestLaterSigintTerminates() {
    CheckLaterSigintTerminates(kRunnerHandler, SIGINT);
    CheckLaterSigintTerminates(kRunnerHandler, SIGTERM);
    CheckLaterSigintTerminates(kCliHandler, SIGINT);
    CheckLaterSigintTerminates(kCliHandler, SIGTERM);
}

/// --display's q / ESC calls RequestStop(): it is the first request, the
/// same as a first Ctrl-C. A SIGINT within kInterruptCoalesceMs of it is the
/// same request (the process lives); one later than that is a new press and
/// terminates, so a wind-down stuck after q can still be ended with Ctrl-C.
void TestSigintAfterRequestStop() {
    ChildOutcome soon = RunInChild(
        [](int fd) {
            dxapp::graph::cli::InstallInterruptHandler();
            dxapp::graph::cli::RequestStop();
            raise(SIGINT);  // coalesced with the request
            WriteByte(fd, dxapp::graph::cli::Interrupted() ? kFlagSet : 0);
            _exit(0);
        },
        NoAction, 5000);
    std::printf("  [cli] RequestStop, then SIGINT at once\n");
    CHILD_CHECK(soon, soon.reported && soon.report == kFlagSet);
    CHILD_CHECK(soon, ExitedCleanly(soon));

    ChildOutcome later = RunInChild(
        [](int fd) {
            dxapp::graph::cli::InstallInterruptHandler();
            dxapp::graph::cli::RequestStop();
            WriteByte(fd, dxapp::graph::cli::Interrupted() ? kFlagSet : 0);
            SleepMs(kLaterMs);
            raise(SIGINT);  // must terminate
            _exit(99);
        },
        NoAction, 5000);
    std::printf("  [cli] RequestStop, then SIGINT %lld ms later\n", kLaterMs);
    CHILD_CHECK(later, later.reported && later.report == kFlagSet);
    CHILD_CHECK(later, TerminatedBy(later, SIGINT));
}

/// `timeout 20 ./binary` sends SIGTERM twice at once, and a supervisor may
/// repeat it later: SIGTERM never escalates (the sender escalates with
/// SIGKILL itself). Both handlers: the runners' and the graph CLI's.
void CheckRepeatedSigtermStaysGraceful(const InterruptHandler& handler) {
    ChildOutcome outcome = RunInChild(
        [&handler](int fd) {
            handler.install();
            raise(SIGTERM);
            raise(SIGTERM);
            SleepMs(kLaterMs);
            raise(SIGTERM);
            WriteByte(fd, handler.flag() ? kFlagSet : 0);
            _exit(0);
        },
        NoAction, 5000);
    std::printf("  [%s] SIGTERM x2, then again %lld ms later\n", handler.name, kLaterMs);
    CHILD_CHECK(outcome, outcome.reported && outcome.report == kFlagSet);
    CHILD_CHECK(outcome, ExitedCleanly(outcome));
}

void TestRepeatedSigtermStaysGraceful() {
    CheckRepeatedSigtermStaysGraceful(kRunnerHandler);
    CheckRepeatedSigtermStaysGraceful(kCliHandler);
}

/// An install that fails is reported on stderr and does not abort.
/// sigaction(SIGKILL) always fails (EINVAL).
void TestFailedInstallWarnsOnce() {
    ChildOutcome outcome = RunInChild(
        [](int fd) {
            bool ok = dxapp::installHandlerOrWarn(SIGKILL, dxapp::signalHandler,
                                                  "SIGKILL");
            WriteByte(fd, ok ? 1 : 0);
            _exit(0);
        },
        NoAction, 5000);
    const std::string needle = "could not install the SIGKILL handler";
    CHILD_CHECK(outcome, outcome.reported && outcome.report == 0);
    CHILD_CHECK(outcome, ExitedCleanly(outcome));
    CHILD_CHECK(outcome, CountOccurrences(outcome.child_stderr, needle) == 1);
}

/// True when the child ended in gtk_init(): without a display the GTK backend
/// of OpenCV < 4.5 (apt 4.2 on Ubuntu 20.04) prints "cannot open display" and
/// exit(1)s on the first window, before showOutput() can report kNoWindow.
/// Newer GTK builds throw instead, and Qt has the offscreen platform.
bool GtkCouldNotOpenDisplay(const ChildOutcome& outcome) {
    return outcome.forked && !outcome.reported && !outcome.timed_out &&
           WIFEXITED(outcome.status) && WEXITSTATUS(outcome.status) == 1 &&
           outcome.child_stderr.find("cannot open display") != std::string::npos;
}

/// R5. The sync runners' image mode shows the result and then waits
///     while (!dxapp::windowShouldClose("Output")) sleep(10 ms);
/// With QT_QPA_PLATFORM=offscreen (or any window nobody closes) that wait
/// never ended on SIGTERM/SIGINT: the handler set g_interrupted() but the
/// loop only looked at the window. The child reproduces the runner: real
/// handlers, a real (offscreen) window, the same loop. `send` delivers one
/// request; the loop must end gracefully (normal exit, flag set).
void CheckImageHoldLoopEnds(const char* label, void (*send)(pid_t)) {
    const unsigned char kReady = 'R';
    const unsigned char kNoWindow = 'S';
    ChildOutcome outcome = RunInChild(
        [kReady, kNoWindow](int fd) {
            SetEnv("QT_QPA_PLATFORM", "offscreen");
            dxapp::installSignalHandlers();
            cv::Mat frame(48, 64, CV_8UC3, cv::Scalar(0, 128, 255));
            dxapp::showOutput(frame);
            if (dxapp::_displayClosed()) {  // no GUI backend at all
                WriteByte(fd, kNoWindow);
                _exit(0);
            }
            WriteByte(fd, kReady);
            while (!dxapp::windowShouldClose("Output")) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // The runner's wind-down (summary, teardown) takes a while too;
            // forwarded copies of the request arrive during it.
            SleepMs(150);
            _exit(dxapp::g_interrupted().load() ? 0 : 3);
        },
        [kReady, send](pid_t pid, unsigned char byte) {
            if (byte != kReady) return;
            SleepMs(200);  // let the child settle into the wait
            send(pid);
        },
        5000);
    std::printf("  hold loop: %s\n", label);
    if (outcome.reported && outcome.report == kNoWindow) {
        Skip("CheckImageHoldLoopEnds",
             "no OpenCV GUI backend could open an offscreen window");
        return;
    }
    if (GtkCouldNotOpenDisplay(outcome)) {
        Skip("CheckImageHoldLoopEnds",
             "OpenCV's GTK backend exits in gtk_init() without a display");
        return;
    }
    CHILD_CHECK(outcome, outcome.reported && outcome.report == kReady);
    CHILD_CHECK(outcome, ExitedCleanly(outcome));
}

void SendOneSigterm(pid_t pid) { kill(pid, SIGTERM); }

/// One Ctrl-C forwarded by `timeout`: three SIGINTs, spaced 20 ms apart so
/// the kernel cannot merge them into one pending signal (then each is
/// really delivered), yet all well inside the coalescing window.
void SendSigintBurst(pid_t pid) {
    kill(pid, SIGINT);
    SleepMs(20);
    kill(pid, SIGINT);
    SleepMs(20);
    kill(pid, SIGINT);
}

void TestImageHoldLoopEndsOnOneRequest() {
    CheckImageHoldLoopEnds("one SIGTERM", SendOneSigterm);
    CheckImageHoldLoopEnds("3 SIGINTs within 40 ms", SendSigintBurst);
}

/// The hold-loop skip recognises gtk_init()'s exit and nothing else. The
/// child does what gtk_init() does without a display.
void TestGtkExitWithoutADisplayIsRecognised() {
    ChildOutcome gtk = RunInChild(
        [](int) {
            const char kMessage[] =
                "Unable to init server: Could not connect: Connection refused\n"
                "(Output:1): Gtk-WARNING **: 07:15:53.621: cannot open display: \n";
            ssize_t written = write(2, kMessage, sizeof(kMessage) - 1);
            (void)written;
            _exit(1);
        },
        NoAction, 5000);
    COMMON_CHECK(GtkCouldNotOpenDisplay(gtk));
    // Any other early exit stays a failure of the case, not a skip.
    ChildOutcome other = RunInChild([](int) { _exit(1); }, NoAction, 5000);
    COMMON_CHECK(!GtkCouldNotOpenDisplay(other));
    ChildOutcome reported = RunInChild(
        [](int fd) {
            WriteByte(fd, 'R');
            const char kMessage[] = "cannot open display: \n";
            ssize_t written = write(2, kMessage, sizeof(kMessage) - 1);
            (void)written;
            _exit(1);
        },
        NoAction, 5000);
    COMMON_CHECK(!GtkCouldNotOpenDisplay(reported));
}

#endif  // !_WIN32

void TestInterrupts() {
#ifdef _WIN32
    Skip("TestInterrupts", "fork()-based; POSIX only");
#else
    TestSigintBurstIsOneRequest();
    TestLaterSigintTerminates();
    TestSigintAfterRequestStop();
    TestRepeatedSigtermStaysGraceful();
    TestFailedInstallWarnsOnce();
    TestImageHoldLoopEndsOnOneRequest();
    TestGtkExitWithoutADisplayIsRecognised();
#endif
}

// U-43 (and SP6 U-37, which reuses this test): the coalescing rule on an
// injected clock, through the 200 ms boundary.
void TestNoteInterruptRequestAtCoalescesThroughTheWindow() {
    std::atomic<long long> first{0};
    COMMON_CHECK(!dxapp::noteInterruptRequestAt(first, 0));  // the first request
    COMMON_CHECK(!dxapp::noteInterruptRequestAt(first, dxapp::kInterruptCoalesceMs));
    COMMON_CHECK(dxapp::noteInterruptRequestAt(first, dxapp::kInterruptCoalesceMs + 1));
    COMMON_CHECK(dxapp::noteInterruptRequestAt(first, 10 * dxapp::kInterruptCoalesceMs));
}

// U-42: with no working CLOCK_MONOTONIC every repeat escalates (the
// count-based rule from before coalescing), and the first request does not.
void TestABrokenClockEscalatesEveryRepeat() {
#ifndef _WIN32
    std::atomic<long long> first{0};
    COMMON_CHECK(!dxapp::noteInterruptRequestAt(first, dxapp::detail::brokenClockMs()));
    COMMON_CHECK(dxapp::noteInterruptRequestAt(first, dxapp::detail::brokenClockMs()));
    COMMON_CHECK(dxapp::noteInterruptRequestAt(first, dxapp::detail::brokenClockMs()));
#endif
}

// SP6 U-37: the portable body of the Windows console control routine
// (the coalescing rule itself is TestNoteInterruptRequestAtCoalescesThroughTheWindow).
int g_console_calls = 0;

void CountingSigintHandler(int sig) {
    if (sig == SIGINT) ++g_console_calls;
}

void EscalatingSigintHandler(int) {
    ++g_console_calls;
    dxapp::detail::consoleTerminateRequested().store(true);  // what terminateBySignal does on Windows
}

void TestConsoleCtrlDispatchRunsOnlyCtrlC() {
    using dxapp::detail::dispatchConsoleCtrl;
    std::atomic<dxapp::SignalHandlerFn>& slot = dxapp::detail::consoleInterruptHandler();
    const dxapp::SignalHandlerFn saved = slot.load();
    slot.store(nullptr);
    COMMON_CHECK(!dispatchConsoleCtrl(dxapp::detail::kConsoleCtrlC));  // nothing installed
    slot.store(&CountingSigintHandler);
    g_console_calls = 0;
    COMMON_CHECK(dispatchConsoleCtrl(dxapp::detail::kConsoleCtrlC));   // handled, keeps running
    COMMON_CHECK(g_console_calls == 1);
    COMMON_CHECK(!dispatchConsoleCtrl(dxapp::detail::kConsoleCtrlBreak));  // default routine
    COMMON_CHECK(!dispatchConsoleCtrl(2));                                  // CTRL_CLOSE_EVENT
    COMMON_CHECK(g_console_calls == 1);
    slot.store(&EscalatingSigintHandler);
    COMMON_CHECK(!dispatchConsoleCtrl(dxapp::detail::kConsoleCtrlC));  // escalated
    COMMON_CHECK(g_console_calls == 2);
    dxapp::detail::consoleTerminateRequested().store(false);
    slot.store(saved);
}

// =====================================================================
// U-30: DXAPP_VERIFY writer - async runners dumped from dxrt's completion
// threads into one file with a plain truncate-and-write, so a record
// could be torn, and only the last frame survived.
// =====================================================================

std::string MakeTempDir(const std::string& tag) {
#ifdef _WIN32
    const long long pid = static_cast<long long>(_getpid());
#else
    const long long pid = static_cast<long long>(getpid());
#endif
    const fs::path dir = fs::temp_directory_path() /
        ("dxapp_common_unit_" + tag + "_" + std::to_string(pid));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir.string();
}

/// Changes the working directory for one scope and restores it on exit.
class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(const std::string& dir) : saved_(fs::current_path()) {
        fs::current_path(dir);
    }
    ~ScopedCurrentPath() {
        std::error_code ignored;
        fs::current_path(saved_, ignored);
    }
    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
    fs::path saved_;
};

// N6: a gallery names its images repo-relative ("sample/reid/gallery/...").
// The retrieval visualizer opens them as resolveGalleryFile opens the gallery:
// as given, else under PROJECT_ROOT_DIR, so a run from outside the
// repository still shows its thumbnails.
void TestRetrievalThumbnailsResolveAgainstTheRepository() {
    const std::string image = "sample/img/sample_dog.jpg";
    const std::string root = PROJECT_ROOT_DIR;
    COMMON_CHECK(std::ifstream(root + "/" + image).good());
    const std::string elsewhere = MakeTempDir("retrieval_cwd");
    {
        ScopedCurrentPath cwd(elsewhere);
        COMMON_CHECK(!std::ifstream(image).good());
        COMMON_CHECK(dxapp::retrieval_vis_detail::resolvePath(image) == root + "/" + image);
        // Absent everywhere: returned as given, for the "missing" tile.
        COMMON_CHECK(dxapp::retrieval_vis_detail::resolvePath("sample/no_such.jpg") ==
                     "sample/no_such.jpg");
    }
    {
        // Found as given: kept as given.
        ScopedCurrentPath cwd(root);
        COMMON_CHECK(dxapp::retrieval_vis_detail::resolvePath(image) == image);
    }
    fs::remove_all(elsewhere);
}

// U-75: a graph stage's config.json is optional; without one the factory's
// defaults apply and nothing is printed.
void TestLoadOptionalConfigIsSilentWithoutAFile() {
    const std::string dir = MakeTempDir("optional_config");
    StderrCapture capture;
    const dxapp::ModelConfig absent = dxapp::LoadOptionalConfig(dir + "/config.json");
    const std::string text = capture.Stop();
    COMMON_CHECK(!absent.isLoaded());
    COMMON_CHECK(text.find("Config file not found") == std::string::npos);
    {
        std::ofstream out((dir + "/config.json").c_str());
        out << "{\"top_k\": 3}";
    }
    const dxapp::ModelConfig present = dxapp::LoadOptionalConfig(dir + "/config.json");
    COMMON_CHECK(present.isLoaded() && present.get<int>("top_k", 0) == 3);
    fs::remove_all(dir);
}

// U-75: a config.json that exists but cannot be read is not an absent one:
// ModelConfig warns, and the stage does not silently run on the defaults.
void TestLoadOptionalConfigWarnsOnAnUnreadableFile() {
#ifdef _WIN32
    Skip("TestLoadOptionalConfigWarnsOnAnUnreadableFile", "chmod 000 is POSIX-only");
#else
    if (geteuid() == 0) {
        Skip("TestLoadOptionalConfigWarnsOnAnUnreadableFile", "root reads a mode-000 file");
        return;
    }
    const std::string dir = MakeTempDir("unreadable_config");
    const std::string path = dir + "/config.json";
    {
        std::ofstream out(path.c_str());
        out << "{\"top_k\": 3}";
    }
    COMMON_CHECK(chmod(path.c_str(), 0) == 0);
    StderrCapture capture;
    const dxapp::ModelConfig config = dxapp::LoadOptionalConfig(path);
    const std::string text = capture.Stop();
    COMMON_CHECK(!config.isLoaded());
    COMMON_CHECK(text.find(path) != std::string::npos);
    chmod(path.c_str(), 0600);
    fs::remove_all(dir);
#endif
}

std::vector<std::string> ReadLines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

int CountTempFiles(const std::string& dir) {
    int count = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().filename().string().find(".tmp.") != std::string::npos) ++count;
    }
    return count;
}

std::vector<dxapp::DetectionResult> OneDetection(int a, int b) {
    return std::vector<dxapp::DetectionResult>(1, dxapp::DetectionResult(
        {static_cast<float>(a), static_cast<float>(b), 1.f, 1.f}, 0.5f, a, "probe"));
}

void TestVerifyDumpsFromManyThreadsAreWholeAndNumbered() {
    const std::string dir = MakeTempDir("verify_threads");
    SetEnv("DXAPP_VERIFY", "1");
    SetEnv("DXAPP_VERIFY_DIR", dir.c_str());
    const std::string json_path = dir + "/threads_probe.json";
    const std::string frames_path = dir + "/threads_probe.frames.jsonl";

    std::atomic<bool> writing{true};
    std::atomic<int> torn{0};
    std::atomic<int> reads{0};
    std::thread reader([&]() {
        while (writing.load()) {
            std::ifstream in(json_path);
            if (!in.is_open()) continue;
            std::stringstream text;
            text << in.rdbuf();
            ++reads;
            try {
                const auto parsed = dxapp::verify::json::parse(text.str());
                (void)parsed;
            } catch (const std::exception&) {
                ++torn;  // a reader saw a half-written file
            }
        }
    });

    std::stringstream sink;  // 400 "[VERIFY] Dumped" lines
    std::streambuf* saved = std::cout.rdbuf(sink.rdbuf());
    std::vector<std::thread> writers;
    for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([t]() {
            for (int i = 0; i < kCallsPerThread; ++i) {
                dxapp::verify::dumpVerifyJson(OneDetection(t, i), "/models/threads_probe.dxnn",
                                              "object_detection", 10, 20);
            }
        });
    }
    for (auto& writer : writers) writer.join();
    writing = false;
    reader.join();
    std::cout.rdbuf(saved);
    UnsetEnv("DXAPP_VERIFY");
    UnsetEnv("DXAPP_VERIFY_DIR");

    COMMON_CHECK(reads.load() > 0);
    COMMON_CHECK(torn.load() == 0);
    const std::vector<std::string> lines = ReadLines(frames_path);
    COMMON_CHECK(lines.size() == static_cast<std::size_t>(kThreads * kCallsPerThread));
    std::set<long long> frames;
    bool whole = true;
    for (const auto& line : lines) {
        try {
            const auto record = dxapp::verify::json::parse(line);
            frames.insert(record.at("frame").get<long long>());
            if (record.at("detections").size() != 1) whole = false;
        } catch (const std::exception&) {
            whole = false;
        }
    }
    COMMON_CHECK(whole);
    COMMON_CHECK(frames.size() == lines.size());
    COMMON_CHECK(!frames.empty() && *frames.begin() == 0 &&
                 *frames.rbegin() == kThreads * kCallsPerThread - 1);
    COMMON_CHECK(CountTempFiles(dir) == 0);
    fs::remove_all(dir);
}

void TestVerifyJsonKeepsItsFormatAndFramesFileAddsTheIndex() {
    const std::string dir = MakeTempDir("verify_format");
    SetEnv("DXAPP_VERIFY", "1");
    SetEnv("DXAPP_VERIFY_DIR", dir.c_str());
    std::stringstream sink;
    std::streambuf* saved = std::cout.rdbuf(sink.rdbuf());
    const std::string written = dxapp::verify::dumpVerifyJson(
        OneDetection(3, 4), "/models/format_probe.dxnn", "object_detection", 10, 20);
    std::cout.rdbuf(saved);
    UnsetEnv("DXAPP_VERIFY");
    UnsetEnv("DXAPP_VERIFY_DIR");

    COMMON_CHECK(written == dir + "/format_probe.json");
    std::ifstream in(dir + "/format_probe.json");
    std::stringstream text;
    text << in.rdbuf();
    const auto record = dxapp::verify::json::parse(text.str());
    COMMON_CHECK(text.str() == record.dump(2) + "\n");
    COMMON_CHECK(record.count("frame") == 0);
    COMMON_CHECK(record.at("task") == "object_detection");
    const std::vector<std::string> lines = ReadLines(dir + "/format_probe.frames.jsonl");
    COMMON_CHECK(lines.size() == 1);
    if (lines.size() == 1) {
        auto line = dxapp::verify::json::parse(lines[0]);
        COMMON_CHECK(line.at("frame") == 0);
        line.erase("frame");
        COMMON_CHECK(line == record);
    }
    fs::remove_all(dir);
}

void TestVerifySerializesDetection3D() {
    dxapp::Detection3DResult d;
    d.class_id = 1; d.class_name = "Car"; d.confidence = 0.5f;
    d.bev_x = 1.f; d.bev_y = 2.f; d.bev_w = 3.f; d.bev_h = 4.f;
    d.x3d = 5.f; d.y3d = 6.f; d.z3d = 7.f;
    d.dim_h = 8.f; d.dim_w = 9.f; d.dim_l = 10.f; d.yaw = 0.25f;
    const auto j = dxapp::verify::serializeDetection3D(
        std::vector<dxapp::Detection3DResult>(1, d), 608, 600);
    COMMON_CHECK(j.at("image_height") == 608 && j.at("image_width") == 600);
    COMMON_CHECK(j.at("detections").size() == 1);
    const auto& e = j.at("detections")[0];
    COMMON_CHECK(e.at("class_id") == 1 && e.at("class_name") == "Car");
    COMMON_CHECK(e.at("conf") == 0.5);
    COMMON_CHECK(e.at("bev") == dxapp::verify::json::array({1.0, 2.0, 3.0, 4.0}));
    COMMON_CHECK(e.at("center") == dxapp::verify::json::array({5.0, 6.0, 7.0}));
    COMMON_CHECK(e.at("dims") == dxapp::verify::json::array({8.0, 9.0, 10.0}));
    COMMON_CHECK(e.at("yaw") == 0.25);
}

// Review Focus 5: an unusable DXAPP_VERIFY_DIR must not stop a run.
void TestVerifyIntoAFileInsteadOfADirectoryWarnsAndContinues() {
    const std::string dir = MakeTempDir("verify_blocked");
    const std::string blocker = dir + "/not_a_directory";
    { std::ofstream(blocker) << "x"; }
    SetEnv("DXAPP_VERIFY", "1");
    SetEnv("DXAPP_VERIFY_DIR", blocker.c_str());
    StderrCapture capture;
    bool threw = false;
    std::string written = "unset";
    try {
        written = dxapp::verify::dumpVerifyJson(OneDetection(1, 1), "/models/blocked_probe.dxnn",
                                                "object_detection", 1, 1);
    } catch (...) {
        threw = true;
    }
    const std::string err = capture.Stop();
    UnsetEnv("DXAPP_VERIFY");
    UnsetEnv("DXAPP_VERIFY_DIR");
    COMMON_CHECK(!threw);
    COMMON_CHECK(written.empty());
    COMMON_CHECK(err.find("[DXAPP] [WARN] verify_serialize") != std::string::npos);
    COMMON_CHECK(CountTempFiles(dir) == 0);
    fs::remove_all(dir);
}

// A frames.jsonl that cannot be opened must not use up a frame number: the
// next dump has to start the file (truncate), not append to an old run's file.
void TestVerifyFramesFileThatCannotBeOpenedIsStartedByTheNextDump() {
    const std::string dir = MakeTempDir("verify_frames_blocked");
    const std::string json_path = dir + "/frames_blocked_probe.json";
    const std::string frames_path = dir + "/frames_blocked_probe.frames.jsonl";
    fs::create_directories(frames_path);  // a directory: opening it as a file fails
    SetEnv("DXAPP_VERIFY", "1");
    SetEnv("DXAPP_VERIFY_DIR", dir.c_str());
    std::stringstream sink;
    std::streambuf* saved = std::cout.rdbuf(sink.rdbuf());
    StderrCapture capture;
    const std::string first = dxapp::verify::dumpVerifyJson(
        OneDetection(1, 1), "/models/frames_blocked_probe.dxnn", "object_detection", 1, 1);
    const std::string err = capture.Stop();
    fs::remove_all(frames_path);
    { std::ofstream(frames_path) << "{\"frame\": 0, \"old_run\": true}\n"; }
    const std::string second = dxapp::verify::dumpVerifyJson(
        OneDetection(2, 2), "/models/frames_blocked_probe.dxnn", "object_detection", 1, 1);
    std::cout.rdbuf(saved);
    UnsetEnv("DXAPP_VERIFY");
    UnsetEnv("DXAPP_VERIFY_DIR");

    COMMON_CHECK(first == json_path);  // the JSON is still written
    COMMON_CHECK(err.find("[DXAPP] [WARN] verify_serialize: cannot write " + frames_path) !=
                 std::string::npos);
    COMMON_CHECK(second == json_path);
    const std::vector<std::string> lines = ReadLines(frames_path);
    COMMON_CHECK(lines.size() == 1);
    if (lines.size() == 1) {
        const auto record = dxapp::verify::json::parse(lines[0]);
        COMMON_CHECK(record.at("frame") == 0);
        COMMON_CHECK(record.count("old_run") == 0);
    }
    COMMON_CHECK(CountTempFiles(dir) == 0);
    fs::remove_all(dir);
}

// POSIX rename() replaces the target itself; removing the target and retrying
// is only for Windows. On POSIX a failed rename must leave the target alone.
void TestVerifyFailedRenameLeavesTheTargetInPlace() {
#ifndef _WIN32
    const std::string dir = MakeTempDir("verify_rename");
    const std::string json_path = dir + "/rename_probe.json";
    fs::create_directories(json_path);  // rename() of a file onto a directory fails
    SetEnv("DXAPP_VERIFY", "1");
    SetEnv("DXAPP_VERIFY_DIR", dir.c_str());
    StderrCapture capture;
    const std::string written = dxapp::verify::dumpVerifyJson(
        OneDetection(1, 1), "/models/rename_probe.dxnn", "object_detection", 1, 1);
    const std::string err = capture.Stop();
    UnsetEnv("DXAPP_VERIFY");
    UnsetEnv("DXAPP_VERIFY_DIR");

    COMMON_CHECK(written.empty());
    COMMON_CHECK(fs::is_directory(json_path));  // not removed to make room
    COMMON_CHECK(err.find("[DXAPP] [WARN] verify_serialize: cannot write " + json_path) !=
                 std::string::npos);
    COMMON_CHECK(CountTempFiles(dir) == 0);
    fs::remove_all(dir);
#endif
}

// =====================================================================
// U-02: YOLOPv2 - each frame's masks travel in that frame's result.
// =====================================================================

/// Owns the buffers the tensors point at (dxrt::Tensor does not copy).
struct YOLOPv2Tensors {
    std::vector<float> head0, head1, head2, drivable, lane;
    dxrt::TensorPtrs outputs;
};

/// A 64x32 model input: three detection heads with no vehicle anywhere
/// (all logits -20), a drivable map that is 1 on the left or the right
/// half, and a lane map that is 1 on row 0 when `lane_top` is set.
std::unique_ptr<YOLOPv2Tensors> MakeYOLOPv2Outputs(bool drivable_left, bool lane_top) {
    const int H = 32, W = 64;
    std::unique_ptr<YOLOPv2Tensors> t(new YOLOPv2Tensors);
    t->head0.assign(255 * 4 * 8, -20.f);
    t->head1.assign(255 * 2 * 4, -20.f);
    t->head2.assign(255 * 1 * 2, -20.f);
    t->drivable.assign(2 * H * W, 0.f);
    t->lane.assign(H * W, -20.f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const bool on = drivable_left ? (x < W / 2) : (x >= W / 2);
            t->drivable[H * W + y * W + x] = on ? 1.f : -1.f;  // channel 1 > channel 0 (0): drivable
        }
    }
    if (lane_top) {
        for (int x = 0; x < W; ++x) t->lane[x] = 20.f;
    }
    auto add = [&t](const char* name, std::vector<int64_t> shape, std::vector<float>& buffer) {
        t->outputs.push_back(std::make_shared<dxrt::Tensor>(
            name, shape, dxrt::DataType::FLOAT, buffer.data()));
    };
    add("det0", {1, 255, 4, 8}, t->head0);
    add("det1", {1, 255, 2, 4}, t->head1);
    add("det2", {1, 255, 1, 2}, t->head2);
    add("drivable", {1, 2, H, W}, t->drivable);
    add("lane", {1, 1, H, W}, t->lane);
    return t;
}

dxapp::PreprocessContext YOLOPv2Ctx() {
    dxapp::PreprocessContext ctx;
    ctx.scale = 1.f;
    ctx.original_width = 64;
    ctx.original_height = 32;
    ctx.input_width = 64;
    ctx.input_height = 32;
    return ctx;
}

void TestYOLOPv2FrameMasksStayWithTheirFrame() {
    dxapp::v_yolopv2_384x640::Yolopv2Factory factory;
    auto post = factory.createPanopticPostprocessor(64, 32);
    const auto a = MakeYOLOPv2Outputs(true, false);
    const auto b = MakeYOLOPv2Outputs(false, true);
    const dxapp::PreprocessContext ctx = YOLOPv2Ctx();
    const auto ra = post->process(a->outputs, ctx);
    const auto rb = post->process(b->outputs, ctx);  // decoded after A, as an async runner may
    COMMON_CHECK(ra.size() == 1 && rb.size() == 1);
    if (ra.size() != 1 || rb.size() != 1) return;
    COMMON_CHECK(ra[0].drivable.at<uchar>(5, 0) == 1 && ra[0].drivable.at<uchar>(5, 63) == 0);
    COMMON_CHECK(rb[0].drivable.at<uchar>(5, 0) == 0 && rb[0].drivable.at<uchar>(5, 63) == 1);
    COMMON_CHECK(cv::countNonZero(ra[0].lane) == 0);
    COMMON_CHECK(rb[0].lane.at<uchar>(0, 10) == 1);
    // A's frame is drawn with A's masks although B was decoded last.
    auto viz = factory.createPanopticVisualizer();
    const cv::Mat black(32, 64, CV_8UC3, cv::Scalar::all(0));
    const cv::Mat drawn = viz->draw(black, ra, ctx);
    COMMON_CHECK(drawn.at<cv::Vec3b>(5, 0)[1] > 0);                 // green: A's drivable half
    COMMON_CHECK(drawn.at<cv::Vec3b>(5, 63) == cv::Vec3b(0, 0, 0));  // B's half stays black
}

// Review Focus 2: a frame without a vehicle still carries its masks.
void TestYOLOPv2FrameWithoutVehiclesStillCarriesMasks() {
    dxapp::v_yolopv2_384x640::Yolopv2Factory factory;
    const auto a = MakeYOLOPv2Outputs(true, true);
    const dxapp::PreprocessContext ctx = YOLOPv2Ctx();
    const auto panoptic = factory.createPanopticPostprocessor(64, 32)->process(a->outputs, ctx);
    COMMON_CHECK(panoptic.size() == 1);
    if (panoptic.size() != 1) return;
    COMMON_CHECK(panoptic[0].detections.empty());
    COMMON_CHECK(panoptic[0].drivable.rows == 32 && panoptic[0].drivable.cols == 64);
    COMMON_CHECK(cv::countNonZero(panoptic[0].lane) == 64);
    // The graph's boxes-only path decodes the same frame to the same (no) boxes.
    COMMON_CHECK(factory.createPostprocessor(64, 32)->process(a->outputs, ctx).empty());
    const auto record = dxapp::verify::serializePanoptic(panoptic, 32, 64);
    COMMON_CHECK(record.at("detections").empty());
    COMMON_CHECK(std::fabs(record.at("drivable_stats").at("mean").get<double>() - 0.5) < 1e-9);
    COMMON_CHECK(record.at("lane_stats").at("max") == 1.0);
}

// =====================================================================
// U-03: OrderedQueue - async runners submit frames in order but get them
// back on dxrt's completion threads in any order.
// =====================================================================

using Ms = std::chrono::milliseconds;

void TestOrderedQueueDeliversShuffledPushesInTicketOrder() {
    const int kFrames = 400;
    dxapp::OrderedQueue<int> q(100);
    std::vector<std::uint64_t> tickets;
    for (int i = 0; i < kFrames; ++i) tickets.push_back(q.issueTicket());
    // Four "completion threads", each handing over every 4th frame, shuffled
    // within windows of 8: a plain FIFO would deliver them out of order.
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&q, &tickets, t]() {
            std::mt19937 rng(1234u + static_cast<unsigned>(t));
            std::vector<std::uint64_t> mine;
            for (int i = t; i < kFrames; i += 4) mine.push_back(tickets[i]);
            for (std::size_t b = 0; b < mine.size(); b += 8) {
                std::shuffle(mine.begin() + b, mine.begin() + std::min(mine.size(), b + 8), rng);
            }
            for (std::uint64_t k : mine) q.push(k, static_cast<int>(k));
        });
    }
    std::vector<int> got;
    int v = 0;
    while (static_cast<int>(got.size()) < kFrames && q.try_pop(v, Ms(2000))) got.push_back(v);
    for (auto& p : producers) p.join();
    COMMON_CHECK(static_cast<int>(got.size()) == kFrames);
    bool in_order = true;
    for (int i = 0; i < static_cast<int>(got.size()); ++i) {
        if (got[i] != i) in_order = false;
    }
    COMMON_CHECK(in_order);
}

void TestOrderedQueuePushBlocksWhenTooFarAhead() {
    dxapp::OrderedQueue<int> q(4);
    for (int i = 0; i < 8; ++i) q.issueTicket();
    for (int i = 0; i < 4; ++i) COMMON_CHECK(q.push(static_cast<std::uint64_t>(i), i));
    std::atomic<bool> done{false};
    std::thread ahead([&]() { q.push(6, 6); done = true; });  // admitted once next >= 3
    std::this_thread::sleep_for(Ms(50));
    COMMON_CHECK(!done.load());
    int v = -1;
    COMMON_CHECK(q.try_pop(v, Ms(100)) && v == 0);
    COMMON_CHECK(q.try_pop(v, Ms(100)) && v == 1);  // next = 2: 6 is still too far ahead
    std::this_thread::sleep_for(Ms(50));
    COMMON_CHECK(!done.load());
    COMMON_CHECK(q.try_pop(v, Ms(100)) && v == 2);  // next = 3: admitted
    ahead.join();
    COMMON_CHECK(done.load());
    q.shutdown();
}

void TestOrderedQueueGivesUpAMissingFrameAfterTheGapTimeout() {
    dxapp::OrderedQueue<int> q(8, Ms(50));
    for (int i = 0; i < 3; ++i) q.issueTicket();
    q.push(1, 1);
    q.push(2, 2);
    StderrCapture capture;
    int v = -1;
    const auto t0 = std::chrono::steady_clock::now();
    const bool popped = q.try_pop(v, Ms(1000));
    const auto waited = std::chrono::steady_clock::now() - t0;
    const std::string err = capture.Stop();
    COMMON_CHECK(popped && v == 1);
    COMMON_CHECK(waited >= Ms(45));
    COMMON_CHECK(CountOccurrences(err, "[DXAPP] [WARN] frame 0 did not come back") == 1);
    COMMON_CHECK(q.try_pop(v, Ms(10)) && v == 2);
    COMMON_CHECK(!q.push(0, 0));  // too late: its slot was given up
}

void TestOrderedQueueSkipReleasesTheNextFrame() {
    dxapp::OrderedQueue<int> q(8, Ms(10000));
    q.issueTicket();
    q.issueTicket();
    q.push(1, 1);
    q.skip(0);
    int v = -1;
    COMMON_CHECK(q.try_pop(v, Ms(100)) && v == 1);
}

void TestOrderedQueueShutdownDeliversOnlyTheUnbrokenPrefix() {
    dxapp::OrderedQueue<int> q(8, Ms(10000));
    for (int i = 0; i < 6; ++i) q.issueTicket();
    q.push(0, 0);
    q.push(1, 1);
    q.push(3, 3);  // 2 is still in flight when the runner shuts down
    q.push(4, 4);
    int v = -1;
    COMMON_CHECK(q.try_pop(v, Ms(10)) && v == 0);
    COMMON_CHECK(q.try_pop(v, Ms(10)) && v == 1);
    COMMON_CHECK(!q.try_pop(v, Ms(20)));  // 2 is missing and the gap timeout is far
    q.shutdown();
    COMMON_CHECK(!q.try_pop(v, Ms(10)));  // 3 and 4 wait behind the missing 2: dropped
    COMMON_CHECK(q.empty());              // so a "running_ || !empty()" loop ends
    COMMON_CHECK(!q.push(2, 2));
}

void TestOrderedQueueNeverDeadlocksWhenProducersWaitBehindAMissingFrame() {
    // Every producer is blocked (tickets 2..5 >= next 0 + capacity 2) and
    // frames 0 and 1 never come: the consumer must give them up.
    dxapp::OrderedQueue<int> q(2, Ms(50));
    for (int i = 0; i < 6; ++i) q.issueTicket();
    std::vector<std::thread> producers;
    for (int t = 2; t < 6; ++t) {
        producers.emplace_back([&q, t]() { q.push(static_cast<std::uint64_t>(t), t); });
    }
    StderrCapture capture;
    std::vector<int> got;
    int v = -1;
    const auto t0 = std::chrono::steady_clock::now();
    while (got.size() < 4 && std::chrono::steady_clock::now() - t0 < Ms(3000)) {
        if (q.try_pop(v, Ms(100))) got.push_back(v);
    }
    q.shutdown();
    for (auto& p : producers) p.join();
    (void)capture.Stop();
    COMMON_CHECK((got == std::vector<int>{2, 3, 4, 5}));
}

void TestOrderedQueueHoldsMoveOnlyItems() {
    dxapp::OrderedQueue<std::unique_ptr<int>> q(4);
    q.issueTicket();
    q.push(0, std::unique_ptr<int>(new int(7)));
    std::unique_ptr<int> out;
    COMMON_CHECK(q.try_pop(out, Ms(10)) && out && *out == 7);
}

struct ReorderItem {
    uint64_t frame_index;
};

// I23: an index that arrives after the cap force-flushed past it is emitted
// at once, and it does not stay at begin() holding every later frame back
// until drain() (2eb1350e's frame_reorder.hpp fix).
void TestFrameReorderEmitsALateIndexAfterAGapFlush() {
    dxapp::FrameReorderBuffer<ReorderItem> reorder(2);
    std::vector<uint64_t> emitted;
    auto emit = [&emitted](ReorderItem& item) { emitted.push_back(item.frame_index); };

    // Index 0 is late: 1, 2 wait for it; 3 exceeds the cap and flushes 1..3.
    for (uint64_t i = 1; i <= 3; ++i) reorder.push(ReorderItem{i}, emit);
    COMMON_CHECK((emitted == std::vector<uint64_t>{1, 2, 3}));

    // The late 0 is emitted at once.
    reorder.push(ReorderItem{0}, emit);
    COMMON_CHECK((emitted == std::vector<uint64_t>{1, 2, 3, 0}));

    // begin() is not pinned by it: the next in-order index flows straight out.
    reorder.push(ReorderItem{4}, emit);
    COMMON_CHECK((emitted == std::vector<uint64_t>{1, 2, 3, 0, 4}));

    // Nothing is left behind for drain().
    std::size_t drained = 0;
    reorder.drain([&drained](ReorderItem&) { ++drained; });
    COMMON_CHECK(drained == 0);
}

void TestVitPoseDecodesHeatmapPeaksToTheOriginalImage() {
    const int K = 17, H = 4, W = 3;
    std::vector<float> heat(K * H * W, 0.f);
    for (int k = 0; k < K; ++k) heat[k * H * W + (k % H) * W + (k % W)] = 0.5f + 0.01f * k;
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "1095", std::vector<int64_t>{1, K, H, W}, dxrt::DataType::FLOAT, heat.data()));
    dxapp::PreprocessContext ctx;
    ctx.original_width = 300;
    ctx.original_height = 400;
    dxapp::VitPosePostprocessor post(192, 256);
    const std::vector<dxapp::PoseResult> poses = post.process(outputs, ctx);
    COMMON_CHECK(poses.size() == 1);
    if (poses.size() != 1) return;
    COMMON_CHECK(poses[0].keypoints.size() == 17);
    COMMON_CHECK(poses[0].confidence == 1.f);
    for (int k = 0; k < K && k < static_cast<int>(poses[0].keypoints.size()); ++k) {
        const dxapp::Keypoint& p = poses[0].keypoints[k];
        // Python's VitPosePostprocessor: x_hm / hm_w * orig_w, y_hm / hm_h * orig_h.
        COMMON_CHECK(std::fabs(p.x - static_cast<float>(k % W) / W * 300.f) < 1e-3f);
        COMMON_CHECK(std::fabs(p.y - static_cast<float>(k % H) / H * 400.f) < 1e-3f);
        COMMON_CHECK(std::fabs(p.confidence - (0.5f + 0.01f * k)) < 1e-6f);
    }
}

// 2eb1350e's VitPosePostprocessor (pose_postprocessor.hpp) answers the model
// name vit_pose_small_bn, and an output with no tensor decodes to no pose
// (its contract), not a throw.
void TestVitPoseNameAndEmptyOutput() {
    dxapp::VitPosePostprocessor post(192, 256);
    COMMON_CHECK(post.getModelName() == "vit_pose_small_bn");
    const dxapp::PreprocessContext ctx;
    COMMON_CHECK(post.process(dxrt::TensorPtrs(), ctx).empty());
}

// The graph trait (top-down pose is an ROI consumer) is not the header's:
// it lives in scripts/gen_model_registry.py's GRAPH_TRAITS (R5), and
// graph_engine_test's TestVitPoseIsATopDownRoiConsumer pins it.
void TestVitPoseFactoryWiresTheHeatmapDecoder() {
    dxapp::v_vitpose_s_256x192::VitposeFactory factory;
    auto post = factory.createPostprocessor(192, 256);
    COMMON_CHECK(dynamic_cast<dxapp::VitPosePostprocessor*>(post.get()) != nullptr);
    auto pre = factory.createPreprocessor(192, 256);
    COMMON_CHECK(dynamic_cast<dxapp::SimpleResizePreprocessor*>(pre.get()) != nullptr);
}

// [RULED Q2-2] dark-hrnet-w32_256x192 is the analogy copy of vitpose (spec
// N11): its heatmaps decode with the same VitPose postprocessor.
void TestDarkHrnetFactoryWiresTheHeatmapDecoder() {
    dxapp::v_dark_hrnet_w32_256x192::DarkHrnetFactory factory;
    auto post = factory.createPostprocessor(192, 256);
    COMMON_CHECK(dynamic_cast<dxapp::VitPosePostprocessor*>(post.get()) != nullptr);
    auto pre = factory.createPreprocessor(192, 256);
    COMMON_CHECK(dynamic_cast<dxapp::SimpleResizePreprocessor*>(pre.get()) != nullptr);
}

// yolov5-s6-pose's CPU task emits one decoded "detections" tensor
// [1, N, 57] = [cx, cy, w, h, obj, cls, 17 x (x, y, conf)]: the anchor
// decoder, not the anchor-free YOLOv8 one (which read it as 33 one-keypoint
// boxes on sample_people.jpg).
void TestYolov5PoseFactoryDecodesTheAnchorDetectionsOutput() {
    dxapp::v_yolov5_s6_pose_640x640::Yolov5PoseFactory factory;
    auto post = factory.createPostprocessor(640, 640, true);
    COMMON_CHECK(dynamic_cast<dxapp::YOLOv5PosePostprocessor*>(post.get()) != nullptr);
    if (dynamic_cast<dxapp::YOLOv5PosePostprocessor*>(post.get()) == nullptr) return;
    std::vector<float> rows(2 * 57, 0.f);
    const float person[6] = {320.f, 320.f, 100.f, 200.f, 0.9f, 0.9f};
    std::copy(person, person + 6, rows.begin());
    for (int k = 0; k < 17; ++k) {
        rows[6 + 3 * k] = 300.f + k;
        rows[7 + 3 * k] = 250.f + k;
        rows[8 + 3 * k] = 0.8f;
    }
    rows[57 + 4] = 0.1f;  // second row: objectness below the threshold
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(2);
    shape.push_back(57);
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "detections", shape, dxrt::DataType::FLOAT, rows.data()));
    const dxapp::PreprocessContext identity(0, 0, 1.f, 640, 640, 640, 640);
    const std::vector<dxapp::PoseResult> poses = post->process(outputs, identity);
    COMMON_CHECK(poses.size() == 1);
    if (poses.size() != 1) return;
    COMMON_CHECK(poses[0].keypoints.size() == 17);
    COMMON_CHECK(poses[0].box == std::vector<float>({270.f, 220.f, 370.f, 420.f}));
    if (poses[0].keypoints.size() != 17) return;
    COMMON_CHECK(poses[0].keypoints[16].x == 316.f && poses[0].keypoints[16].y == 266.f);
}

// yolov5-s6-pose's config.json has an empty "config" object, so the
// factory's defaults are what the runner and a graph stage use: Python's
// (obj 0.25, score 0.3, nms 0.45).
void TestYolov5PoseFactoryDefaultsArePythons() {
    dxapp::v_yolov5_s6_pose_640x640::Yolov5PoseFactory factory;
    auto post = factory.createPostprocessor(640, 640, true);
    std::vector<float> rows(3 * 57, 0.f);
    const float kept[6] = {320.f, 320.f, 100.f, 200.f, 0.9f, 0.9f};       // conf 0.81
    const float overlap[6] = {350.f, 320.f, 100.f, 200.f, 0.9f, 0.85f};   // IoU 0.54 with kept
    const float weak[6] = {100.f, 100.f, 50.f, 50.f, 0.9f, 0.3f};          // conf 0.27
    std::copy(kept, kept + 6, rows.begin());
    std::copy(overlap, overlap + 6, rows.begin() + 57);
    std::copy(weak, weak + 6, rows.begin() + 2 * 57);
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(3);
    shape.push_back(57);
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "detections", shape, dxrt::DataType::FLOAT, rows.data()));
    const dxapp::PreprocessContext identity(0, 0, 1.f, 640, 640, 640, 640);
    const std::vector<dxapp::PoseResult> poses = post->process(outputs, identity);
    COMMON_CHECK(poses.size() == 1);  // the overlap goes at nms 0.45, the weak row at score 0.3
    if (poses.size() == 1) {
        COMMON_CHECK(poses[0].box == std::vector<float>({270.f, 220.f, 370.f, 420.f}));
    }
}

// The case above cannot tell score 0.3 (Python) from 8d0b748's 0.5: its
// rows sit at 0.81 and 0.27. A row at 0.36 can.
void TestYolov5PoseFactoryDefaultScoreIsPythons() {
    dxapp::v_yolov5_s6_pose_640x640::Yolov5PoseFactory factory;
    auto post = factory.createPostprocessor(640, 640, true);
    std::vector<float> rows(57, 0.f);
    const float mid[6] = {320.f, 320.f, 100.f, 200.f, 0.6f, 0.6f};  // obj 0.6, conf 0.36
    std::copy(mid, mid + 6, rows.begin());
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(57);
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "detections", shape, dxrt::DataType::FLOAT, rows.data()));
    const dxapp::PreprocessContext identity(0, 0, 1.f, 640, 640, 640, 640);
    COMMON_CHECK(post->process(outputs, identity).size() == 1);  // 0.36 >= 0.3
}

void TestSrMergeLumaIsTheRunnersMerge() {
    cv::Mat bgr(6, 8, CV_8UC3);
    cv::randu(bgr, 0, 255);
    cv::Mat sr_y(12, 16, CV_8UC1);
    cv::randu(sr_y, 0, 255);
    // The runner's inline code before the extraction, verbatim.
    cv::Mat lr_ycrcb; dxapp::colorspace::bgrToYCrCbLimited(bgr, lr_ycrcb);
    std::vector<cv::Mat> ch; cv::split(lr_ycrcb, ch);
    cv::Mat cr_up, cb_up;
    cv::resize(ch[1], cr_up, cv::Size(16, 12), 0, 0, cv::INTER_CUBIC);
    cv::resize(ch[2], cb_up, cv::Size(16, 12), 0, 0, cv::INTER_CUBIC);
    cv::Mat merged; cv::merge(std::vector<cv::Mat>{sr_y, cr_up, cb_up}, merged);
    cv::Mat expected; dxapp::colorspace::ycrcbLimitedToBgr(merged, expected);
    const cv::Mat got = dxapp::srtiling::mergeSrLuma(bgr, sr_y);
    COMMON_CHECK(got.size() == expected.size() && got.type() == expected.type());
    if (got.size() == expected.size() && got.type() == expected.type()) {
        COMMON_CHECK(cv::norm(got, expected, cv::NORM_INF) == 0);
    }
}

void TestSrPrepareLowResPadsAndConvertsLikeTheRunner() {
    cv::Mat bgr(10, 7, CV_8UC3);
    cv::randu(bgr, 0, 255);
    cv::Mat padded;
    cv::copyMakeBorder(bgr, padded, 0, 26 - 10, 0, 17 - 7, cv::BORDER_REPLICATE);
    cv::Mat expected_gray; dxapp::colorspace::bgrToYLimited(padded, expected_gray);
    cv::Mat lr_bgr, lr_gray;
    dxapp::srtiling::prepareLowRes(bgr, 26, 17, lr_bgr, lr_gray);
    COMMON_CHECK(lr_bgr.size() == cv::Size(17, 26) && cv::norm(lr_bgr, padded, cv::NORM_INF) == 0);
    COMMON_CHECK(lr_gray.size() == cv::Size(17, 26) && cv::norm(lr_gray, expected_gray, cv::NORM_INF) == 0);
}

/// Stands in for dxrt::InferenceEngine in runTilesPipelined: RunAsync throws
/// on call number `throw_on` (1-based); Wait checks that the tile buffer it
/// was given at submit still holds the tile's bytes.
class ThrowingTileEngine {
public:
    explicit ThrowingTileEngine(int throw_on) : throw_on_(throw_on), calls_(0) {}

    int RunAsync(void* input) {
        ++calls_;
        if (calls_ == throw_on_) throw std::runtime_error("RunAsync failed");
        const int job = static_cast<int>(submitted_.size());
        submitted_.push_back(static_cast<const unsigned char*>(input));
        copies_.push_back(std::vector<unsigned char>(
            static_cast<const unsigned char*>(input),
            static_cast<const unsigned char*>(input) + kTileBytes));
        return job;
    }

    dxrt::TensorPtrs Wait(int job) {
        waited_.push_back(job);
        intact_.push_back(std::equal(copies_[job].begin(), copies_[job].end(),
                                     submitted_[job]));
        return dxrt::TensorPtrs();
    }

    static const int kTile = 4;
    static const int kTileBytes = kTile * kTile;
    std::vector<int> waited_;
    std::vector<bool> intact_;
    std::size_t submitted() const { return submitted_.size(); }

private:
    int throw_on_;
    int calls_;
    std::vector<const unsigned char*> submitted_;
    std::vector<std::vector<unsigned char> > copies_;
};

// A RunAsync that throws partway through the submit loop: every job already
// submitted is waited for, while its tile buffer is still alive, and the
// error still reaches the caller.
void TestSrPipelinedTilesWaitForSubmittedJobsWhenRunAsyncThrows() {
    const int tile = ThrowingTileEngine::kTile;
    cv::Mat plane(tile, tile * 5, CV_8UC1);
    cv::randu(plane, 0, 255);
    std::vector<dxapp::srtiling::TilePlan> plans(5);
    for (int i = 0; i < 5; ++i) {
        plans[i].win_x = i * tile;
        plans[i].win_y = 0;
    }
    ThrowingTileEngine engine(3);
    std::vector<dxrt::TensorPtrs> outputs;
    bool threw = false;
    try {
        dxapp::srtiling::runTilesPipelined(engine, plane, plans, tile, tile, outputs, 16);
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()) == "RunAsync failed";
    }
    COMMON_CHECK(threw);
    COMMON_CHECK(engine.submitted() == 2);
    COMMON_CHECK(engine.waited_.size() == 2);
    if (engine.waited_.size() == 2) {
        COMMON_CHECK(engine.waited_[0] == 0 && engine.waited_[1] == 1);
        COMMON_CHECK(engine.intact_[0] && engine.intact_[1]);
    }
}

/// A nearest-neighbour x2 "model" for 17x17 uint8 tiles: each output is 34x34
/// floats in [0, 1]. At every Wait() it records how many outputs it handed out
/// earlier are still referenced by the caller.
class UpscalingTileEngine {
public:
    static const int kTile = 17;
    static const int kScale = 2;
    static const int kOutTile = kTile * kScale;

    int RunAsync(void* input) {
        const unsigned char* bytes = static_cast<const unsigned char*>(input);
        jobs_.push_back(std::vector<unsigned char>(bytes, bytes + kTile * kTile));
        return static_cast<int>(jobs_.size()) - 1;
    }

    dxrt::TensorPtrs Wait(int job) {
        int alive = 0;
        for (const auto& ref : handed_out_) alive += ref.expired() ? 0 : 1;
        max_alive_ = std::max(max_alive_, alive);

        auto storage = std::make_shared<std::vector<float>>(kOutTile * kOutTile);
        for (int y = 0; y < kOutTile; ++y) {
            for (int x = 0; x < kOutTile; ++x) {
                (*storage)[y * kOutTile + x] =
                    jobs_[job][(y / kScale) * kTile + x / kScale] / 255.0f;
            }
        }
        std::vector<int64_t> shape;
        shape.push_back(1);
        shape.push_back(1);
        shape.push_back(kOutTile);
        shape.push_back(kOutTile);
        // The tensor does not own its data; the deleter keeps the buffer alive with it.
        std::shared_ptr<dxrt::Tensor> tensor(
            new dxrt::Tensor("sr", shape, dxrt::DataType::FLOAT, storage->data()),
            [storage](dxrt::Tensor* t) { delete t; });
        handed_out_.push_back(tensor);
        return dxrt::TensorPtrs(1, tensor);
    }

    int max_alive_ = 0;

private:
    std::vector<std::vector<unsigned char>> jobs_;
    std::vector<std::weak_ptr<dxrt::Tensor>> handed_out_;
};

/// A 40x61 random plane cut into 17x17 tiles with a halo of 4.
void MakeSrTilingCase(int& padded_h, int& padded_w,
                      std::vector<dxapp::srtiling::TilePlan>& plans, cv::Mat& plane) {
    const int tile = UpscalingTileEngine::kTile;
    dxapp::srtiling::planTiles(40, 61, tile, tile, 4, padded_h, padded_w, plans);
    plane.create(padded_h, padded_w, CV_8UC1);
    cv::randu(plane, 0, 256);
}

// Streaming assembly (each tile stitched as its output arrives) gives the same
// image as collect-then-assemble, pixel for pixel.
void TestSrStreamingAssemblyMatchesCollectThenAssemble() {
    const int tile = UpscalingTileEngine::kTile;
    const int scale = UpscalingTileEngine::kScale;
    const int out_tile = UpscalingTileEngine::kOutTile;
    int padded_h = 0, padded_w = 0;
    std::vector<dxapp::srtiling::TilePlan> plans;
    cv::Mat plane;
    MakeSrTilingCase(padded_h, padded_w, plans, plane);

    UpscalingTileEngine collecting;
    std::vector<dxrt::TensorPtrs> outputs;
    dxapp::srtiling::runTilesPipelined(collecting, plane, plans, tile, tile, outputs, 16);
    cv::Mat expected;
    const int expected_done = dxapp::srtiling::assembleTiles(
        plans, outputs, padded_h * scale, padded_w * scale, scale, scale, out_tile, expected);

    UpscalingTileEngine streaming;
    cv::Mat streamed;
    const int streamed_done = dxapp::srtiling::runAndAssembleTiles(
        streaming, plane, plans, tile, tile, padded_h * scale, padded_w * scale,
        scale, scale, out_tile, streamed, 16);

    COMMON_CHECK(expected_done == static_cast<int>(plans.size()));
    COMMON_CHECK(streamed_done == expected_done);
    COMMON_CHECK(streamed.size() == expected.size() && streamed.type() == expected.type());
    COMMON_CHECK(cv::norm(streamed, expected, cv::NORM_INF) == 0);
}

// Streaming keeps no finished tile output alive; collect-then-assemble holds
// every one (25,560 at 1080p for ESPCN x4, ~0.5 GB).
void TestSrStreamingAssemblyKeepsNoFinishedOutputAlive() {
    const int tile = UpscalingTileEngine::kTile;
    const int scale = UpscalingTileEngine::kScale;
    int padded_h = 0, padded_w = 0;
    std::vector<dxapp::srtiling::TilePlan> plans;
    cv::Mat plane;
    MakeSrTilingCase(padded_h, padded_w, plans, plane);

    UpscalingTileEngine streaming;
    cv::Mat sr_y;
    dxapp::srtiling::runAndAssembleTiles(streaming, plane, plans, tile, tile,
                                         padded_h * scale, padded_w * scale, scale, scale,
                                         UpscalingTileEngine::kOutTile, sr_y, 4);
    COMMON_CHECK(streaming.max_alive_ == 0);

    UpscalingTileEngine collecting;
    std::vector<dxrt::TensorPtrs> outputs;
    dxapp::srtiling::runTilesPipelined(collecting, plane, plans, tile, tile, outputs, 4);
    COMMON_CHECK(collecting.max_alive_ == static_cast<int>(plans.size()) - 1);
}

// =====================================================================
// I17: one ROI crop rule (common/utility/roi_crop.hpp) for the graph engine.
// =====================================================================

// The graph engine's rule before the extraction, verbatim from
// common/graph/roi_router.cpp (ApplyPad, ClipToFrame).
cv::Rect2f GraphApplyPadBefore(const cv::Rect2f& box, float pad) {
    if (pad <= 0.f) return box;
    const float dx = box.width * pad;
    const float dy = box.height * pad;
    return cv::Rect2f(box.x - dx, box.y - dy, box.width + dx * 2.f,
                      box.height + dy * 2.f);
}

cv::Rect GraphClipToFrameBefore(const cv::Rect2f& box, const cv::Mat& source) {
    const float x1 = std::max(0.f, box.x);
    const float y1 = std::max(0.f, box.y);
    const float x2 = std::min(static_cast<float>(source.cols), box.x + box.width);
    const float y2 = std::min(static_cast<float>(source.rows), box.y + box.height);
    if (x2 <= x1 || y2 <= y1) return cv::Rect();
    return cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
}

void TestPaddedCropRectMatchesTheGraphRule() {
    const cv::Mat frame(480, 640, CV_8UC3);
    const float xs[] = {-3.7f, 0.f, 10.2f, 637.9f};
    const float sizes[] = {0.4f, 5.5f, 100.25f};
    const float pads[] = {0.f, 0.1f, 0.25f};
    int cases = 0;
    int mismatches = 0;
    for (const float x : xs) {
        for (const float y : xs) {
            for (const float w : sizes) {
                for (const float h : sizes) {
                    for (const float pad : pads) {
                        const cv::Rect2f box(x, y, w, h);
                        const cv::Rect2f padded = dxapp::PadBox(box, pad);
                        const cv::Rect2f expected_pad = GraphApplyPadBefore(box, pad);
                        const cv::Rect expected =
                            GraphClipToFrameBefore(expected_pad, frame);
                        ++cases;
                        if (padded != expected_pad ||
                            dxapp::ClipBoxToFrame(padded, frame.cols, frame.rows) != expected ||
                            dxapp::PaddedCropRect(box, pad, frame.cols, frame.rows) != expected) {
                            ++mismatches;
                            std::printf("  crop rule differs: box %g,%g %gx%g pad %g\n",
                                        x, y, w, h, pad);
                        }
                    }
                }
            }
        }
    }
    COMMON_CHECK(cases == 4 * 4 * 3 * 3 * 3);
    COMMON_CHECK(mismatches == 0);
}

void TestPaddedCropRectEdgeCases() {
    const int cols = 640;
    const int rows = 480;
    // Fractional coordinates: clamp in float, truncate the corner and the
    // size once. y 10.7 .. 20.2 is 9 rows (truncating each corner first
    // would cut 10 rows, 10 .. 20).
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(3.5f, 10.7f, 5.f, 9.5f), 0.f, cols, rows) ==
                 cv::Rect(3, 10, 5, 9));
    // Over the right border: x 634.7 + 8.5 is clipped at 640.
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(634.7f, 200.f, 8.5f, 20.f), 0.f, cols, rows) ==
                 cv::Rect(634, 200, 5, 20));
    // Over the top-left corner: starts at 0.
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(-5.f, -2.5f, 15.f, 12.5f), 0.f, cols, rows) ==
                 cv::Rect(0, 0, 10, 10));
    // Exactly the frame, and a box larger than the frame on every side.
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(0.f, 0.f, 640.f, 480.f), 0.f, cols, rows) ==
                 cv::Rect(0, 0, cols, rows));
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(-10.f, -10.f, 700.f, 500.f), 0.1f, cols, rows) ==
                 cv::Rect(0, 0, cols, rows));
    // Pad grows each side by pad x size: 100 x 50 at pad 0.25 -> 25 / 12.5.
    COMMON_CHECK(dxapp::PadBox(cv::Rect2f(100.f, 100.f, 100.f, 50.f), 0.25f) ==
                 cv::Rect2f(75.f, 87.5f, 150.f, 75.f));
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(100.f, 100.f, 100.f, 50.f), 0.25f, cols, rows) ==
                 cv::Rect(75, 87, 150, 75));
    // A negative or zero pad leaves the box alone.
    COMMON_CHECK(dxapp::PadBox(cv::Rect2f(1.5f, 2.5f, 3.f, 4.f), -0.5f) ==
                 cv::Rect2f(1.5f, 2.5f, 3.f, 4.f));
    // Zero-size and inverted boxes, and boxes wholly outside: empty.
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(10.f, 10.f, 0.f, 5.f), 0.f, cols, rows).empty());
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(10.f, 10.f, 5.f, 0.f), 0.25f, cols, rows).empty());
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(10.f, 10.f, -4.f, 5.f), 0.f, cols, rows).empty());
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(640.f, 10.f, 5.f, 5.f), 0.f, cols, rows).empty());
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(-20.f, 10.f, 20.f, 5.f), 0.f, cols, rows).empty());
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(10.f, 480.5f, 5.f, 5.f), 0.f, cols, rows).empty());
    // A sliver under one pixel inside the frame truncates to a zero size.
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(10.2f, 10.f, 0.4f, 5.f), 0.f, cols, rows).width == 0);
    // An empty frame gives no crop.
    COMMON_CHECK(dxapp::PaddedCropRect(cv::Rect2f(0.f, 0.f, 5.f, 5.f), 0.f, 0, 0).empty());
}

// =====================================================================
// U-77: ModelConfig reads JSON strings as JSON does.
// =====================================================================

void TestModelConfigBracketsInsideStringsAreText() {
    const dxapp::ModelConfig config(
        "{\"class_names\": [\"a]\", \"b[\", \"c\"], \"score_threshold\": 0.25, "
        "\"label\": \"x]y\", \"after\": 3}", dxapp::ConfigSource::kText);
    const std::vector<std::string> names = config.get_string_list("class_names");
    COMMON_CHECK(names.size() == 3);
    if (names.size() == 3) COMMON_CHECK(names[0] == "a]" && names[1] == "b[" && names[2] == "c");
    COMMON_CHECK(config.get<float>("score_threshold", 0.f) == 0.25f);
    COMMON_CHECK(config.get<std::string>("label", "") == "x]y");
    COMMON_CHECK(config.get<int>("after", 0) == 3);
}

void TestModelConfigDecodesJsonEscapes() {
    const dxapp::ModelConfig config(
        "{\"q\": \"say \\\"hi\\\"\", \"b\": \"a\\\\b\", \"c\": \"t\\tn\\n/\\/\", "
        "\"u\": \"\\u00e9\\ud83d\\ude00\", \"odd\": \"\\x\", "
        "\"list\": [\"x\\\"y\", \"z\\\\w\"]}", dxapp::ConfigSource::kText);
    COMMON_CHECK(config.get<std::string>("q", "") == "say \"hi\"");
    COMMON_CHECK(config.get<std::string>("b", "") == "a\\b");
    COMMON_CHECK(config.get<std::string>("c", "") == "t\tn\n//");
    COMMON_CHECK(config.get<std::string>("u", "") == "\xc3\xa9\xf0\x9f\x98\x80");  // é 😀
    COMMON_CHECK(config.get<std::string>("odd", "") == "\\x");  // unknown escape kept
    const std::vector<std::string> list = config.get_string_list("list");
    COMMON_CHECK(list.size() == 2);
    if (list.size() == 2) COMMON_CHECK(list[0] == "x\"y" && list[1] == "z\\w");
}

void TestModelConfigSkipsNestedObjectsWholeStringsIncluded() {
    const dxapp::ModelConfig config(
        "{\"obj\": {\"k\": \"}]\", \"n\": [1, 2]}, \"after\": 7}", dxapp::ConfigSource::kText);
    COMMON_CHECK(config.get<int>("after", 0) == 7);
    COMMON_CHECK(config.get<std::string>("k", "absent") == "absent");  // inner keys are not top-level
    // A lone '}' in a nested string: counted as structure, it ends the
    // object early and the next key is read from the wrong quote.
    const dxapp::ModelConfig lone("{\"obj\": {\"k\": \"}\"}, \"after\": 7}", dxapp::ConfigSource::kText);
    COMMON_CHECK(lone.get<int>("after", 0) == 7);
}

void TestModelConfigDecodesTheRemainingEscapes() {
    const dxapp::ModelConfig config(
        "{\"bfr\": \"\\b\\f\\r\", \"nul\": \"a\\u0000b\", \"bad\": \"\\u12G4\", "
        "\"high\": \"x\\ud83dy\", \"low\": \"\\ude00\", \"unpaired\": \"\\ud83d\\u0041\"}",
        dxapp::ConfigSource::kText);
    COMMON_CHECK(config.get<std::string>("bfr", "") == "\b\f\r");
    COMMON_CHECK(config.get<std::string>("nul", "") == std::string("a\0b", 3));
    COMMON_CHECK(config.get<std::string>("bad", "") == "\\u12G4");  // not hex: kept as written
    // A lone surrogate is not a code point: U+FFFD, as JSON decoders do,
    // never the 3-byte CESU form, which is not UTF-8.
    COMMON_CHECK(config.get<std::string>("high", "") == "x\xef\xbf\xbdy");
    COMMON_CHECK(config.get<std::string>("low", "") == "\xef\xbf\xbd");
    COMMON_CHECK(config.get<std::string>("unpaired", "") == "\xef\xbf\xbd" "A");
}

/// Every nested object's keys, at any depth (objects inside arrays too).
void CollectNestedKeys(const nlohmann::json& value, std::set<std::string>* keys) {
    if (value.is_object()) {
        for (nlohmann::json::const_iterator kv = value.begin(); kv != value.end(); ++kv) {
            keys->insert(kv.key());
            CollectNestedKeys(kv.value(), keys);
        }
    } else if (value.is_array()) {
        for (std::size_t i = 0; i < value.size(); ++i) CollectNestedKeys(value[i], keys);
    }
}

/// What ModelConfig reads from a parsed document (C3): its top-level
/// members, with a top-level "config" object's members laid over them -
/// the nested value wins - except the objects among them, which stay
/// unread like every other nested object.
nlohmann::json EffectiveConfigView(const nlohmann::json& document) {
    nlohmann::json view = document;
    if (!document.is_object()) return view;
    const nlohmann::json::const_iterator config = document.find("config");
    if (config == document.end() || !config->is_object()) return view;
    for (nlohmann::json::const_iterator kv = config->begin(); kv != config->end(); ++kv) {
        if (!kv.value().is_object()) view[kv.key()] = kv.value();
    }
    return view;
}

/// The keys where ModelConfig reads `text` differently from nlohmann::json's
/// EffectiveConfigView ("<parse error: ...>" when the reference parser
/// rejects it). Strings, numbers, booleans and null by value; arrays by
/// their raw JSON text (numbers and nesting included); any other nested
/// object's keys must not read back as top-level ones.
std::vector<std::string> KeysModelConfigReadsDifferently(const std::string& text) {
    std::vector<std::string> differ;
    nlohmann::json reference;
    try {
        reference = EffectiveConfigView(nlohmann::json::parse(text));
    } catch (const std::exception& e) {
        differ.push_back(std::string("<parse error: ") + e.what() + ">");
        return differ;
    }
    const dxapp::ModelConfig config(text, dxapp::ConfigSource::kText);
    const std::map<std::string, std::string>& arrays = config.rawArrays();
    std::set<std::string> nested;
    for (nlohmann::json::const_iterator kv = reference.begin(); kv != reference.end(); ++kv) {
        const nlohmann::json& value = kv.value();
        bool same = true;
        if (value.is_string()) {
            same = config.get<std::string>(kv.key(), "<absent>") == value.get<std::string>();
        } else if (value.is_number()) {
            same = config.get<double>(kv.key(), -1e300) == value.get<double>();
        } else if (value.is_boolean()) {
            same = config.get<std::string>(kv.key(), "") == (value.get<bool>() ? "true" : "false");
        } else if (value.is_null()) {
            same = config.get<std::string>(kv.key(), "<absent>") == "null";
        } else if (value.is_array()) {
            const std::map<std::string, std::string>::const_iterator raw = arrays.find(kv.key());
            same = raw != arrays.end() && nlohmann::json::parse(raw->second) == value;
            CollectNestedKeys(value, &nested);
        } else if (value.is_object()) {
            CollectNestedKeys(value, &nested);
        }
        if (!same) differ.push_back(kv.key());
    }
    for (std::set<std::string>::const_iterator key = nested.begin(); key != nested.end(); ++key) {
        if (reference.count(*key) != 0) continue;  // also a top-level key: compared above
        if (config.get<std::string>(*key, "<absent>") != "<absent>" || arrays.count(*key) != 0) {
            differ.push_back("nested key leaked: " + *key);
        }
    }
    return differ;
}

void PrintDifferences(const std::string& where, const std::vector<std::string>& differ) {
    for (std::size_t i = 0; i < differ.size(); ++i) {
        std::printf("      %s: %s differs\n", where.c_str(), differ[i].c_str());
    }
}

// The guard below compares every JSON shape, though no shipped config.json
// has a null, a non-string list or a nested object today.
void TestModelConfigAgreesWithAJsonParserOnEveryShape() {
    const std::string text =
        "{\"s\": \"x\", \"n\": -1.5e3, \"t\": true, \"f\": false, \"z\": null, "
        "\"nums\": [1, 2.5, -3], \"grid\": [[1, 2], [3]], \"mixed\": [\"a\", 1, null, {\"in\": 1}], "
        "\"obj\": {\"inner\": \"v\", \"deep\": {\"deeper\": [1]}}, \"after\": 7}";
    const std::vector<std::string> differ = KeysModelConfigReadsDifferently(text);
    PrintDifferences("every-shape document", differ);
    COMMON_CHECK(differ.empty());
    const std::vector<std::string> broken = KeysModelConfigReadsDifferently("{\"a\": ");
    COMMON_CHECK(broken.size() == 1 && broken[0].find("<parse error") == 0);
}

// Review Focus 1: the rewrite keeps every shipped config.json's meaning,
// the nested "config" object included - one per variant,
// <task>/<family>/<variant>/config.json.
void TestModelConfigAgreesWithAJsonParserOnEveryShippedConfig() {
    const fs::path root = fs::path(PROJECT_ROOT_DIR) / "src" / "cpp_example";
    int files = 0;
    int variant_files = 0;
    for (fs::recursive_directory_iterator it(root), end; it != end; ++it) {
        if (it->path().filename() != "config.json") continue;
        ++files;
        if (it.depth() == 3) ++variant_files;
        std::ifstream in(it->path().string().c_str());
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        const std::vector<std::string> differ = KeysModelConfigReadsDifferently(text);
        PrintDifferences(it->path().string(), differ);
        COMMON_CHECK(differ.empty());
    }
    COMMON_CHECK(variant_files >= 499);  // 499 at 8d0b748: the walk found them
    COMMON_CHECK(files == variant_files);
}

// =====================================================================
// Camera capture: USB webcam + Raspberry Pi CSI camera (utility/camera.hpp)
// =====================================================================

const char* const kRpicamListTwoCameras =
    "Available cameras\n"
    "-----------------\n"
    "0 : imx708_wide [4608x2592 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a)\n"
    "    Modes: 'SRGGB10_CSI2P' : 1536x864 [120.13 fps - (768, 432)/3072x1728 crop]\n"
    "                             2304x1296 [56.03 fps - (0, 0)/4608x2592 crop]\n"
    "\n"
    "1 : imx219 [3280x2464 8-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@80000/imx219@10)\n"
    "    Modes: 'SRGGB8' : 640x480 [103.33 fps - (1000, 752)/1280x960 crop]\n";

void TestCameraRaspberryPiModelString() {
    COMMON_CHECK(dxapp::isRaspberryPiModel(std::string("Raspberry Pi 5 Model B Rev 1.0") + '\0'));
    COMMON_CHECK(dxapp::isRaspberryPiModel("raspberry pi 4 model b"));
    COMMON_CHECK(!dxapp::isRaspberryPiModel("Radxa ROCK 5B"));
    COMMON_CHECK(!dxapp::isRaspberryPiModel(""));
}

void TestCameraCsiNodeNames() {
    COMMON_CHECK(dxapp::isRpiCsiDeviceName("rp1-cfe"));
    COMMON_CHECK(dxapp::isRpiCsiDeviceName("unicam"));
    COMMON_CHECK(dxapp::isRpiCsiDeviceName("pispbe"));
    COMMON_CHECK(dxapp::isRpiCsiDeviceName("bcm2835-isp"));
    // The legacy V4L2 capture driver opens through OpenCV's V4L2 backend.
    COMMON_CHECK(!dxapp::isRpiCsiDeviceName("bcm2835-v4l2"));
    COMMON_CHECK(!dxapp::isRpiCsiDeviceName("UVC Camera (046d:0825)"));
    COMMON_CHECK(!dxapp::isRpiCsiDeviceName(""));
}

void TestCameraLibcameraIdComesFromTheCameraList() {
    COMMON_CHECK(dxapp::parseLibcameraCameraId(kRpicamListTwoCameras, 0) ==
                 "/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a");
    COMMON_CHECK(dxapp::parseLibcameraCameraId(kRpicamListTwoCameras, 1) ==
                 "/base/axi/pcie@120000/rp1/i2c@80000/imx219@10");
    COMMON_CHECK(dxapp::parseLibcameraCameraId(kRpicamListTwoCameras, 2).empty());
    COMMON_CHECK(dxapp::parseLibcameraCameraId("No cameras available!\n", 0).empty());
    COMMON_CHECK(dxapp::parseLibcameraCameraId("", 0).empty());
}

void TestCameraLibcameraPipelineString() {
    COMMON_CHECK(dxapp::buildLibcameraPipeline("", 1280, 720) ==
                 "libcamerasrc ! video/x-raw,width=1280,height=720,format=NV12"
                 " ! videoconvert ! video/x-raw,format=BGR"
                 " ! appsink drop=true max-buffers=2");
    COMMON_CHECK(dxapp::buildLibcameraPipeline("/base/soc/i2c0mux/i2c@1/imx477@1a", 640, 480) ==
                 "libcamerasrc camera-name=\"/base/soc/i2c0mux/i2c@1/imx477@1a\""
                 " ! video/x-raw,width=640,height=480,format=NV12"
                 " ! videoconvert ! video/x-raw,format=BGR"
                 " ! appsink drop=true max-buffers=2");
}

void TestCameraGStreamerFromBuildInformation() {
    COMMON_CHECK(dxapp::buildInfoHasGStreamer("  Video I/O:\n    GStreamer:                   YES (1.26.0)\n"));
    COMMON_CHECK(!dxapp::buildInfoHasGStreamer("  Video I/O:\n    GStreamer:                   NO\n"));
    COMMON_CHECK(!dxapp::buildInfoHasGStreamer("  Video I/O:\n    FFMPEG:                      YES\n"));
}

// A hung or unrunnable rpicam-hello (`timeout` exits 124-127) is "no listing",
// like camera.py's None, so camera 0 still opens libcamerasrc's first camera.
void TestCameraListingNeedsTheToolToFinish() {
#ifdef _WIN32
    Skip("TestCameraListingNeedsTheToolToFinish", "libcamera listing is POSIX only");
#else
    COMMON_CHECK(dxapp::cameraListingCompleted(std::system("exit 0")));
    COMMON_CHECK(dxapp::cameraListingCompleted(std::system("exit 1")));
    COMMON_CHECK(!dxapp::cameraListingCompleted(std::system("timeout 1 sleep 5")));
    COMMON_CHECK(!dxapp::cameraListingCompleted(std::system("exit 126")));
    COMMON_CHECK(!dxapp::cameraListingCompleted(std::system("exit 127")));
    COMMON_CHECK(!dxapp::cameraListingCompleted(-1));  // pclose() failed
#endif
}

void TestCameraOffAPiOpensLikeVideoCapture() {
    if (dxapp::isRaspberryPi()) {
        Skip("TestCameraOffAPiOpensLikeVideoCapture", "running on a Raspberry Pi");
        return;
    }
    cv::VideoCapture video;
    COMMON_CHECK(!dxapp::openCameraCapture(video, 9999));
    COMMON_CHECK(!video.isOpened());
}

// =====================================================================
// MediaPipe hand landmark: a keypoint's confidence is the hand presence
// score, never the landmark's relative depth (z), which is negative for
// most points and made every consumer that thresholds on confidence (the
// graph's keypoint drawing) drop all 21 points.
// =====================================================================

/// HandLandmarkLite's four outputs: [1,63] landmarks in input pixels with a
/// negative z each, [1,1] presence, [1,1] handedness, [1,63] world landmarks.
dxrt::TensorPtrs HandLandmarkOutputs(std::vector<std::vector<float> >& storage,
                                     float presence, std::size_t count) {
    storage.assign(4, std::vector<float>());
    for (int i = 0; i < 21; ++i) {
        storage[0].push_back(20.f + static_cast<float>(i) * 8.f);   // x
        storage[0].push_back(200.f - static_cast<float>(i) * 7.f);  // y
        storage[0].push_back(-0.5f - static_cast<float>(i));        // z (relative depth)
    }
    storage[1].push_back(presence);
    storage[2].push_back(0.9f);
    storage[3].assign(63, 0.01f);
    const int64_t sizes[] = {63, 1, 1, 63};
    const char* names[] = {"Identity", "Identity_1", "Identity_2", "Identity_3"};
    dxrt::TensorPtrs outputs;
    for (std::size_t t = 0; t < count; ++t) {
        std::vector<int64_t> shape;
        shape.push_back(1);
        shape.push_back(sizes[t]);
        outputs.push_back(std::make_shared<dxrt::Tensor>(
            names[t], shape, dxrt::DataType::FLOAT, storage[t].data()));
    }
    return outputs;
}

dxapp::PreprocessContext HandContext() {
    dxapp::PreprocessContext ctx;
    ctx.original_width = 224;
    ctx.original_height = 224;
    return ctx;
}

void TestHandLandmarkKeypointsCarryPresenceNotDepth() {
    std::vector<std::vector<float> > storage;
    const dxrt::TensorPtrs outputs = HandLandmarkOutputs(storage, 0.93f, 4);
    dxapp::HandLandmarkPostprocessor post(224, 224);
    const std::vector<dxapp::HandLandmarkResult> hands = post.process(outputs, HandContext());
    COMMON_CHECK(hands.size() == 1);
    if (hands.size() != 1) return;
    COMMON_CHECK(hands[0].landmarks.size() == 21);
    COMMON_CHECK(hands[0].confidence == 0.93f);
    COMMON_CHECK(hands[0].handedness == "Right");
    for (std::size_t i = 0; i < hands[0].landmarks.size(); ++i) {
        const dxapp::Keypoint& kp = hands[0].landmarks[i];
        COMMON_CHECK(kp.confidence >= 0.f && kp.confidence <= 1.f);
        COMMON_CHECK(kp.confidence == 0.93f);
    }
    COMMON_CHECK(hands[0].landmarks[20].x == 180.f);
    COMMON_CHECK(hands[0].landmarks[20].y == 60.f);
}

/// Landmarks alone (no presence output): nothing scores the points, so each
/// one is fully confident rather than carrying its depth.
void TestHandLandmarkKeypointsWithoutPresenceAreFullyConfident() {
    std::vector<std::vector<float> > storage;
    const dxrt::TensorPtrs outputs = HandLandmarkOutputs(storage, 0.f, 1);
    dxapp::HandLandmarkPostprocessor post(224, 224);
    const std::vector<dxapp::HandLandmarkResult> hands = post.process(outputs, HandContext());
    COMMON_CHECK(hands.size() == 1);
    if (hands.size() != 1) return;
    COMMON_CHECK(hands[0].confidence == 1.f);
    for (std::size_t i = 0; i < hands[0].landmarks.size(); ++i) {
        COMMON_CHECK(hands[0].landmarks[i].confidence == 1.f);
    }
}

/// The C++ result has no depth field, so the verify dump does not invent one
/// from the keypoint's confidence.
void TestHandLandmarkDumpDoesNotReportConfidenceAsDepth() {
    std::vector<std::vector<float> > storage;
    const dxrt::TensorPtrs outputs = HandLandmarkOutputs(storage, 0.93f, 4);
    dxapp::HandLandmarkPostprocessor post(224, 224);
    const std::vector<dxapp::HandLandmarkResult> hands = post.process(outputs, HandContext());
    const nlohmann::json dump = dxapp::verify::serializeHandLandmark(hands, 224, 224);
    COMMON_CHECK(dump["detections"].size() == 1);
    if (dump["detections"].size() != 1) return;
    const nlohmann::json& point = dump["detections"][0]["landmarks"][0];
    COMMON_CHECK(point.contains("x") && point.contains("y"));
    COMMON_CHECK(!point.contains("z"));
}


// =====================================================================
// PoseVisualizer joins keypoints by what they are, read from their count:
// COCO body (17), MediaPipe hand (21), 68-point face, CenterPose box (8).
// Any other count, and unordered points (SuperPoint), is drawn as points
// only - never with the COCO body limbs between unrelated indices.
// =====================================================================

/// `count` points evenly on a circle of radius 220 around (240, 240): 68 of
/// them are 20 px apart, so a probe between two never touches either dot.
dxapp::PoseResult PoseOnCircle(std::size_t count, float confidence) {
    dxapp::PoseResult pose;
    for (std::size_t i = 0; i < count; ++i) {
        const double angle = 2.0 * 3.14159265358979 * static_cast<double>(i) / static_cast<double>(count);
        pose.keypoints.push_back(dxapp::Keypoint(static_cast<float>(240.0 + 220.0 * std::cos(angle)),
                                                 static_cast<float>(240.0 + 220.0 * std::sin(angle)),
                                                 confidence));
    }
    return pose;
}

cv::Mat DrawPose(const dxapp::PoseResult& pose, bool unordered) {
    dxapp::PoseVisualizer visualizer;
    visualizer.setUnorderedKeypoints(unordered);
    dxapp::PreprocessContext ctx;
    ctx.original_width = 480;
    ctx.original_height = 480;
    return visualizer.draw(cv::Mat::zeros(480, 480, CV_8UC3),
                           std::vector<dxapp::PoseResult>(1, pose), ctx);
}

/// Brightest channel in the 5x5 window around the midpoint of keypoints a
/// and b: lines are anti-aliased, so one exact pixel is too strict a probe.
int PeakBetween(const cv::Mat& canvas, const dxapp::PoseResult& pose, int a, int b) {
    const cv::Point mid(cvRound((pose.keypoints[a].x + pose.keypoints[b].x) / 2.f),
                        cvRound((pose.keypoints[a].y + pose.keypoints[b].y) / 2.f));
    const cv::Rect window = cv::Rect(mid.x - 2, mid.y - 2, 5, 5) & cv::Rect(0, 0, canvas.cols, canvas.rows);
    double peak = 0.0;
    cv::minMaxLoc(canvas(window).reshape(1), NULL, &peak);
    return static_cast<int>(peak);
}

int PointsDrawn(const cv::Mat& canvas, const dxapp::PoseResult& pose) {
    int drawn = 0;
    for (std::size_t i = 0; i < pose.keypoints.size(); ++i) {
        const int k = static_cast<int>(i);
        if (PeakBetween(canvas, pose, k, k) > 100) ++drawn;
    }
    return drawn;
}

void TestPoseVisualizerJoinsCocoBodyLimbs() {
    const dxapp::PoseResult pose = PoseOnCircle(17, 0.9f);
    const cv::Mat canvas = DrawPose(pose, false);
    COMMON_CHECK(PeakBetween(canvas, pose, 5, 11) > 100);  // left shoulder -> left hip
    COMMON_CHECK(PeakBetween(canvas, pose, 0, 1) > 100);   // nose -> left eye
    COMMON_CHECK(PeakBetween(canvas, pose, 9, 10) == 0);   // wrists are not joined
    COMMON_CHECK(PointsDrawn(canvas, pose) == 17);
}

void TestPoseVisualizerDrawsTheFaceOutlineFor68Points() {
    const dxapp::PoseResult pose = PoseOnCircle(68, 1.f);
    const cv::Mat canvas = DrawPose(pose, false);
    COMMON_CHECK(PeakBetween(canvas, pose, 0, 1) > 100);    // jaw
    COMMON_CHECK(PeakBetween(canvas, pose, 41, 36) > 100);  // left eye closes
    COMMON_CHECK(PeakBetween(canvas, pose, 59, 48) > 100);  // outer lip closes
    COMMON_CHECK(PeakBetween(canvas, pose, 67, 60) > 100);  // inner lip closes
    COMMON_CHECK(PeakBetween(canvas, pose, 16, 17) == 0);   // jaw end -> brow: no line
    COMMON_CHECK(PeakBetween(canvas, pose, 21, 22) == 0);   // brow -> brow: no line
    COMMON_CHECK(PeakBetween(canvas, pose, 5, 11) == 0);    // no COCO limb
    COMMON_CHECK(PeakBetween(canvas, pose, 6, 12) == 0);    // no COCO limb
    COMMON_CHECK(PointsDrawn(canvas, pose) == 68);
}

void TestPoseVisualizerDrawsTheCenterPoseBox() {
    const dxapp::PoseResult pose = PoseOnCircle(8, 0.9f);
    const cv::Mat canvas = DrawPose(pose, false);
    COMMON_CHECK(PeakBetween(canvas, pose, 3, 0) > 100);  // front face closes
    COMMON_CHECK(PeakBetween(canvas, pose, 0, 4) > 100);  // front -> back
}

void TestPoseVisualizerDrawsOtherCountsAsPointsOnly() {
    const std::size_t counts[] = {5, 30, 133};
    for (std::size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c) {
        const dxapp::PoseResult pose = PoseOnCircle(counts[c], 0.9f);
        const cv::Mat canvas = DrawPose(pose, false);
        int lines = 0;
        for (std::size_t i = 0; i + 1 < pose.keypoints.size(); ++i) {
            const int k = static_cast<int>(i);
            if (PeakBetween(canvas, pose, k, k + 1) > 0) ++lines;
        }
        COMMON_CHECK(lines == 0 || counts[c] == 133);  // 133 points touch each other
        COMMON_CHECK(PeakBetween(canvas, pose, 0, static_cast<int>(counts[c] / 2)) == 0);
        COMMON_CHECK(PointsDrawn(canvas, pose) == static_cast<int>(counts[c]));
    }
}

/// SuperPoint's points: never joined, even 17 or 21 of them, and each one is
/// drawn - the model's own threshold already kept it.
void TestPoseVisualizerDrawsUnorderedPointsOnly() {
    const std::size_t counts[] = {17, 21, 68};
    for (std::size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c) {
        const dxapp::PoseResult pose = PoseOnCircle(counts[c], 0.05f);
        const cv::Mat canvas = DrawPose(pose, true);
        COMMON_CHECK(PeakBetween(canvas, pose, 0, 1) == 0);
        COMMON_CHECK(PeakBetween(canvas, pose, 5, 11) == 0);
        COMMON_CHECK(PeakBetween(canvas, pose, 0, 5) == 0);
        COMMON_CHECK(PointsDrawn(canvas, pose) == static_cast<int>(counts[c]));
    }
}
}  // namespace


// ---------------------------------------------------------------------------
// PicoDetPostprocessor: PP-PicoDet's per-level score and box tensors.
// ---------------------------------------------------------------------------

dxrt::TensorPtr PicoTensor(const char* name, int anchors, int cols, std::vector<float>& storage) {
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(anchors);
    shape.push_back(cols);
    return std::make_shared<dxrt::Tensor>(name, shape, dxrt::DataType::FLOAT, storage.data());
}

/// 64x64 input, strides 8/16/32/64 (grids 8/4/2/1), one object on the stride-16 level at
/// row 1, column 2 whose four sides all peak at bin 2 (reg_max 7): centre (40, 24), each
/// side 2 x 16 = 32 px away, so x1y1x2y2 (8, -8, 72, 56) before clipping to the 64x64 frame.
/// The tensors arrive boxes first and out of level order: they are paired by anchor count.
void TestPicoDetDecodesPerLevelTensors() {
    const int grids[4] = {8, 4, 2, 1};
    std::vector<std::vector<float> > scores(4), boxes(4);
    for (int l = 0; l < 4; ++l) {
        const int anchors = grids[l] * grids[l];
        scores[l].assign(anchors, 0.01f);
        boxes[l].assign(static_cast<std::size_t>(anchors) * 32, 0.f);
    }
    const int anchor = 1 * 4 + 2;              // stride-16 level, row 1, column 2
    scores[1][anchor] = 0.9f;                  // already sigmoid, as the export emits it
    for (int side = 0; side < 4; ++side) boxes[1][anchor * 32 + side * 8 + 2] = 12.f;

    dxrt::TensorPtrs outputs;
    outputs.push_back(PicoTensor("box16", 16, 32, boxes[1]));
    outputs.push_back(PicoTensor("score8", 64, 1, scores[0]));
    outputs.push_back(PicoTensor("box64", 1, 32, boxes[3]));
    outputs.push_back(PicoTensor("score16", 16, 1, scores[1]));
    outputs.push_back(PicoTensor("box8", 64, 32, boxes[0]));
    outputs.push_back(PicoTensor("score64", 1, 1, scores[3]));
    outputs.push_back(PicoTensor("score32", 4, 1, scores[2]));
    outputs.push_back(PicoTensor("box32", 4, 32, boxes[2]));

    dxapp::PreprocessContext ctx;
    ctx.original_width = 64;
    ctx.original_height = 64;
    std::vector<std::string> names(1, "mainbody");
    dxapp::PicoDetPostprocessor post(64, 64, 0.4f, 0.5f, 1, names);
    const std::vector<dxapp::DetectionResult> found = post.process(outputs, ctx);
    COMMON_CHECK(found.size() == 1);
    if (found.size() != 1) return;
    COMMON_CHECK(std::fabs(found[0].confidence - 0.9f) < 1e-6f);
    COMMON_CHECK(found[0].class_id == 0 && found[0].class_name == "mainbody");
    COMMON_CHECK(std::fabs(found[0].box[0] - 8.f) < 0.05f);
    COMMON_CHECK(found[0].box[1] == 0.f);                 // -8 clipped to the frame
    COMMON_CHECK(std::fabs(found[0].box[2] - 63.f) < 1e-4f);  // 72 clipped to width - 1
    COMMON_CHECK(std::fabs(found[0].box[3] - 56.f) < 0.05f);
}

/// NanoDet's single concatenated tensor is not PicoDet's packaging: no level pairs, no boxes.
void TestPicoDetIgnoresASingleConcatenatedTensor() {
    std::vector<float> storage(static_cast<std::size_t>(64) * 33, 0.9f);
    dxrt::TensorPtrs outputs;
    outputs.push_back(PicoTensor("nanodet", 64, 33, storage));
    dxapp::PreprocessContext ctx;
    ctx.original_width = 64;
    ctx.original_height = 64;
    dxapp::PicoDetPostprocessor post(64, 64);
    COMMON_CHECK(post.process(outputs, ctx).empty());
}

int main() {
    // Must stay first: see the ORDER NOTE on this test.
    TestAnchorFaceDebugHeaderOncePerInstance();
    TestPicoDetDecodesPerLevelTensors();
    TestPicoDetIgnoresASingleConcatenatedTensor();

    TestUnserializedProbeSeesOverlap();
    TestSerializedPostprocessorAdmitsOneCallAtATime();
    TestSerializedPostprocessorForwardsEveryVirtual();
    TestSerializedPostprocessorReleasesLockOnThrow();
    TestSerializeRejectsNull();

    TestClassificationSoftmaxProbabilitiesInRangeSumToOneOrderPreserved();
    TestClassificationTopClassNameMatchesImageNetLabelZero();
    TestClassificationDistributionGuardAvoidsDoubleSoftmax();
    TestClassificationNonImageNetClassCountLeavesNameEmpty();
    TestClassificationTopKFieldFilled();
    TestClassificationShortTensorNeverReadPast();
    TestClassificationIntegerOutputReadAtItsWidth();

    TestInterrupts();
    TestNoteInterruptRequestAtCoalescesThroughTheWindow();
    TestABrokenClockEscalatesEveryRepeat();
    TestConsoleCtrlDispatchRunsOnlyCtrlC();

    TestVerifyDumpsFromManyThreadsAreWholeAndNumbered();
    TestVerifyJsonKeepsItsFormatAndFramesFileAddsTheIndex();
    TestVerifySerializesDetection3D();
    TestVerifyIntoAFileInsteadOfADirectoryWarnsAndContinues();
    TestVerifyFramesFileThatCannotBeOpenedIsStartedByTheNextDump();
    TestVerifyFailedRenameLeavesTheTargetInPlace();

    TestYOLOPv2FrameMasksStayWithTheirFrame();
    TestYOLOPv2FrameWithoutVehiclesStillCarriesMasks();

    TestOrderedQueueDeliversShuffledPushesInTicketOrder();
    TestOrderedQueuePushBlocksWhenTooFarAhead();
    TestOrderedQueueGivesUpAMissingFrameAfterTheGapTimeout();
    TestOrderedQueueSkipReleasesTheNextFrame();
    TestOrderedQueueShutdownDeliversOnlyTheUnbrokenPrefix();
    TestOrderedQueueNeverDeadlocksWhenProducersWaitBehindAMissingFrame();
    TestOrderedQueueHoldsMoveOnlyItems();
    TestFrameReorderEmitsALateIndexAfterAGapFlush();

    TestVitPoseDecodesHeatmapPeaksToTheOriginalImage();
    TestVitPoseNameAndEmptyOutput();
    TestVitPoseFactoryWiresTheHeatmapDecoder();
    TestDarkHrnetFactoryWiresTheHeatmapDecoder();
    TestYolov5PoseFactoryDecodesTheAnchorDetectionsOutput();
    TestYolov5PoseFactoryDefaultsArePythons();
    TestYolov5PoseFactoryDefaultScoreIsPythons();

    TestSrMergeLumaIsTheRunnersMerge();
    TestSrPrepareLowResPadsAndConvertsLikeTheRunner();
    TestSrPipelinedTilesWaitForSubmittedJobsWhenRunAsyncThrows();
    TestSrStreamingAssemblyMatchesCollectThenAssemble();
    TestSrStreamingAssemblyKeepsNoFinishedOutputAlive();

    TestPaddedCropRectMatchesTheGraphRule();
    TestPaddedCropRectEdgeCases();

    TestModelConfigBracketsInsideStringsAreText();
    TestModelConfigDecodesJsonEscapes();
    TestModelConfigSkipsNestedObjectsWholeStringsIncluded();
    TestModelConfigDecodesTheRemainingEscapes();
    TestModelConfigAgreesWithAJsonParserOnEveryShape();
    TestModelConfigAgreesWithAJsonParserOnEveryShippedConfig();
    TestLoadOptionalConfigIsSilentWithoutAFile();
    TestRetrievalThumbnailsResolveAgainstTheRepository();
    TestLoadOptionalConfigWarnsOnAnUnreadableFile();

    TestCameraRaspberryPiModelString();
    TestCameraCsiNodeNames();
    TestCameraLibcameraIdComesFromTheCameraList();
    TestCameraLibcameraPipelineString();
    TestCameraGStreamerFromBuildInformation();
    TestCameraListingNeedsTheToolToFinish();
    TestCameraOffAPiOpensLikeVideoCapture();

    TestHandLandmarkKeypointsCarryPresenceNotDepth();
    TestHandLandmarkKeypointsWithoutPresenceAreFullyConfident();
    TestHandLandmarkDumpDoesNotReportConfidenceAsDepth();

    TestPoseVisualizerJoinsCocoBodyLimbs();
    TestPoseVisualizerDrawsTheFaceOutlineFor68Points();
    TestPoseVisualizerDrawsTheCenterPoseBox();
    TestPoseVisualizerDrawsOtherCountsAsPointsOnly();
    TestPoseVisualizerDrawsUnorderedPointsOnly();

    std::printf("%d checks, %d failures, %d skipped\n",
                g_checks, g_failures, g_skipped);
    return g_failures == 0 ? 0 : 1;
}
