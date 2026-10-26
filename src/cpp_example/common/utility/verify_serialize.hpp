/**
 * @file verify_serialize.hpp
 * @brief Serialize postprocess results to JSON for numerical verification.
 *
 * C++ port of Python's verify_serialize.py.
 * Activated by DXAPP_VERIFY=1 environment variable.
 * Writes the last frame to `{dir}/{model_stem}.json` and every frame to
 * `{dir}/{model_stem}.frames.jsonl` ({dir} = $DXAPP_VERIFY_DIR, default
 * logs/verify); writes are serialized and the JSON is replaced atomically.
 *
 * Requires nlohmann/json (bundled at common/third_party/nlohmann_json.hpp).
 */

#ifndef VERIFY_SERIALIZE_HPP
#define VERIFY_SERIALIZE_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <opencv2/opencv.hpp>
#include "common/third_party/nlohmann_json.hpp"
#include "common/base/i_processor.hpp"

namespace dxapp {
namespace verify {

using json = nlohmann::json;

// ============================================================================
// Environment check
// ============================================================================

inline bool isVerifyEnabled() {
    const char* v = std::getenv("DXAPP_VERIFY");
    return v && std::string(v) == "1";
}

inline std::string getVerifyDir() {
    const char* d = std::getenv("DXAPP_VERIFY_DIR");
    std::string dir = d ? d : "logs/verify";
    fs::create_directories(dir);
    return dir;
}

// ============================================================================
// cv::Mat statistics (mirrors Python's _np_stats)
// ============================================================================

inline json matStats(const cv::Mat& mat) {
    if (mat.empty()) return json::object();

    cv::Mat f64;
    mat.convertTo(f64, CV_64F);
    f64 = f64.reshape(1);  // flatten channels

    double minVal = 0;
    double maxVal = 0;
    cv::minMaxLoc(f64, &minVal, &maxVal);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(f64, mean, stddev);

    bool has_nan = false;
    bool has_inf = false;
    for (int r = 0; r < f64.rows; ++r) {
        const double* row = f64.ptr<double>(r);
        for (int c = 0; c < f64.cols; ++c) {
            if (std::isnan(row[c])) has_nan = true;
            if (std::isinf(row[c])) has_inf = true;
            if (has_nan && has_inf) break;
        }
        if (has_nan && has_inf) break;
    }

    std::vector<int> shape;
    shape.push_back(mat.rows);
    shape.push_back(mat.cols);
    if (mat.channels() > 1) shape.push_back(mat.channels());

    std::string dtype_str;
    if (mat.depth() == CV_32F) dtype_str = "float32";
    else if (mat.depth() == CV_64F) dtype_str = "float64";
    else dtype_str = "uint8";

    return {
        {"shape", shape},
        {"dtype", dtype_str},
        {"min", minVal},
        {"max", maxVal},
        {"mean", mean[0]},
        {"std", stddev[0]},
        {"has_nan", has_nan},
        {"has_inf", has_inf}
    };
}

// ============================================================================
// Serializers per result type
// ============================================================================

inline json serializeDetection(const std::vector<DetectionResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        dets.push_back({
            {"bbox", d.box},
            {"conf", d.confidence},
            {"class_id", d.class_id},
            {"class_name", d.class_name}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializeFace(const std::vector<FaceDetectionResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        json kpts = json::array();
        for (const auto& kp : d.landmarks) {
            kpts.push_back({{"x", kp.x}, {"y", kp.y}, {"conf", kp.confidence}});
        }
        dets.push_back({
            {"bbox", d.box},
            {"conf", d.confidence},
            {"keypoints", kpts}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializePose(const std::vector<PoseResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        json kpts = json::array();
        for (const auto& kp : d.keypoints) {
            kpts.push_back({{"x", kp.x}, {"y", kp.y}, {"conf", kp.confidence}});
        }
        dets.push_back({
            {"bbox", d.box},
            {"conf", d.confidence},
            {"keypoints", kpts}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializeInstanceSeg(const std::vector<InstanceSegmentationResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        bool has_mask = !d.mask.empty();
        std::vector<int> mask_shape;
        if (has_mask) { mask_shape.push_back(d.mask.rows); mask_shape.push_back(d.mask.cols); }
        dets.push_back({
            {"bbox", d.box},
            {"conf", d.confidence},
            {"class_id", d.class_id},
            {"class_name", d.class_name},
            {"has_mask", has_mask},
            {"mask_shape", mask_shape}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializeDepth(const std::vector<DepthResult>& items, int img_h, int img_w) {
    json result = {{"image_height", img_h}, {"image_width", img_w}};
    if (!items.empty()) {
        result["output_stats"] = matStats(items[0].depth_map);
    }
    return result;
}

inline json serializeAnomaly(const std::vector<AnomalyResult>& items, int img_h, int img_w) {
    json result = {{"image_height", img_h}, {"image_width", img_w}};
    if (!items.empty()) {
        result["output_stats"] = matStats(items[0].heatmap);
        result["score"] = items[0].score;
        result["channels"] = items[0].channels;
    }
    return result;
}

inline json serializeSegmentation(const std::vector<SegmentationResult>& items, int img_h, int img_w) {
    json result = {{"image_height", img_h}, {"image_width", img_w}};
    if (!items.empty()) {
        const auto& s = items[0];
        // Count unique classes
        std::set<int> unique_set(s.class_ids.begin(), s.class_ids.end());
        result["mask_shape"] = {s.height, s.width};
        result["unique_classes"] = static_cast<int>(unique_set.size());
        result["class_ids"] = s.class_ids;
    }
    return result;
}

inline json serializeClassification(const std::vector<ClassificationResult>& items, int img_h, int img_w) {
    json result = {{"image_height", img_h}, {"image_width", img_w}};
    if (!items.empty()) {
        const auto& c = items[0];
        result["classifications"] = json::array({
            {{"class_id", c.class_id}, {"class_name", c.class_name}, {"conf", c.confidence}}
        });
        json top_k_confs = json::array();
        for (const auto& p : c.top_k) {
            top_k_confs.push_back(p.second);
        }
        result["top_k_confs"] = top_k_confs;
    }
    return result;
}

inline json serializeOBB(const std::vector<OBBResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        dets.push_back({
            {"cx", d.cx}, {"cy", d.cy},
            {"width", d.width}, {"height", d.height},
            {"angle", d.angle},
            {"conf", d.confidence},
            {"class_id", d.class_id},
            {"class_name", d.class_name}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializeEmbedding(const std::vector<EmbeddingResult>& items, int img_h, int img_w) {
    json result = {{"image_height", img_h}, {"image_width", img_w}};
    if (!items.empty()) {
        const auto& e = items[0];
        double l2_norm = 0.0;
        bool has_nan = false;
        for (float v : e.embedding) {
            l2_norm += static_cast<double>(v) * v;
            if (std::isnan(v)) has_nan = true;
        }
        l2_norm = std::sqrt(l2_norm);
        result["embedding"] = {
            {"dim", e.dimension},
            {"l2_norm", l2_norm},
            {"has_nan", has_nan}
        };
        // Retrieval variants also carry a ranking. Emitted with the same keys the
        // Python tree's _ser_retrieval writes, so a cross-tree diff compares like
        // with like: a matching descriptor with a different order means the two
        // galleries differ, not the two models.
        if (!e.matches.empty() || !e.gallery_name.empty()) {
            result["gallery"] = {{"name", e.gallery_name},
                                 {"size", e.gallery_size}};
            json matches = json::array();
            for (const auto& m : e.matches) {
                matches.push_back({{"rank", m.rank},
                                   {"score", m.score},
                                   {"path", m.path},
                                   {"label", m.label}});
            }
            result["matches"] = matches;
        }
    }
    return result;
}

inline json serializeFaceAlignment(const std::vector<FaceAlignmentResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        json lm2d = json::array();
        for (const auto& kp : d.landmarks_2d) {
            lm2d.push_back({{"x", kp.x}, {"y", kp.y}, {"conf", kp.confidence}});
        }
        dets.push_back({
            {"landmarks_2d", lm2d},
            {"pose", d.pose},
            {"params_size", static_cast<int>(d.params.size())}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializeHandLandmark(const std::vector<HandLandmarkResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        json lm = json::array();
        for (const auto& kp : d.landmarks) {
            // Keypoint has no depth; its confidence is the hand's presence.
            lm.push_back({{"x", kp.x}, {"y", kp.y}});
        }
        dets.push_back({
            {"confidence", d.confidence},
            {"handedness", d.handedness},
            {"landmarks", lm}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializeRestoration(const std::vector<RestorationResult>& items, int img_h, int img_w) {
    json result = {{"image_height", img_h}, {"image_width", img_w}};
    if (!items.empty() && !items[0].restored_image.empty()) {
        const auto& r = items[0];
        result["output_shape"] = {r.restored_image.rows, r.restored_image.cols, r.restored_image.channels()};
        result["input_image_shape"] = {img_h, img_w};
        result["output_stats"] = matStats(r.restored_image);
    }
    return result;
}

inline json serializeDetection3D(const std::vector<Detection3DResult>& items, int img_h, int img_w) {
    json dets = json::array();
    for (const auto& d : items) {
        dets.push_back({
            {"class_id", d.class_id},
            {"class_name", d.class_name},
            {"conf", d.confidence},
            {"bev", {d.bev_x, d.bev_y, d.bev_w, d.bev_h}},
            {"center", {d.x3d, d.y3d, d.z3d}},
            {"dims", {d.dim_h, d.dim_w, d.dim_l}},
            {"yaw", d.yaw}
        });
    }
    return {{"image_height", img_h}, {"image_width", img_w}, {"detections", dets}};
}

inline json serializePanoptic(const std::vector<PanopticResult>& items, int img_h, int img_w) {
    if (items.empty()) return serializeDetection(std::vector<DetectionResult>(), img_h, img_w);
    json result = serializeDetection(items.front().detections, img_h, img_w);
    result["drivable_stats"] = matStats(items.front().drivable);
    result["lane_stats"] = matStats(items.front().lane);
    return result;
}

// ============================================================================
// Internal write helper
// ============================================================================

namespace detail {

/// Every verify write in the process goes through this lock: async runners
/// used to dump from dxrt's completion threads, and two frames' truncate-
/// and-rewrite of one file could interleave.
struct VerifyWriteState {
    std::mutex mutex;
    std::map<std::string, long long> next_frame;  // per .frames.jsonl path
};

inline VerifyWriteState& verifyWriteState() {
    static VerifyWriteState state;
    return state;
}

inline long long currentProcessId() {
#ifdef _WIN32
    return static_cast<long long>(_getpid());
#else
    return static_cast<long long>(getpid());
#endif
}

/// Writes `text` to a sibling temp file and renames it over `path`, so a
/// reader sees the previous file or the new one, never a torn one.
inline bool writeFileAtomically(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp." + std::to_string(currentProcessId());
    {
        std::ofstream ofs(tmp, std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) return false;
        ofs << text;
        ofs.close();
        if (!ofs) {
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
#ifdef _WIN32
        std::remove(path.c_str());  // Windows: rename() does not replace a file
        if (std::rename(tmp.c_str(), path.c_str()) == 0) return true;
#endif
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

/// `<stem>.json` holds the last dumped frame (unchanged format);
/// `<stem>.frames.jsonl` gets one record per dump with "frame": 0, 1, 2, ...
inline std::string writeVerifyJson(
    json data,
    const std::string& modelPath,
    const std::string& task) {

    data["task"] = task;
    data["model"] = fs::path(modelPath).filename().string();
    data["model_path"] = modelPath;

    VerifyWriteState& state = verifyWriteState();
    std::lock_guard<std::mutex> lock(state.mutex);

    const std::string verifyDir = getVerifyDir();
    const std::string modelStem = fs::path(modelPath).stem().string();
    const std::string jsonPath = verifyDir + "/" + modelStem + ".json";
    const std::string framesPath = verifyDir + "/" + modelStem + ".frames.jsonl";

    if (!writeFileAtomically(jsonPath, data.dump(2) + "\n")) {
        std::cerr << "[DXAPP] [WARN] verify_serialize: cannot write " << jsonPath << std::endl;
        return "";
    }

    auto found = state.next_frame.find(framesPath);
    const bool first = (found == state.next_frame.end());
    const long long frame = first ? 0 : found->second;
    json record = data;
    record["frame"] = frame;
    std::ofstream frames(framesPath, first ? (std::ios::out | std::ios::trunc)
                                           : (std::ios::out | std::ios::app));
    if (frames.is_open()) frames << record.dump() << "\n" << std::flush;
    if (frames.is_open() && frames) {
        // Only a written record uses up its number: until the first write
        // succeeds, the next dump still truncates (never appends to an old run).
        state.next_frame[framesPath] = frame + 1;
    } else {
        std::cerr << "[DXAPP] [WARN] verify_serialize: cannot write " << framesPath << std::endl;
    }

    std::cout << "[VERIFY] Dumped -> " << jsonPath << std::endl;
    return jsonPath;
}

}  // namespace detail

// ============================================================================
// Overloaded dump functions per result type
// ============================================================================

#define DXAPP_VERIFY_DUMP_IMPL(ResultType, serializeFn) \
inline std::string dumpVerifyJson( \
    const std::vector<ResultType>& results, \
    const std::string& modelPath, \
    const std::string& task, \
    int img_h, int img_w) { \
    if (!isVerifyEnabled()) return ""; \
    try { \
        return detail::writeVerifyJson(serializeFn(results, img_h, img_w), modelPath, task); \
    } catch (const std::exception& e) { \
        std::cerr << "[DXAPP] [WARN] verify_serialize failed: " << e.what() << std::endl; \
        return ""; \
    } \
}

DXAPP_VERIFY_DUMP_IMPL(DetectionResult, serializeDetection)
DXAPP_VERIFY_DUMP_IMPL(FaceDetectionResult, serializeFace)
DXAPP_VERIFY_DUMP_IMPL(PoseResult, serializePose)
DXAPP_VERIFY_DUMP_IMPL(InstanceSegmentationResult, serializeInstanceSeg)
DXAPP_VERIFY_DUMP_IMPL(DepthResult, serializeDepth)
DXAPP_VERIFY_DUMP_IMPL(AnomalyResult, serializeAnomaly)
DXAPP_VERIFY_DUMP_IMPL(SegmentationResult, serializeSegmentation)
DXAPP_VERIFY_DUMP_IMPL(ClassificationResult, serializeClassification)
DXAPP_VERIFY_DUMP_IMPL(OBBResult, serializeOBB)
DXAPP_VERIFY_DUMP_IMPL(EmbeddingResult, serializeEmbedding)
DXAPP_VERIFY_DUMP_IMPL(FaceAlignmentResult, serializeFaceAlignment)
DXAPP_VERIFY_DUMP_IMPL(HandLandmarkResult, serializeHandLandmark)
DXAPP_VERIFY_DUMP_IMPL(RestorationResult, serializeRestoration)
DXAPP_VERIFY_DUMP_IMPL(Detection3DResult, serializeDetection3D)
DXAPP_VERIFY_DUMP_IMPL(PanopticResult, serializePanoptic)

#undef DXAPP_VERIFY_DUMP_IMPL

}  // namespace verify
}  // namespace dxapp

#endif  // VERIFY_SERIALIZE_HPP
