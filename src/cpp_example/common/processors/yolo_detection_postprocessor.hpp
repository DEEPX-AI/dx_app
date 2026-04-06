/**
 * @file yolo_detection_postprocessor.hpp
 * @brief Unified YOLO Detection Postprocessors for v3 interface
 * 
 * Groups all YOLO-based object detection postprocessors:
 *   - YOLOv5 family (6 args): YOLOv5, YOLOv7, YOLOX
 *   - YOLOv8 family (5 args): YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLOv26
 */

#ifndef YOLO_DETECTION_POSTPROCESSOR_HPP
#define YOLO_DETECTION_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/result_converters.hpp"

// Merged postprocess headers
#include "anchor_detection_postprocessor.hpp"
#include "anchorless_dfl_detection_postprocessor.hpp"

namespace dxapp {

// ============================================================================
// Base template for coordinate scaling (shared by all detection postprocessors)
// ============================================================================
namespace detail {

inline void scaleDetectionResults(std::vector<DetectionResult>& results,
                                   const PreprocessContext& ctx) {
    for (auto& det : results) {
        scaleBox(det.box, ctx);
    }
}

}  // namespace detail

// ============================================================================
// Unified Anchor-YOLO Postprocessor
//
// Single class that handles all anchor-based YOLO models (v3/v4/v5/v6/v7/X).
// Preset enum selects the correct anchors, NPU tensor names, and thresholds.
// Backward-compatible aliases (YOLOv5Postprocessor, etc.) are provided below.
// ============================================================================

class AnchorYOLOPostprocessor : public IPostprocessor<DetectionResult> {
public:
    /** @brief Model presets — selects anchors, NPU output names, defaults */
    enum class Preset { YOLOV5, YOLOV7, YOLOX };

    /**
     * @brief Construct with explicit preset.
     * @param preset   YOLOV5 | YOLOV7 | YOLOX
     * @param input_width   Model input width  (default set per preset)
     * @param input_height  Model input height (default set per preset)
     * @param obj_threshold objectness threshold (default set per preset)
     * @param score_threshold class score threshold
     * @param nms_threshold  NMS IoU threshold
     * @param is_ort_configured  ORT mode flag
     */
    AnchorYOLOPostprocessor(Preset preset,
                            int input_width, int input_height,
                            float obj_threshold,
                            float score_threshold,
                            float nms_threshold,
                            bool is_ort_configured)
        : impl_(input_width, input_height,
                obj_threshold, score_threshold, nms_threshold,
                is_ort_configured,
                cpu_names_for(preset),
                npu_names_for(preset),
                anchors_for(preset),
                npu_supported_for(preset)),
          preset_(preset) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        std::vector<DetectionResult> results = convertAll(legacy_results);
        detail::scaleDetectionResults(results, ctx);
        return results;
    }

    std::string getModelName() const override {
        switch (preset_) {
            case Preset::YOLOV7: return "YOLOv7";
            case Preset::YOLOX:  return "YOLOX";
            default:             return "YOLOv5";
        }
    }

    Preset getPreset() const { return preset_; }

private:
    AnchorYOLOPostProcess impl_;
    Preset preset_;

    // ---- Preset look-up helpers (inline, header-only) ----------------------
    using AnchorMap = std::map<int, std::vector<std::pair<int,int>>>;

    static std::vector<std::string> cpu_names_for(Preset) {
        return {"output"};
    }

    static std::vector<std::string> npu_names_for(Preset p) {
        switch (p) {
            case Preset::YOLOV5: return {"378", "439", "500"};
            case Preset::YOLOV7: return {"onnx::Reshape_491", "onnx::Reshape_525",
                                         "onnx::Reshape_559"};
            case Preset::YOLOX:  return {};
        }
        return {};
    }

    static AnchorMap anchors_for(Preset p) {
        switch (p) {
            case Preset::YOLOV5:
                return {{8,  {{10,13},{16,30},{33,23}}},
                        {16, {{30,61},{62,45},{59,119}}},
                        {32, {{116,90},{156,198},{373,326}}}};
            case Preset::YOLOV7:
                return {{8,  {{12,16},{19,36},{40,28}}},
                        {16, {{36,75},{76,55},{72,146}}},
                        {32, {{142,110},{192,243},{459,401}}}};
            case Preset::YOLOX:
                return {{8, {}}, {16, {}}, {32, {}}};
        }
        return {};
    }

    static bool npu_supported_for(Preset p) {
        return p != Preset::YOLOX;
    }
};

// ============================================================================
// YOLOv8 Family Postprocessor Template
//
// Each YOLOv8+ variant (v8/v9/v10/v11/v12/v26) has its own legacy
// PostProcess class with different tensor parsing logic.
// This template provides the common process() → legacy → convert → scale flow.
// Subclasses override getModelName() only.
// ============================================================================
template<typename LegacyPostProcess>
class YOLOv8FamilyPostprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOv8FamilyPostprocessor(int input_width = 640, int input_height = 640,
                              float score_threshold = 0.3f,
                              float nms_threshold = 0.45f,
                              bool is_ort_configured = false)
        : impl_(input_width, input_height, score_threshold, nms_threshold,
                is_ort_configured) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        std::vector<DetectionResult> results = convertAll(legacy_results);
        detail::scaleDetectionResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLO"; }

private:
    LegacyPostProcess impl_;
};

// ============================================================================
// ============================================================================
// Backward-compatible Alias Subclasses (anchor-based)
//
// These preserve the old API so that all existing factories continue to
// compile without changes:
//   make_unique<YOLOv5Postprocessor>(w, h, obj, score, nms, ort)
// ============================================================================

class YOLOv5Postprocessor : public AnchorYOLOPostprocessor {
public:
    YOLOv5Postprocessor(int w = 640, int h = 640,
                        float obj = 0.25f, float score = 0.3f, float nms = 0.45f,
                        bool ort = false)
        : AnchorYOLOPostprocessor(Preset::YOLOV5, w, h, obj, score, nms, ort) {}
};

class YOLOv7Postprocessor : public AnchorYOLOPostprocessor {
public:
    YOLOv7Postprocessor(int w = 640, int h = 640,
                        float obj = 0.3f, float score = 0.4f, float nms = 0.5f,
                        bool ort = false)
        : AnchorYOLOPostprocessor(Preset::YOLOV7, w, h, obj, score, nms, ort) {}
};

class YOLOXPostprocessor : public AnchorYOLOPostprocessor {
public:
    YOLOXPostprocessor(int w = 512, int h = 512,
                       float obj = 0.25f, float score = 0.3f, float nms = 0.45f,
                       bool ort = true)
        : AnchorYOLOPostprocessor(Preset::YOLOX, w, h, obj, score, nms, ort) {}
};

// YOLOv8 Family (5 args)
class YOLOv8Postprocessor : public YOLOv8FamilyPostprocessor<YOLOv8PostProcess> {
public:
    using YOLOv8FamilyPostprocessor::YOLOv8FamilyPostprocessor;
    std::string getModelName() const override { return "YOLOv8"; }
};

class YOLOv9Postprocessor : public YOLOv8FamilyPostprocessor<YOLOv9PostProcess> {
public:
    using YOLOv8FamilyPostprocessor::YOLOv8FamilyPostprocessor;
    std::string getModelName() const override { return "YOLOv9"; }
};

class YOLOv10Postprocessor : public YOLOv8FamilyPostprocessor<YOLOv10PostProcess> {
public:
    using YOLOv8FamilyPostprocessor::YOLOv8FamilyPostprocessor;
    std::string getModelName() const override { return "YOLOv10"; }
};

class YOLOv11Postprocessor : public YOLOv8FamilyPostprocessor<YOLOv11PostProcess> {
public:
    using YOLOv8FamilyPostprocessor::YOLOv8FamilyPostprocessor;
    std::string getModelName() const override { return "YOLOv11"; }
};

class YOLOv12Postprocessor : public YOLOv8FamilyPostprocessor<YOLOv12PostProcess> {
public:
    using YOLOv8FamilyPostprocessor::YOLOv8FamilyPostprocessor;
    std::string getModelName() const override { return "YOLOv12"; }
};

class YOLOv26Postprocessor : public YOLOv8FamilyPostprocessor<YOLOv26PostProcess> {
public:
    using YOLOv8FamilyPostprocessor::YOLOv8FamilyPostprocessor;
    std::string getModelName() const override { return "YOLOv26"; }
};

}  // namespace dxapp

#endif  // YOLO_DETECTION_POSTPROCESSOR_HPP
